from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import queue
import sqlite3

import numpy as np
import pandas as pd
import torch

from .config import DEVICE, INPUT_FEATURES, PROCESS_FEATURES, TARGETS, WINDOW_EVENTS
from .monitoring import SimpleDriftMonitor


class SQLiteEventStore:
    """Persist source events and model predictions in SQLite."""

    def __init__(self, db_path="physical_ai_stream.sqlite"):
        self.db_path = str(db_path)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prediction_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def write_event(self, event_id, job_id, timestamp, payload):
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?)",
                (str(event_id), str(job_id), str(timestamp), json.dumps(payload, default=str)),
            )

    def write_prediction(self, event_id, job_id, timestamp, prediction):
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?)",
                (str(event_id), str(job_id), str(timestamp), json.dumps(prediction, default=str)),
            )

    def read_predictions(self, limit=20):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, job_id, timestamp, prediction_json "
                "FROM predictions ORDER BY rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"event_id": row[0], "job_id": row[1], "timestamp": row[2], **json.loads(row[3])}
            for row in rows
        ]


class StreamingLTService:
    """Validate incoming events, run V5 inference, and persist results."""

    def __init__(
        self,
        model,
        x_scaler,
        y_scaler,
        physics_anchor,
        reference_X,
        db_path="physical_ai_stream.sqlite",
        window_events=WINDOW_EVENTS,
        drift_threshold=3.0,
    ):
        self.model = model.eval()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.physics_anchor = physics_anchor
        self.window_events = int(window_events)
        self.store = SQLiteEventStore(db_path)
        self.buffers = defaultdict(lambda: deque(maxlen=self.window_events))
        self.last_time = {}
        self.event_counter = 0
        self.monitor = SimpleDriftMonitor(reference_X, threshold=drift_threshold)

    @staticmethod
    def _timestamp(event):
        value = event.get("timestamp")
        return datetime.now(timezone.utc).isoformat() if value is None else str(value)

    def _delta_t(self, event, job_id, timestamp):
        previous = self.last_time.get(job_id)
        self.last_time[job_id] = timestamp
        if event.get("delta_t") is not None:
            return max(0.0, float(event["delta_t"]))
        if previous is None:
            return 0.0
        try:
            return max(0.0, float((pd.Timestamp(timestamp) - pd.Timestamp(previous)).total_seconds()))
        except Exception:
            return 0.0

    @torch.no_grad()
    def process_event(self, event):
        event = dict(event)
        job_id = str(event.get("job_id", "job_default"))
        timestamp = self._timestamp(event)
        self.event_counter += 1
        event_id = str(event.get("event_id", f"{job_id}:{event.get('global_index', self.event_counter)}"))
        required = PROCESS_FEATURES + ["sensor_1", "sensor_2", "sensor_3"]
        missing = [name for name in required if name not in event]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        clean = dict(event)
        clean.update({"job_id": job_id, "timestamp": timestamp})
        clean["delta_t"] = self._delta_t(event, job_id, timestamp)
        for name in INPUT_FEATURES:
            clean[name] = float(clean[name])
            if not np.isfinite(clean[name]):
                raise ValueError(f"Non-finite numeric input: {name}")

        anchor_raw = self.physics_anchor.predict(pd.DataFrame([clean], columns=PROCESS_FEATURES))
        anchor_scaled = self.y_scaler.transform(np.log1p(anchor_raw)).astype(np.float32)
        input_raw = np.array([[clean[name] for name in INPUT_FEATURES]], dtype=float)
        input_scaled = self.x_scaler.transform(input_raw).astype(np.float32)
        clean.update({f"z_{name}": float(input_scaled[0, i]) for i, name in enumerate(INPUT_FEATURES)})
        self.buffers[job_id].append(clean)
        history = list(self.buffers[job_id])
        sequence = np.array(
            [[row[f"z_{name}"] for name in INPUT_FEATURES] for row in history], dtype=np.float32
        )
        x_tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)
        mask = torch.ones((1, len(history)), dtype=torch.bool, device=DEVICE)
        anchor_tensor = torch.from_numpy(anchor_scaled).to(DEVICE)
        prediction_scaled, log_variance, loops_scaled = self.model(
            x_tensor, mask, anchor_tensor, return_loop_preds=True
        )

        prediction_log = self.y_scaler.inverse_transform(prediction_scaled.cpu().numpy())
        prediction_raw = np.maximum(np.expm1(prediction_log), 0.0)[0]
        loop_log = self.y_scaler.inverse_transform(loops_scaled[0].cpu().numpy())
        loop_raw = np.maximum(np.expm1(loop_log), 0.0)
        sigma_std = torch.exp(0.5 * log_variance).cpu().numpy()[0]
        sigma_log1p = sigma_std * self.y_scaler.scale_
        sigma_raw = np.exp(prediction_log[0]) * sigma_log1p
        drift = self.monitor.update(np.array([clean[name] for name in INPUT_FEATURES], dtype=float))

        output = {
            "event_id": event_id,
            "job_id": job_id,
            "timestamp": timestamp,
            "history_events": len(history),
            "anchor": {TARGETS[i]: float(anchor_raw[0, i]) for i in range(3)},
            "prediction": {TARGETS[i]: float(prediction_raw[i]) for i in range(3)},
            "uncertainty_sigma_approx": {TARGETS[i]: float(sigma_raw[i]) for i in range(3)},
            "loop_predictions": [
                {TARGETS[i]: float(loop_raw[loop, i]) for i in range(3)}
                for loop in range(loop_raw.shape[0])
            ],
            "drift_score": float(drift["score"]),
            "drift_warning": bool(drift["warning"]),
            "model_updated": False,
        }
        payload = {key: value for key, value in clean.items() if not key.startswith("z_")}
        self.store.write_event(event_id, job_id, timestamp, payload)
        self.store.write_prediction(event_id, job_id, timestamp, output)
        return output


def run_stream_consumer(service, event_queue, stop_event, result_queue=None, timeout_s=0.5):
    """Blocking consumer for a worker thread; it is not started implicitly."""
    while not stop_event.is_set():
        try:
            event = event_queue.get(timeout=timeout_s)
        except queue.Empty:
            continue
        try:
            result = service.process_event(event)
            if result_queue is not None:
                result_queue.put(result)
        except Exception as exc:
            if result_queue is not None:
                result_queue.put({
                    "event_id": event.get("event_id"),
                    "error": repr(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        finally:
            event_queue.task_done()