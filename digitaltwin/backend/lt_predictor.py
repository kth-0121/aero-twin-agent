import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

from .lt_contracts import GeometryPrediction, PROCESS_FEATURES
from .sensors import SyntheticSensorProvider


class V5StreamingLTPredictor:
    """Adapt the trained V5 streaming service to the digital-twin contract."""

    def __init__(self, service, sensor_provider):
        self.service = service
        self.sensor_provider = sensor_provider
        self.event_counter = 0

    @classmethod
    def from_artifacts(cls, artifact_dir=None, db_path=None):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import joblib
        import torch

        from model.LoopedTransformer.v5.config import (
            DEVICE,
            INPUT_FEATURES,
            PROCESS_FEATURES as MODEL_PROCESS_FEATURES,
            TARGETS,
            WINDOW_EVENTS,
            SEED,
        )
        from model.LoopedTransformer.v5.data import load_dataset
        from model.LoopedTransformer.v5.model import AdaptivePhysicsResidualLT
        from model.LoopedTransformer.v5.physics import RosenthalPhysicsAnchor
        from model.LoopedTransformer.v5.service import StreamingLTService
        from model.LoopedTransformer.v5.simulation import make_simulated_stream
        from model.LoopedTransformer.v5.windows import SEQUENCE_FEATURES

        if tuple(MODEL_PROCESS_FEATURES) != PROCESS_FEATURES:
            raise ValueError("Digital-twin process features do not match the LT model")
        artifact_dir = Path(
            artifact_dir
            or os.environ.get("LT_ARTIFACT_DIR", repo_root / "model/LoopedTransformer/V5_stream")
        ).resolve()
        checkpoint_path = artifact_dir / "looped_transformer_v5_st.pt"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        if checkpoint.get("input_features") != INPUT_FEATURES:
            raise ValueError("LT checkpoint input feature contract does not match this code")
        if checkpoint.get("targets") != TARGETS:
            raise ValueError("LT checkpoint target contract does not match this code")
        if checkpoint.get("seq_features") != SEQUENCE_FEATURES:
            raise ValueError("LT checkpoint sequence feature contract does not match this code")

        x_scaler = joblib.load(artifact_dir / "x_scaler.joblib")
        y_scaler = joblib.load(artifact_dir / "y_scaler.joblib")
        if list(x_scaler.feature_names_in_) != INPUT_FEATURES:
            raise ValueError("LT input scaler feature order does not match the checkpoint")
        if list(y_scaler.feature_names_in_) != TARGETS:
            raise ValueError("LT output scaler target order does not match the checkpoint")

        model = AdaptivePhysicsResidualLT(
            input_dim=checkpoint["input_dim"],
            hidden=checkpoint["hidden"],
            n_loops=checkpoint["n_loops"],
            max_delta=checkpoint["max_delta"],
        ).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        dataset = load_dataset()
        replay_reference = make_simulated_stream(dataset, n_jobs=8, seed=SEED)
        reference_x = replay_reference[INPUT_FEATURES].to_numpy(dtype=float)
        anchor = RosenthalPhysicsAnchor(checkpoint["eta_effective"])
        service = StreamingLTService(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            physics_anchor=anchor,
            reference_X=reference_x,
            db_path=db_path or os.environ.get("LT_EVENT_DB_PATH", "physical_ai_stream.sqlite"),
            window_events=checkpoint.get("window_events", WINDOW_EVENTS),
        )
        sensor_provider = SyntheticSensorProvider(
            dataset[list(PROCESS_FEATURES)].to_numpy(dtype=float), seed=SEED
        )
        return cls(service, sensor_provider)

    def predict(self, event: Mapping[str, object]) -> GeometryPrediction:
        from model.LoopedTransformer.v5.config import TARGETS

        self.event_counter += 1
        process = {name: float(event[name]) for name in PROCESS_FEATURES}
        sensor_values = self.sensor_provider.observe(process)
        service_event = {
            **event,
            **process,
            **sensor_values,
            "event_id": str(event.get("event_id", f"dt:{self.event_counter}")),
            "job_id": str(event.get("job_id", "digital-twin")),
        }
        result = self.service.process_event(service_event)
        prediction = result["prediction"]
        width = float(prediction[TARGETS[0]])
        depth = float(prediction[TARGETS[1]])
        area = float(prediction[TARGETS[2]])
        if not np.isfinite([width, depth, area]).all() or min(width, depth, area) <= 0:
            raise RuntimeError("LT returned non-finite or non-positive geometry")

        return GeometryPrediction(
            width_um=width,
            depth_um=depth,
            area_um2=area,
            drift_score=float(result["drift_score"]),
            drift_warning=bool(result["drift_warning"]),
            uncertainty_sigma_approx=result["uncertainty_sigma_approx"],
            loop_predictions=tuple(result["loop_predictions"]),
            sensor_source=self.sensor_provider.source,
        )