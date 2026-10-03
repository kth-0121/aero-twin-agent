import asyncio
import math
import tempfile
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..build_job import BuildPlan, create_build_plan
from ..fipy_dt import FiPyThermalTwin
from ..process import demo_316l_settings
from .lt_contracts import GeometryPredictor, LTRequest
from .lt_predictor import V5StreamingLTPredictor
from sim.decision import recommend_action_from_prediction
from sim.lpbf import evaluate_physical_state

app = FastAPI(title="Physical AI Digital Twin")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@dataclass
class EquipmentState:
    timestamp: float
    equipment_id: str
    state: str
    position: dict
    velocity: dict
    force: dict
    process: dict
    build: dict
    machine: dict
    physical_state: dict
    decision: dict


class BuildReview(BaseModel):
    approved: bool


class ControlUpdate(BaseModel):
    running: bool | None = None
    laser_power_w: float | None = Field(default=None, ge=50, le=500)
    scan_speed_mm_s: float | None = Field(default=None, ge=100, le=2000)
    mode: Literal["measured_replay", "what_if"] | None = None


class LPBFCellSimulation:
    def __init__(self, geometry_predictor: GeometryPredictor):
        self.geometry_predictor = geometry_predictor
        self.elapsed_s = 0.0
        self.event_counter = 0
        self.start_layer = 41
        self.layer_count = 120
        self.track_count = 120
        self.track_length_mm = 76.0
        self.track_pitch_mm = 0.6
        self.laser_power_w = 250.0
        self.scan_speed_mm_s = 800.0
        self.layer_path_mm = self.track_count * self.track_length_mm
        self.running = False
        self.replay_row_provider = getattr(geometry_predictor, "next_replay_row", None)
        self.mode = "measured_replay" if getattr(geometry_predictor, "replay_rows", ()) else "what_if"
        self.replay_index = 0
        self._lock = Lock()
        self._latest_state = None

    def configure(self, running=None, laser_power_w=None, scan_speed_mm_s=None, mode=None):
        with self._lock:
            if running is not None:
                self.running = running
            if mode is not None:
                self.mode = mode
            elif laser_power_w is not None or scan_speed_mm_s is not None:
                self.mode = "what_if"
            if laser_power_w is not None:
                self.laser_power_w = laser_power_w
            if scan_speed_mm_s is not None:
                self.scan_speed_mm_s = scan_speed_mm_s
            return {
                "running": self.running,
                "laser_power_w": self.laser_power_w,
                "scan_speed_mm_s": self.scan_speed_mm_s,
                "mode": self.mode,
            }

    def controls(self):
        with self._lock:
            return {
                "running": self.running,
                "laser_power_w": self.laser_power_w,
                "scan_speed_mm_s": self.scan_speed_mm_s,
                "mode": self.mode,
            }

    def step(self, dt=1/30):
        with self._lock:
            if not self.running and self._latest_state is not None:
                return replace(
                    self._latest_state,
                    state="PAUSED",
                    process={
                        **self._latest_state.process,
                        "laser_power": self.laser_power_w,
                        "scan_speed": self.scan_speed_mm_s,
                    },
                    machine={**self._latest_state.machine, "laser": "STANDBY"},
                )
            if self.running:
                self.elapsed_s += dt
            elapsed_s = self.elapsed_s
            laser_power_w = self.laser_power_w
            scan_speed_mm_s = self.scan_speed_mm_s
            running = self.running
            mode = self.mode

        measured = None
        if mode == "measured_replay" and self.replay_row_provider is not None:
            measured = self.replay_row_provider()
            if measured is None:
                mode = "what_if"
            else:
                self.replay_index += 1
                laser_power_w = float(measured["P_laser"])
                scan_speed_mm_s = float(measured["v_scan"])
        if measured is None:
            measured = {}

        total_distance_mm = elapsed_s * scan_speed_mm_s
        completed_layers = int(total_distance_mm // self.layer_path_mm)
        job_cycle, layer_index = divmod(self.start_layer + completed_layers, self.layer_count)
        layer_offset_mm = total_distance_mm % self.layer_path_mm
        track_index = int(layer_offset_mm // self.track_length_mm)
        scan_offset_mm = layer_offset_mm % self.track_length_mm
        reverse = (track_index + layer_index) % 2 == 1
        x = 12.0 + (self.track_length_mm - scan_offset_mm if reverse else scan_offset_mm)
        y = 12.0 + track_index * self.track_pitch_mm
        z = (layer_index + 1) * 0.03
        layer_progress = layer_offset_mm / self.layer_path_mm
        build_progress = (layer_index + layer_progress) / self.layer_count * 100
        chamber_temperature = 28.0 + 0.35 * math.sin(elapsed_s / 18)
        oxygen_ppm = 82.0 + 3.0 * math.sin(elapsed_s / 7)
        timestamp = time.time()
        job_id = f"LPBF-316L-{42 + job_cycle:03d}"
        self.event_counter += 1
        process_input = {
            "P_laser": laser_power_w,
            "v_scan": scan_speed_mm_s,
            "d_laser": float(measured.get("d_laser", 0.1)),
            "t_powder": float(measured.get("t_powder", 30.0)),
        }
        geometry = self.geometry_predictor.predict({
            **process_input,
            "event_id": f"{job_id}:{self.event_counter}",
            "job_id": job_id,
            "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "delta_t": dt,
            **(
                {"source_row_index": int(measured.get("source_row_index", self.replay_index - 1))}
                if measured else {}
            ),
        })
        physical_state = evaluate_physical_state(
            laser_power_w,
            scan_speed_mm_s,
            process_input["d_laser"],
            100.0,
            process_input["t_powder"],
            geometry.width_um,
            geometry.depth_um,
        )
        decision = recommend_action_from_prediction(
            laser_power_w,
            scan_speed_mm_s,
            process_input["d_laser"],
            100.0,
            process_input["t_powder"],
            geometry.width_um,
            geometry.depth_um,
        )
        prediction_fields = geometry.as_process_fields()
        if measured:
            prediction_fields["measured_geometry"] = {
                "source_row_index": int(measured.get("source_row_index", self.replay_index - 1)),
                "melt_width_um": float(measured["melt_width"]),
                "melt_depth_um": float(measured["melt_depth"]),
                "melt_area_um2": float(measured["melt_area"]),
            }

        state = EquipmentState(
            timestamp=timestamp,
            equipment_id="LBFP-01",
            state="RUNNING" if running else "PAUSED",
            position={"x": x, "y": y, "z": z},
            velocity={
                "x": -scan_speed_mm_s if reverse else scan_speed_mm_s,
                "y": 0.0,
                "z": 0.0,
            },
            force={"fx": 0.0, "fy": 0.0, "fz": 0.0},
            process={
                "laser_power": process_input["P_laser"],
                "scan_speed": scan_speed_mm_s,
                "spot_diameter": process_input["d_laser"],
                "powder_layer_thickness_um": process_input["t_powder"],
                "track_spacing": 100.0,
                **prediction_fields,
            },
            build={
                "job_id": job_id,
                "material": "316L",
                "layer_current": layer_index + 1,
                "layer_total": self.layer_count,
                "track_current": track_index + 1,
                "track_total": self.track_count,
                "progress_pct": min(build_progress, 100.0),
                "elapsed_s": 468.0 + elapsed_s,
            },
            machine={
                "source": "316L_MEASURED_REPLAY" if measured else "WHAT_IF",
                "laser": "SCANNING" if running else "STANDBY",
                "recoater": "STANDBY",
                "chamber_temperature": chamber_temperature,
                "oxygen_ppm": oxygen_ppm,
                "filter_load_pct": 68.0,
                "interlock": "CLOSED",
            },
            physical_state=physical_state,
            decision=decision,
        )
        with self._lock:
            self._latest_state = state
        return state


@dataclass
class BuildReplaySession:
    build_id: str
    plan: BuildPlan
    events: tuple
    source_units: str
    phase: str = "sliced"
    thermal_cursor: int = 0
    agent_cursor: int = 0
    thermal_twin: FiPyThermalTwin | None = None
    thermal_hot_fraction_sum: float = 0.0
    thermal_melt_threshold_fraction_sum: float = 0.0
    risk_counts: dict[str, int] = field(
        default_factory=lambda: {"normal": 0, "lof_risk": 0, "keyhole_risk": 0}
    )
    action_votes: dict[tuple[str, float, float], dict[str, float]] = field(default_factory=dict)
    risk_history: list[str] = field(default_factory=list)
    last_result: dict | None = None
    recommendation: dict | None = None

    def status(self):
        event_count = len(self.events)
        thermal_count = self.thermal_cursor
        return {
            "build_id": self.build_id,
            "status": self.phase,
            "phase": self.phase,
            "event_index": self.agent_cursor,
            "event_count": event_count,
            "thermal_event_index": self.thermal_cursor,
            "thermal_progress_pct": self.thermal_cursor / event_count * 100 if event_count else 0.0,
            "agent_progress_pct": self.agent_cursor / event_count * 100 if event_count else 0.0,
            "thermal_summary": {
                "hot_cell_fraction_avg": (
                    self.thermal_hot_fraction_sum / thermal_count if thermal_count else None
                ),
                "reference_threshold_fraction_avg": (
                    self.thermal_melt_threshold_fraction_sum / thermal_count
                    if thermal_count else None
                ),
            },
            "layer_count": self.plan.cli_job.layer_count,
            "source_units": self.source_units,
            "dimensions_mm": {
                "width": self.plan.conversion.build_width_mm,
                "depth": self.plan.conversion.build_depth_mm,
                "height": self.plan.conversion.build_height_mm,
            },
            "risk_counts": dict(self.risk_counts),
            "recommendation": self.recommendation,
        }

    def ensure_thermal_twin(self):
        if self.thermal_twin is not None:
            return self.thermal_twin
        conversion = self.plan.conversion
        if conversion is None:
            raise RuntimeError("STL conversion metadata is missing")
        resolution_mm = 0.25
        padding_mm = max(self.plan.settings.process.spot_diameter_mm * 2, resolution_mm)
        self.thermal_twin = FiPyThermalTwin(
            self.plan.settings,
            build_width_mm=max(conversion.build_width_mm, resolution_mm) + 2 * padding_mm,
            build_depth_mm=max(conversion.build_depth_mm, resolution_mm) + 2 * padding_mm,
            resolution_mm=resolution_mm,
        )
        return self.thermal_twin

    def finish_agent(self):
        event_count = max(self.agent_cursor, 1)
        recent_window = 5
        if len(self.risk_history) < recent_window * 2:
            direction = "insufficient_data"
        else:
            prior_risk_rate = sum(state != "normal" for state in self.risk_history[-2 * recent_window:-recent_window]) / recent_window
            recent_risk_rate = sum(state != "normal" for state in self.risk_history[-recent_window:]) / recent_window
            direction = (
                "risk_increasing"
                if recent_risk_rate > prior_risk_rate + 0.1
                else "risk_decreasing"
                if prior_risk_rate > recent_risk_rate + 0.1
                else "stable"
            )

        candidates = [
            (key, vote)
            for key, vote in self.action_votes.items()
            if key[0] != "공정 유지"
        ]
        resolved = [(key, vote) for key, vote in candidates if vote["resolved"] > 0]
        selected = max(
            resolved or candidates,
            key=lambda item: (item[1]["resolved"], item[1]["count"]),
            default=None,
        )
        if selected is None:
            first_process = self.events[0].process
            action, power, speed = (
                "공정 유지",
                first_process["laser_power_w"],
                first_process["scan_speed_mm_s"],
            )
            resolved_count = 0
            supporting_vectors = event_count
        else:
            (action, power, speed), vote = selected
            resolved_count = int(vote["resolved"])
            supporting_vectors = int(vote["count"])
        self.recommendation = {
            "action": action,
            "laser_power_w": round(power, 1),
            "scan_speed_mm_s": round(speed, 1),
            "supporting_vectors": supporting_vectors,
            "resolved_vectors": resolved_count,
            "risk_counts": dict(self.risk_counts),
            "predicted_direction": direction,
            "direction_basis": f"최근 {recent_window}개 벡터와 직전 {recent_window}개의 물리 위험 비율 비교",
            "model_boundary": "LT 형상 예측 + Rosenthal ROM 행동 검증; 장비 실측/품질 보증 아님",
        }
        self.phase = "review"


lt_predictor: V5StreamingLTPredictor | None = None
active_build: BuildReplaySession | None = None
approved_setpoints: dict[str, float] | None = None

@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/capabilities")
async def get_capabilities():
    replay_count = len(getattr(lt_predictor, "replay_rows", ())) if lt_predictor else 0
    return {
        "model": "Looped Transformer V5",
        "primary_data": "data/316L_meltpool.csv",
        "measured_rows": replay_count,
        "nist_am_bench": "not configured",
        "sensor_source": "not connected",
        "machine_actuation": False,
        "agent_loaded": lt_predictor is not None,
    }


@app.post("/api/builds")
async def upload_build(request: Request, source_units: Literal["mm", "cm", "m", "in"] = "mm"):
    global active_build, lt_predictor
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="The STL file is empty")
    if len(body) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="STL upload limit is 15 MiB")

    try:
        with tempfile.TemporaryDirectory(prefix="aero-twin-build-") as work_dir:
            stl_path = Path(work_dir) / "upload.stl"
            cli_path = Path(work_dir) / "build.cli"
            stl_path.write_bytes(body)
            settings = demo_316l_settings()
            if approved_setpoints is not None:
                settings = replace(
                    settings,
                    process=replace(
                        settings.process,
                        laser_power_w=approved_setpoints["laser_power_w"],
                        scan_speed_mm_s=approved_setpoints["scan_speed_mm_s"],
                    ),
                )
            plan = create_build_plan(stl_path, cli_path, settings, source_units=source_units)
            events = tuple(plan.iter_events())
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(events) > 100_000:
        raise HTTPException(status_code=422, detail="STL creates more than 100,000 scan vectors")
    if plan.conversion is None:
        raise HTTPException(status_code=500, detail="STL conversion metadata is missing")
    active_build = BuildReplaySession(
        build_id=uuid4().hex,
        plan=plan,
        events=events,
        source_units=source_units,
    )
    lt_predictor = None
    return {
        **active_build.status(),
        "machine_id": settings.machine.machine_id,
        "material": settings.material,
        "process": {
            "laser_power_w": settings.process.laser_power_w,
            "scan_speed_mm_s": settings.process.scan_speed_mm_s,
            "layer_thickness_um": settings.process.layer_thickness_um,
            "hatch_spacing_mm": settings.process.hatch_spacing_mm,
        },
        "profile_origin": "illustrative 316L demo profile",
        "thermal_model": "FiPy reduced-order 2D heat equation",
        "approved_setpoints": approved_setpoints,
    }


@app.get("/api/builds/{build_id}")
async def get_build(build_id: str):
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    return {**active_build.status(), "last_result": active_build.last_result}


@app.post("/api/builds/{build_id}/thermal-step")
async def step_thermal_analysis(
    build_id: str,
    count: int = Query(default=16, ge=1, le=64),
):
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    if active_build.phase not in {"sliced", "thermal_analysis"}:
        raise HTTPException(status_code=409, detail="Thermal analysis is not the active workflow phase")
    active_build.phase = "thermal_analysis"
    thermal_results = []
    segments = []
    try:
        twin = active_build.ensure_thermal_twin()
        for _ in range(count):
            if active_build.thermal_cursor >= len(active_build.events):
                break
            event = active_build.events[active_build.thermal_cursor]
            result = await asyncio.to_thread(twin.consume, event)
            snapshot = asdict(result)
            thermal_results.append(snapshot)
            segments.append({
                "start_mm": event.start_mm,
                "end_mm": event.end_mm,
                "layer_index": event.layer_index,
                "path_type": event.path_type,
            })
            active_build.thermal_cursor += 1
            active_build.thermal_hot_fraction_sum += result.hot_cell_fraction
            active_build.thermal_melt_threshold_fraction_sum += result.reference_threshold_cell_fraction
    except (ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Thermal analysis failed: {exc}") from exc

    if active_build.thermal_cursor >= len(active_build.events):
        active_build.phase = "thermal_complete"
    last_thermal = thermal_results[-1] if thermal_results else None
    active_build.last_result = {"thermal": last_thermal}
    return {
        **active_build.status(),
        "segments": segments,
        "thermal_results": thermal_results,
    }


@app.post("/api/builds/{build_id}/agent/start")
async def start_agent(build_id: str):
    global lt_predictor
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    if active_build.phase != "thermal_complete":
        raise HTTPException(status_code=409, detail="Complete thermal analysis before starting the AI Agent")
    if lt_predictor is None:
        try:
            lt_predictor = await asyncio.to_thread(V5StreamingLTPredictor.from_artifacts)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"Unable to load the LT Agent: {exc}") from exc
    active_build.phase = "agent_running"
    return active_build.status()


def _record_agent_result(session: BuildReplaySession, event, geometry):
    process = event.process
    physical_state = evaluate_physical_state(
        process["laser_power_w"],
        process["scan_speed_mm_s"],
        process["spot_diameter_mm"],
        process["hatch_spacing_mm"] * 1000,
        process["layer_thickness_um"],
        geometry.width_um,
        geometry.depth_um,
    )
    decision = recommend_action_from_prediction(
        process["laser_power_w"],
        process["scan_speed_mm_s"],
        process["spot_diameter_mm"],
        process["hatch_spacing_mm"] * 1000,
        process["layer_thickness_um"],
        geometry.width_um,
        geometry.depth_um,
    )
    session.risk_counts[physical_state["status"]] += 1
    session.risk_history.append(physical_state["status"])
    if decision["recommended_action"] != "공정 유지":
        selected = next(
            (
                candidate
                for candidate in decision["candidates_evaluated"]
                if candidate["action"] == decision["recommended_action"]
            ),
            None,
        )
        vote_key = (
            decision["recommended_action"],
            float(selected["laser_power_w"]) if selected else process["laser_power_w"],
            float(selected["scan_speed_mm_s"]) if selected else process["scan_speed_mm_s"],
        )
        vote = session.action_votes.setdefault(vote_key, {"count": 0.0, "resolved": 0.0})
        vote["count"] += 1
        vote["resolved"] += int(selected["resolved"]) if selected else 0
    return {
        "segment": {
            "start_mm": event.start_mm,
            "end_mm": event.end_mm,
            "layer_index": event.layer_index,
            "path_type": event.path_type,
        },
        "prediction": geometry.as_process_fields(),
        "physical_state": physical_state,
        "decision": decision,
        "thermal_model": "FiPy qualitative field + Rosenthal ROM; no quality guarantee",
    }


@app.post("/api/builds/{build_id}/agent/step")
async def step_agent(build_id: str, count: int = Query(default=16, ge=1, le=64)):
    if lt_predictor is None:
        raise HTTPException(status_code=503, detail="The LT predictor is not ready")
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    if active_build.phase != "agent_running":
        raise HTTPException(status_code=409, detail="Start the AI Agent after thermal analysis")
    if active_build.agent_cursor >= len(active_build.events):
        active_build.finish_agent()
        return {**active_build.status(), "last_result": active_build.last_result}

    results = []
    segments = []
    for _ in range(count):
        if active_build.agent_cursor >= len(active_build.events):
            break
        event = active_build.events[active_build.agent_cursor]
        process = event.process
        try:
            geometry = await asyncio.to_thread(
                lt_predictor.predict,
                LTRequest(
                    process={
                        "P_laser": process["laser_power_w"],
                        "v_scan": process["scan_speed_mm_s"],
                        "d_laser": process["spot_diameter_mm"],
                        "t_powder": process["layer_thickness_um"],
                    },
                    event_id=event.event_id,
                    job_id=event.job_id,
                    delta_t=event.duration_s,
                    metadata={"build_layer": event.layer_index, "path_type": event.path_type},
                ).as_event(),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=f"LT inference failed: {exc}") from exc
        result = _record_agent_result(active_build, event, geometry)
        results.append(result)
        segments.append(result["segment"])
        active_build.agent_cursor += 1
        active_build.last_result = result
    if active_build.agent_cursor >= len(active_build.events):
        active_build.finish_agent()
    return {
        **active_build.status(),
        "segments": segments,
        "results": results,
        "last_result": active_build.last_result,
    }


@app.post("/api/builds/{build_id}/review")
async def review_recommendation(build_id: str, review: BuildReview):
    global approved_setpoints
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    if active_build.phase != "review" or active_build.recommendation is None:
        raise HTTPException(status_code=409, detail="There is no completed recommendation to review")

    recommendation = active_build.recommendation
    if review.approved:
        approved_setpoints = {
            "laser_power_w": recommendation["laser_power_w"],
            "scan_speed_mm_s": recommendation["scan_speed_mm_s"],
        }
        active_build.phase = "approved"
    else:
        active_build.phase = "rejected"
    active_build.recommendation = {**recommendation, "approved": review.approved}
    return active_build.status()
