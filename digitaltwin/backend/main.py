import asyncio
from array import array
import logging
import math
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..build_job import BuildPlan, create_build_plan, load_cli_build_plan
from ..fipy_dt import FiPyThermalTwin
from ..process import demo_316l_settings
from .lt_contracts import GeometryPredictor, LTRequest
from .lt_predictor import V5StreamingLTPredictor
from sim.decision import recommend_action_from_prediction
from sim.lpbf import evaluate_physical_state

app = FastAPI(title="Physical AI Digital Twin")
logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
PREVIEW_CLI_PATH = Path(__file__).resolve().parents[1] / "nozzle.preview.cli"
PREVIEW_STL_PATH = Path(__file__).resolve().parents[1] / "nozzle.stl"
MAX_ANALYSIS_VECTORS = 100_000
MAX_DISPLAY_VECTORS = 15_000
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
    plan: BuildPlan | None
    events: tuple
    source_units: str
    source_name: str = "STL"
    machine_build_volume_mm: dict[str, float] = field(default_factory=dict)
    phase: str = "slicing"
    machine_id: str = ""
    material: str = "316L"
    process: dict = field(default_factory=dict)
    error: str | None = None
    preview_toolpath: bytes | None = None
    slicing_layer: int = 0
    slicing_layer_count: int = 0
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
        conversion = self.plan.conversion if self.plan is not None else None
        plan_vector_count = (
            getattr(self.plan.cli_job, "source_segment_count", 0)
            if self.plan is not None else 0
        )
        source_vector_count = (
            plan_vector_count
            if plan_vector_count
            else len(self.events)
        )
        return {
            "build_id": self.build_id,
            "status": self.phase,
            "phase": self.phase,
            "event_index": self.agent_cursor,
            "event_count": event_count,
            "source_vector_count": source_vector_count,
            "analysis_sampled": source_vector_count > event_count,
            "source_name": self.source_name,
            "machine_build_volume_mm": dict(self.machine_build_volume_mm),
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
            "layer_count": self.plan.cli_job.layer_count if self.plan is not None else None,
            "source_units": self.source_units,
            "dimensions_mm": {
                "width": conversion.build_width_mm,
                "depth": conversion.build_depth_mm,
                "height": conversion.build_height_mm,
            } if conversion is not None else None,
            "risk_counts": dict(self.risk_counts),
            "recommendation": self.recommendation,
            "error": self.error,
            "slicing_layer": self.slicing_layer,
            "slicing_layer_count": self.slicing_layer_count,
            "slicing_progress_pct": (
                self.slicing_layer / self.slicing_layer_count * 100
                if self.slicing_layer_count else 0.0
            ),
            "machine_id": self.machine_id,
            "material": self.material,
            "process": dict(self.process),
        }

    def ensure_thermal_twin(self):
        if self.thermal_twin is not None:
            return self.thermal_twin
        if self.plan is None:
            raise RuntimeError("STL slicing has not completed")
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
build_preparation_tasks: dict[str, asyncio.Task] = {}

@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/capabilities")
async def get_capabilities():
    replay_count = len(getattr(lt_predictor, "replay_rows", ())) if lt_predictor else 0
    machine = demo_316l_settings().machine
    return {
        "model": "Looped Transformer V5",
        "primary_data": "data/316L_meltpool.csv",
        "measured_rows": replay_count,
        "nist_am_bench": "not configured",
        "sensor_source": "not connected",
        "machine_actuation": False,
        "agent_loaded": lt_predictor is not None,
        "preview_cli_available": PREVIEW_CLI_PATH.is_file(),
        "preview_mesh_available": PREVIEW_STL_PATH.is_file(),
        "preview_cli_size_bytes": PREVIEW_CLI_PATH.stat().st_size if PREVIEW_CLI_PATH.is_file() else 0,
        "machine_build_volume_mm": {
            "width": machine.build_width_mm,
            "depth": machine.build_depth_mm,
            "height": machine.build_height_mm,
        },
    }


def _build_settings_for_next_job():
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
    return settings


def _create_uploaded_build(body: bytes, source_units: str, settings, progress_callback=None):
    with tempfile.TemporaryDirectory(prefix="aero-twin-build-") as work_dir:
        stl_path = Path(work_dir) / "upload.stl"
        cli_path = Path(work_dir) / "build.cli"
        stl_path.write_bytes(body)
        plan = create_build_plan(
            stl_path,
            cli_path,
            settings,
            source_units=source_units,
            slicing_progress=progress_callback,
        )
        events = tuple(plan.iter_events())
    if len(events) > 100_000:
        raise ValueError("STL creates more than 100,000 scan vectors")
    if plan.conversion is None:
        raise RuntimeError("STL conversion metadata is missing")
    return plan, events


async def _prepare_build_in_background(
    session: BuildReplaySession,
    body: bytes,
    source_units: str,
    settings,
):
    def report_progress(layer: int, layer_count: int):
        if active_build is session:
            session.slicing_layer = layer
            session.slicing_layer_count = layer_count

    try:
        plan, events = await asyncio.to_thread(
            _create_uploaded_build,
            body,
            source_units,
            settings,
            report_progress,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if active_build is session:
            session.phase = "slice_failed"
            session.error = str(exc)
    else:
        if active_build is session:
            session.plan = plan
            session.events = events
            session.phase = "sliced"
    finally:
        build_preparation_tasks.pop(session.build_id, None)


def _build_preparation_finished(session: BuildReplaySession, task: asyncio.Task):
    if task.cancelled():
        if active_build is session:
            session.phase = "slice_failed"
            session.error = "STL slicing task was cancelled"
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Unexpected STL preparation failure for build %s",
            session.build_id,
            exc_info=(type(error), error, error.__traceback__),
        )
        if active_build is session:
            session.phase = "slice_failed"
            session.error = f"Unexpected STL preparation failure: {error}"


async def _prepare_preview_in_background(
    session: BuildReplaySession,
    settings,
):
    packed_path = array("h")

    def collect_segment(segment):
        packed_path.extend(
            round(coordinate * 100)
            for coordinate in (
                segment.start_mm[0],
                segment.start_mm[1],
                segment.z_mm,
                segment.end_mm[0],
                segment.end_mm[1],
                segment.z_mm,
            )
        )

    try:
        plan = await asyncio.to_thread(
            load_cli_build_plan,
            PREVIEW_CLI_PATH,
            settings,
            max_segments=MAX_ANALYSIS_VECTORS,
            segment_callback=collect_segment,
        )
        events = tuple(plan.iter_events())
        if packed_path.itemsize != 2:
            raise RuntimeError("16-bit path preview encoding is not supported on this platform")
        center_x, center_y = plan.cli_job.source_center_mm
        center_x_ticks = round(center_x * 100)
        center_y_ticks = round(center_y * 100)
        for index in range(0, len(packed_path), 6):
            packed_path[index] -= center_x_ticks
            packed_path[index + 1] -= center_y_ticks
            packed_path[index + 3] -= center_x_ticks
            packed_path[index + 4] -= center_y_ticks
        if sys.byteorder != "little":
            packed_path.byteswap()
        preview_toolpath = packed_path.tobytes()
    except (OSError, OverflowError, ValueError, RuntimeError) as exc:
        if active_build is session:
            session.phase = "slice_failed"
            session.error = str(exc)
    else:
        if active_build is session:
            session.plan = plan
            session.events = events
            session.preview_toolpath = preview_toolpath
            session.phase = "sliced"
    finally:
        build_preparation_tasks.pop(session.build_id, None)


@app.post("/api/builds", status_code=status.HTTP_202_ACCEPTED)
async def upload_build(request: Request, source_units: Literal["mm", "cm", "m", "in"] = "mm"):
    global active_build, lt_predictor
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="The STL file is empty")
    if len(body) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="STL upload limit is 15 MiB")

    settings = _build_settings_for_next_job()
    active_build = BuildReplaySession(
        build_id=uuid4().hex,
        plan=None,
        events=(),
        source_units=source_units,
        source_name="STL upload",
        machine_build_volume_mm={
            "width": settings.machine.build_width_mm,
            "depth": settings.machine.build_depth_mm,
            "height": settings.machine.build_height_mm,
        },
        machine_id=settings.machine.machine_id,
        material=settings.material,
        process={
            "laser_power_w": settings.process.laser_power_w,
            "scan_speed_mm_s": settings.process.scan_speed_mm_s,
            "layer_thickness_um": settings.process.layer_thickness_um,
            "hatch_spacing_mm": settings.process.hatch_spacing_mm,
        },
    )
    lt_predictor = None
    preparation_task = asyncio.create_task(
        _prepare_build_in_background(active_build, body, source_units, settings)
    )
    build_preparation_tasks[active_build.build_id] = preparation_task
    preparation_task.add_done_callback(
        lambda task, session=active_build: _build_preparation_finished(session, task)
    )
    return {
        **active_build.status(),
        "profile_origin": "illustrative 316L demo profile",
        "thermal_model": "FiPy reduced-order 2D heat equation",
        "approved_setpoints": approved_setpoints,
    }


@app.post("/api/builds/preview", status_code=status.HTTP_202_ACCEPTED)
async def start_preview_build():
    global active_build, lt_predictor
    if not PREVIEW_CLI_PATH.is_file():
        raise HTTPException(status_code=404, detail="Bundled nozzle.preview.cli is not available")
    settings = _build_settings_for_next_job()
    active_build = BuildReplaySession(
        build_id=uuid4().hex,
        plan=None,
        events=(),
        source_units="mm",
        source_name=PREVIEW_CLI_PATH.name,
        machine_build_volume_mm={
            "width": settings.machine.build_width_mm,
            "depth": settings.machine.build_depth_mm,
            "height": settings.machine.build_height_mm,
        },
        machine_id=settings.machine.machine_id,
        material=settings.material,
        process={
            "laser_power_w": settings.process.laser_power_w,
            "scan_speed_mm_s": settings.process.scan_speed_mm_s,
            "layer_thickness_um": settings.process.layer_thickness_um,
            "hatch_spacing_mm": settings.process.hatch_spacing_mm,
        },
    )
    lt_predictor = None
    preparation_task = asyncio.create_task(
        _prepare_preview_in_background(active_build, settings)
    )
    build_preparation_tasks[active_build.build_id] = preparation_task
    preparation_task.add_done_callback(
        lambda task, session=active_build: _build_preparation_finished(session, task)
    )
    return {
        **active_build.status(),
        "preview_mesh_url": "/api/preview/nozzle.stl" if PREVIEW_STL_PATH.is_file() else None,
        "profile_origin": "illustrative 316L demo profile",
        "thermal_model": "FiPy reduced-order 2D heat equation",
        "approved_setpoints": approved_setpoints,
    }


@app.get("/api/preview/nozzle.stl")
async def get_preview_mesh():
    if not PREVIEW_STL_PATH.is_file():
        raise HTTPException(status_code=404, detail="Bundled nozzle.stl is not available")
    return FileResponse(PREVIEW_STL_PATH, media_type="model/stl", filename=PREVIEW_STL_PATH.name)


@app.get("/api/builds/{build_id}")
async def get_build(build_id: str):
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    return {
        **active_build.status(),
        "last_result": active_build.last_result,
        "profile_origin": "illustrative 316L demo profile",
        "thermal_model": "FiPy reduced-order 2D heat equation",
        "approved_setpoints": approved_setpoints,
    }


@app.get("/api/builds/{build_id}/toolpath")
async def get_build_toolpath(build_id: str):
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    stride = max(1, math.ceil(len(active_build.events) / MAX_DISPLAY_VECTORS))
    segments = [
        {
            "start_mm": list(event.start_mm),
            "end_mm": list(event.end_mm),
        }
        for index, event in enumerate(active_build.events)
        if index % stride == 0
    ]
    return {
        "segments": segments,
        "source_vector_count": active_build.status()["source_vector_count"],
        "displayed_vector_count": len(segments),
        "display_sampled": len(segments) < len(active_build.events),
    }


@app.get("/api/builds/{build_id}/toolpath/full")
async def get_full_build_toolpath(build_id: str):
    if active_build is None or active_build.build_id != build_id:
        raise HTTPException(status_code=404, detail="Build job not found")
    if active_build.phase != "sliced" or active_build.preview_toolpath is None:
        raise HTTPException(status_code=409, detail="Full bundled toolpath is not ready")
    return Response(
        content=active_build.preview_toolpath,
        media_type="application/octet-stream",
        headers={
            "X-Vector-Count": str(len(active_build.preview_toolpath) // 12),
            "X-Coordinate-Scale": "0.01",
        },
    )


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
