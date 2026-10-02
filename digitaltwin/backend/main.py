import asyncio
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .lt_contracts import GeometryPredictor
from .lt_predictor import V5StreamingLTPredictor

app = FastAPI(title="Physical AI Digital Twin")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
        self.scan_speed_mm_s = 800.0
        self.layer_path_mm = self.track_count * self.track_length_mm

    def step(self, dt=1/30):
        self.elapsed_s += dt
        total_distance_mm = self.elapsed_s * self.scan_speed_mm_s
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
        chamber_temperature = 28.0 + 0.35 * math.sin(self.elapsed_s / 18)
        oxygen_ppm = 82.0 + 3.0 * math.sin(self.elapsed_s / 7)
        timestamp = time.time()
        job_id = f"LPBF-316L-{42 + job_cycle:03d}"
        self.event_counter += 1
        process_input = {
            "P_laser": 250.0,
            "v_scan": self.scan_speed_mm_s,
            "d_laser": 0.1,
            "t_powder": 30.0,
        }
        geometry = self.geometry_predictor.predict({
            **process_input,
            "event_id": f"{job_id}:{self.event_counter}",
            "job_id": job_id,
            "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "delta_t": dt,
        })

        return EquipmentState(
            timestamp=timestamp,
            equipment_id="LBFP-01",
            state="RUNNING",
            position={"x": x, "y": y, "z": z},
            velocity={
                "x": -self.scan_speed_mm_s if reverse else self.scan_speed_mm_s,
                "y": 0.0,
                "z": 0.0,
            },
            force={"fx": 0.0, "fy": 0.0, "fz": 0.0},
            process={
                "laser_power": process_input["P_laser"],
                "scan_speed": self.scan_speed_mm_s,
                "spot_diameter": process_input["d_laser"],
                "powder_layer_thickness_um": process_input["t_powder"],
                "track_spacing": self.track_pitch_mm * 1000,
                **geometry.as_process_fields(),
            },
            build={
                "job_id": job_id,
                "material": "316L",
                "layer_current": layer_index + 1,
                "layer_total": self.layer_count,
                "track_current": track_index + 1,
                "track_total": self.track_count,
                "progress_pct": min(build_progress, 100.0),
                "elapsed_s": 468.0 + self.elapsed_s,
            },
            machine={
                "source": "SIMULATION",
                "laser": "SCANNING",
                "recoater": "STANDBY",
                "chamber_temperature": chamber_temperature,
                "oxygen_ppm": oxygen_ppm,
                "filter_load_pct": 68.0,
                "interlock": "CLOSED",
            },
        )

dt = None
clients: Set[WebSocket] = set()
latest = None

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.websocket("/ws/dt")
async def websocket_dt(ws: WebSocket):
    global latest
    await ws.accept()
    clients.add(ws)
    try:
        if latest:
            await ws.send_json(asdict(latest))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)

async def simulation_loop():
    global latest
    while True:
        latest = await asyncio.to_thread(dt.step)
        dead = []
        payload = asdict(latest)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
        await asyncio.sleep(1/30)

@app.on_event("startup")
async def startup():
    global dt
    dt = LPBFCellSimulation(V5StreamingLTPredictor.from_artifacts())
    asyncio.create_task(simulation_loop())
