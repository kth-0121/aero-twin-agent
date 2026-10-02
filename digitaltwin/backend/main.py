import asyncio
import time
import threading
from dataclasses import dataclass, asdict
from typing import Set

import pybullet as p
import pybullet_data
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

class PyBulletDT:
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.body = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
        self.t = 0.0
        self.lock = threading.Lock()

    def step(self, dt=1/30):
        with self.lock:
            # Demo motion: replace with the actual LBFP motion model later.
            x = 0.5 * __import__('math').sin(self.t)
            p.resetBasePositionAndOrientation(self.body, [x, 0, 0.5], [0, 0, 0, 1])
            p.stepSimulation()
            pos, _ = p.getBasePositionAndOrientation(self.body)
            vel, _ = p.getBaseVelocity(self.body)
            self.t += dt

            return EquipmentState(
                timestamp=time.time(),
                equipment_id="LBFP-001",
                state="RUNNING",
                position={"x": pos[0], "y": pos[1], "z": pos[2]},
                velocity={"x": vel[0], "y": vel[1], "z": vel[2]},
                force={"fx": 0.0, "fy": 0.0, "fz": 0.0},
                process={
                    "laser_power": 250.0,
                    "scan_speed": 800.0,
                    "melt_width": 117.3,
                    "aspect_ratio": 0.824,
                    "melt_area": 9203.4,
                },
            )

dt = PyBulletDT()
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
        latest = dt.step()
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
    asyncio.create_task(simulation_loop())
