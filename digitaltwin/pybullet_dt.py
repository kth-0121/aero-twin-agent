"""PyBullet build-plate, STL, and scan-track scene for CLI job replay."""

from collections import deque
from math import atan2, hypot
from pathlib import Path

from .build_job import BuildEvent
from .process import BuildSettings


class PyBulletBuildScene:
    def __init__(
        self,
        settings: BuildSettings,
        *,
        connection_mode: int | None = None,
        history_limit: int = 800,
    ):
        try:
            import pybullet as bullet
        except ImportError as exc:
            raise RuntimeError("PyBullet DT requires pybullet; install digitaltwin/requirements.txt") from exc
        if history_limit < 1:
            raise ValueError("history_limit must be positive")

        self.bullet = bullet
        self.settings = settings
        self.history_limit = history_limit
        self._tracks = deque()
        self._laser_marker = None
        mode = bullet.DIRECT if connection_mode is None else connection_mode
        self.client_id = bullet.connect(mode)
        if self.client_id < 0:
            raise RuntimeError("could not connect to PyBullet")
        bullet.resetSimulation(physicsClientId=self.client_id)
        bullet.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        self._create_build_plate()

    def _create_build_plate(self) -> None:
        bullet = self.bullet
        machine = self.settings.machine
        thickness_m = 0.01
        shape = bullet.createCollisionShape(
            bullet.GEOM_BOX,
            halfExtents=(machine.build_width_mm / 2000, machine.build_depth_mm / 2000, thickness_m / 2),
            physicsClientId=self.client_id,
        )
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=(machine.build_width_mm / 2000, machine.build_depth_mm / 2000, thickness_m / 2),
            rgbaColor=(0.28, 0.34, 0.36, 1.0),
            physicsClientId=self.client_id,
        )
        self.plate_id = bullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=(0.0, 0.0, -thickness_m / 2),
            physicsClientId=self.client_id,
        )

    def add_stl(self, stl_path: str | Path, *, source_units: str = "mm") -> int:
        """Add the part as a visual mesh, aligned to the centered CLI frame."""
        unit_to_m = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
        if source_units not in unit_to_m:
            raise ValueError(f"source_units must be one of {tuple(unit_to_m)}")
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError("STL scene loading requires trimesh") from exc
        mesh = trimesh.load_mesh(stl_path, force="mesh", process=True)
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        scale = unit_to_m[source_units]
        visual = self.bullet.createVisualShape(
            self.bullet.GEOM_MESH,
            fileName=str(Path(stl_path).resolve()),
            meshScale=(scale, scale, scale),
            rgbaColor=(0.42, 0.70, 0.63, 0.55),
            physicsClientId=self.client_id,
        )
        self.part_id = self.bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=(-center[0] * scale, -center[1] * scale, -bounds[0][2] * scale),
            physicsClientId=self.client_id,
        )
        return self.part_id

    def consume(self, event: BuildEvent) -> None:
        bullet = self.bullet
        x0, y0, z = event.start_mm
        x1, y1, _ = event.end_mm
        dx, dy = x1 - x0, y1 - y0
        length_mm = hypot(dx, dy)
        if length_mm <= 0:
            return
        process = self.settings.process
        half_extents = (
            length_mm / 2000,
            max(process.hatch_spacing_mm / 2, 0.025),
            max(process.layer_thickness_um / 2000, 0.015),
        )
        color = (0.25, 0.78, 0.67, 0.9) if event.path_type == "hatch" else (0.36, 0.70, 0.84, 0.9)
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client_id,
        )
        body = bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=((x0 + x1) / 2000, (y0 + y1) / 2000, z / 1000),
            baseOrientation=bullet.getQuaternionFromEuler((0.0, 0.0, atan2(dy, dx))),
            physicsClientId=self.client_id,
        )
        self._tracks.append(body)
        if len(self._tracks) > self.history_limit:
            bullet.removeBody(self._tracks.popleft(), physicsClientId=self.client_id)

        if self._laser_marker is not None:
            bullet.removeBody(self._laser_marker, physicsClientId=self.client_id)
        marker = bullet.createVisualShape(
            bullet.GEOM_SPHERE,
            radius=0.0005,
            rgbaColor=(1.0, 0.25, 0.16, 1.0),
            physicsClientId=self.client_id,
        )
        self._laser_marker = bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker,
            basePosition=(x1 / 1000, y1 / 1000, z / 1000),
            physicsClientId=self.client_id,
        )
        bullet.stepSimulation(physicsClientId=self.client_id)

    def close(self) -> None:
        if self.client_id >= 0:
            self.bullet.disconnect(physicsClientId=self.client_id)
            self.client_id = -1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()