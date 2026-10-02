"""Compose the pre-LT build-event stream with FiPy and PyBullet twins."""

from dataclasses import dataclass
from typing import Iterator

from .build_job import BuildEvent, BuildPlan
from .fipy_dt import FiPyThermalTwin, ThermalSnapshot
from .pybullet_dt import PyBulletBuildScene


@dataclass(frozen=True)
class PreLTResult:
    event: BuildEvent
    thermal: ThermalSnapshot


class PreLTDigitalTwin:
    """Replay a validated job through thermal and geometric DT modules."""

    def __init__(
        self,
        plan: BuildPlan,
        *,
        source_stl: str | None = None,
        source_units: str = "mm",
        connection_mode: int | None = None,
        thermal_resolution_mm: float = 0.25,
    ):
        self.plan = plan
        if plan.conversion is not None:
            width_mm = plan.conversion.build_width_mm
            depth_mm = plan.conversion.build_depth_mm
        else:
            x_values = [
                coordinate
                for segment in plan.cli_job.segments
                for coordinate in (segment.start_mm[0], segment.end_mm[0])
            ]
            y_values = [
                coordinate
                for segment in plan.cli_job.segments
                for coordinate in (segment.start_mm[1], segment.end_mm[1])
            ]
            width_mm = max(x_values) - min(x_values)
            depth_mm = max(y_values) - min(y_values)
        padding_mm = max(plan.settings.process.spot_diameter_mm * 2, thermal_resolution_mm)
        domain_width_mm = max(width_mm, thermal_resolution_mm) + 2 * padding_mm
        domain_depth_mm = max(depth_mm, thermal_resolution_mm) + 2 * padding_mm

        self.thermal = FiPyThermalTwin(
            plan.settings,
            build_width_mm=domain_width_mm,
            build_depth_mm=domain_depth_mm,
            resolution_mm=thermal_resolution_mm,
        )
        self.scene = PyBulletBuildScene(
            plan.settings,
            connection_mode=connection_mode,
        )
        if source_stl is not None:
            self.scene.add_stl(source_stl, source_units=source_units)
        self._closed = False

    def consume(self, event: BuildEvent) -> PreLTResult:
        thermal = self.thermal.consume(event)
        self.scene.consume(event)
        return PreLTResult(event=event, thermal=thermal)

    def run(self, *, event_limit: int | None = None) -> Iterator[PreLTResult]:
        for event_index, event in enumerate(self.plan.iter_events()):
            if event_limit is not None and event_index >= event_limit:
                break
            yield self.consume(event)

    def close(self) -> None:
        if not self._closed:
            self.scene.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()