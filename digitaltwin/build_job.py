"""Turn a CLI toolpath and machine profile into pre-LT process events."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

from .cli_toolpath import CLIJob, PathSegment, read_cli
from .process import BuildSettings
from .stl_slicer import STLConversionResult, slice_stl_to_cli


@dataclass(frozen=True)
class BuildEvent:
    event_id: str
    job_id: str
    event_index: int
    layer_index: int
    path_type: str
    start_mm: tuple[float, float, float]
    end_mm: tuple[float, float, float]
    duration_s: float
    process: dict[str, float]
    context: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "event_index": self.event_index,
            "layer_index": self.layer_index,
            "path_type": self.path_type,
            "start_mm": list(self.start_mm),
            "end_mm": list(self.end_mm),
            "duration_s": self.duration_s,
            "process": dict(self.process),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class BuildPlan:
    settings: BuildSettings
    cli_job: CLIJob
    conversion: STLConversionResult | None = None

    def iter_events(self) -> Iterator[BuildEvent]:
        for event_index, segment in enumerate(self.cli_job.segments):
            yield _make_event(event_index, segment, self.settings)


def _make_event(event_index: int, segment: PathSegment, settings: BuildSettings) -> BuildEvent:
    process = settings.process
    powder = settings.powder
    chamber = settings.chamber
    machine = settings.machine
    return BuildEvent(
        event_id=f"{settings.job_id}:{event_index:08d}",
        job_id=settings.job_id,
        event_index=event_index,
        layer_index=segment.layer_index,
        path_type=segment.path_type,
        start_mm=(segment.start_mm[0], segment.start_mm[1], segment.z_mm),
        end_mm=(segment.end_mm[0], segment.end_mm[1], segment.z_mm),
        duration_s=segment.length_mm / process.scan_speed_mm_s,
        process={
            "laser_power_w": process.laser_power_w,
            "scan_speed_mm_s": process.scan_speed_mm_s,
            "hatch_spacing_mm": process.hatch_spacing_mm,
            "layer_thickness_um": process.layer_thickness_um,
            "spot_diameter_mm": process.spot_diameter_mm,
            "preheat_temperature_c": process.preheat_temperature_c,
        },
        context={
            "material": settings.material,
            "powder_lot_id": powder.lot_id,
            "powder_d10_um": powder.particle_d10_um,
            "powder_d50_um": powder.particle_d50_um,
            "powder_d90_um": powder.particle_d90_um,
            "powder_oxygen_ppm": powder.oxygen_ppm,
            "powder_reuse_count": powder.reuse_count,
            "powder_data_origin": powder.data_origin,
            "support_density_percent": settings.supports.density_percent,
            "support_contact_diameter_mm": settings.supports.contact_diameter_mm,
            "support_removal_access_required": settings.supports.removal_access_required,
            "shielding_gas": chamber.shielding_gas,
            "oxygen_limit_ppm": chamber.oxygen_limit_ppm,
            "oxygen_measured_ppm": chamber.oxygen_measured_ppm,
            "machine_id": machine.machine_id,
            "support_strategy": settings.supports.strategy,
        },
    )


def _validate_plan(settings: BuildSettings, cli_job: CLIJob) -> None:
    settings.validate()
    if not settings.chamber.purge_complete:
        raise ValueError("chamber oxygen is above the configured limit; purge is incomplete")
    if not cli_job.segments:
        raise ValueError("CLI job has no scan vectors")
    x_values = [coordinate for segment in cli_job.segments for coordinate in (segment.start_mm[0], segment.end_mm[0])]
    y_values = [coordinate for segment in cli_job.segments for coordinate in (segment.start_mm[1], segment.end_mm[1])]
    z_values = [segment.z_mm for segment in cli_job.segments]
    if max(x_values) - min(x_values) > settings.machine.build_width_mm:
        raise ValueError("CLI toolpath width exceeds machine build volume")
    if max(y_values) - min(y_values) > settings.machine.build_depth_mm:
        raise ValueError("CLI toolpath depth exceeds machine build volume")
    if max(z_values) > settings.machine.build_height_mm:
        raise ValueError("CLI toolpath height exceeds machine build volume")


def _center_cli_job(cli_job: CLIJob) -> CLIJob:
    x_values = [coordinate for segment in cli_job.segments for coordinate in (segment.start_mm[0], segment.end_mm[0])]
    y_values = [coordinate for segment in cli_job.segments for coordinate in (segment.start_mm[1], segment.end_mm[1])]
    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2
    centered_segments = tuple(
        replace(
            segment,
            start_mm=(segment.start_mm[0] - center_x, segment.start_mm[1] - center_y),
            end_mm=(segment.end_mm[0] - center_x, segment.end_mm[1] - center_y),
        )
        for segment in cli_job.segments
    )
    return replace(cli_job, segments=centered_segments)


def load_cli_build_plan(cli_path: str | Path, settings: BuildSettings) -> BuildPlan:
    """Load an existing CLI file and validate it against the selected job."""
    cli_job = _center_cli_job(read_cli(cli_path))
    _validate_plan(settings, cli_job)
    return BuildPlan(settings=settings, cli_job=cli_job)


def create_build_plan(
    stl_path: str | Path,
    cli_path: str | Path,
    settings: BuildSettings,
    *,
    source_units: str = "mm",
) -> BuildPlan:
    """Run STL -> CLI -> validated event plan for one configured build job."""
    settings.validate()
    if not settings.chamber.purge_complete:
        raise ValueError("chamber oxygen is above the configured limit; purge is incomplete")
    conversion = slice_stl_to_cli(
        stl_path,
        cli_path,
        layer_thickness_um=settings.process.layer_thickness_um,
        hatch_spacing_mm=settings.process.hatch_spacing_mm,
        source_units=source_units,
        label=settings.job_id,
    )
    machine = settings.machine
    if conversion.build_width_mm > machine.build_width_mm:
        raise ValueError("STL width exceeds machine build volume")
    if conversion.build_depth_mm > machine.build_depth_mm:
        raise ValueError("STL depth exceeds machine build volume")
    if conversion.build_height_mm > machine.build_height_mm:
        raise ValueError("STL height exceeds machine build volume")
    cli_job = _center_cli_job(read_cli(conversion.cli_path))
    _validate_plan(settings, cli_job)
    return BuildPlan(settings=settings, cli_job=cli_job, conversion=conversion)