"""Run an STL/CLI job through the pre-LT digital-twin modules."""

import argparse
import time
from pathlib import Path

from .build_job import create_build_plan, load_cli_build_plan
from .pre_lt_dt import PreLTDigitalTwin
from .process import demo_316l_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=Path, help="watertight part STL")
    parser.add_argument("--source-units", choices=("mm", "cm", "m", "in"), default="mm")
    parser.add_argument("--cli-input", type=Path, help="use a CLI file produced by an external slicer")
    parser.add_argument("--cli-output", type=Path, help="where generated CLI is written (defaults beside STL)")
    parser.add_argument("--event-limit", type=int, default=100)
    parser.add_argument("--thermal-resolution-mm", type=float, default=0.25)
    parser.add_argument("--gui", action="store_true", help="show PyBullet GUI and play at scan-time speed")
    args = parser.parse_args()
    if args.event_limit < 1:
        parser.error("--event-limit must be at least 1")
    if not args.stl.is_file():
        parser.error(f"STL file not found: {args.stl}; pass the path to an existing STL file")
    if args.cli_input and not args.cli_input.is_file():
        parser.error(f"CLI file not found: {args.cli_input}; pass the path to an existing CLI file")

    settings = demo_316l_settings()
    if args.cli_input:
        plan = load_cli_build_plan(args.cli_input, settings)
    else:
        cli_output = args.cli_output or args.stl.with_suffix(".cli")
        plan = create_build_plan(
            args.stl,
            cli_output,
            settings,
            source_units=args.source_units,
        )

    print(
        f"job={settings.job_id} material={settings.material} "
        f"layers={plan.cli_job.layer_count} vectors={len(plan.cli_job.segments)}"
    )
    print(
        f"powder={settings.powder.lot_id} ({settings.powder.data_origin}); "
        f"purge={'ready' if settings.chamber.purge_complete else 'blocked'}"
    )

    connection_mode = None
    if args.gui:
        import pybullet

        connection_mode = pybullet.GUI
    with PreLTDigitalTwin(
        plan,
        source_stl=str(args.stl),
        source_units=args.source_units,
        connection_mode=connection_mode,
        thermal_resolution_mm=args.thermal_resolution_mm,
    ) as twin:
        for result in twin.run(event_limit=args.event_limit):
            event = result.event
            print(
                f"{event.event_index:06d} layer={event.layer_index + 1:04d} "
                f"{event.path_type:7s} peak={result.thermal.peak_temperature_c:7.1f} C"
            )
            if args.gui:
                time.sleep(event.duration_s)


if __name__ == "__main__":
    main()