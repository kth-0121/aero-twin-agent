"""Focused tests for the STL/CLI path and pre-LT DT boundary."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import trimesh
from shapely.geometry import Polygon

from digitaltwin.build_job import create_build_plan, load_cli_build_plan
from digitaltwin.cli_toolpath import LayerPaths, read_cli, write_cli
from digitaltwin.pre_lt_dt import PreLTDigitalTwin
from digitaltwin.process import demo_316l_settings, load_build_settings
from digitaltwin.stl_slicer import _hatch_polygon


class PreLTPipelineTests(unittest.TestCase):
    def test_hatching_keeps_scan_order_and_holes_for_both_directions(self):
        polygon = Polygon(
            [(0, 0), (4, 0), (4, 3), (0, 3)],
            holes=[[(1, 1), (3, 1), (3, 2), (1, 2)]],
        )

        horizontal = _hatch_polygon(polygon, 0.5, 0)
        vertical = _hatch_polygon(polygon, 0.5, 90)

        self.assertEqual(len(horizontal), 8)
        self.assertEqual(len(vertical), 12)
        self.assertEqual(horizontal[0], ((0.0, 0.25), (4.0, 0.25)))
        self.assertEqual(horizontal[1], ((4.0, 0.75), (0.0, 0.75)))
        self.assertEqual(vertical[0], ((0.25, 3.0), (0.25, 0.0)))
        self.assertEqual(vertical[1], ((0.75, 0.0), (0.75, 3.0)))

    def test_slicer_reports_layer_progress(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stl_path = root / "part.stl"
            trimesh.creation.box(extents=(1.0, 1.0, 0.06)).export(stl_path)
            progress = []
            plan = create_build_plan(
                stl_path,
                root / "part.cli",
                demo_316l_settings(),
                slicing_progress=lambda layer, count: progress.append((layer, count)),
            )

        self.assertEqual(progress[0], (0, 2))
        self.assertEqual(progress[-1], (2, 2))
        self.assertEqual([layer for layer, _ in progress], sorted(layer for layer, _ in progress))
        self.assertIsNotNone(plan.conversion)

    def test_json_build_profile_roundtrip(self):
        settings = demo_316l_settings()
        with TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps(asdict(settings)), encoding="utf-8")
            loaded = load_build_settings(profile_path)
        self.assertEqual(loaded, settings)

    def test_cli_roundtrip_and_centered_events(self):
        with TemporaryDirectory() as directory:
            cli_path = Path(directory) / "offset.cli"
            write_cli(
                cli_path,
                [LayerPaths(0, 0.03, (), (((9.0, 4.0), (10.0, 4.0)),))],
            )
            job = read_cli(cli_path)
            self.assertEqual(job.layer_count, 1)
            plan = load_cli_build_plan(cli_path, demo_316l_settings())
            self.assertEqual(plan.cli_job.bounds_mm, ((-0.5, 0.0, 0.03), (0.5, 0.0, 0.03)))
            event = next(plan.iter_events())
            self.assertEqual(event.start_mm[:2], (-0.5, 0.0))
            self.assertEqual(event.end_mm[:2], (0.5, 0.0))
            with PreLTDigitalTwin(plan, thermal_resolution_mm=0.25) as twin:
                self.assertLess(twin.thermal.nx, 20)

    def test_cli_sampling_is_ordered_and_preserves_full_path_bounds(self):
        with TemporaryDirectory() as directory:
            cli_path = Path(directory) / "sampled.cli"
            write_cli(
                cli_path,
                [
                    LayerPaths(
                        0,
                        0.03,
                        (),
                        tuple(((float(index), 0.0), (float(index + 1), 0.0)) for index in range(10)),
                    ),
                    LayerPaths(1, 0.06, (), (((20.0, 0.0), (21.0, 0.0)),)),
                ],
            )

            job = read_cli(cli_path, max_segments=3)

        self.assertEqual(job.source_segment_count, 11)
        self.assertEqual(len(job.segments), 3)
        self.assertEqual(job.layer_count, 2)
        self.assertEqual(job.bounds_mm, ((0.0, 0.0, 0.03), (21.0, 0.0, 0.06)))
        source_positions = [segment.start_mm[0] for segment in job.segments]
        self.assertEqual(source_positions, sorted(source_positions))

    def test_purge_gate_blocks_job(self):
        settings = demo_316l_settings()
        settings = replace(
            settings,
            chamber=replace(settings.chamber, oxygen_measured_ppm=101.0),
        )
        with TemporaryDirectory() as directory:
            cli_path = Path(directory) / "ready.cli"
            write_cli(
                cli_path,
                [LayerPaths(0, 0.03, (), (((0.0, 0.0), (0.2, 0.0)),))],
            )
            with self.assertRaisesRegex(ValueError, "purge is incomplete"):
                load_cli_build_plan(cli_path, settings)

    def test_stl_to_cli_to_fipy_and_pybullet(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stl_path = root / "part.stl"
            trimesh.creation.box(extents=(1.0, 1.0, 0.06)).export(stl_path)
            plan = create_build_plan(
                stl_path,
                root / "part.cli",
                demo_316l_settings(),
            )
            self.assertEqual(plan.conversion.layer_count, 2)
            self.assertGreater(len(plan.cli_job.segments), 0)

            with PreLTDigitalTwin(plan, source_stl=str(stl_path)) as twin:
                results = list(twin.run(event_limit=2))

            self.assertEqual(len(results), 2)
            for result in results:
                self.assertGreaterEqual(result.thermal.hot_cell_fraction, 0.0)
                self.assertLessEqual(result.thermal.hot_cell_fraction, 1.0)
                self.assertGreaterEqual(result.thermal.reference_threshold_cell_fraction, 0.0)
                self.assertLessEqual(result.thermal.reference_threshold_cell_fraction, 1.0)
                self.assertFalse(hasattr(result.thermal, "peak_temperature_c"))


if __name__ == "__main__":
    unittest.main()