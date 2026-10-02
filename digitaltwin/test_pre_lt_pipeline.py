"""Focused tests for the STL/CLI path and pre-LT DT boundary."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import trimesh

from digitaltwin.build_job import create_build_plan, load_cli_build_plan
from digitaltwin.cli_toolpath import LayerPaths, read_cli, write_cli
from digitaltwin.pre_lt_dt import PreLTDigitalTwin
from digitaltwin.process import demo_316l_settings


class PreLTPipelineTests(unittest.TestCase):
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
            event = next(plan.iter_events())
            self.assertEqual(event.start_mm[:2], (-0.5, 0.0))
            self.assertEqual(event.end_mm[:2], (0.5, 0.0))
            with PreLTDigitalTwin(plan, thermal_resolution_mm=0.25) as twin:
                self.assertLess(twin.thermal.nx, 20)

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
                self.assertTrue(np.isfinite(result.thermal.peak_temperature_c))
                self.assertGreaterEqual(result.thermal.peak_temperature_c, 100.0)


if __name__ == "__main__":
    unittest.main()