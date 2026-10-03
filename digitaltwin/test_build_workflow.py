"""Tests for the operator-gated STL analysis workflow."""

import asyncio
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from digitaltwin.backend import main
from digitaltwin.backend.main import BuildReplaySession
from digitaltwin.cli_toolpath import LayerPaths, write_cli
from digitaltwin.fipy_dt import ThermalSnapshot


def make_session(phase="sliced"):
    process = {"laser_power_w": 200.0, "scan_speed_mm_s": 800.0}
    event = SimpleNamespace(
        process=process,
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(1.0, 0.0, 0.0),
        layer_index=0,
        path_type="hatch",
    )
    plan = SimpleNamespace(
        cli_job=SimpleNamespace(layer_count=1),
        conversion=SimpleNamespace(
            build_width_mm=1.0,
            build_depth_mm=1.0,
            build_height_mm=0.1,
        ),
    )
    return BuildReplaySession(
        build_id="test-build",
        plan=plan,
        events=(event,),
        source_units="mm",
        phase=phase,
    )


class BuildWorkflowTests(unittest.TestCase):
    def test_upload_acknowledges_before_slicing_finishes(self):
        slicing_started = Event()
        finish_slicing = Event()
        plan = SimpleNamespace(
            cli_job=SimpleNamespace(layer_count=1),
            conversion=SimpleNamespace(
                build_width_mm=1.0,
                build_depth_mm=1.0,
                build_height_mm=0.1,
            ),
        )

        def delayed_slice(*args):
            progress_callback = args[-1]
            progress_callback(0, 1)
            slicing_started.set()
            if not finish_slicing.wait(timeout=5):
                raise RuntimeError("test slicing timed out")
            progress_callback(1, 1)
            return plan, (object(),)

        with (
            patch.object(main, "active_build", None),
            patch.object(main, "lt_predictor", None),
            patch.object(main, "_create_uploaded_build", side_effect=delayed_slice),
            TestClient(main.app) as client,
        ):
            try:
                response = client.post(
                    "/api/builds?source_units=mm",
                    content=b"uploaded STL bytes",
                    headers={"content-type": "model/stl"},
                )
                self.assertEqual(response.status_code, 202)
                build = response.json()
                self.assertEqual(build["phase"], "slicing")
                self.assertTrue(slicing_started.wait(timeout=2))
                in_progress = client.get(f"/api/builds/{build['build_id']}").json()
                self.assertEqual(in_progress["slicing_layer_count"], 1)
                self.assertEqual(in_progress["slicing_progress_pct"], 0.0)
            finally:
                finish_slicing.set()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status_response = client.get(f"/api/builds/{build['build_id']}")
                self.assertEqual(status_response.status_code, 200)
                status = status_response.json()
                if status["phase"] != "slicing":
                    break
                time.sleep(0.02)

        self.assertEqual(status["phase"], "sliced")
        self.assertEqual(status["event_count"], 1)
        self.assertEqual(status["slicing_progress_pct"], 100.0)

    def test_preview_cli_starts_sliced_job_with_bounded_analysis_sample(self):
        with TemporaryDirectory() as directory:
            preview_cli = Path(directory) / "nozzle.preview.cli"
            write_cli(
                preview_cli,
                [
                    LayerPaths(
                        0,
                        0.03,
                        (),
                        tuple(((float(index), 0.0), (float(index + 1), 0.0)) for index in range(5)),
                    )
                ],
            )
            with (
                patch.object(main, "active_build", None),
                patch.object(main, "lt_predictor", None),
                patch.object(main, "PREVIEW_CLI_PATH", preview_cli),
                patch.object(main, "PREVIEW_STL_PATH", Path(directory) / "nozzle.stl"),
                patch.object(main, "MAX_ANALYSIS_VECTORS", 2),
                TestClient(main.app) as client,
            ):
                response = client.post("/api/builds/preview")
                self.assertEqual(response.status_code, 202)
                build = response.json()
                deadline = time.monotonic() + 5
                while build["phase"] == "slicing" and time.monotonic() < deadline:
                    time.sleep(0.01)
                    build = client.get(f"/api/builds/{build['build_id']}").json()
                toolpath = client.get(f"/api/builds/{build['build_id']}/toolpath").json()
                full_toolpath = client.get(f"/api/builds/{build['build_id']}/toolpath/full")

        self.assertEqual(build["phase"], "sliced")
        self.assertEqual(build["source_name"], "nozzle.preview.cli")
        self.assertEqual(build["event_count"], 2)
        self.assertEqual(build["source_vector_count"], 5)
        self.assertTrue(build["analysis_sampled"])
        self.assertEqual(toolpath["displayed_vector_count"], 2)
        self.assertFalse(toolpath["display_sampled"])
        self.assertEqual(full_toolpath.headers["x-vector-count"], "5")
        self.assertEqual(len(full_toolpath.content), 5 * 6 * 2)

    def test_new_job_waits_at_slicing_stage(self):
        session = make_session()

        self.assertEqual(session.status()["phase"], "sliced")
        self.assertEqual(session.status()["thermal_event_index"], 0)
        self.assertEqual(session.status()["agent_progress_pct"], 0.0)
        self.assertIsNone(session.thermal_twin)

    def test_agent_cannot_start_before_thermal_analysis_finishes(self):
        session = make_session()
        with patch.object(main, "active_build", session), patch.object(main, "lt_predictor", None):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(main.start_agent("test-build"))

        self.assertEqual(error.exception.status_code, 409)

    def test_thermal_analysis_advances_only_when_operator_requests_it(self):
        session = make_session()

        class FakeThermalTwin:
            def consume(self, _event):
                return ThermalSnapshot(
                    event_id="vector-1",
                    hot_cell_count=2,
                    reference_threshold_cell_count=1,
                    hot_cell_fraction=0.2,
                    reference_threshold_cell_fraction=0.1,
                    grid_shape=(5, 4),
                    cell_size_mm=(0.25, 0.25),
                )

        with (
            patch.object(main, "active_build", session),
            patch.object(session, "ensure_thermal_twin", return_value=FakeThermalTwin()),
        ):
            result = asyncio.run(main.step_thermal_analysis("test-build", count=1))

        self.assertEqual(result["phase"], "thermal_complete")
        self.assertEqual(result["thermal_event_index"], 1)
        self.assertAlmostEqual(result["thermal_summary"]["hot_cell_fraction_avg"], 0.2)

    def test_recommendation_is_created_only_after_all_agent_vectors(self):
        session = make_session()
        session.agent_cursor = 1
        session.risk_counts["lof_risk"] = 1
        session.risk_history = ["normal"] * 5 + ["lof_risk"] * 5
        session.action_votes[("레이저 출력 증가", 230.0, 800.0)] = {
            "count": 3.0,
            "resolved": 2.0,
        }

        session.finish_agent()

        self.assertEqual(session.phase, "review")
        self.assertEqual(session.recommendation["action"], "레이저 출력 증가")
        self.assertEqual(session.recommendation["predicted_direction"], "risk_increasing")
        self.assertEqual(session.recommendation["laser_power_w"], 230.0)

    def test_background_slicing_failure_is_visible_in_job_status(self):
        session = make_session()
        session.plan = None
        session.events = ()

        with (
            patch.object(main, "active_build", session),
            patch.object(main, "build_preparation_tasks", {"test-build": object()}),
            patch.object(main, "_create_uploaded_build", side_effect=ValueError("invalid mesh")),
        ):
            asyncio.run(
                main._prepare_build_in_background(
                    session,
                    b"stl",
                    "mm",
                    SimpleNamespace(),
                )
            )

        self.assertEqual(session.status()["phase"], "slice_failed")
        self.assertEqual(session.status()["error"], "invalid mesh")


if __name__ == "__main__":
    unittest.main()
