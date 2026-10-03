"""Tests for the operator-gated STL analysis workflow."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from digitaltwin.backend import main
from digitaltwin.backend.main import BuildReplaySession
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


if __name__ == "__main__":
    unittest.main()
