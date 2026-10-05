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
        event_id="vector-1",
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
    plan.iter_events = lambda: iter(())
    return BuildReplaySession(
        build_id="test-build",
        plan=plan,
        events=(event,),
        source_units="mm",
        phase=phase,
    )


class BuildWorkflowTests(unittest.TestCase):
    def test_build_status_reports_registration_in_progress(self):
        registration = main.BuildRegistration(
            slicing_percent=25,
            layers_completed=1,
            total_layers=4,
        )
        with (
            patch.dict(main.build_registrations, {"pending-build": registration}),
            patch.object(main, "active_build", None),
        ):
            result = asyncio.run(main.get_build("pending-build"))

        self.assertEqual(result["build_id"], "pending-build")
        self.assertEqual(result["registration_status"], "processing")
        self.assertEqual(result["phase"], "slicing")
        self.assertEqual(result["slicing_percent"], 25)
        self.assertEqual(result["layers_completed"], 1)
        self.assertEqual(result["total_layers"], 4)

    def test_build_registration_runs_preparation_without_blocking_endpoint(self):
        registration = main.BuildRegistration()
        plan = SimpleNamespace(
            cli_job=SimpleNamespace(layer_count=1),
            conversion=SimpleNamespace(
                build_width_mm=1.0,
                build_depth_mm=1.0,
                build_height_mm=0.1,
            ),
        )
        settings = SimpleNamespace(
            machine=SimpleNamespace(machine_id="test-machine"),
            material="316L",
            process=SimpleNamespace(
                laser_power_w=200.0,
                scan_speed_mm_s=800.0,
                layer_thickness_um=30.0,
                hatch_spacing_mm=0.1,
            ),
        )

        async def prepare_build():
            await main._process_build_registration(
                "ready-build", b"stl", settings, "mm", None
            )

        with (
            patch.dict(main.build_registrations, {"ready-build": registration}),
            patch.object(main, "active_build", None),
            patch.object(main, "lt_predictor", None),
            patch.object(main, "_prepare_uploaded_build", return_value=(plan, (), settings, 1)),
        ):
            asyncio.run(prepare_build())
            result = asyncio.run(main.get_build("ready-build"))

        self.assertEqual(registration.state, "complete")
        self.assertEqual(result["registration_status"], "complete")
        self.assertEqual(result["event_count"], 0)
        self.assertFalse(result["slicing_adjusted"])
        self.assertEqual(result["slicing_attempts"], 1)

    def test_large_build_is_resliced_with_coarser_resolution(self):
        settings = main.demo_316l_settings()
        used_settings = []
        progress_updates = []
        vector_counts = iter((main.MAX_SCAN_VECTORS + 1, main.MAX_SCAN_VECTORS))

        def make_plan(stl_path, cli_path, current_settings, **options):
            self.assertTrue(stl_path.is_file())
            self.assertEqual(cli_path.name, "build.cli")
            self.assertEqual(options["source_units"], "mm")
            used_settings.append(current_settings)
            options["progress_callback"](0, 1)
            options["progress_callback"](1, 1)
            count = next(vector_counts)
            plan = SimpleNamespace(
                cli_job=SimpleNamespace(segments=[None] * count),
                iter_events=lambda: iter(("event",)),
            )
            return plan

        with patch.object(main, "create_build_plan", side_effect=make_plan):
            plan, events, final_settings, attempts = main._prepare_uploaded_build(
                b"stl",
                settings,
                "mm",
                lambda *update: progress_updates.append(update),
            )

        self.assertEqual(attempts, 2)
        self.assertEqual(len(plan.cli_job.segments), main.MAX_SCAN_VECTORS)
        self.assertEqual(events, ("event",))
        self.assertGreater(
            final_settings.process.hatch_spacing_mm,
            settings.process.hatch_spacing_mm,
        )
        self.assertGreater(
            final_settings.process.layer_thickness_um,
            settings.process.layer_thickness_um,
        )
        self.assertEqual(used_settings[0], settings)
        self.assertEqual(used_settings[1], final_settings)
        self.assertEqual(
            [update[:2] for update in progress_updates],
            [(1, main.MAX_SLICING_ATTEMPTS)] * 2
            + [(2, main.MAX_SLICING_ATTEMPTS)] * 2,
        )

    def test_build_still_fails_if_vector_cap_is_exceeded_after_adjustments(self):
        settings = main.demo_316l_settings()

        def make_oversized_plan(stl_path, cli_path, current_settings, **options):
            self.assertTrue(stl_path.is_file())
            self.assertEqual(cli_path.name, "build.cli")
            current_settings.validate()
            options["progress_callback"](0, 1)
            return SimpleNamespace(
                cli_job=SimpleNamespace(
                    segments=[None] * (main.MAX_SCAN_VECTORS + 1)
                ),
            )

        with (
            patch.object(main, "MAX_SLICING_ATTEMPTS", 2),
            patch.object(main, "create_build_plan", side_effect=make_oversized_plan),
            self.assertRaisesRegex(ValueError, "after 1 automatic resolution adjustments"),
        ):
            main._prepare_uploaded_build(b"stl", settings, "mm")

    def test_build_registration_exposes_slicing_errors(self):
        registration = main.BuildRegistration()
        with (
            patch.dict(main.build_registrations, {"bad-build": registration}),
            patch.object(main, "_prepare_uploaded_build", side_effect=ValueError("invalid STL")),
        ):
            asyncio.run(
                main._process_build_registration("bad-build", b"stl", None, "mm", None)
            )
            with self.assertRaises(HTTPException) as error:
                asyncio.run(main.get_build("bad-build"))

        self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(error.exception.detail, "invalid STL")

    def test_new_job_waits_at_slicing_stage(self):
        session = make_session()

        self.assertEqual(session.status()["phase"], "sliced")
        self.assertEqual(session.status()["thermal_event_index"], 0)
        self.assertEqual(session.status()["agent_progress_pct"], 0.0)
        self.assertIsNone(session.thermal_twin)

    def test_large_build_uses_bounded_thermal_grid_and_substeps(self):
        session = make_session()
        session.plan.settings = SimpleNamespace(
            process=SimpleNamespace(spot_diameter_mm=0.1)
        )
        fake_twin = object()

        with patch.object(main, "FiPyThermalTwin", return_value=fake_twin) as factory:
            result = session.ensure_thermal_twin()

        self.assertIs(result, fake_twin)
        self.assertEqual(factory.call_args.kwargs["max_cells"], 10_000)
        self.assertEqual(factory.call_args.kwargs["max_substeps_per_vector"], 2)

    def test_agent_cannot_start_before_thermal_analysis_finishes(self):
        session = make_session()
        with patch.object(main, "active_build", session), patch.object(main, "lt_predictor", None):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(main.start_agent("test-build"))

        self.assertEqual(error.exception.status_code, 409)
        self.assertIn("0/1 paths complete", error.exception.detail)

    def test_agent_start_uses_completed_thermal_cursor_as_authoritative(self):
        session = make_session(phase="thermal_analysis")
        session.thermal_cursor = len(session.events)
        predictor = object()

        with (
            patch.object(main, "active_build", session),
            patch.object(main, "lt_predictor", predictor),
        ):
            result = asyncio.run(main.start_agent("test-build"))

        self.assertEqual(result["phase"], "agent_running")

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

    def test_thermal_analysis_processes_only_small_batch_per_request(self):
        session = make_session(phase="thermal_analysis")
        session.events = (session.events[0], session.events[0], session.events[0])

        class FakeThermalTwin:
            def consume(self, event):
                return ThermalSnapshot(
                    event_id=event.event_id,
                    hot_cell_count=1,
                    reference_threshold_cell_count=0,
                    hot_cell_fraction=0.1,
                    reference_threshold_cell_fraction=0.0,
                    grid_shape=(5, 4),
                    cell_size_mm=(0.25, 0.25),
                )

        with (
            patch.object(main, "active_build", session),
            patch.object(session, "ensure_thermal_twin", return_value=FakeThermalTwin()),
        ):
            result = asyncio.run(main.step_thermal_analysis("test-build", count=2))

        self.assertEqual(result["phase"], "thermal_analysis")
        self.assertEqual(result["thermal_event_index"], 2)
        self.assertEqual(len(result["thermal_results"]), 2)

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
