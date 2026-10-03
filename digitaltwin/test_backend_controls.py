"""Tests for the simulation controls exposed by the digital-twin API."""

from types import SimpleNamespace
import unittest

from digitaltwin.backend.main import LPBFCellSimulation


class FakeGeometryPredictor:
    def __init__(self):
        self.inputs = []

    def predict(self, event):
        self.inputs.append(event)
        return SimpleNamespace(
            width_um=120.0,
            depth_um=80.0,
            area_um2=9000.0,
            as_process_fields=lambda: {
                "melt_width": 120.0,
                "melt_depth": 80.0,
                "aspect_ratio": 80.0 / 120.0,
                "melt_area": 9000.0,
            }
        )


class SimulationControlTests(unittest.TestCase):
    def test_process_setpoints_reach_live_prediction(self):
        predictor = FakeGeometryPredictor()
        simulation = LPBFCellSimulation(predictor)
        simulation.configure(laser_power_w=300.0, scan_speed_mm_s=1200.0)

        state = simulation.step()

        self.assertEqual(predictor.inputs[-1]["P_laser"], 300.0)
        self.assertEqual(predictor.inputs[-1]["v_scan"], 1200.0)
        self.assertEqual(state.process["laser_power"], 300.0)
        self.assertEqual(state.process["scan_speed"], 1200.0)

    def test_pause_holds_build_progress_and_resume_continues(self):
        simulation = LPBFCellSimulation(FakeGeometryPredictor())
        simulation.step()
        elapsed_before_pause = simulation.elapsed_s

        simulation.configure(running=False)
        paused = simulation.step()
        self.assertEqual(paused.state, "PAUSED")
        self.assertEqual(paused.machine["laser"], "STANDBY")
        self.assertEqual(simulation.elapsed_s, elapsed_before_pause)

        simulation.configure(running=True)
        resumed = simulation.step()
        self.assertEqual(resumed.state, "RUNNING")
        self.assertGreater(simulation.elapsed_s, elapsed_before_pause)

    def test_measured_dataset_row_reaches_prediction_and_physical_state(self):
        predictor = FakeGeometryPredictor()
        predictor.replay_rows = ({"row": 1},)
        predictor.next_replay_row = lambda: {
            "P_laser": 220.0,
            "v_scan": 900.0,
            "d_laser": 0.12,
            "t_powder": 30.0,
            "melt_width": 130.0,
            "melt_depth": 65.0,
            "melt_area": 6600.0,
            "source_row_index": 17,
        }
        simulation = LPBFCellSimulation(predictor)

        state = simulation.step()

        self.assertEqual(state.machine["source"], "316L_MEASURED_REPLAY")
        self.assertEqual(predictor.inputs[-1]["P_laser"], 220.0)
        self.assertEqual(predictor.inputs[-1]["v_scan"], 900.0)
        self.assertEqual(predictor.inputs[-1]["source_row_index"], 17)
        self.assertEqual(state.process["measured_geometry"]["source_row_index"], 17)
        self.assertEqual(state.process["measured_geometry"]["melt_width_um"], 130.0)
        self.assertIn(state.physical_state["status"], {"normal", "lof_risk", "keyhole_risk"})
        self.assertIn("recommended_action", state.decision)


if __name__ == "__main__":
    unittest.main()
