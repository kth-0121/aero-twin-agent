import unittest

from digitaltwin.backend.lt_contracts import LTRequest
from digitaltwin.backend.lt_predictor import V5StreamingLTPredictor


class FakeSensorProvider:
    source = "TEST"

    def observe(self, process):
        return {"sensor_1": 0.0, "sensor_2": 0.0, "sensor_3": 0.0}


class FakeLTService:
    def process_event(self, event):
        return {
            "prediction": {"melt_width": 100.0, "melt_depth": 50.0, "melt_area": 4000.0},
            "drift_score": 0.2,
            "drift_warning": False,
            "uncertainty_sigma_approx": {},
            "loop_predictions": [],
            "anchor": {},
            "history_events": 0,
        }


class FailingLTService:
    def process_event(self, event):
        raise ValueError("inference failed")


class LTAdapterTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "P_laser": 250,
            "v_scan": 800,
            "d_laser": 0.1,
            "t_powder": 30,
            "event_id": "event-1",
            "operator_note": "retained",
        }

    def test_common_request_and_success_response(self):
        request = LTRequest.from_event(self.event)
        predictor = V5StreamingLTPredictor(
            FakeLTService(), FakeSensorProvider(), model_version="LT-V5-test"
        )

        response = predictor.predict_result(request)

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.model_version, "LT-V5-test")
        self.assertEqual(response.event_id, "event-1")
        self.assertEqual(response.prediction.width_um, 100.0)
        self.assertEqual(response.prediction.model_version, "LT-V5-test")
        self.assertEqual(request.as_event()["operator_note"], "retained")

    def test_failure_is_returned_as_error_status(self):
        predictor = V5StreamingLTPredictor(
            FailingLTService(), FakeSensorProvider(), model_version="LT-V5-test"
        )

        response = predictor.predict_result(self.event)

        self.assertEqual(response.status, "error")
        self.assertEqual(response.model_version, "LT-V5-test")
        self.assertIn("inference failed", response.error)
        self.assertIsNone(response.prediction)

    def test_missing_required_input_returns_error_status(self):
        predictor = V5StreamingLTPredictor(FakeLTService(), FakeSensorProvider())

        response = predictor.predict_result({"P_laser": 250})

        self.assertEqual(response.status, "error")
        self.assertIn("v_scan", response.error)
        self.assertIsNone(response.prediction)

    def test_legacy_predict_raises_for_failed_inference(self):
        predictor = V5StreamingLTPredictor(
            FailingLTService(), FakeSensorProvider()
        )

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            predictor.predict(self.event)


if __name__ == "__main__":
    unittest.main()