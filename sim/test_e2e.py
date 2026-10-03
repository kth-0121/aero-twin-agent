"""대표 E2E 경로(판단 -> 행동 후보 -> 재검증 -> API/DT 이벤트) 자동 테스트.

실행: python -m unittest sim.test_e2e -v   (저장소 루트, KMP_DUPLICATE_LIB_OK=TRUE)
Ollama가 없어도 통과한다 (Agent는 200 또는 503 둘 다 허용).
"""
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sim import decision
from sim.decision import assess_state, recommend_action


def _client():
    from digitaltwin.backend.decision_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class DecisionScenarios(unittest.TestCase):
    def test_normal_keeps_process(self):
        r = recommend_action(200, 800)
        self.assertEqual(r["current_state"]["status"], "normal")
        self.assertEqual(r["recommended_action"], "공정 유지")
        self.assertEqual(r["candidates_evaluated"], [])

    def test_risk_states_get_a_resolving_action(self):
        for p, v, expected in [(80, 1500, "lof_risk"), (120, 1000, "lof_risk"),
                               (450, 300, "keyhole_risk"), (300, 500, "keyhole_risk")]:
            with self.subTest(p=p, v=v):
                r = recommend_action(p, v)
                self.assertEqual(r["current_state"]["status"], expected)
                best = next(c for c in r["candidates_evaluated"] if c["action"] == r["recommended_action"])
                self.assertTrue(best["resolved"])
                # 추천 조건을 처음부터 다시 판단해도 정상이어야 함 (자기모순 없음)
                again = assess_state(best["new_power_w"], best["new_speed_mm_s"])
                self.assertEqual(again["status"], "normal")

    def test_recommendation_stays_inside_training_range(self):
        r = recommend_action(450, 300)
        best = next(c for c in r["candidates_evaluated"] if c["action"] == r["recommended_action"])
        self.assertTrue(decision.SEARCH_POWER_W[0] <= best["new_power_w"] <= decision.SEARCH_POWER_W[1])
        self.assertTrue(decision.SEARCH_SPEED_MM_S[0] <= best["new_speed_mm_s"] <= decision.SEARCH_SPEED_MM_S[1])

    def test_layer_thickness_changes_lof(self):
        a = assess_state(80, 1500, layer_thickness_um=30.0)["lof_criterion"]
        b = assess_state(80, 1500, layer_thickness_um=60.0)["lof_criterion"]
        self.assertNotEqual(a, b)

    def test_rom_fallback_when_lt_unavailable(self):
        original = decision._LT_AVAILABLE
        decision._LT_AVAILABLE = False
        try:
            s = assess_state(200, 800)
            self.assertEqual(s["source"], "rom_fallback")
            self.assertIn(s["status"], {"normal", "lof_risk", "keyhole_risk"})
        finally:
            decision._LT_AVAILABLE = original

    def test_monotonic_power_depth(self):
        import numpy as np

        for v in (400.0, 800.0, 1200.0):
            _, d, _ = decision._predict_width_depth(np.linspace(50, 500, 40), np.full(40, v), 100.0, 30.0)
            self.assertGreaterEqual(np.mean(np.diff(d) >= 0), 0.9, f"출력↑ 깊이↑ 단조성 @v={v}")


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _client()

    def test_assess_and_model_provenance(self):
        out = self.client.post("/api/assess", json={"power_w": 200, "speed_mm_s": 800}).json()
        self.assertEqual(out["state"]["status"], "normal")
        self.assertIn("Looped Transformer", out["prediction_model"])
        self.assertIn("미제공", out["uncertainty"])
        self.assertEqual(out["warnings"], [])

    def test_out_of_range_warns(self):
        out = self.client.post("/api/assess", json={"power_w": 900, "speed_mm_s": 300}).json()
        self.assertTrue(out["warnings"])

    def test_recommend_has_verification_note(self):
        out = self.client.post("/api/recommend", json={"power_w": 80, "speed_mm_s": 1500}).json()
        self.assertTrue(any(c["resolved"] for c in out["candidates_evaluated"]))
        self.assertIn("ROM", out["verification"])

    def test_invalid_input_rejected(self):
        for body in ({"power_w": -5, "speed_mm_s": 800}, {"power_w": 200}, {"power_w": "x", "speed_mm_s": 800}):
            self.assertEqual(self.client.post("/api/recommend", json=body).status_code, 422)

    def test_scenarios_match_their_labels(self):
        expected = {"normal": "normal", "lof": "lof_risk", "keyhole": "keyhole_risk"}
        for s in self.client.get("/api/scenarios").json():
            out = self.client.post("/api/assess", json={"power_w": s["power_w"], "speed_mm_s": s["speed_mm_s"]}).json()
            self.assertEqual(out["state"]["status"], expected[s["id"]], s)

    def test_agent_degrades_gracefully(self):
        # Ollama 유무와 무관하게 200(응답) 또는 503(명시적 오류)이어야 하며 500이면 안 됨
        self.assertIn(self.client.post("/api/agent", json={"question": "안전해?"}).status_code, (200, 503))

    def test_decision_page_served(self):
        res = self.client.get("/decision")
        self.assertEqual(res.status_code, 200)
        self.assertIn("의사결정 지원", res.text)


class DigitalTwinEvents(unittest.TestCase):
    def test_build_events_flow_through_decision(self):
        from digitaltwin.build_job import load_cli_build_plan
        from digitaltwin.cli_toolpath import LayerPaths, write_cli
        from digitaltwin.process import demo_316l_settings
        from sim.digital_twin_loop import run_decision_loop

        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "t.cli"
            write_cli(cli, [LayerPaths(0, 0.03, (), (((0.0, 0.0), (5.0, 0.0)), ((0.0, 0.1), (5.0, 0.1))))])
            results = list(run_decision_loop(load_cli_build_plan(cli, demo_316l_settings())))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["current_state"]["source"] in {"lt_v3", "rom_fallback"} for r in results))


if __name__ == "__main__":
    unittest.main()
