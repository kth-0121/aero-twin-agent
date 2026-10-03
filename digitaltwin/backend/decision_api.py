"""의사결정 지원 API (판단·행동결정·가상검증·Agent).

DT 콘솔(main.py)에 router 한 줄로 붙는다. 기존 콘솔 코드는 건드리지 않고 이 파일에서만:
  POST /api/assess     공정조건 -> 상태판단 (LT 예측 + LOF/regime), 학습범위 밖 경고
  POST /api/recommend  상태판단 + 행동 후보 생성 + 후보별 재검증
  POST /api/agent      로컬 sLLM Agent 질의 (Ollama 없으면 503)
  GET  /api/scenarios  시연용 프리셋 조건 (정상 / LOF 위험 / keyhole 위험)
  GET  /decision       통합 화면(/)의 의사결정 지원 패널로 이동

검증기는 sim.decision의 ROM 재평가 방식이다 (PyBullet 물리 솔버가 아님) -- 응답의
`verification` 필드에 그 사실을 명시한다.
"""
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.decision import assess_state, recommend_action  # noqa: E402

router = APIRouter()

SCENARIOS = [
    {"id": "normal", "label": "정상 조건", "power_w": 250, "speed_mm_s": 800},
    {"id": "lof", "label": "LOF 위험 (저출력·고속)", "power_w": 80, "speed_mm_s": 1500},
    {"id": "keyhole", "label": "Keyhole 위험 (고출력·저속)", "power_w": 450, "speed_mm_s": 300},
]


class Condition(BaseModel):
    power_w: float = Field(gt=0, le=2000, description="레이저 출력 [W]")
    speed_mm_s: float = Field(gt=0, le=10000, description="스캔 속도 [mm/s]")
    spot_diameter_um: float = Field(100.0, gt=0, le=1000)
    hatch_um: float = Field(100.0, gt=0, le=1000)
    layer_thickness_um: float = Field(30.0, gt=0, le=500)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _training_range():
    """학습 데이터(316L 실측)의 공정변수 범위 -- 이 밖의 입력은 외삽이라 경고한다."""
    from model.LoopedTransformer.legacy.dataLT import load_316l_csv

    df = load_316l_csv(_REPO_ROOT / "data" / "316L_meltpool.csv")
    cols = {"power_w": "P_laser", "speed_mm_s": "v_scan", "spot_diameter_um": "d_laser"}
    scale = {"power_w": 1.0, "speed_mm_s": 1.0, "spot_diameter_um": 1000.0}  # d_laser는 mm
    return {k: (df[c].min() * scale[k], df[c].max() * scale[k]) for k, c in cols.items()}


def _range_warnings(c: Condition) -> list[str]:
    names = {"power_w": "레이저 출력(W)", "speed_mm_s": "스캔 속도(mm/s)", "spot_diameter_um": "스폿 직경(um)"}
    out = []
    for key, (lo, hi) in _training_range().items():
        value = getattr(c, key)
        if not lo <= value <= hi:
            out.append(f"{names[key]} {value:g}는 학습 데이터 범위({lo:g}~{hi:g}) 밖 -- 외삽 예측이라 신뢰도 낮음")
    return out


def _model_info(state: dict) -> dict:
    names = {
        "lt_v3": "Looped Transformer v3 (공식)",
        "lt_v5": "Looped Transformer V5 (스트리밍, 정상상태 평가 모드)",
    }
    return {
        "prediction_model": names.get(state.get("source"), "Rosenthal ROM (LT 로드 실패 폴백)"),
        "uncertainty": "미제공 -- 점추정이며 안전 보장이 아님",
    }


@router.get("/api/scenarios")
def scenarios():
    return SCENARIOS


@router.post("/api/assess")
def assess(c: Condition):
    state = assess_state(c.power_w, c.speed_mm_s, c.spot_diameter_um, c.hatch_um, c.layer_thickness_um)
    return {"state": state, "warnings": _range_warnings(c), **_model_info(state)}


@router.post("/api/recommend")
def recommend(c: Condition):
    result = recommend_action(c.power_w, c.speed_mm_s, c.spot_diameter_um, c.hatch_um, c.layer_thickness_um)
    return {
        **result,
        "warnings": _range_warnings(c),
        "verification": "후보 조건을 같은 예측 모델로 재평가 (ROM 기반 가상검증, 물리 시뮬레이터 아님)",
        **_model_info(result["current_state"]),
    }


@router.post("/api/agent")
def agent(q: Question):
    try:
        from agent.loop import ask_agent

        return {"answer": ask_agent(q.question)}
    except Exception as exc:  # Ollama 미기동 등
        raise HTTPException(503, f"Agent 호출 실패 (Ollama 서버 확인): {exc}")


@router.get("/decision")
def decision_page():
    """예전 주소 호환: 통합 화면의 의사결정 지원 패널로 이동."""
    return RedirectResponse("/#decision-panel")


def demo():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    states = {x["id"]: client.post("/api/assess", json={"power_w": x["power_w"], "speed_mm_s": x["speed_mm_s"]}).json()
              for x in SCENARIOS}
    assert states["normal"]["state"]["status"] == "normal", states["normal"]
    assert states["lof"]["state"]["status"] == "lof_risk", states["lof"]
    assert states["keyhole"]["state"]["status"] == "keyhole_risk", states["keyhole"]

    out = client.post("/api/assess", json={"power_w": 900, "speed_mm_s": 300}).json()
    assert out["warnings"], "학습범위 밖 입력은 경고해야 함"
    assert client.post("/api/assess", json={"power_w": -5, "speed_mm_s": 800}).status_code == 422

    rec = client.post("/api/recommend", json={"power_w": 80, "speed_mm_s": 1500}).json()
    assert rec["candidates_evaluated"] and rec["verification"]
    print(f"demo ok: 시나리오 3종 판정 일치, 범위 밖 경고, 입력검증(422), 후보 {len(rec['candidates_evaluated'])}개 재검증")


if __name__ == "__main__":
    demo()
