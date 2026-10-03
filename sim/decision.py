"""공정 상태별 판단 + 행동 결정 (태훈 담당: Physics & Digital Twin Lead).

현재 공정 조건을 평가해서 상태(정상/LOF위험/Keyhole위험)를 판단하고, 위험
유형에 맞는 행동 후보를 생성한 뒤, 각 후보를 가상 검증해서 최선의 행동을
추천한다. 개발계획서 "나.개발방법" 문단의 행동결정 예시
(LOF 위험->파워 증가/속도 감소, Keyhole 위험->파워 감소/속도 증가) 그대로.

폭/깊이 예측은 기준님의 공식 Looped Transformer(v3, model/LoopedTransformer/legacy/)를
우선 사용한다 (digitaltwin/looped_transformer_integration.md 데이터 계약 참고).
체크포인트가 없거나 로드에 실패하면 ROM(sim.lpbf)으로 자동 폴백한다 -- LOF
판정·regime 분류 자체는 폭/깊이 소스와 무관하게 항상 ROM의 공식을 그대로 쓴다.

2026-10-03: 기준님이 model/LoopedTransformer/를 리팩토링하면서 dataLT·interfaceLT·
modelLT·physicsLT·v3 체크포인트를 legacy/ 하위로 옮김 (새 model/LoopedTransformer/v5/
패키지는 실시간 스트림용 StreamingLTService라 이 모듈의 단발 호출 용도와는 다름).
그래서 legacy 경로 + 절대 패키지 import로 맞춤.
"""
import os

import pandas as pd

from sim.lpbf import melt_pool_width, melt_pool_depth, lof_risk, classify_regime

_LT_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "LoopedTransformer", "legacy")
_LT_CHECKPOINT = os.path.join(_LT_DIR, "looped_transformer_v3.pt")

try:
    from model.LoopedTransformer.legacy.interfaceLT import RUN as _LTRun
    _lt_runner = _LTRun()
    _LT_AVAILABLE = os.path.exists(_LT_CHECKPOINT)
except Exception:
    _LT_AVAILABLE = False

# dataLT.py가 학습 때 쓴 파생 특성 공식과 동일한 상수. sim/lpbf.py의 ROM 상수
# (RHO=7900, CP=650, T_M=1700 등)와는 다른 값이므로 섞어 쓰면 안 됨.
_ALPHA, _RHO, _CP, _TM, _T0, _ETA = 5.0e-6, 7800.0, 460.0, 1678.0, 293.15, 0.34


def _lt_predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um):
    """공정조건 한 건을 LT 입력 7특성으로 변환해 (width_m, depth_m) 예측."""
    d_mm = spot_diameter_um / 1000.0
    v_mps = speed_mm_s / 1000.0
    d_m = d_mm / 1000.0
    h_s_vol = _RHO * _CP * (_TM - _T0)
    row = pd.DataFrame([{
        "P_laser": power_w, "v_scan": speed_mm_s, "d_laser": d_mm, "t_powder": layer_thickness_um,
        "linear_energy": power_w / speed_mm_s,
        "peclet_number": v_mps * d_m / _ALPHA,
        "normalized_enthalpy": _ETA * power_w / (h_s_vol * (_ALPHA * d_m ** 3 * v_mps) ** 0.5),
    }])
    pred = _lt_runner.getPredict(_LT_CHECKPOINT, row)
    width_um = float(pred["melt_width"].iloc[0])
    ratio = float(pred["aspect_ratio"].iloc[0])
    return width_um * 1e-6, ratio * width_um * 1e-6

ACTION_CANDIDATES = {
    "lof_risk": [
        ("레이저 출력 증가", lambda p, v: (p * 1.15, v)),
        ("스캔 속도 감소", lambda p, v: (p, v * 0.85)),
    ],
    "keyhole_risk": [
        ("레이저 출력 감소", lambda p, v: (p * 0.85, v)),
        ("스캔 속도 증가", lambda p, v: (p, v * 1.15)),
    ],
}


def assess_state(power_w, speed_mm_s, spot_diameter_um=100.0, hatch_um=100.0, layer_thickness_um=30.0):
    """현재 공정조건의 상태를 판단: normal / lof_risk / keyhole_risk.
    폭/깊이는 LT(v3) 예측 우선, 실패 시 ROM 폴백."""
    speed_m_s = speed_mm_s / 1000.0
    source = "lt_v3"
    if _LT_AVAILABLE:
        try:
            w, d = _lt_predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um)
        except Exception:
            w, d, source = melt_pool_width(power_w, speed_m_s), melt_pool_depth(power_w, speed_m_s), "rom_fallback"
    else:
        w, d, source = melt_pool_width(power_w, speed_m_s), melt_pool_depth(power_w, speed_m_s), "rom_fallback"

    risk = lof_risk(hatch_um * 1e-6, layer_thickness_um * 1e-6, w, d)
    regime = str(classify_regime(power_w, speed_m_s, spot_diameter_um * 1e-6))

    if risk > 1.0:
        status = "lof_risk"
    elif regime == "keyhole_or_transition":
        status = "keyhole_risk"
    else:
        status = "normal"

    return {
        "status": status,
        "depth_um": round(float(d) * 1e6, 1),
        "width_um": round(float(w) * 1e6, 1),
        "lof_criterion": round(float(risk), 3),
        "regime": regime,
        "source": source,
    }


def recommend_action(power_w, speed_mm_s, spot_diameter_um=100.0, hatch_um=100.0, layer_thickness_um=30.0):
    """현재 상태를 판단하고, 위험 유형에 맞는 행동 후보들을 ROM으로 가상
    검증(assess_state 재호출)해서 최선을 추천한다."""
    state = assess_state(power_w, speed_mm_s, spot_diameter_um, hatch_um, layer_thickness_um)

    if state["status"] == "normal":
        return {
            "current_state": state,
            "recommended_action": "공정 유지",
            "reason": "현재 조건이 conduction 영역이며 LOF 위험도 없음",
            "candidates_evaluated": [],
        }

    evaluated = []
    for name, adjust in ACTION_CANDIDATES[state["status"]]:
        new_p, new_v = adjust(power_w, speed_mm_s)
        new_state = assess_state(new_p, new_v, spot_diameter_um, hatch_um, layer_thickness_um)
        evaluated.append({
            "action": name,
            "new_power_w": round(new_p, 1),
            "new_speed_mm_s": round(new_v, 1),
            "resulting_state": new_state,
            "resolved": new_state["status"] == "normal",
        })

    resolved = [c for c in evaluated if c["resolved"]]
    best = resolved[0] if resolved else min(evaluated, key=lambda c: c["resulting_state"]["lof_criterion"])
    return {
        "current_state": state,
        "recommended_action": best["action"],
        "reason": (f"{best['action']} 적용 시 "
                   f"{'안전 상태로 회복' if best['resolved'] else '위험이 완화되나 완전 해소는 아님'} "
                   f"(새 조건: {best['new_power_w']}W, {best['new_speed_mm_s']}mm/s)"),
        "candidates_evaluated": evaluated,
    }


def demo():
    lof_case = recommend_action(power_w=80, speed_mm_s=1500)
    assert lof_case["current_state"]["status"] != "normal", "80W/1500mm/s should trigger a risk state"
    assert lof_case["candidates_evaluated"], "risk state should generate action candidates"
    print(f"demo ok (위험조건): {lof_case['recommended_action']} - {lof_case['reason']}")

    normal_case = recommend_action(power_w=500, speed_mm_s=1583.6)
    assert normal_case["current_state"]["status"] == "normal", normal_case
    print(f"demo ok (정상조건): status={normal_case['current_state']['status']}, "
          f"action={normal_case['recommended_action']}")


if __name__ == "__main__":
    demo()
