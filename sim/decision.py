"""공정 상태별 판단 + 행동 결정 (태훈 담당: Physics & Digital Twin Lead).

현재 공정 조건을 평가해서 상태(정상/LOF위험/Keyhole위험)를 판단하고, 위험
유형에 맞는 행동 후보를 생성한 뒤, 각 후보를 ROM(sim.lpbf)으로 가상 검증해서
최선의 행동을 추천한다. 개발계획서 "나.개발방법" 문단의 행동결정 예시
(LOF 위험->파워 증가/속도 감소, Keyhole 위험->파워 감소/속도 증가) 그대로.

ROM 자체가 곧 "Digital Twin 가상검증" 역할을 한다 -- 후보마다 assess_state()를
다시 돌려서 실제로 위험이 해소되는지 확인하므로, 별도의 무거운 DT 컴포넌트를
새로 만들 필요가 없다.
"""
from sim.lpbf import melt_pool_width, melt_pool_depth, lof_risk, classify_regime

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
    """현재 공정조건의 상태를 판단: normal / lof_risk / keyhole_risk."""
    speed_m_s = speed_mm_s / 1000.0
    w = melt_pool_width(power_w, speed_m_s)
    d = melt_pool_depth(power_w, speed_m_s)
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
