"""Tools the AI agent can call. Small enough that a dict dispatch beats a framework."""
from sim.lpbf import melt_pool_width, melt_pool_depth, lof_risk, classify_regime, suggest_process_params
from sim.decision import recommend_action


def predict_melt_pool(laser_power_w, scan_speed_mm_s, spot_diameter_um=100.0,
                       hatch_um=100.0, layer_thickness_um=30.0):
    """Rosenthal-ROM estimate of melt pool depth/width for given LPBF settings,
    plus LOF (Lack-of-Fusion) safety and conduction/keyhole regime. All
    physical lengths returned in um."""
    speed_m_s = scan_speed_mm_s / 1000.0
    width_m = melt_pool_width(laser_power_w, speed_m_s)
    depth_m = melt_pool_depth(laser_power_w, speed_m_s)
    risk = lof_risk(hatch_um * 1e-6, layer_thickness_um * 1e-6, width_m, depth_m)
    regime = str(classify_regime(laser_power_w, speed_m_s, spot_diameter_um * 1e-6))
    return {
        "depth_um": round(float(depth_m) * 1e6, 1),
        "width_um": round(float(width_m) * 1e6, 1),
        "lof_criterion": round(float(risk), 3),
        "safe_fusion": bool(risk <= 1.0),
        "regime": regime,
    }


def suggest_params_for_target_depth(target_depth_um):
    """Inverse design: given a target melt pool depth, suggest a safe,
    conduction-regime laser power/scan speed combo (team's '목표 제어' flow)."""
    return suggest_process_params(target_depth_um)


def recommend_process_action(laser_power_w, scan_speed_mm_s, spot_diameter_um=100.0,
                             hatch_um=100.0, layer_thickness_um=30.0):
    """현재 공정 조건의 상태(정상/LOF위험/Keyhole위험)를 판단하고, 위험하면
    ROM으로 가상검증한 행동 후보 중 최선을 추천한다. 형상 입력의 단위는 um이다."""
    return recommend_action(
        laser_power_w,
        scan_speed_mm_s,
        spot_diameter_um=spot_diameter_um,
        hatch_um=hatch_um,
        layer_thickness_um=layer_thickness_um,
    )


TOOLS = {
    "predict_melt_pool": predict_melt_pool,
    "suggest_params_for_target_depth": suggest_params_for_target_depth,
    "recommend_process_action": recommend_process_action,
}


def demo():
    result = predict_melt_pool(laser_power_w=200, scan_speed_mm_s=800)
    assert "depth_um" in result and "width_um" in result and "regime" in result
    assert result["depth_um"] > 0 and result["width_um"] > 0
    print(f"demo ok: {result}")

    suggestion = suggest_params_for_target_depth(target_depth_um=60.0)
    assert abs(suggestion["predicted_depth_um"] - 60.0) < 1.0
    print(f"demo ok: {suggestion}")

    action = recommend_process_action(laser_power_w=80, scan_speed_mm_s=1500)
    assert "recommended_action" in action
    thicker_layer_action = recommend_process_action(
        laser_power_w=80,
        scan_speed_mm_s=1500,
        layer_thickness_um=60.0,
    )
    assert (
        thicker_layer_action["current_state"]["lof_criterion"]
        != action["current_state"]["lof_criterion"]
    )
    print(f"demo ok: {action}")


if __name__ == "__main__":
    demo()
