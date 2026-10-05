"""공정 상태별 판단 + 행동 결정.

현재 공정 조건을 평가해서 상태(정상/LOF위험/Keyhole위험)를 판단하고, 위험
유형에 맞는 행동 후보를 생성한 뒤, 각 후보를 가상 검증해서 최선의 행동을
추천한다. 개발계획서 "나.개발방법" 문단의 행동결정 예시
(LOF 위험->파워 증가/속도 감소, Keyhole 위험->파워 감소/속도 증가) 그대로.

폭/깊이 예측은 공식 Looped Transformer(v3, model/LoopedTransformer/legacy/)를
우선 사용한다 (digitaltwin/looped_transformer_integration.md 데이터 계약 참고).
체크포인트가 없거나 로드에 실패하면 ROM(sim.lpbf)으로 자동 폴백한다 -- LOF
판정·regime 분류 자체는 폭/깊이 소스와 무관하게 항상 ROM의 공식을 그대로 쓴다.

2026-10-03: model/LoopedTransformer/ 리팩토링으로 dataLT·interfaceLT·
modelLT·physicsLT·v3 체크포인트를 legacy/ 하위로 옮김 (새 model/LoopedTransformer/v5/
패키지는 실시간 스트림용 StreamingLTService라 이 모듈의 단발 호출 용도와는 다름).
그래서 legacy 경로 + 절대 패키지 import로 맞춤.
"""
import os

import numpy as np
import pandas as pd

from sim.lpbf import evaluate_physical_state, melt_pool_width, melt_pool_depth, lof_risk, classify_regime

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
    """공정조건(스칼라 또는 배열)을 LT 입력 7특성으로 변환해 (width_m, depth_m) 배열 예측.
    배열이면 LT를 한 번만 호출한다 (격자 탐색용)."""
    p, v = np.broadcast_arrays(np.atleast_1d(np.asarray(power_w, float)), np.atleast_1d(np.asarray(speed_mm_s, float)))
    d_mm = spot_diameter_um / 1000.0
    v_mps = v / 1000.0
    d_m = d_mm / 1000.0
    h_s_vol = _RHO * _CP * (_TM - _T0)
    rows = pd.DataFrame({
        "P_laser": p, "v_scan": v, "d_laser": d_mm, "t_powder": layer_thickness_um,
        "linear_energy": p / v,
        "peclet_number": v_mps * d_m / _ALPHA,
        "normalized_enthalpy": _ETA * p / (h_s_vol * (_ALPHA * d_m ** 3 * v_mps) ** 0.5),
    })
    pred = _lt_runner.getPredict(_LT_CHECKPOINT, rows)
    width_um = pred["melt_width"].to_numpy(float)
    return width_um * 1e-6, pred["aspect_ratio"].to_numpy(float) * width_um * 1e-6


# 판단에 쓸 예측 모델: "v3"(공식 legacy LT, 기본) | "v5"(스트리밍 LT의 정상상태 평가 모드).
# 최종 모델 확정 시 이 값(또는 환경변수 DECISION_MODEL)만 바꾸면 된다. 실패하면 v3 -> ROM 순으로 폴백.
MODEL_BACKEND = os.environ.get("DECISION_MODEL", "v3")
_v5_model = None


def _v5_predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um):
    """V5를 서비스(이력·SQLite·난수 센서) 없이 정상상태로 배치 평가. (width_m, depth_m)."""
    global _v5_model
    if _v5_model is None:
        from sim.v5_condition_model import V5ConditionModel

        _v5_model = V5ConditionModel()
    w_um, d_um, _ = _v5_model.predict(power_w, speed_mm_s, spot_diameter_um / 1000.0, layer_thickness_um)
    return w_um * 1e-6, d_um * 1e-6


def _predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um, backend=None):
    """선택된 LT 우선, 실패 시 v3 -> ROM 폴백. (width_m, depth_m, source) -- 입력이 배열이어도 동작.
    backend: "v3"/"v5" (None이면 MODEL_BACKEND)."""
    if (backend or MODEL_BACKEND) == "v5":
        try:
            w, d = _v5_predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um)
            return w, d, "lt_v5"
        except Exception:
            pass
    if _LT_AVAILABLE:
        try:
            w, d = _lt_predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um)
            return w, d, "lt_v3"
        except Exception:
            pass
    p = np.atleast_1d(np.asarray(power_w, float))
    v = np.atleast_1d(np.asarray(speed_mm_s, float)) / 1000.0
    return melt_pool_width(p, v), melt_pool_depth(p, v), "rom_fallback"


# 조건 탐색 범위 = 학습 데이터(316L 실측) 범위. 이 밖은 외삽이라 추천하지 않는다.
SEARCH_POWER_W = (50.0, 500.0, 10.0)        # (min, max, step)
SEARCH_SPEED_MM_S = (225.0, 1500.0, 25.0)

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
    w, d, source = _predict_width_depth(power_w, speed_mm_s, spot_diameter_um, layer_thickness_um)
    status, risk, regime = _classify(w, d, power_w, speed_mm_s, spot_diameter_um, hatch_um, layer_thickness_um)
    return {
        "status": str(status[0]),
        "depth_um": round(float(d[0]) * 1e6, 1),
        "width_um": round(float(w[0]) * 1e6, 1),
        "lof_criterion": round(float(risk[0]), 3),
        "regime": str(regime[0]),
        "source": source,
    }


def _classify(w, d, power_w, speed_mm_s, spot_diameter_um, hatch_um, layer_thickness_um):
    """폭/깊이(m) 배열 -> (status, lof_risk, regime) 배열. 스칼라 판단과 격자 탐색이 공용."""
    risk = lof_risk(hatch_um * 1e-6, layer_thickness_um * 1e-6, w, d)
    regime = classify_regime(power_w, np.asarray(speed_mm_s, float) / 1000.0, spot_diameter_um * 1e-6)
    status = np.where(risk > 1.0, "lof_risk", np.where(regime == "keyhole_or_transition", "keyhole_risk", "normal"))
    return np.atleast_1d(status), np.atleast_1d(risk), np.atleast_1d(regime)


def find_safe_conditions(power_w, speed_mm_s, spot_diameter_um=100.0, hatch_um=100.0, layer_thickness_um=30.0,
                         lof_margin=0.95, backend=None):
    """학습 데이터 범위 안 (출력, 속도) 격자에서, 현재 조건에서 가장 작게 바꿔서 normal이 되는
    조건을 찾는다. LT를 한 번만 호출. 없으면 None. lof_margin: 경계값 근처 조건은 제외."""
    p_axis = np.arange(SEARCH_POWER_W[0], SEARCH_POWER_W[1] + 1e-9, SEARCH_POWER_W[2])
    v_axis = np.arange(SEARCH_SPEED_MM_S[0], SEARCH_SPEED_MM_S[1] + 1e-9, SEARCH_SPEED_MM_S[2])
    p, v = (a.ravel() for a in np.meshgrid(p_axis, v_axis))
    w, d, _ = _predict_width_depth(p, v, spot_diameter_um, layer_thickness_um, backend)
    status, risk, _ = _classify(w, d, p, v, spot_diameter_um, hatch_um, layer_thickness_um)
    ok = (status == "normal") & (risk <= lof_margin)
    if not ok.any():
        return None
    dist = np.hypot((p - power_w) / power_w, (v - speed_mm_s) / speed_mm_s)  # 상대 변경량
    i = int(np.argmin(np.where(ok, dist, np.inf)))
    return {"power_w": float(p[i]), "speed_mm_s": float(v[i]), "change_distance": round(float(dist[i]), 3)}


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

    if not any(c["resolved"] for c in evaluated):
        # 한 축 ±15% 조정으로 해소가 안 되면, 가장 작게 바꿔서 안전해지는 (출력, 속도) 조합을 탐색
        safe = find_safe_conditions(power_w, speed_mm_s, spot_diameter_um, hatch_um, layer_thickness_um)
        if safe:
            new_state = assess_state(safe["power_w"], safe["speed_mm_s"], spot_diameter_um, hatch_um, layer_thickness_um)
            evaluated.append({
                "action": "안전 조건으로 재설정 (출력·속도 동시 조정)",
                "new_power_w": safe["power_w"],
                "new_speed_mm_s": safe["speed_mm_s"],
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


def recommend_action_from_prediction(
    power_w,
    speed_mm_s,
    spot_diameter_mm,
    hatch_spacing_um,
    layer_thickness_um,
    predicted_width_um,
    predicted_depth_um,
):
    """Use LT geometry for diagnosis and the ROM to validate bounded actions."""
    state = evaluate_physical_state(
        power_w,
        speed_mm_s,
        spot_diameter_mm,
        hatch_spacing_um,
        layer_thickness_um,
        predicted_width_um,
        predicted_depth_um,
    )
    if state["status"] == "normal":
        return {
            "current_state": state,
            "recommended_action": "공정 유지",
            "reason": "LT 예측 형상과 물리 제약이 설정된 정상 범위입니다.",
            "candidates_evaluated": [],
            "verification_model": "Rosenthal ROM",
        }

    bounds = {"power_w": (50.0, 500.0), "speed_mm_s": (100.0, 2000.0)}
    candidates = []
    for name, adjust in ACTION_CANDIDATES[state["status"]]:
        candidate_power, candidate_speed = adjust(power_w, speed_mm_s)
        if not (
            bounds["power_w"][0] <= candidate_power <= bounds["power_w"][1]
            and bounds["speed_mm_s"][0] <= candidate_speed <= bounds["speed_mm_s"][1]
        ):
            continue
        rom_state = assess_state(
            candidate_power,
            candidate_speed,
            spot_diameter_um=spot_diameter_mm * 1000.0,
            hatch_um=hatch_spacing_um,
            layer_thickness_um=layer_thickness_um,
        )
        candidates.append({
            "action": name,
            "laser_power_w": round(candidate_power, 1),
            "scan_speed_mm_s": round(candidate_speed, 1),
            "rom_state": rom_state,
            "resolved": rom_state["status"] == "normal",
        })

    if not candidates:
        return {
            "current_state": state,
            "recommended_action": "운전자 검토 필요",
            "reason": "안전 범위 안에서 검증 가능한 행동 후보가 없습니다.",
            "candidates_evaluated": [],
            "verification_model": "Rosenthal ROM",
        }

    resolved = [candidate for candidate in candidates if candidate["resolved"]]
    best = resolved[0] if resolved else min(
        candidates,
        key=lambda candidate: candidate["rom_state"]["lof_criterion"],
    )
    return {
        "current_state": state,
        "recommended_action": best["action"],
        "reason": (
            f"{best['action']} 후보를 Rosenthal ROM으로 검증했습니다. "
            f"{'정상 범위' if best['resolved'] else '위험 완화 여부를 작업자가 확인해야 합니다'}."
        ),
        "candidates_evaluated": candidates,
        "verification_model": "Rosenthal ROM",
    }


def demo():
    lof_case = recommend_action(power_w=80, speed_mm_s=1500)
    assert lof_case["current_state"]["status"] != "normal", "80W/1500mm/s should trigger a risk state"
    assert lof_case["candidates_evaluated"], "risk state should generate action candidates"
    assert any(c["resolved"] for c in lof_case["candidates_evaluated"]), "LOF 위험은 해소하는 조건을 찾아야 함"
    print(f"demo ok (위험조건): {lof_case['recommended_action']} - {lof_case['reason']}")

    keyhole_case = recommend_action(power_w=450, speed_mm_s=300)
    assert keyhole_case["current_state"]["status"] == "keyhole_risk", keyhole_case
    assert any(c["resolved"] for c in keyhole_case["candidates_evaluated"]), "keyhole 위험도 해소 조건을 찾아야 함"
    print(f"demo ok (keyhole): {keyhole_case['recommended_action']} - {keyhole_case['reason']}")

    normal_case = recommend_action(power_w=250, speed_mm_s=800)
    assert normal_case["current_state"]["status"] == "normal", normal_case
    print(f"demo ok (정상조건): status={normal_case['current_state']['status']}, "
          f"action={normal_case['recommended_action']}")


if __name__ == "__main__":
    demo()
