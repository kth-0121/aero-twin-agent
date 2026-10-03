"""E2E 판단 파이프라인 평가. 개발계획서 정성 목표를 수치로 잰다:
  - 상태 변화에 따른 적절한 행동 후보 생성 후 선택  -> 위험 조건에서 '해소되는' 추천을 내는 비율
  - 주요 공정변수 변화에 대한 단조성 재현            -> 출력↑ 깊이↑, 속도↑ 깊이↓ 비율
  - 학습에 포함되지 않은 조건 처리                    -> 범위 밖 입력 경고
정답 라벨이 필요한 정확도 평가가 아니라, 파이프라인이 자기모순 없이 동작하는지(일관성)를 본다.
LT 예측 정확도 자체는 model/Baseline의 평가 결과를 참고.

실행: python -m sim.evaluate_e2e [N]   (저장소 루트, KMP_DUPLICATE_LIB_OK=TRUE)
"""
import sys
import time

import numpy as np

from sim.decision import SEARCH_POWER_W, SEARCH_SPEED_MM_S, _predict_width_depth, assess_state, recommend_action

P_RANGE, V_RANGE = SEARCH_POWER_W[:2], SEARCH_SPEED_MM_S[:2]


def evaluate(n=120, seed=0):
    rng = np.random.default_rng(seed)
    conds = list(zip(rng.uniform(*P_RANGE, n).round(0), rng.uniform(*V_RANGE, n).round(0)))

    stats = {"normal": 0, "lof_risk": 0, "keyhole_risk": 0}
    risk_total = resolved = reverified = in_range = 0
    normal_clean = 0
    changes, secs = [], []
    for p, v in conds:
        t = time.time()
        r = recommend_action(p, v)
        secs.append(time.time() - t)
        status = r["current_state"]["status"]
        stats[status] += 1
        if status == "normal":
            normal_clean += r["recommended_action"] == "공정 유지" and not r["candidates_evaluated"]
            continue
        risk_total += 1
        best = next(c for c in r["candidates_evaluated"] if c["action"] == r["recommended_action"])
        if best["resolved"]:
            resolved += 1
            reverified += assess_state(best["new_power_w"], best["new_speed_mm_s"])["status"] == "normal"
            in_range += P_RANGE[0] <= best["new_power_w"] <= P_RANGE[1] and V_RANGE[0] <= best["new_speed_mm_s"] <= V_RANGE[1]
            changes.append(np.hypot((best["new_power_w"] - p) / p, (best["new_speed_mm_s"] - v) / v))

    # 단조성: 속도 고정 출력 스윕 / 출력 고정 속도 스윕, 전 구간 LT 배치 호출
    mono_p, mono_v = [], []
    for v in np.linspace(*V_RANGE, 12):
        _, d, _ = _predict_width_depth(np.linspace(*P_RANGE, 40), np.full(40, v), 100.0, 30.0)
        mono_p.append(np.mean(np.diff(d) >= 0))
    for p in np.linspace(*P_RANGE, 12):
        _, d, _ = _predict_width_depth(np.full(40, p), np.linspace(*V_RANGE, 40), 100.0, 30.0)
        mono_v.append(np.mean(np.diff(d) <= 0))

    return {
        "n": n, "status_counts": stats, "risk_total": risk_total,
        "resolve_rate": resolved / risk_total if risk_total else float("nan"),
        "reverify_rate": reverified / resolved if resolved else float("nan"),
        "in_range_rate": in_range / resolved if resolved else float("nan"),
        "normal_clean_rate": normal_clean / stats["normal"] if stats["normal"] else float("nan"),
        "median_change": float(np.median(changes)) if changes else float("nan"),
        "monotonic_power_depth": float(np.mean(mono_p)),
        "monotonic_speed_depth": float(np.mean(mono_v)),
        "latency_median_s": float(np.median(secs)), "latency_p95_s": float(np.percentile(secs, 95)),
    }


def report(m):
    s = m["status_counts"]
    return "\n".join([
        f"# E2E 판단 파이프라인 평가 (학습 범위 내 무작위 {m['n']}개 조건, seed=0)",
        "",
        f"- 상태 분포: 정상 {s['normal']} / LOF 위험 {s['lof_risk']} / Keyhole 위험 {s['keyhole_risk']}",
        "",
        "| 지표 | 값 | 의미 |",
        "|---|---|---|",
        f"| 위험 해소율 | {m['resolve_rate']:.1%} ({m['risk_total']}건 중) | 위험 조건에서 '해소되는' 추천을 낸 비율 |",
        f"| 재검증 일치율 | {m['reverify_rate']:.1%} | 추천 조건을 다시 판단해도 정상인 비율 (자기모순 없음) |",
        f"| 추천의 학습범위 준수율 | {m['in_range_rate']:.1%} | 추천 조건이 학습 데이터 범위 안(외삽 아님)인 비율 |",
        f"| 정상 조건 오추천 없음 | {m['normal_clean_rate']:.1%} | 정상 조건에 '공정 유지'만 낸 비율 |",
        f"| 추천 변경량(중앙값) | {m['median_change']:.2f} | 현재 조건 대비 상대 변경 거리 (작을수록 최소 개입) |",
        f"| 단조성: 출력↑→깊이↑ | {m['monotonic_power_depth']:.1%} | 12개 속도에서 출력 스윕 시 인접 구간 중 비감소 비율 |",
        f"| 단조성: 속도↑→깊이↓ | {m['monotonic_speed_depth']:.1%} | 12개 출력에서 속도 스윕 시 인접 구간 중 비증가 비율 |",
        f"| 추천 응답시간 | 중앙 {m['latency_median_s']:.2f}s / p95 {m['latency_p95_s']:.2f}s | 판단+후보 생성+재검증 전체 |",
        "",
        "한계: ROM/동일 모델 재평가 기반의 일관성 평가이며 실제 설비 결과와의 비교(Sim-to-Real)가 아님. "
        "LT 점추정은 불확실성을 제공하지 않음.",
    ])


if __name__ == "__main__":
    print(report(evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 120)))
