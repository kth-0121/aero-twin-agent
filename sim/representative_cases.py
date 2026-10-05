"""대표 테스트 5건 실행 (기술설명서 '대표 테스트 5건 결과'와 같은 번호).

  1  정상 조건      250W / 800mm/s   -> 정상, 공정 유지
  2  LOF 위험       80W / 1500mm/s   -> 위험 판정 후 해소 조건 추천, 재검증 정상
  3  keyhole 위험   450W / 300mm/s   -> 위험 판정 후 해소 조건 추천, 재검증 정상
  4  무작위 120조건 (학습 범위 안)   -> 위험 조건 전부 해소, 재검증 일치
  5  STL 전체 흐름  demo_ring.stl    -> 등록·슬라이싱 -> 열해석 -> AI 분석 -> 권고 -> 승인

실행 (저장소 루트):
  python -m sim.representative_cases          # 1~4 (약 1분)
  python -m sim.representative_cases --stl    # 1~5 (5는 FiPy 열해석 포함, 수십 분)

재검증은 같은 예측 모델로 다시 판정하는 것이라 독립 검증이 아니다.
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sim.decision import recommend_action
from sim.evaluate_e2e import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_STL = os.path.join(ROOT, "digitaltwin", "demo_model", "demo_ring.stl")


def _chosen(r):
    return next((c for c in r["candidates_evaluated"] if c["action"] == r["recommended_action"]), None)


def single_case(power_w, speed_mm_s, expected_status):
    r = recommend_action(power_w, speed_mm_s)
    s = r["current_state"]
    c = _chosen(r)
    if expected_status == "normal":
        ok = s["status"] == "normal" and c is None
        result = f"{s['status']} (LOF {s['lof_criterion']}) -> {r['recommended_action']}"
    else:
        ok = s["status"] == expected_status and c is not None and c["resulting_state"]["status"] == "normal"
        after = c["resulting_state"] if c else {}
        result = (f"{s['status']} (LOF {s['lof_criterion']}) -> {c['new_power_w']}W / {c['new_speed_mm_s']}mm/s, "
                  f"재검증 {after.get('status')} (LOF {after.get('lof_criterion')})" if c else f"{s['status']} -> 추천 없음")
    return ok, result


def random_case(n=120):
    m = evaluate(n)
    ok = m["resolve_rate"] == 1.0 and m["reverify_rate"] == 1.0
    return ok, (f"위험 {m['risk_total']}건 중 해소 {m['resolve_rate']:.0%}, 재검증 일치 {m['reverify_rate']:.0%}, "
                f"응답 중앙 {m['latency_median_s']:.2f}s")


def stl_case(path=DEMO_STL):
    from fastapi.testclient import TestClient
    from digitaltwin.backend.main import app

    with TestClient(app) as c:  # with 블록 안에서만 업로드 후 백그라운드 슬라이싱이 진행됨
        t0 = time.time()
        with open(path, "rb") as f:
            bid = c.post("/api/builds?source_units=mm", content=f.read()).json()["build_id"]
        while True:
            b = c.get(f"/api/builds/{bid}").json()
            if "event_count" in b:
                break
            if b.get("registration_status") == "failed":
                return False, f"STL 등록 실패: {b}"
            time.sleep(0.5)
        n = b["event_count"]
        t1 = time.time()
        while True:  # 서버가 한 번에 최대 2개 벡터만 받는다
            r = c.post(f"/api/builds/{bid}/thermal-step?count=2")
            r.raise_for_status()
            if r.json()["phase"] == "thermal_complete":
                break
        t2 = time.time()
        c.post(f"/api/builds/{bid}/agent/start").raise_for_status()
        while True:
            r = c.post(f"/api/builds/{bid}/agent/step?count=64")
            r.raise_for_status()
            s = r.json()
            if s["phase"] == "review":
                break
        t3 = time.time()
        rec = s["recommendation"]
        final = c.post(f"/api/builds/{bid}/review", json={"approved": True}).json()
    ok = final["phase"] == "approved" and sum(s["risk_counts"].values()) == n
    return ok, (f"{n}벡터 · 등록·슬라이싱 {t1 - t0:.1f}s · 열해석 {t2 - t1:.0f}s · 분석 {t3 - t2:.0f}s "
                f"({s['risk_counts']}) -> {rec['action']} {rec['laser_power_w']}W/{rec['scan_speed_mm_s']}mm/s -> {final['phase']}")


def main(with_stl=False):
    cases = [
        ("1 정상 조건", lambda: single_case(250, 800, "normal")),
        ("2 LOF 위험", lambda: single_case(80, 1500, "lof_risk")),
        ("3 keyhole 위험", lambda: single_case(450, 300, "keyhole_risk")),
        ("4 무작위 120조건", random_case),
    ]
    if with_stl:
        cases.append(("5 STL 전체 흐름", stl_case))
    results = []
    for name, fn in cases:
        ok, text = fn()
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {text}", flush=True)
    if not with_stl:
        print("[SKIP] 5 STL 전체 흐름: --stl 옵션으로 실행 (FiPy·trimesh 필요, 수십 분)")
    print(f"\n{sum(results)}/{len(results)} 통과")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main("--stl" in sys.argv) else 1)
