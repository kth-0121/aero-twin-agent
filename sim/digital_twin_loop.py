"""STL/CLI 빌드 이벤트 -> 판단 -> 행동결정 루프.

digitaltwin/ pre-LT 파이프라인(STL -> 슬라이싱 -> 공정이벤트 -> FiPy/PyBullet)은
README에 명시된 대로 "deliberately has no LT call"로 끝난다. 이 모듈이 그
빈 칸을 메운다: BuildEvent 하나하나를 sim.decision(LT v3 기반 상태판단+행동추천)에
흘려서, "STL 업로드 -> Agent 처리 -> 판단 -> 장비제어명령" 루프의 "판단" 단계를 구현.

digitaltwin.build_job/cli_toolpath/process만 쓰고 trimesh/fipy/pybullet 의존이
없다 -- 그건 시각화·열해석 레이어라 여기(판단 단계)에는 필요 없음.
"""
from digitaltwin.build_job import BuildEvent, BuildPlan, load_cli_build_plan

from sim.decision import recommend_action


def _decision_kwargs(event: BuildEvent) -> dict:
    """BuildEvent.process(mm 단위) -> sim.decision 인자(um 단위)로 변환."""
    p = event.process
    return {
        "power_w": p["laser_power_w"],
        "speed_mm_s": p["scan_speed_mm_s"],
        "spot_diameter_um": p["spot_diameter_mm"] * 1000.0,
        "hatch_um": p["hatch_spacing_mm"] * 1000.0,
        "layer_thickness_um": p["layer_thickness_um"],
    }


def decide(event: BuildEvent) -> dict:
    """이벤트 한 건에 대한 상태판단 + 행동추천. 장비제어명령 직전 단계 출력."""
    decision = recommend_action(**_decision_kwargs(event))
    return {
        "event_id": event.event_id,
        "layer_index": event.layer_index,
        "path_type": event.path_type,
        **decision,
    }


def run_decision_loop(plan: BuildPlan, event_limit: int | None = None):
    """BuildPlan의 이벤트를 순서대로 판단한다."""
    for i, event in enumerate(plan.iter_events()):
        if event_limit is not None and i >= event_limit:
            break
        yield decide(event)


def demo():
    import tempfile
    from pathlib import Path

    from digitaltwin.cli_toolpath import LayerPaths, write_cli
    from digitaltwin.process import demo_316l_settings

    with tempfile.TemporaryDirectory() as tmp:
        cli_path = Path(tmp) / "demo.cli"
        write_cli(cli_path, [
            LayerPaths(layer_index=0, z_mm=0.03, contours=(),
                       hatches=(((0.0, 0.0), (5.0, 0.0)), ((0.0, 0.1), (5.0, 0.1)))),
        ])
        plan = load_cli_build_plan(cli_path, demo_316l_settings())
        results = list(run_decision_loop(plan))

    assert len(results) == 2, results
    assert all(r["current_state"]["status"] == "normal" for r in results), results
    print(f"demo ok: {len(results)}개 이벤트 판단 완료, 전부 status=normal "
          f"(200W/800mm/s -> conduction) action={results[0]['recommended_action']} "
          f"source={results[0]['current_state']['source']}")


if __name__ == "__main__":
    demo()
