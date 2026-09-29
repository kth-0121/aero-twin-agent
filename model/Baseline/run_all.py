"""[실행] 평가 방식(scheme) 하나에 대해 모델 × seed 5개를 돌려 결과를 저장한다.

실행 (aero-twin-agent 폴더에서): python model/Baseline/run_all.py <scheme>
    scheme = val | test | group | regime   (설명은 common.py)
결과: model/Baseline/results/<scheme>/raw.csv (모델 × seed 별 지표), summary.csv (평균 ± 표준편차)
4가지를 모두 돌린 뒤 report.py 로 통합 문서(results/results.md)를 만든다.
"""
import sys
from functools import partial
from pathlib import Path

import pandas as pd

import gpr
import linear
import looped
import mlp
import rom
import tree_gbm
import tree_rf
from common import SEEDS, evaluate, load_data

BASELINES = {
    "Linear": linear.run,
    "RandomForest": tree_rf.run,
    "GradientBoosting": tree_gbm.run,
    "GPR": gpr.run,
    "MLP": mlp.run,
}
MODELS = {
    **BASELINES,
    "ROM": rom.run,
    **{name: partial(looped.run, loop_steps=n, train_physics=tp, infer_physics=ip)
       for name, (n, tp, ip) in looped.VARIANTS.items()},
}


def main(scheme):
    data = load_data(scheme)
    # InferOff(조건 불일치) 는 val 에서 이미 문제를 확인했으므로 나머지 평가에서는 생략
    models = {k: v for k, v in MODELS.items() if scheme == "val" or not k.endswith("InferOff")}
    print(f"[{scheme}] train {len(data.x_train)} / eval {len(data.x_eval)}", flush=True)

    rows = []
    for name, run in models.items():
        for seed in SEEDS:
            print(f"[{scheme}] running {name} seed={seed}", flush=True)
            rows.append({"model": name, "seed": seed, **evaluate(run(data, seed), data)})

    out = Path(__file__).resolve().parent / "results" / scheme
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    g = raw.drop(columns="seed").groupby("model", sort=False)
    (g.mean().map("{:.4g}".format) + " ± " + g.std(ddof=0).map("{:.2g}".format)).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
