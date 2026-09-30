"""[추가 실험 c] 기준이 노트북 방식의 모델 선택으로 바꾸면 결과가 달라지는가.

- 기준이 분할(70/15/15, seed 42)의 train 으로 학습
- 매 epoch 끝에 val 데이터 loss 를 재서 가장 낮은 epoch 의 가중치 사용, 20 epoch 개선 없으면 중단
- 채점은 test (val 점수는 epoch 선택에 쓰였으므로 참고용)
- epoch 가 있는 신경망(MLP, LT)만 해당. 나머지 baseline 은 results_v3/test 결과와 같다

실행: python model/Baseline/select_val.py
결과: results_v3/select_val/raw.csv (모델 × seed × 평가), summary.csv (평균 ± 표준편차)
"""
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

import looped
import mlp
from common import RESULTS, SEEDS, evaluate, load_data

MODELS = {
    "MLP": mlp.run,
    "LT-v3": partial(looped.run, physics=True),
    "LT-v3-NoPhys": partial(looped.run, physics=False),
}


def main():
    dv, dt = load_data("val"), load_data("test")   # train 과 표준화 기준은 두 쪽이 같다
    n = len(dv.x_eval)
    # 한 번 학습으로 val · test 를 함께 예측하려고 평가 입력을 이어 붙임 (행마다 따로 예측하므로 결과 같음)
    d = SimpleNamespace(**vars(dt))
    d.x_eval = np.vstack([dv.x_eval, dt.x_eval])
    d.x_sel, d.y_sel = dv.x_eval, dv.y_eval

    rows = []
    for name, run in MODELS.items():
        for seed in SEEDS:
            print(f"[select_val] running {name} seed={seed}", flush=True)
            pred = run(d, seed)
            rows.append({"model": name, "seed": seed, "eval": "val", **evaluate(pred[:n], dv)})
            rows.append({"model": name, "seed": seed, "eval": "test", **evaluate(pred[n:], dt)})

    out = RESULTS / "select_val"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    g = raw.drop(columns="seed").groupby(["eval", "model"], sort=False)
    (g.mean().map("{:.4g}".format) + " ± " + g.std(ddof=0).map("{:.2g}".format)).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
