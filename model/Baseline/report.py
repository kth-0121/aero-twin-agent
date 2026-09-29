"""[통합 문서] 4가지 평가 결과(results/<scheme>/raw.csv)를 모아 results/results.md 하나로 정리한다.
실행: python model/Baseline/report.py   (run_all.py 로 4가지 scheme 을 먼저 돌려야 함)
"""
from pathlib import Path

import pandas as pd

from common import load_data

RES = Path(__file__).resolve().parent / "results"
BASELINES = ["Linear", "RandomForest", "GradientBoosting", "GPR", "MLP"]
LT_EVAL = ["LT-3", "LT-6", "LT-12"]
SHOW = ["std_MSE", "mean_MAPE", "W_MAPE", "D_MAPE", "A_MAPE", "W_MAE", "D_MAE", "A_MAE"]

SCHEMES = {   # scheme: (이름, 무엇을 확인하나)
    "val": ("① val 평가", "기본 비교 (기준이 분할)"),
    "test": ("② test 평가", "val 에만 맞춰진 결과가 아닌지 (한 번도 안 쓴 데이터)"),
    "group": ("③ 공정조건 단위 분할", "같은 조건 반복 실험을 외워서 맞히는 효과를 뺀 성능"),
    "regime": ("④ 영역 이동 (conduction → keyhole)", "처음 보는 영역에서도 버티는지 (외삽)"),
}

HEADER = """# Baseline 비교 결과 (4가지 평가 통합)

## 평가 방식 한눈에
| 평가 | 설명 |
|---|---|
| ① val | 기준이가 나눈 방식 그대로 train 473개로 학습하고 val 101개로 채점. 지금까지 해온 기본 비교 |
| ② test | 같은 train 으로 학습하고, 한 번도 쓰지 않은 test 103개로 채점. val 결과가 우연히 그 시험지에만 잘 맞은 것은 아닌지 확인 |
| ③ 공정조건 단위 분할 | 같은 조건(P, v, d, t)으로 두 번 실험한 쌍이 train/test 에 나뉘지 않게 묶어서 분할. 거의 같은 문제를 미리 보고 맞히는 효과를 뺀 엄격한 평가 |
| ④ 영역 이동 | 얕은 용융풀(conduction, D/W < 0.5)로만 학습하고 깊은 용융풀(keyhole, D/W ≥ 0.5)로 채점. 처음 보는 영역에서도 예측이 버티는지(외삽) 보는 가장 어려운 평가 |

## 공통 실험 조건
- 입력 7개 (P, v, d, t, E_L, Pe, H*) → 출력 3개 (W, R = D/W, A). 표준화 기준은 각 평가의 train 으로만 계산
- 반복: 모델 seed 0~4 로 5회 → 평균 ± 표준편차 (데이터 분할은 평가마다 고정)
- 신경망(MLP, LT): AdamW, lr 1e-3, batch 64, 100 epoch 고정 (평가 데이터로 best epoch 를 고르지 않음)

## 지표
| 지표 | 뜻 |
|---|---|
| std_MSE | 표준화 공간의 평균제곱오차 (W · R · A 평균). 단위가 없어 항목을 합칠 수 있음 → **10% 판정 기준** |
| MAPE (%) | 평균 상대오차. mean_MAPE = W · D · A 의 MAPE 평균 |
| MAE | 평균 절대오차 (W, D: um / A: um^2) |
| 개선률 (%) | (baseline std_MSE − LT std_MSE) / baseline std_MSE × 100. **10 이상이면 목표 달성** |

## 모델
| 모델 | 어떤 모델인가 | 비고 |
|---|---|---|
| Linear | 입력의 가중합으로 예측하는 가장 단순한 기준선 | 무작위성 없음 |
| RandomForest | 결정트리 여러 개의 예측을 평균 | scikit-learn 기본값 |
| GradientBoosting | 앞 트리의 오차를 다음 트리가 보완하며 쌓음 | W, R, A 각각 별도 모델 |
| GPR | 비슷한 입력 → 비슷한 출력 가정(kernel). 소량 데이터에 강함 | RBF + 노이즈 |
| MLP | 3층 신경망 (기준이 `StateMLP`) | LT 와 같은 학습 설정 |
| ROM | 학습 없이 물리식만으로 계산 (Rosenthal W, H*·Pe 깊이, 반타원 A) | 참고용 |
| LT-k | Looped Transformer, loop k 회. 학습 · 추론 모두 물리 피드백 | **평가 대상** |
| LT-3-NoPhys | 학습 · 추론 모두 물리 피드백 없음 | 내부 비교: LT-3 과의 차이 = 물리 피드백 효과 |
| LT-k-InferOff | 학습은 물리 피드백, 추론은 끔 (`interfaceLT.py` 방식) | 조건 불일치 참고용 (① 에서만 실행) |
"""


def to_md(df, index_name="model"):
    lines = ["| " + " | ".join([index_name, *map(str, df.columns)]) + " |",
             "|" + "---|" * (len(df.columns) + 1)]
    lines += ["| " + " | ".join([str(i), *map(str, r)]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(lines) + "\n"


def main():
    raws = {s: pd.read_csv(RES / s / "raw.csv") for s in SCHEMES}
    means = {s: r.drop(columns="seed").groupby("model", sort=False).mean() for s, r in raws.items()}
    stds = {s: r.drop(columns="seed").groupby("model", sort=False).std(ddof=0) for s, r in raws.items()}

    # 평가 방식 설명표
    rows = []
    for s, (name, purpose) in SCHEMES.items():
        d = load_data(s)
        rows.append([name, len(d.x_train), len(d.x_eval), purpose])
    schemes = pd.DataFrame(rows, columns=["평가", "train", "평가 데이터", "확인하는 것"]).set_index("평가")

    # 종합표 1: 모델 × 평가 std_MSE
    overall = pd.DataFrame({SCHEMES[s][0]: means[s]["std_MSE"].map("{:.3f}".format)
                            + " ± " + stds[s]["std_MSE"].map("{:.3f}".format) for s in SCHEMES}).fillna("-")

    # 종합표 2: 평가별 LT 개선률 (가장 강한 baseline 대비)
    judge = {}
    for s in SCHEMES:
        m = means[s]["std_MSE"]
        best = m[BASELINES].idxmin()
        judge[SCHEMES[s][0]] = {f"{lt} vs 최강 baseline({best})": round((m[best] - m[lt]) / m[best] * 100, 1)
                                for lt in LT_EVAL}
    judge = pd.DataFrame(judge).T

    parts = [HEADER, "\n## 평가 방식 4가지\n\n", to_md(schemes, "평가"),
             "\n## 종합 1: 평가별 std_MSE (낮을수록 좋음, 평균 ± 표준편차)\n\n", to_md(overall),
             "\n## 종합 2: 가장 강한 baseline 대비 LT 개선률 (%) — 10 이상이면 목표 달성\n\n", to_md(judge, "평가")]

    for s, (name, purpose) in SCHEMES.items():
        m, sd = means[s], stds[s]
        table = m[SHOW].map("{:.4g}".format) + " ± " + sd[SHOW].map("{:.2g}".format)
        improve = pd.DataFrame({lt: {b: round((m.loc[b, "std_MSE"] - m.loc[lt, "std_MSE"])
                                              / m.loc[b, "std_MSE"] * 100, 1) for b in BASELINES}
                                for lt in LT_EVAL})
        parts += [f"\n---\n\n## {name}\n{purpose}\n\n### 주요 지표\n\n", to_md(table),
                  "\n### baseline 별 LT 개선률 (%)\n\n", to_md(improve, "baseline")]

    (RES / "results.md").write_text("".join(parts), encoding="utf-8")
    print("written:", RES / "results.md")


if __name__ == "__main__":
    main()
