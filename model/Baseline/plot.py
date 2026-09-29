"""[그래프] 평가 방식별 모델 비교 그래프 → results/figures/<scheme>.png
실행: python model/Baseline/plot.py   (run_all.py 로 결과를 먼저 만들어야 함)

그래프 1장 = 평가 1개. 왼쪽 std_MSE(10% 판정 기준), 오른쪽 평균 MAPE(%).
- 막대 = seed 5회 평균, 가는 선 = ± 표준편차, 막대 끝 숫자 = 평균값
- 색 = 모델 종류 (baseline / LT 평가 대상 / 내부 비교 / 물리식)
- 왼쪽 점선 2개 = 가장 좋은 baseline 값, 그보다 10% 낮은 값(= LT 목표선)
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

RES = Path(__file__).resolve().parent / "results"
SCHEMES = {
    "val": "① val 평가",
    "test": "② test 평가",
    "group": "③ 공정조건 단위 분할",
    "regime": "④ 영역 이동 (conduction → keyhole)",
}
BASELINES = ["Linear", "RandomForest", "GradientBoosting", "GPR", "MLP"]

# 색: dataviz 기본 팔레트 slot 1~3 (검증 통과) + 참고용 회색
GROUPS = {   # 종류: (색, 해당 모델)
    "Baseline": ("#2a78d6", BASELINES),
    "LT (평가 대상)": ("#eb6834", ["LT-3", "LT-6", "LT-12"]),
    "LT 내부 비교 (물리 없음)": ("#1baf7a", ["LT-3-NoPhys"]),
    "물리식만 (ROM, 참고)": ("#9a9893", ["ROM"]),
}
COLOR = {m: c for c, models in GROUPS.values() for m in models}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def panel(ax, mean, std, col, fmt, title):
    y = range(len(mean))
    ax.barh(y, mean[col], height=0.6, color=[COLOR[m] for m in mean.index], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(mean[col], y, xerr=std[col], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for i, (v, e) in enumerate(zip(mean[col], std[col])):
        ax.text(v + e + ax.get_xlim()[1] * 0.01, i, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_yticks(list(y), mean.index, fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)             # 숫자 라벨 공간


def plot(scheme, title):
    raw = pd.read_csv(RES / scheme / "raw.csv")
    raw = raw[~raw["model"].str.endswith("InferOff")]     # 조건 불일치 모델은 축을 망가뜨려 제외 (표 참고)
    g = raw.drop(columns="seed").groupby("model")
    mean, std = g.mean(), g.std(ddof=0)
    order = mean["std_MSE"].sort_values().index          # std_MSE 좋은 순, 두 패널 같은 순서
    mean, std = mean.loc[order], std.loc[order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, facecolor=SURFACE,
                                 gridspec_kw={"wspace": 0.08})
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
    panel(a1, mean, std, "std_MSE", "{:.3f}", "std_MSE (낮을수록 좋음 · 10% 판정 기준)")
    panel(a2, mean, std, "mean_MAPE", "{:.1f}%", "평균 MAPE, % (W · D · A 평균 상대오차)")
    a1.invert_yaxis()                                    # 가장 좋은 모델이 위 (sharey 라 한 번만)

    # 10% 목표선: 가장 좋은 baseline 값 × 0.9
    best = mean.loc[BASELINES, "std_MSE"].idxmin()
    b = mean.loc[best, "std_MSE"]
    a1.axvline(b, color=MUTED, linestyle="--", linewidth=1)
    a1.axvline(b * 0.9, color=INK, linestyle=":", linewidth=1.2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in GROUPS.values()]
    handles += [plt.Line2D([], [], color=MUTED, linestyle="--"), plt.Line2D([], [], color=INK, linestyle=":")]
    labels = [*GROUPS.keys(), f"최강 baseline ({best})", "10% 목표선"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{title}  —  모델 비교 (seed 5회 평균 ± 표준편차)", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.01, "막대 끝 숫자 = 평균, 가는 선 = ±표준편차. 점선(회색) = 최강 baseline, 점선(검정) = LT 가 넘어야 할 10% 목표선."
             + (" InferOff(학습·추론 조건 불일치) 모델은 값이 너무 커서 제외 — results.md 표 참고." if scheme == "val" else ""),
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.12)

    out = RES / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{scheme}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved:", out / f"{scheme}.png")


if __name__ == "__main__":
    for s, t in SCHEMES.items():
        plot(s, t)
