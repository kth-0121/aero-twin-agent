"""
[Baseline 공통 모듈] 모든 모델이 같은 데이터 · 같은 분할 · 같은 지표를 쓰도록 한 곳에 모아둔 파일

- 데이터 준비: 기준이의 dataLT.py 를 그대로 재사용
    입력 x 7개 (공정조건 4 + 파생 3), 출력 y 3개 (W, R, A), train 으로만 표준화
- 평가 방식 (scheme) 4가지 — 학습 데이터와 평가 데이터를 어떻게 나누는가
    val    : 기준이 분할(70/15/15, seed 42)의 train → val 로 평가
    test   : 같은 train → test 로 평가
    group  : 같은 공정조건(P, v, d, t) 행을 한 묶음으로 70/15/15 분할 → test 로 평가
    regime : conduction (실측 D/W < 0.5) 으로 학습 → keyhole/transition (D/W >= 0.5) 으로 평가
- 신경망 학습 설정: 기준이 L_T.ipynb 와 동일 (AdamW, lr 1e-3, batch 64, gamma 0.8, grad clip 1.0)
    단, val 로 best epoch 를 고르지 않고 100 epoch 고정 학습
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]                     # aero-twin-agent/
sys.path.insert(0, str(ROOT / "model" / "LoopedTransformer"))   # dataLT, modelLT 등을 import 하기 위함

import dataLT                                    # noqa: E402
from lossLT import DeepSupervisionLoss           # noqa: E402

DATA_PATH = ROOT / "data" / "316L_meltpool.csv"
RESULTS = Path(__file__).resolve().parent / "results_v3"   # V1 결과는 results/ 에 보존
SPLIT_SEED = 42            # 데이터 분할 seed (기준이 설정과 동일, 고정)
SEEDS = [0, 1, 2, 3, 4]    # 반복 실험용 모델 seed
EPOCHS = 100


def _group_split(df):
    """같은 공정조건 행이 train / test 에 나뉘지 않도록 조건 단위로 70/15/15 분할 → (train, test)."""
    groups = df[dataLT.BASE_FEATURES].astype(str).agg("_".join, axis=1)
    tr, rest = next(GroupShuffleSplit(train_size=0.70, random_state=SPLIT_SEED).split(df, groups=groups))
    rest_df = df.iloc[rest]
    _, te = next(GroupShuffleSplit(train_size=0.50, random_state=SPLIT_SEED)
                 .split(rest_df, groups=groups.iloc[rest]))
    return df.iloc[tr], rest_df.iloc[te]


def load_data(scheme="val"):
    """학습/평가 데이터 준비. x, y 는 표준화된 numpy 배열."""
    df = dataLT.load_316l_csv(DATA_PATH)
    if scheme in ("val", "test"):
        train_df, val_df, test_df = dataLT.make_iid_split(df, seed=SPLIT_SEED)
        eval_df = val_df if scheme == "val" else test_df
    elif scheme == "group":
        train_df, eval_df = _group_split(df)
    elif scheme == "regime":
        keyhole = df["aspect_ratio"] >= 0.5
        train_df, eval_df = df[~keyhole], df[keyhole]
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    scalers = dataLT.fit_scalers(train_df)
    f, t = scalers.features, scalers.targets
    return SimpleNamespace(
        x_train=scalers.x.transform(train_df[f]).astype(np.float32),
        y_train=scalers.y.transform(train_df[t]).astype(np.float32),
        x_eval=scalers.x.transform(eval_df[f]).astype(np.float32),
        y_eval=scalers.y.transform(eval_df[t]).astype(np.float32),
        eval_df=eval_df,
        scalers=scalers,
    )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_nn(model, data, forward):
    """신경망 공통 학습 루프.
    forward(model, x) 는 (예측 리스트, 추가 loss) 를 반환해야 한다
    (MLP 는 ([pred], 0), LT 는 (루프 단계별 예측 전체, physics_alpha * 물리 loss)).
    loss = 기준이 코드와 같은 DeepSupervisionLoss + 추가 loss."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.y_train)),
        batch_size=64, shuffle=True,
    )
    loss_fn = DeepSupervisionLoss(gamma=0.8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            preds, extra_loss = forward(model, xb)
            loss = loss_fn(preds, yb) + extra_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def _item_metrics(name, pred, true):
    err = pred - true
    return {
        f"{name}_MAE": np.abs(err).mean(),
        f"{name}_RMSE": np.sqrt((err ** 2).mean()),
        f"{name}_MAPE": (np.abs(err) / np.abs(true)).mean() * 100,
    }


def evaluate(pred_std, data):
    """평가 데이터 예측(표준화된 W, R, A)을 받아 지표 dict 반환.
    - std_MSE : 표준화 공간 MSE (W, R, A 평균). 단위가 없어 항목을 합칠 수 있음 → 10% 판정 기준
    - 항목별  : W [um], D [um] (= W*R), A [um^2], R [-] 의 MAE, RMSE, MAPE(%)
    - mean_MAPE : W, D, A 의 MAPE 평균"""
    out = {"std_MSE": float(((pred_std - data.y_eval) ** 2).mean())}

    W, R, A = data.scalers.y.inverse_transform(pred_std).T     # 원래 단위로 복원
    v = data.eval_df
    out |= _item_metrics("W", W, v["melt_width"].to_numpy())
    out |= _item_metrics("D", W * R, v["melt_depth"].to_numpy())
    out |= _item_metrics("A", A, v["melt_area"].to_numpy())
    out |= _item_metrics("R", R, v["aspect_ratio"].to_numpy())
    out["mean_MAPE"] = np.mean([out["W_MAPE"], out["D_MAPE"], out["A_MAPE"]])
    return out


def run_single(run_fn, seed=0):
    """모델 파일 단독 실행용: seed 하나로 돌려 지표 출력."""
    data = load_data()
    for k, v in evaluate(run_fn(data, seed), data).items():
        print(f"{k:>10}: {v:.4f}")
