"""
v9: 물리기반 합성 keyhole 데이터 증강 (Gan et al. 2021 Ke/e* + Rosenthal width).
MDPI Applied Sci 2025 "Physics-Based Data Augmentation..." (316L, 36개 실측으로도
효과봄)와 같은 전략: loss-time 보정(v7/v8, 실패)이 아니라 학습 데이터 자체를 늘림.

conduction만 학습(기존 도메인시프트 설정과 동일) + 합성 keyhole 행 추가 -> 실측
keyhole 325행으로 평가. plain(증강 없음) vs v9(증강)을 10-seed로 공정 비교.
실행 위치: 저장소 루트 (GPU 권장)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(LT_DIR, "legacy"))
import dataLT
import lossLT
import modelLT
import physicsLT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

SEED_BASE = 42
FEATURES = dataLT.FEATURES
TARGETS = dataLT.TARGETS
KEYHOLE_THRESHOLD = 4.4e6
N_SEEDS = 10
N_SYNTHETIC = 200

# physicsLT.py와 동일 상수 (RosenthalPhysicsEvaluator 기본값)
RHO, CP, ALPHA, TM, T0, ABSORPTIVITY = 7800.0, 460.0, 5.0e-6, 1678.0, 293.15, 0.34


def generate_synthetic_keyhole_rows(n, seed=0, noise_std=0.08):
    """Ke/e*(Gan 2021) + Rosenthal width(physicsLT 기본)로 keyhole 영역 합성 행 생성."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(50.0, 500.0, n)
    v_mm_s = rng.uniform(225.0, 1500.0, n)
    d_mm = rng.uniform(0.05, 0.14, n)
    t_powder = rng.choice([0.0, 30.0, 60.0], n)

    v_mps = v_mm_s / 1000.0
    r0_m = (d_mm / 1000.0) / 2.0

    deltaT = TM - T0
    width_m = np.sqrt((8.0 / (np.pi * np.e)) * ABSORPTIVITY * P / (RHO * CP * v_mps * deltaT))
    width_um = width_m * 1e6

    Ke = (ABSORPTIVITY * P) / (deltaT * np.pi * RHO * CP * np.sqrt(ALPHA * v_mps * r0_m ** 3))
    e_star = np.clip(0.4 * (Ke - 1.4), 0.0, None)
    depth_um = e_star * r0_m * 1e6

    # keyhole/transition만 사용 (Ke>1.4), conduction 쪽은 이미 실측 데이터로 충분
    mask = Ke > 1.4
    P, v_mm_s, d_mm, t_powder = P[mask], v_mm_s[mask], d_mm[mask], t_powder[mask]
    width_um, depth_um = width_um[mask], depth_um[mask]
    v_mps, r0_m = v_mps[mask], r0_m[mask]

    # 노이즈 (합성 데이터를 "정답"처럼 과신하지 않도록)
    width_um = width_um * rng.normal(1.0, noise_std, len(width_um))
    depth_um = depth_um * rng.normal(1.0, noise_std, len(depth_um))
    depth_um = np.clip(depth_um, 1.0, None)

    aspect_ratio = depth_um / np.maximum(width_um, 1e-6)
    area_um2 = (np.pi / 4.0) * width_um * depth_um

    h_s_vol = RHO * CP * (TM - T0)
    d_m = d_mm / 1000.0
    linear_energy = P / v_mm_s
    peclet_number = v_mps * d_m / ALPHA
    normalized_enthalpy = ABSORPTIVITY * P / (h_s_vol * np.sqrt(ALPHA * d_m ** 3 * v_mps))

    return pd.DataFrame({
        "P_laser": P, "v_scan": v_mm_s, "d_laser": d_mm, "t_powder": t_powder,
        "linear_energy": linear_energy, "peclet_number": peclet_number, "normalized_enthalpy": normalized_enthalpy,
        "melt_width": width_um, "aspect_ratio": aspect_ratio, "melt_area": area_um2,
    })


def train_one(seed, train_df, val_df, x_scaler, y_scaler, epochs=300, patience=30):
    torch.manual_seed(seed)
    xtr = torch.tensor(x_scaler.transform(train_df[FEATURES]), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_scaler.transform(train_df[TARGETS]), dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x_scaler.transform(val_df[FEATURES]), dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_scaler.transform(val_df[TARGETS]), dtype=torch.float32, device=DEVICE)

    model = modelLT.LoopedTransformer(n_features=len(FEATURES), feedback_dim=7, loop_steps=10).to(DEVICE)
    pe = physicsLT.FluidThermalPhysicsEvaluator(
        x_mean=x_scaler.mean_, x_scale=x_scaler.scale_, y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
    )
    data_loss = lossLT.DeepSupervisionLoss(gamma=0.8, state_weights=(1.0, 2.0, 1.0))
    phys_loss = lossLT.PhysicsConsistencyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, best_state, bad = float("inf"), None, 0
    n = xtr.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            out = model(xtr[idx], physics_evaluator=pe)
            loss = data_loss(out.predictions, ytr[idx]) + 0.1 * phys_loss(out.feedbacks, out.confidences)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xv, physics_evaluator=pe)
            vl = data_loss(out.predictions, yv).item()
        if vl < best_val - 1e-5:
            best_val, best_state, bad = vl, {k: v.clone().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pe


def predict(model, pe, x_scaler, y_scaler, feat_df, loop_steps=12):
    x = torch.tensor(x_scaler.transform(feat_df[FEATURES]), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(x, physics_evaluator=pe, loop_steps=loop_steps)
    return y_scaler.inverse_transform(out.predictions[-1].cpu().numpy())


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    df = dataLT.load_316l_csv(os.path.join(ROOT_DIR, "data", "316L_meltpool.csv"))
    v_m_s = df["v_scan"] / 1000.0
    d_m = df["d_laser"] / 1000.0
    ed = df["P_laser"] / (v_m_s * d_m)
    df["regime"] = np.where(ed > KEYHOLE_THRESHOLD, "keyhole", "conduction")

    cond_df = df[df["regime"] == "conduction"].reset_index(drop=True)
    key_df = df[df["regime"] == "keyhole"].reset_index(drop=True)
    cond_train, cond_val = train_test_split(cond_df, test_size=0.15, random_state=42)

    synthetic = generate_synthetic_keyhole_rows(N_SYNTHETIC, seed=42)
    print(f"실측 conduction train: {len(cond_train)}  합성 keyhole 추가: {len(synthetic)}  실측 val(conduction): {len(cond_val)}  평가(실측 keyhole): {len(key_df)}")

    augmented_train = pd.concat([cond_train, synthetic], ignore_index=True)

    # plain: 증강 없음, v9: 증강 포함. scaler는 각자 train 기준으로.
    x_scaler_plain = StandardScaler().fit(cond_train[FEATURES])
    y_scaler_plain = StandardScaler().fit(cond_train[TARGETS])
    x_scaler_v9 = StandardScaler().fit(augmented_train[FEATURES])
    y_scaler_v9 = StandardScaler().fit(augmented_train[TARGETS])

    true = key_df[TARGETS].to_numpy()
    preds_plain, preds_v9 = [], []
    t0 = time.time()
    for i in range(N_SEEDS):
        m_p, pe_p = train_one(SEED_BASE + i, cond_train, cond_val, x_scaler_plain, y_scaler_plain)
        preds_plain.append(predict(m_p, pe_p, x_scaler_plain, y_scaler_plain, key_df))

        m_v9, pe_v9 = train_one(SEED_BASE + i, augmented_train, cond_val, x_scaler_v9, y_scaler_v9)
        preds_v9.append(predict(m_v9, pe_v9, x_scaler_v9, y_scaler_v9, key_df))
        print(f"  seed {i+1}/{N_SEEDS} done ({time.time()-t0:.1f}s)")

    preds_plain = np.stack(preds_plain)
    preds_v9 = np.stack(preds_v9)
    ens_plain = preds_plain.mean(axis=0)
    ens_v9 = preds_v9.mean(axis=0)

    print("\n=== 실측 keyhole 325행 평가: plain(증강없음) vs v9(합성keyhole 증강) ===")
    for i, t in enumerate(TARGETS):
        rp_each = np.mean([rmse(true[:, i], preds_plain[s, :, i]) for s in range(N_SEEDS)])
        rv_each = np.mean([rmse(true[:, i], preds_v9[s, :, i]) for s in range(N_SEEDS)])
        rp_ens = rmse(true[:, i], ens_plain[:, i])
        rv_ens = rmse(true[:, i], ens_v9[:, i])
        print(f"  {t:14s} plain(개별={rp_each:7.3f}, 앙상블={rp_ens:7.3f})   v9(개별={rv_each:7.3f}, 앙상블={rv_ens:7.3f})")
    print(f"\n소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
