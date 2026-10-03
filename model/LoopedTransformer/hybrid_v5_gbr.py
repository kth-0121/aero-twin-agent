"""
v5 LoopedTransformer + GradientBoosting 평균 앙상블.
~/buruG/aero-twin-agent 저장소 루트에서 실행 (model/LoopedTransformer 코드 재사용).

사용법: python hybrid_v5_gbr.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

LT_DIR = os.path.join(os.path.dirname(__file__), "model", "LoopedTransformer")
sys.path.insert(0, LT_DIR)
import dataLT  # noqa: E402
import modelLT  # noqa: E402
import physicsLT  # noqa: E402

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "316L_meltpool.csv")
V5_CHECKPOINT = os.path.join(LT_DIR, "looped_transformer_v5.pt")
GBR_MODEL_PATH = os.path.join(LT_DIR, "gbr_hybrid.pkl")

FEATURES = dataLT.FEATURES
TARGETS = dataLT.TARGETS


def load_v5():
    ckpt = torch.load(V5_CHECKPOINT, weights_only=False, map_location="cpu")
    x_scaler, y_scaler = ckpt["x_scaler"], ckpt["y_scaler"]
    model = modelLT.LoopedTransformer(n_features=len(FEATURES), feedback_dim=7, loop_steps=10)
    model.load_state_dict(ckpt["model"])
    model.eval()
    physics_eval = physicsLT.FluidThermalPhysicsEvaluator(
        x_mean=x_scaler.mean_, x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
    )
    return model, physics_eval, x_scaler, y_scaler


def predict_v5(model, physics_eval, x_scaler, y_scaler, feat_df, loop_steps=12):
    x = torch.tensor(x_scaler.transform(feat_df[FEATURES]), dtype=torch.float32)
    with torch.no_grad():
        out = model(x, physics_evaluator=physics_eval, loop_steps=loop_steps)
        pred_scaled = out.predictions[-1].numpy()
    return y_scaler.inverse_transform(pred_scaled)


def hybrid_predict(power_w, speed_mm_s, spot_diameter_mm=0.11, t_powder_um=30.0):
    row = pd.DataFrame([{
        "P_laser": power_w, "v_scan": speed_mm_s, "d_laser": spot_diameter_mm, "t_powder": t_powder_um,
    }])
    v_mps = speed_mm_s / 1000.0
    d_m = spot_diameter_mm / 1000.0
    alpha, rho, cp, Tm, T0, absorptivity = 5.0e-6, 7800.0, 460.0, 1678.0, 293.15, 0.34
    row["linear_energy"] = power_w / speed_mm_s
    row["peclet_number"] = v_mps * d_m / alpha
    h_s_vol = rho * cp * (Tm - T0)
    row["normalized_enthalpy"] = absorptivity * power_w / (h_s_vol * np.sqrt(alpha * d_m ** 3 * v_mps))

    model, physics_eval, x_scaler, y_scaler = load_v5()
    pred_lt = predict_v5(model, physics_eval, x_scaler, y_scaler, row)[0]
    gbr = joblib.load(GBR_MODEL_PATH)
    pred_gbr = gbr.predict(row[FEATURES])[0]
    return dict(zip(TARGETS, (pred_lt + pred_gbr) / 2.0))


def benchmark():
    df = dataLT.load_316l_csv(DATA_CSV)
    train_df, val_df, test_df = dataLT.make_iid_split(df, seed=42)
    held_out = pd.concat([val_df, test_df], ignore_index=True)
    true = held_out[TARGETS].to_numpy()
    print(f"train={len(train_df)} held_out={len(held_out)}")

    gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    gbr.fit(train_df[FEATURES], train_df[TARGETS])
    joblib.dump(gbr, GBR_MODEL_PATH)
    pred_gbr = gbr.predict(held_out[FEATURES])

    model, physics_eval, x_scaler, y_scaler = load_v5()
    pred_lt = predict_v5(model, physics_eval, x_scaler, y_scaler, held_out)
    pred_hybrid = (pred_lt + pred_gbr) / 2.0

    print(f"\n{'target':<14}{'v5':>10}{'GBR':>10}{'hybrid':>10}   (RMSE)")
    for i, t in enumerate(TARGETS):
        r_lt = mean_squared_error(true[:, i], pred_lt[:, i]) ** 0.5
        r_gbr = mean_squared_error(true[:, i], pred_gbr[:, i]) ** 0.5
        r_hy = mean_squared_error(true[:, i], pred_hybrid[:, i]) ** 0.5
        print(f"{t:<14}{r_lt:>10.3f}{r_gbr:>10.3f}{r_hy:>10.3f}")
    print(f"\nGBR 저장: {GBR_MODEL_PATH}")


if __name__ == "__main__":
    benchmark()
    print("\n--- 실시간 예시 (200W, 800mm/s) ---")
    print(hybrid_predict(200, 800))
