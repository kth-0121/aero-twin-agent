"""
looped_transformer_*_ensemble10.pt (10-seed 앙상블 체크포인트) 로더 + 추론 유틸.

사용법:
    from ensemble_predict import load_ensemble, predict_ensemble
    bundle = load_ensemble("model/LoopedTransformer/looped_transformer_v9_ensemble10.pt",
                            evaluator_cls=KeyholeAwarePhysicsEvaluator)
    pred = predict_ensemble(bundle, feat_df)  # (n, 3) numpy array, 10개 모델 평균
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
import modelLT


def load_ensemble(path, evaluator_cls, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    models = []
    for state in ckpt["states"]:
        m = modelLT.LoopedTransformer(
            n_features=len(ckpt["features"]), feedback_dim=cfg["feedback_dim"], loop_steps=cfg["loop_steps"]
        )
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    x_scaler, y_scaler = ckpt["x_scaler"], ckpt["y_scaler"]
    physics_eval = evaluator_cls(
        x_mean=x_scaler.mean_, x_scale=x_scaler.scale_, y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
    )
    return {
        "models": models, "physics_eval": physics_eval,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "features": ckpt["features"], "targets": ckpt["targets"],
    }


def predict_ensemble(bundle, feat_df, loop_steps=12):
    x = torch.tensor(bundle["x_scaler"].transform(feat_df[bundle["features"]]), dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for m in bundle["models"]:
            out = m(x, physics_evaluator=bundle["physics_eval"], loop_steps=loop_steps)
            preds.append(out.predictions[-1].numpy())
    avg_scaled = np.mean(preds, axis=0)
    return bundle["y_scaler"].inverse_transform(avg_scaled)
