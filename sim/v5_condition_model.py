"""V5 스트리밍 LT를 "공정조건 정상상태 평가"용으로 쓰는 상태 없는 래퍼 (태훈 담당).

V5 서비스(StreamingLTService)는 이벤트 이력 버퍼·SQLite 저장·난수 센서를 가진 상태 있는
서비스라, 조건 수천 개를 한 번에 평가하는 격자 탐색에는 쓸 수 없다. 모델 자체
(AdaptivePhysicsResidualLT.forward)는 순수 함수이므로, 서비스를 거치지 않고 같은 전처리로
배치 호출한다. 한 조건을 "같은 이벤트 8개가 반복된 윈도우"로 보고, 합성 센서는 난수를 끈
기댓값(latent=0, 잡음 0)을 쓴다 -> 같은 입력은 항상 같은 출력.

주의: V5의 센서 3개는 합성 프록시(공정값의 함수 + 난수)이며 실측 센서가 아니다. delta_t는
V5 학습 분포(1~10초 균등, 평균 5.5초)에 맞춘다 (DT 어댑터가 쓰는 1/30초와 다름).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.LoopedTransformer.v5.config import DEVICE, INPUT_FEATURES, PROCESS_FEATURES, TARGETS, WINDOW_EVENTS  # noqa: E402
from model.LoopedTransformer.v5.data import load_dataset  # noqa: E402
from model.LoopedTransformer.v5.model import AdaptivePhysicsResidualLT  # noqa: E402
from model.LoopedTransformer.v5.physics import RosenthalPhysicsAnchor  # noqa: E402

ARTIFACT_DIR = _ROOT / "model" / "LoopedTransformer" / "V5_stream"
TRAIN_DELTA_T_S = 5.5   # simulate_stream: delta_t ~ U(1, 10)


class V5ConditionModel:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        ckpt = torch.load(artifact_dir / "looped_transformer_v5_st.pt", map_location=DEVICE, weights_only=True)
        if ckpt.get("input_features") != INPUT_FEATURES or ckpt.get("targets") != TARGETS:
            raise ValueError("V5 체크포인트의 입출력 계약이 코드와 다름")
        self.x_scaler = joblib.load(artifact_dir / "x_scaler.joblib")
        self.y_scaler = joblib.load(artifact_dir / "y_scaler.joblib")
        self.model = AdaptivePhysicsResidualLT(
            input_dim=ckpt["input_dim"], hidden=ckpt["hidden"], n_loops=ckpt["n_loops"], max_delta=ckpt["max_delta"],
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.anchor = RosenthalPhysicsAnchor(ckpt["eta_effective"])
        self.window = int(ckpt.get("window_events", WINDOW_EVENTS))
        self.version = str(ckpt.get("model_version", "LT-V5"))

        ref = load_dataset()[PROCESS_FEATURES].to_numpy(float)   # DT 센서 프록시와 같은 기준
        self.median = np.median(ref, axis=0)
        q25, q75 = np.percentile(ref, [25, 75], axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-6)

    def _sensors(self, proc):
        """합성 센서의 기댓값 (SyntheticSensorProvider.observe에서 latent=0, 잡음=0)."""
        s = (proc - self.median) / self.iqr
        return np.stack([0.45 * s[:, 0] - 0.25 * s[:, 1],
                         0.20 * s[:, 1] + 0.30 * s[:, 2],
                         0.25 * s[:, 0] + 0.15 * s[:, 3]], axis=1)

    @torch.no_grad()
    def predict(self, power_w, speed_mm_s, spot_diameter_mm, layer_thickness_um, chunk=512):
        """스칼라/배열 입력 -> (width_um, depth_um, area_um2) 배열. 같은 입력은 항상 같은 출력."""
        p, v = np.broadcast_arrays(np.atleast_1d(np.asarray(power_w, float)), np.atleast_1d(np.asarray(speed_mm_s, float)))
        proc = np.stack([p, v, np.full_like(p, spot_diameter_mm), np.full_like(p, layer_thickness_um)], axis=1)
        out = []
        for lo in range(0, len(proc), chunk):
            part = proc[lo:lo + chunk]
            sensors = self._sensors(part)
            delta_t = np.where(np.arange(self.window) == 0, 0.0, TRAIN_DELTA_T_S)          # [T]
            feat = np.concatenate([part, sensors], axis=1)                                   # [B, 7]
            raw = np.concatenate([np.repeat(feat[:, None, :], self.window, axis=1),
                                  np.broadcast_to(delta_t[None, :, None], (len(part), self.window, 1))], axis=2)
            flat = pd.DataFrame(raw.reshape(-1, len(INPUT_FEATURES)), columns=INPUT_FEATURES)
            x = torch.from_numpy(self.x_scaler.transform(flat).astype(np.float32)).reshape(len(part), self.window, -1).to(DEVICE)
            mask = torch.ones(len(part), self.window, dtype=torch.bool, device=DEVICE)
            anchor_raw = self.anchor.predict(pd.DataFrame(part[:, :4], columns=PROCESS_FEATURES))
            anchor = torch.from_numpy(self.y_scaler.transform(np.log1p(anchor_raw)).astype(np.float32)).to(DEVICE)
            pred, _ = self.model(x, mask, anchor)
            out.append(np.maximum(np.expm1(self.y_scaler.inverse_transform(pred.cpu().numpy())), 0.0))
        return np.concatenate(out, axis=0).T


def demo():
    m = V5ConditionModel()
    w, d, a = m.predict(np.array([200.0, 200.0, 450.0]), np.array([800.0, 800.0, 300.0]), 0.1, 30.0)
    assert np.allclose(w[0], w[1]) and np.allclose(d[0], d[1]), "같은 입력은 같은 출력이어야 함"
    assert (w > 0).all() and (d > 0).all() and (a > 0).all()
    assert d[2] > d[0], "고출력·저속이 정상 조건보다 깊어야 함"
    print(f"demo ok: 200W/800 W={w[0]:.1f} D={d[0]:.1f} A={a[0]:.0f} | 450W/300 W={w[2]:.1f} D={d[2]:.1f}")


if __name__ == "__main__":
    demo()
