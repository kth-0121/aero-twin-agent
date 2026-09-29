"""[참고] 물리식만으로 예측 (ROM) — 학습 없이 공정조건을 식에 넣어 바로 계산.
기준이 physicsLT.py 의 RosenthalPhysicsEvaluator 식을 그대로 사용한다.
    W = Rosenthal 폭            (P, v 사용)
    D = H*·Pe scaling 깊이       (P, v, d 사용)
    A = pi/4 * W * D            (반타원 면적)
무작위성이 없어서 seed 를 바꿔도 결과가 같다."""
import math

import numpy as np
import torch

from common import run_single
from physicsLT import RosenthalPhysicsEvaluator


def run(data, seed):
    s = data.scalers
    phys = RosenthalPhysicsEvaluator(s.x.mean_, s.x.scale_, s.y.mean_, s.y.scale_)
    x = torch.from_numpy(data.x_eval)
    W = phys.rosenthal_width(x).numpy()    # [um]
    D = phys.scaling_depth(x).numpy()      # [um]
    A = math.pi / 4 * W * D                # [um^2]
    pred = np.stack([W, D / W, A], axis=1)  # 모델 출력 형식 (W, R, A) 에 맞춤
    return s.y.transform(pred)              # 다른 모델과 같은 표준화 공간으로 변환


if __name__ == "__main__":
    run_single(run)
