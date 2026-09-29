"""[Baseline] MLP — 기준이 modelLT.py 의 StateMLP 를 그대로 사용 (7 → 64 → 64 → 3).
Looped Transformer 와 같은 신경망 계열이면서 loop · 물리 피드백이 없는 모델.
학습 설정은 common.train_nn (LT 와 동일)."""
from common import run_single, set_seed, train_nn

import torch
from modelLT import StateMLP


def run(data, seed):
    set_seed(seed)
    model = StateMLP(n_features=data.x_train.shape[1], state_dim=3)
    train_nn(model, data, lambda m, x: [m(x)])   # 예측이 1개뿐이라 리스트로 감싸 loss 에 전달
    with torch.no_grad():
        return model(torch.from_numpy(data.x_eval)).numpy()


if __name__ == "__main__":
    run_single(run)
