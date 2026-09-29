"""[Baseline] Gaussian Process Regression — 비슷한 입력은 비슷한 출력을 낸다는 가정(kernel)으로 예측.
데이터가 적을 때 자주 쓰는 모델.

kernel 구성
- ConstantKernel * RBF : 입력 간 거리 기반 유사도. length_scale 을 feature 7개마다 따로 학습 (어떤 feature 가 중요한지 반영)
- WhiteKernel          : 측정 노이즈 (같은 조건 반복 실험에서도 값이 다름)"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from common import run_single


def run(data, seed):
    n_features = data.x_train.shape[1]
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(0.1)
    model = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    model.fit(data.x_train, data.y_train)   # 출력 3개는 같은 kernel 을 공유
    return model.predict(data.x_eval)


if __name__ == "__main__":
    run_single(run)
