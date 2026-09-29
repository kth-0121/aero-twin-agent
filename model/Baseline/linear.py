"""[Baseline] 선형 회귀 — 가장 단순한 기준선. 이것보다 못하면 모델에 문제가 있다는 뜻.
무작위성이 없어서 seed 를 바꿔도 결과가 같다 (표준편차 0)."""
from sklearn.linear_model import LinearRegression

from common import run_single


def run(data, seed):
    model = LinearRegression().fit(data.x_train, data.y_train)   # 출력 3개(W, R, A)를 한 번에 학습
    return model.predict(data.x_eval)


if __name__ == "__main__":
    run_single(run)
