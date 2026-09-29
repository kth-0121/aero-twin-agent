"""[Baseline] Gradient Boosting — 앞 트리의 오차를 다음 트리가 보완하며 순서대로 쌓는 방식.
출력을 1개만 지원하므로 MultiOutputRegressor 로 W, R, A 각각 별도 모델을 학습한다."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from common import run_single


def run(data, seed):
    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=seed))
    model.fit(data.x_train, data.y_train)
    return model.predict(data.x_eval)


if __name__ == "__main__":
    run_single(run)
