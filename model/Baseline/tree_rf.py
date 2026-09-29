"""[Baseline] Random Forest — 결정트리 여러 개의 평균으로 예측. 표 형태의 소량 데이터에서 강한 편.
seed 에 따라 트리를 만드는 무작위 샘플링이 달라진다."""
from sklearn.ensemble import RandomForestRegressor

from common import run_single


def run(data, seed):
    model = RandomForestRegressor(random_state=seed).fit(data.x_train, data.y_train)   # 출력 3개 동시 지원
    return model.predict(data.x_eval)


if __name__ == "__main__":
    run_single(run)
