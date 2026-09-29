"""[평가 대상] Looped Transformer — 기준이 modelLT.py 를 그대로 사용.

조건 (학습과 추론의 물리 피드백 조건을 일치시키는 것이 기본)
- LT-k            : 학습 · 추론 모두 Rosenthal 물리 피드백 사용 → 10% 판정 대상
- LT-3-NoPhys     : 학습 · 추론 모두 물리 피드백 없음 → 내부 비교 (LT-3 과의 차이 = 물리 피드백의 효과)
- LT-k-InferOff   : 학습은 물리 피드백, 추론은 끔 (interfaceLT.py 방식) → 조건 불일치 참고용
- loop 3 = 기준이 설정 / 6, 12 = 조사 기반 후보 (6: TRM 논문, 12: Yang et al. 2023 의 최소 학습 loop)
evaluator 는 공정조건과 예측값만 사용하고 실측값은 쓰지 않으므로 추론 때 켜도 정답을 보는 것이 아니다.
"""
import torch

from common import run_single, set_seed, train_nn
from modelLT import LoopedTransformer
from physicsLT import RosenthalPhysicsEvaluator

# 이름: (loop 횟수, 학습 시 물리 피드백, 추론 시 물리 피드백)
VARIANTS = {
    "LT-3": (3, True, True),
    "LT-6": (6, True, True),
    "LT-12": (12, True, True),
    "LT-3-NoPhys": (3, False, False),
    "LT-3-InferOff": (3, True, False),
    "LT-6-InferOff": (6, True, False),
    "LT-12-InferOff": (12, True, False),
}


def run(data, seed, loop_steps=3, train_physics=True, infer_physics=True):
    set_seed(seed)
    s = data.scalers
    # feedback_dim=3: Rosenthal residual 3개(W, R, A). v3 에서 기본값이 7로 바뀌어 명시
    model = LoopedTransformer(n_features=data.x_train.shape[1], feedback_dim=3, loop_steps=loop_steps)
    evaluator = RosenthalPhysicsEvaluator(s.x.mean_, s.x.scale_, s.y.mean_, s.y.scale_)

    # 학습: 루프 단계별 예측을 모두 loss 에 넣음 (deep supervision). evaluator=None 이면 residual 0
    train_ev = evaluator if train_physics else None
    train_nn(model, data, lambda m, x: m(x, physics_evaluator=train_ev).predictions)

    # 추론: 마지막 루프 단계의 예측 사용
    with torch.no_grad():
        out = model(torch.from_numpy(data.x_eval), physics_evaluator=evaluator if infer_physics else None)
        return out.predictions[-1].numpy()


if __name__ == "__main__":
    run_single(run)   # LT-3 한 번 실행
