"""[평가 대상] Looped Transformer v3 — 기준이 modelLT.py + physicsLT.FluidThermalPhysicsEvaluator 를 그대로 사용.

v3 설정 = looped_transformer_v3.pt 에 저장된 학습 인자: 학습 loop 10, fluid 평가기(residual 7개), physics_alpha 0.1
- LT-v3        : 학습 loop 10 / 추론 loop 10 (학습과 같은 조건) → 10% 판정 대상
- LT-v3-i12    : 학습 loop 10 / 추론 loop 12 (interfaceLT.py 설정) → 10% 판정 대상
- LT-v3-NoPhys : 물리 피드백 · 물리 loss 없음 → 내부 비교 (LT-v3 과의 차이 = 물리의 효과)
학습 · 추론 모두 같은 평가기를 쓴다 (조건 일치). 평가기는 공정조건과 예측값만 쓰므로 추론 때 켜도 정답을 보지 않는다.
V1 (Rosenthal 평가기, loop 3/6/12) 결과는 results/ 에, 그 코드는 PR #6 커밋에 남아 있다.
"""
import torch

from common import run_single, set_seed, train_nn
from lossLT import PhysicsConsistencyLoss
from modelLT import LoopedTransformer
from physicsLT import FluidThermalPhysicsEvaluator

PHYSICS_ALPHA = 0.1   # 최종 loss = 데이터 loss + 0.1 * 물리 loss (v3 체크포인트 설정)

# 이름: (학습 loop, 추론 loop, 물리 사용 여부)
VARIANTS = {
    "LT-v3": (10, 10, True),
    "LT-v3-i12": (10, 12, True),
    "LT-v3-NoPhys": (10, 10, False),
}


def run(data, seed, train_loops=10, infer_loops=10, physics=True):
    set_seed(seed)
    s = data.scalers
    # feedback_dim: fluid 평가기 residual 7개. 평가기가 없으면 modelLT 가 3차원 0 을 넣으므로 3
    # (입력이 전부 0 이라 차원이 달라도 모델이 할 수 있는 일은 같음)
    model = LoopedTransformer(n_features=data.x_train.shape[1], feedback_dim=7 if physics else 3,
                              loop_steps=train_loops)
    evaluator = FluidThermalPhysicsEvaluator(s.x.mean_, s.x.scale_, s.y.mean_, s.y.scale_) if physics else None
    physics_loss = PhysicsConsistencyLoss()

    # 학습: 루프 단계별 예측을 모두 loss 에 넣음 (deep supervision) + 물리 loss. 평가기가 없으면 물리 loss 0
    def forward(m, x):
        out = m(x, physics_evaluator=evaluator)
        return out.predictions, PHYSICS_ALPHA * physics_loss(out.feedbacks, out.confidences)

    train_nn(model, data, forward)

    # 추론: 같은 평가기로 infer_loops 번 돌리고 마지막 루프 단계의 예측 사용
    with torch.no_grad():
        out = model(torch.from_numpy(data.x_eval), physics_evaluator=evaluator, loop_steps=infer_loops)
        return out.predictions[-1].numpy()


if __name__ == "__main__":
    run_single(run)   # LT-v3 한 번 실행
