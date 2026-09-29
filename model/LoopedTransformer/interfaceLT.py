"""
601Solutions
"""

from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np

import dataLT
import modelLT
import physicsLT

FEATURES = dataLT.FEATURES
TARGETS = dataLT.TARGETS

"""
추상클래스 이용해서 인터페이스 구현
사용시에는 from interfaceLT import LoopedTransformer 로 사용.
"""
class LoopedTransformer(ABC): # 추상클래스
    @abstractmethod
    def setData(self, path: str):
        """
        예측에 필요한 입력 데이터를 가져오는 메서드
        """
        pass

    @abstractmethod
    def getPredict(self, path:str, data:pd.DataFrame):
        """
        예측 결과 받아오기
        """
        pass

class RUN(LoopedTransformer):
    def setData(self, path):
        try:
            df = dataLT.load_316l_csv(path)
            return df
        except:
            raise FileNotFoundError("입력 데이터를 찾을 수 없음.")

    def getPredict(self, path, data) -> pd.DataFrame:
        # 학습된 가중치 가져오기
        checkpoint_path = path
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 학습된 스케일러 복원
        scaler_x = checkpoint["x_scaler"]
        scaler_y = checkpoint["y_scaler"]

        # 모델 초기화
        # loop_steps=12: 검증데이터로 3~1000 스캔한 결과 12 근처(8~15)가 최적,
        # 학습값(3)보다도 나음. 1000은 학습때 안 본 구간이라 오차가 크게 나빠짐.
        model = modelLT.LoopedTransformer(n_features=len(FEATURES), feedback_dim=7, loop_steps=12)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # 학습 때 쓴 것과 같은 물리평가기 (FluidThermalPhysicsEvaluator)를
        # 추론 시에도 생성해서 넘겨야 함 -- 없으면 feedback이 zeros(dim=3)로
        # 대체되어 feedback_encoder(dim=7 기대)와 shape mismatch 발생함
        physics_evaluator = physicsLT.FluidThermalPhysicsEvaluator(
            x_mean=scaler_x.mean_, x_scale=scaler_x.scale_,
            y_mean=scaler_y.mean_, y_scale=scaler_y.scale_,
        )

        # 입력 텐서 변환
        x_scaled = scaler_x.transform(data[FEATURES]).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled)

        # 예측
        with torch.no_grad():
            output = model(x_tensor, physics_evaluator=physics_evaluator)
            final_predictions_scaled = output.predictions[-1].numpy()

        final_predictions = scaler_y.inverse_transform(final_predictions_scaled)

        result = pd.DataFrame()
        for i, target_col in enumerate(TARGETS):
            result[f"{target_col}"] = final_predictions[:, i]

        return result
        