"""
601Solutions
"""
# [interfaceLT.py] 학습된 Looped Transformer를 다른 모듈에서 호출하기 위한 인터페이스
# - setData(csv 경로)                 → 입력 DataFrame (원본 + 파생 feature)
# - getPredict(체크포인트 .pt 경로, df) → 예측 DataFrame (melt_width, aspect_ratio, melt_area)
# - 사용: lt = RUN(); df = lt.setData(csv); pred = lt.getPredict("looped_transformer_v1.pt", df)
# - 주의: physics_evaluator 없이 예측하므로 추론 시 physics residual은 0 (loop_steps=1000)

from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np

import dataLT
import modelLT

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
        model = modelLT.LoopedTransformer(n_features=len(FEATURES), loop_steps=1000)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # 입력 텐서 변환
        x_scaled = scaler_x.transform(data[FEATURES]).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled)

        # 예측
        with torch.no_grad():
            output = model(x_tensor)
            final_predictions_scaled = output.predictions[-1].numpy()

        final_predictions = scaler_y.inverse_transform(final_predictions_scaled)

        result = pd.DataFrame()
        for i, target_col in enumerate(TARGETS):
            result[f"{target_col}"] = final_predictions[:, i]

        return result
        