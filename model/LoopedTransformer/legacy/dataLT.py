"""
601Solutions
"""
# [dataLT.py] Looped Transformer 학습용 데이터 준비 모듈
# - 316L melt pool CSV를 읽어 입력 x(공정조건 4 + 파생 feature 3 = 7개)와
#   정답 y(W: 폭, R: 깊이/폭, A: 단면적)를 만든다.
# - 사용 순서: load_316l_csv → make_iid_split → fit_scalers → MeltPoolDataset

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 입력 x 컬럼: CSV 원본 공정조건 4개 + load_316l_csv에서 계산하는 파생 feature 3개
BASE_FEATURES = ["P_laser", "v_scan", "d_laser", "t_powder"]
DERIVED_FEATURES = ["linear_energy", "peclet_number", "normalized_enthalpy"]

# 정답 y 컬럼 (balling은 Dataset에 담기지만 현재 loss에는 미사용)
#   W = melt-pool width [um]
#   R = D/W = melt-pool aspect ratio [-]
#   A = melt-pool cross-sectional area [um^2]
TARGETS = ["melt_width", "aspect_ratio", "melt_area"]
RISK = "balling"


# 입력 feature 목록 반환. True면 파생 feature 3개가 모두 추가된다 (linear_energy만이 아님)
def get_features(include_linear_energy: bool = True) -> list[str]:
    return BASE_FEATURES + (DERIVED_FEATURES if include_linear_energy else [])


FEATURES = get_features(True)


# CSV 로드 + 파생 feature(E_L, Pe, H*) 및 aspect_ratio 계산
# CSV 헤더의 따옴표가 깨져 있어 헤더를 건너뛰고 컬럼명을 직접 붙인다
def load_316l_csv(path: str | Path) -> pd.DataFrame:
    """Load the supplied 316L single-track CSV and derive v3 geometry state."""
    path = Path(path)
    df = pd.read_csv(path, header=None, skiprows=1)
    if df.shape[1] != 9:
        raise ValueError(f"Expected 9 columns including ID, got {df.shape[1]}")

    df.columns = [
        "id", "P_laser", "v_scan", "d_laser", "t_powder",
        "melt_width", "melt_depth", "melt_area", "balling",
    ]
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="raise")
    df["id"] = df["id"].astype(int)
    df["balling"] = df["balling"].astype(np.float32)

    # J/mm: W / (mm/s) = J/mm.
    df["linear_energy"] = df["P_laser"] / df["v_scan"]

    # Conventions: d_laser is treated as the laser spot DIAMETER; v_scan is
    # converted from mm/s to m/s. These conventions match the normalized
    # enthalpy / Peclet definitions used by the scaling-law literature.
    #
    # Pe = v*d/alpha
    # H* = eta*P / (h_s * sqrt(alpha*d^3*v))
    # where h_s = rho*cp*(Tm-T0) is the sensible enthalpy density needed
    # to heat the material from T0 to the melting temperature.
    # 316L 물성 상수 (재료가 바뀌면 이 값들을 바꿔야 함)
    rho_316l = 7800.0              # kg/m^3
    cp_316l = 460.0                # J/(kg K)
    alpha_316l = 5.0e-6            # m^2/s, representative 316L value
    Tm_316l = 1678.0               # K
    T0_316l = 293.15               # K
    absorptivity_316l = 0.34
    h_s_vol = rho_316l * cp_316l * (Tm_316l - T0_316l)

    v_mps = np.maximum(df["v_scan"].to_numpy(dtype=float) / 1000.0, 1e-12)
    diameter_m = np.maximum(df["d_laser"].to_numpy(dtype=float) / 1000.0, 1e-12)
    power_w = np.maximum(df["P_laser"].to_numpy(dtype=float), 1e-12)
    df["peclet_number"] = v_mps * diameter_m / alpha_316l
    df["normalized_enthalpy"] = (
        absorptivity_316l * power_w
        / (h_s_vol * np.sqrt(alpha_316l * diameter_m**3 * v_mps))
    )

    # Aspect ratio is intentionally computed from the measured geometry rather
    # than predicted independently:
    #     R = D / W
    # It will later be used to reconstruct physical depth as D = W * R.
    width = df["melt_width"].clip(lower=1e-6)
    df["aspect_ratio"] = df["melt_depth"] / width

    if not np.isfinite(df["aspect_ratio"]).all():
        raise ValueError("Non-finite aspect_ratio found in melt-pool data")

    return df


# x, y 표준화 기준(scaler)을 묶어 보관. 예측 시 같은 기준으로 변환·역변환하는 데 사용
@dataclass
class DataScalers:
    x: StandardScaler
    y: StandardScaler
    features: list[str]
    targets: list[str]



# 행 단위 무작위 분할: train 70% / val 15% / test 15%
# 주의: 같은 공정조건의 반복 실험 쌍이 train과 test에 나뉠 수 있음
def make_iid_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Initial IID split. A process-space split can be added as a separate experiment."""
    train, temp = train_test_split(df, test_size=1.0 - train_ratio, random_state=seed)
    relative_val = val_ratio / (1.0 - train_ratio)
    val, test = train_test_split(temp, test_size=1.0 - relative_val, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)



# x, y를 평균 0 / 표준편차 1로 맞추는 scaler 생성. val/test 정보 유출을 막기 위해 train으로만 fit
def fit_scalers(train_df: pd.DataFrame, include_linear_energy: bool = True) -> DataScalers:
    features = get_features(include_linear_energy)
    x = StandardScaler().fit(train_df[features])
    y = StandardScaler().fit(train_df[TARGETS])
    return DataScalers(x=x, y=y, features=features, targets=TARGETS.copy())


# PyTorch Dataset. 샘플 하나를 {"x", "y", "balling", "id"} dict로 반환 (x, y는 표준화된 값)
class MeltPoolDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scalers: DataScalers):
        self.df = df.reset_index(drop=True)
        self.x = scalers.x.transform(self.df[scalers.features]).astype(np.float32)
        self.y = scalers.y.transform(self.df[scalers.targets]).astype(np.float32)
        self.balling = self.df[RISK].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.from_numpy(self.y[idx]),
            "balling": torch.tensor(self.balling[idx]),
            "id": int(self.df.iloc[idx]["id"]),
        }
