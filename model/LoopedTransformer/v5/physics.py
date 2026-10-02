import numpy as np
import pandas as pd

from .config import (
    GRID_DEPTH_UM,
    GRID_DY_UM,
    GRID_DZ_UM,
    GRID_HALF_WIDTH_UM,
    PROCESS_FEATURES,
    TARGETS,
    T_AMBIENT_K,
    T_LIQUIDUS_K,
    THERMAL_CONDUCTIVITY_W_MK,
    THERMAL_DIFFUSIVITY_M2_S,
)


class RosenthalPhysicsAnchor:
    """Train-calibrated Rosenthal cross-section anchor for W, D and A."""

    def __init__(self, eta: float):
        self.eta = float(eta)
        self.y_axis_um = np.arange(-GRID_HALF_WIDTH_UM, GRID_HALF_WIDTH_UM + GRID_DY_UM, GRID_DY_UM)
        self.z_axis_um = np.arange(0.0, GRID_DEPTH_UM + GRID_DZ_UM, GRID_DZ_UM)
        self.y_m, self.z_m = np.meshgrid(self.y_axis_um * 1e-6, self.z_axis_um * 1e-6)

    @classmethod
    def fit(cls, train_frame: pd.DataFrame):
        from scipy.optimize import minimize_scalar

        train = train_frame.reset_index(drop=True)
        targets = train[TARGETS].to_numpy(float)
        cache = {}

        def objective(eta):
            key = round(float(eta), 5)
            if key not in cache:
                predictions = cls(key).predict(train)
                cache[key] = float(np.mean((np.log1p(predictions) - np.log1p(targets)) ** 2))
            return cache[key]

        result = minimize_scalar(
            objective,
            bounds=(0.05, 0.95),
            method="bounded",
            options={"xatol": 0.01, "maxiter": 24},
        )
        anchor = cls(float(result.x))
        anchor.calibration_loss = float(result.fun)
        return anchor

    def _predict_one(self, power_w, speed_mm_s, diameter_mm):
        power_w = max(float(power_w), 1e-6)
        speed_m_s = max(float(speed_mm_s) * 1e-3, 1e-6)
        radius_m = max(float(diameter_mm) * 0.5e-3, 5e-6)
        x_m = -radius_m
        distance = np.sqrt(x_m**2 + self.y_m**2 + self.z_m**2)
        travel = np.maximum(x_m + distance, 0.0)
        delta_temperature = (
            self.eta * power_w
            / (2.0 * np.pi * THERMAL_CONDUCTIVITY_W_MK * distance)
            * np.exp(-speed_m_s * travel / (2.0 * THERMAL_DIFFUSIVITY_M2_S))
        )
        molten = T_AMBIENT_K + delta_temperature >= T_LIQUIDUS_K
        if not molten.any():
            return 1.0, 1.0, 1.0

        z_index, y_index = np.where(molten)
        width_um = max(float(self.y_axis_um[y_index].max() - self.y_axis_um[y_index].min() + GRID_DY_UM), GRID_DY_UM)
        depth_um = max(float(self.z_axis_um[z_index].max()), GRID_DZ_UM)
        area_um2 = max(float(molten.sum() * GRID_DY_UM * GRID_DZ_UM), GRID_DY_UM * GRID_DZ_UM)
        return width_um, depth_um, area_um2

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.empty((len(frame), 3), dtype=float)
        for index, row in enumerate(frame[PROCESS_FEATURES].itertuples(index=False, name=None)):
            values[index] = self._predict_one(row[0], row[1], row[2])
        return np.maximum(values, 1.0)