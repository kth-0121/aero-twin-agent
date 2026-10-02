from dataclasses import dataclass
from typing import Mapping, Protocol


PROCESS_FEATURES = ("P_laser", "v_scan", "d_laser", "t_powder")


@dataclass(frozen=True)
class GeometryPrediction:
    width_um: float
    depth_um: float
    area_um2: float
    drift_score: float
    drift_warning: bool
    uncertainty_sigma_approx: Mapping[str, float]
    loop_predictions: tuple[Mapping[str, float], ...]
    sensor_source: str

    @property
    def aspect_ratio(self) -> float:
        return self.depth_um / self.width_um

    def as_process_fields(self) -> dict:
        return {
            "melt_width": self.width_um,
            "melt_depth": self.depth_um,
            "aspect_ratio": self.aspect_ratio,
            "melt_area": self.area_um2,
            "prediction_model": "LT-V5",
            "prediction_source": "MODEL",
            "lt_drift_score": self.drift_score,
            "lt_drift_warning": self.drift_warning,
            "lt_uncertainty_sigma_approx": dict(self.uncertainty_sigma_approx),
            "lt_loop_predictions": [dict(row) for row in self.loop_predictions],
            "lt_sensor_source": self.sensor_source,
        }


class GeometryPredictor(Protocol):
    def predict(self, event: Mapping[str, object]) -> GeometryPrediction:
        """Predict melt-pool geometry from one raw process event."""