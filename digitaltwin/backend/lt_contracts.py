from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol


PROCESS_FEATURES = ("P_laser", "v_scan", "d_laser", "t_powder")
TARGETS = ("melt_width", "melt_depth", "melt_area")


@dataclass(frozen=True)
class LTRequest:
    process: Mapping[str, object]
    event_id: str | None = None
    job_id: str | None = None
    timestamp: str | None = None
    delta_t: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event: Mapping[str, object]) -> "LTRequest":
        return cls(
            process={name: event[name] for name in PROCESS_FEATURES if name in event},
            event_id=None if event.get("event_id") is None else str(event["event_id"]),
            job_id=None if event.get("job_id") is None else str(event["job_id"]),
            timestamp=None if event.get("timestamp") is None else str(event["timestamp"]),
            delta_t=None if event.get("delta_t") is None else float(event["delta_t"]),
            metadata={
                name: value
                for name, value in event.items()
                if name not in PROCESS_FEATURES
                and name not in {"event_id", "job_id", "timestamp", "delta_t"}
            },
        )

    def as_event(self) -> dict[str, object]:
        event = {**self.metadata, **self.process}
        for name in ("event_id", "job_id", "timestamp", "delta_t"):
            value = getattr(self, name)
            if value is not None:
                event[name] = value
        return event


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
    model_version: str = "LT-V5"

    @property
    def aspect_ratio(self) -> float:
        return self.depth_um / self.width_um

    def as_process_fields(self) -> dict:
        return {
            "melt_width": self.width_um,
            "melt_depth": self.depth_um,
            "aspect_ratio": self.aspect_ratio,
            "melt_area": self.area_um2,
            "prediction_model": self.model_version,
            "prediction_source": "MODEL",
            "lt_drift_score": self.drift_score,
            "lt_drift_warning": self.drift_warning,
            "lt_uncertainty_sigma_approx": dict(self.uncertainty_sigma_approx),
            "lt_loop_predictions": [dict(row) for row in self.loop_predictions],
            "lt_sensor_source": self.sensor_source,
        }


@dataclass(frozen=True)
class LTResponse:
    status: Literal["ok", "error"]
    model_version: str
    prediction: GeometryPrediction | None = None
    error: str | None = None
    event_id: str | None = None

    def __post_init__(self):
        if self.status == "ok" and (self.prediction is None or self.error is not None):
            raise ValueError("Successful LT responses require a prediction and no error")
        if self.status == "error" and (self.error is None or self.prediction is not None):
            raise ValueError("Failed LT responses require an error and no prediction")


class GeometryPredictor(Protocol):
    def predict(self, event: Mapping[str, object]) -> GeometryPrediction:
        """Predict melt-pool geometry from one raw process event."""