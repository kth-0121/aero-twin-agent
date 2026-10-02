from typing import Mapping

import numpy as np

from .lt_contracts import PROCESS_FEATURES


class SyntheticSensorProvider:
    """Generate demo-only sensor proxies using the V5 replay data recipe."""

    source = "SYNTHETIC_PROXY"

    def __init__(self, reference_process_values, seed=42):
        reference = np.asarray(reference_process_values, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != len(PROCESS_FEATURES):
            raise ValueError("reference_process_values must have shape [N, 4]")
        self.median = np.median(reference, axis=0)
        q25, q75 = np.percentile(reference, [25, 75], axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-6)
        self.rng = np.random.default_rng(seed)
        self.latent = 0.0

    def observe(self, process: Mapping[str, float]) -> dict[str, float]:
        values = np.asarray([process[name] for name in PROCESS_FEATURES], dtype=float)
        observed = values + self.rng.normal(0.0, 0.015, size=4) * self.iqr
        scaled = (observed - self.median) / self.iqr
        self.latent = 0.92 * self.latent + self.rng.normal(0.0, 0.18)
        sensor_1 = 0.45 * scaled[0] - 0.25 * scaled[1] + 0.35 * self.latent
        sensor_2 = 0.20 * scaled[1] + 0.30 * scaled[2] + 0.25 * self.latent
        sensor_3 = 0.25 * scaled[0] + 0.15 * scaled[3] + 0.20 * self.latent
        return {
            "sensor_1": float(sensor_1 + self.rng.normal(0.0, 0.12)),
            "sensor_2": float(sensor_2 + self.rng.normal(0.0, 0.12)),
            "sensor_3": float(sensor_3 + self.rng.normal(0.0, 0.12)),
        }