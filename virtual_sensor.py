"""Controllable synthetic sensor environment for the 316L LPBF digital twin.

The simulator follows ``process parameters -> physical state -> observation``.
Generated time evolution is synthetic because the source CSV has no timestamps.
Ground-truth columns are retained only when ``include_ground_truth=True``.

Example::

    from virtual_sensor import VirtualSensorSimulator, SimulatorConfig, read_process_csv

    simulator = VirtualSensorSimulator(SimulatorConfig(sequence_length=100), seed=42)
    sensors, states = simulator.generate(read_process_csv("data/316L_meltpool.csv"))
    sensors.to_csv("synthetic_sensor_data.csv", index=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class SensorConfig:
    """Noise and failure parameters for one sensor family."""

    name: str
    noise_std: float = 0.01
    bias: float = 0.0
    bias_std: float = 0.0
    drift_std: float = 0.0
    outlier_probability: float = 0.0
    outlier_scale: float = 4.0
    missing_probability: float = 0.0
    delay_steps: int = 0
    quantization: float | None = None
    response_alpha: float = 0.25


@dataclass
class SimulatorConfig:
    """Top-level simulator configuration."""

    sequence_length: int = 100
    sampling_rate: float = 1000.0
    include_ground_truth: bool = True
    thermal: SensorConfig = field(
        default_factory=lambda: SensorConfig(
            "thermal", noise_std=0.018, bias_std=0.008, drift_std=0.001,
            outlier_probability=0.003, missing_probability=0.005, delay_steps=2,
        )
    )
    optical: SensorConfig = field(
        default_factory=lambda: SensorConfig(
            "optical", noise_std=0.025, bias_std=0.012, drift_std=0.0015,
            outlier_probability=0.006, missing_probability=0.008, delay_steps=1,
        )
    )
    acoustic: SensorConfig = field(
        default_factory=lambda: SensorConfig(
            "acoustic", noise_std=0.04, bias_std=0.02, drift_std=0.003,
            outlier_probability=0.01, missing_probability=0.01,
        )
    )
    process: SensorConfig = field(
        default_factory=lambda: SensorConfig(
            "process", noise_std=0.008, bias_std=0.004, drift_std=0.0005,
            outlier_probability=0.002, missing_probability=0.003,
        )
    )


class VirtualSensorSimulator:
    """Generate correlated virtual sensor streams from LPBF experiment rows.

    Dependencies are intentionally explicit:
    thermal uses power, speed, linear energy and morphology; optical uses W/D/A/R;
    acoustic uses energy, aspect ratio, balling and a shared instability latent;
    process sensors use only measured process parameters. No observation is a
    direct copy of W, D, R or A.
    """

    _ALIASES: Mapping[str, tuple[str, ...]] = {
        "sample_id": ("sample_id", "id", "No", "index"),
        "P_laser": ("P_laser", "P_laser [W]", "laser_power"),
        "v_scan": ("v_scan", "v_scan [mm/s]", "scan_speed"),
        "d_laser": ("d_laser", "d_laser [mm]", "laser_diameter"),
        "t_powder": ("t_powder", "t_powder [um]", "powder_thickness"),
        "W": ("W", "melt_width", "Weldwidth w_w [um]"),
        "D": ("D", "melt_depth", "Penetrationdepth d_w [um]"),
        "A": ("A", "melt_area", "Moltencross-sectionalarea A_m [um]"),
        "Balling": ("Balling", "balling"),
    }

    def __init__(self, config: SimulatorConfig | None = None, seed: int | None = 42):
        self.config = config or SimulatorConfig()
        if self.config.sequence_length < 1 or self.config.sampling_rate <= 0:
            raise ValueError("sequence_length must be positive and sampling_rate must be > 0")
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        process_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(sensor_observations, physical_states)`` in long format."""
        data = self._normalise_input(process_data)
        states = self._physical_states(data)
        sensor_rows: list[pd.DataFrame] = []
        state_rows: list[pd.DataFrame] = []
        for row_index in range(len(data)):
            state = states.iloc[row_index]
            process = data.iloc[row_index]
            base, state_stream = self._latent_stream(process, state)
            observed = self._sensor_streams(process, state, base, state_stream)
            sensor_rows.append(observed)
            truth = state_stream.copy()
            truth.insert(0, "sample_id", process["sample_id"])
            state_rows.append(truth)

        sensors = pd.concat(sensor_rows, ignore_index=True)
        truths = pd.concat(state_rows, ignore_index=True)
        if self.config.include_ground_truth:
            sensors = sensors.merge(truths, on=["sample_id", "time"], how="left")
        return sensors, truths

    def _normalise_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.rename(columns=lambda column: str(column).strip().strip(',"\'').strip())
        rename: dict[str, str] = {}
        for target, aliases in self._ALIASES.items():
            found = next((column for column in frame.columns if column in aliases), None)
            if found is None and target not in {"sample_id", "Balling"}:
                raise ValueError(f"Missing required process column for {target}: {aliases}")
            if found is not None:
                rename[found] = target
        data = frame.rename(columns=rename).copy()
        if "sample_id" not in data:
            data.insert(0, "sample_id", np.arange(len(data)))
        if "Balling" not in data:
            data["Balling"] = 0.0
        for column in ("P_laser", "v_scan", "d_laser", "t_powder", "W", "D", "A", "Balling"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[["P_laser", "v_scan", "W", "D", "A"]].isna().any().any():
            raise ValueError("Required process and physical-state values cannot be missing")
        return data

    def _physical_states(self, data: pd.DataFrame) -> pd.DataFrame:
        width = data["W"].to_numpy(float)
        depth = data["D"].to_numpy(float)
        area = data["A"].to_numpy(float)
        ratio = np.divide(depth, np.maximum(width, 1e-6))
        energy = data["P_laser"].to_numpy(float) / np.maximum(data["v_scan"].to_numpy(float), 1e-6)
        peclet = data["v_scan"].to_numpy(float) * np.maximum(data["d_laser"].to_numpy(float), 1e-6) / 0.004
        enthalpy = energy / np.maximum(data["d_laser"].to_numpy(float) * np.maximum(data["t_powder"].to_numpy(float), 1.0), 1e-6)
        regime = np.select([ratio < 0.3, ratio < 0.8, ratio < 1.5], ["shallow", "intermediate", "deep"], default="high_aspect")
        return pd.DataFrame({
            "true_W": width, "true_D": depth, "true_R": ratio, "true_A": area,
            "linear_energy": energy, "peclet_number": peclet,
            "normalized_enthalpy": enthalpy, "morphology_regime_proxy": regime,
        })

    def _latent_stream(self, process: pd.Series, state: pd.Series) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        n = self.config.sequence_length
        phase = np.linspace(0.0, 1.0, n)
        warmup = 1.0 - 0.12 * np.exp(-phase * 10.0)
        energy_norm = (state["linear_energy"] / 0.8) / (1.0 + state["linear_energy"] / 0.8)
        instability = np.clip(0.12 + 0.20 * float(process["Balling"]) + 0.20 * abs(state["true_R"] - 0.75), 0.05, 1.0)
        shared = self.rng.normal(0, 1, n)
        kernel_size = min(5, n)
        shared = np.convolve(shared, np.ones(kernel_size) / kernel_size, mode="same")
        evolution = warmup * (1.0 + 0.025 * np.sin(2 * np.pi * phase) + 0.018 * shared)
        latent = {"energy_norm": np.full(n, energy_norm), "instability": np.full(n, instability), "evolution": evolution}
        states = pd.DataFrame({
            "time": np.arange(n, dtype=float) / self.config.sampling_rate,
            "true_W": np.full(n, state["true_W"]), "true_D": np.full(n, state["true_D"]),
            "true_R": np.full(n, state["true_R"]), "true_A": np.full(n, state["true_A"]),
            "linear_energy": np.full(n, state["linear_energy"]),
            "morphology_regime_proxy": np.full(n, state["morphology_regime_proxy"]),
        })
        return latent, states

    def _sensor_streams(self, process: pd.Series, state: pd.Series, latent: dict[str, np.ndarray], state_stream: pd.DataFrame) -> pd.DataFrame:
        n = self.config.sequence_length
        energy = latent["energy_norm"] * latent["evolution"]
        instability = latent["instability"] * latent["evolution"]
        # Thermal: power/energy raises temperature; speed lowers energy per length.
        temperature = 0.35 + 0.65 * energy + 0.08 * np.tanh(state["true_D"] / 100.0)
        temperature += 0.015 * instability * self.rng.normal(size=n)
        thermal_intensity = np.clip(temperature**2, 0, None)
        # Optical: morphology determines apparent response, then sensor effects obscure it.
        width_signal = 0.25 + 0.55 * np.tanh(state["true_W"] / 250.0) + 0.05 * energy
        area_signal = 0.20 + 0.65 * np.tanh(state["true_A"] / 20000.0) + 0.06 * energy
        optical_intensity = np.clip(0.15 + 0.70 * temperature + 0.08 * energy, 0, None)
        edge_fluctuation = 0.02 + 0.16 * instability
        # Acoustic is a proxy, not a calibrated physical acoustic-emission quantity.
        acoustic_rms = 0.12 + 0.28 * energy + 0.40 * instability + 0.05 * abs(state["true_R"] - 0.75)
        acoustic_variance = (0.03 + 0.25 * instability) ** 2
        values = {
            "P_laser_sensor": self._apply_failure(np.full(n, process["P_laser"]), self.config.process),
            "v_scan_sensor": self._apply_failure(np.full(n, process["v_scan"]), self.config.process),
            "d_laser_sensor": self._apply_failure(np.full(n, process["d_laser"]), self.config.process),
            "t_powder_sensor": self._apply_failure(np.full(n, process["t_powder"]), self.config.process),
            "temperature_sensor": self._apply_failure(temperature, self.config.thermal),
            "thermal_intensity": self._apply_failure(thermal_intensity, self.config.thermal),
            "optical_width": self._apply_failure(width_signal, self.config.optical),
            "optical_area": self._apply_failure(area_signal, self.config.optical),
            "optical_intensity": self._apply_failure(optical_intensity, self.config.optical),
            "edge_fluctuation": self._apply_failure(edge_fluctuation * (1 + 0.2 * self.rng.normal(size=n)), self.config.optical),
            "acoustic_rms": self._apply_failure(acoustic_rms, self.config.acoustic),
            "acoustic_variance": self._apply_failure(acoustic_variance, self.config.acoustic),
        }
        result = pd.DataFrame(values)
        result.insert(0, "time", state_stream["time"].to_numpy())
        result.insert(0, "sample_id", process["sample_id"])
        validity_columns = {
            "temperature": ["temperature_sensor", "thermal_intensity"],
            "optical": ["optical_width", "optical_area", "optical_intensity", "edge_fluctuation"],
            "acoustic": ["acoustic_rms", "acoustic_variance"],
            "process": ["P_laser_sensor", "v_scan_sensor", "d_laser_sensor", "t_powder_sensor"],
        }
        for name, columns in validity_columns.items():
            result[f"{name}_valid"] = ~result[columns].isna().all(axis=1)
        result["sensor_validity"] = result[["temperature_valid", "optical_valid", "acoustic_valid", "process_valid"]].all(axis=1)
        return result

    def _apply_failure(self, signal: np.ndarray, config: SensorConfig) -> np.ndarray:
        signal = np.asarray(signal, dtype=float).copy()
        n = len(signal)
        bias = config.bias + self.rng.normal(0, config.bias_std)
        drift = np.cumsum(self.rng.normal(0, config.drift_std, n))
        measured = signal + bias + drift + self.rng.normal(0, config.noise_std, n)
        if config.delay_steps > 0:
            delayed = np.full(n, measured[0])
            delayed[config.delay_steps:] = measured[:-config.delay_steps]
            measured = delayed
        if config.response_alpha < 1.0:
            filtered = np.empty(n)
            filtered[0] = measured[0]
            for index in range(1, n):
                filtered[index] = config.response_alpha * measured[index] + (1 - config.response_alpha) * filtered[index - 1]
            measured = filtered
        outliers = self.rng.random(n) < config.outlier_probability
        measured[outliers] += self.rng.normal(0, config.outlier_scale * max(config.noise_std, 1e-6), outliers.sum())
        measured[self.rng.random(n) < config.missing_probability] = np.nan
        if config.quantization and config.quantization > 0:
            measured = np.round(measured / config.quantization) * config.quantization
        return measured


def correlation_analysis(sensors: pd.DataFrame, states: pd.DataFrame | None = None) -> pd.Series:
    """Calculate requested process-to-observation correlations."""
    frame = sensors.copy()
    if states is not None and "true_R" not in frame:
        frame = frame.merge(states[["sample_id", "time", "true_R", "linear_energy"]], on=["sample_id", "time"], how="left")
    return pd.Series({
        "P_laser_vs_temperature": frame[["P_laser_sensor", "temperature_sensor"]].corr().iloc[0, 1],
        "linear_energy_vs_optical_width": frame[["linear_energy", "optical_width"]].corr().iloc[0, 1],
        "R_vs_acoustic_rms": frame[["true_R", "acoustic_rms"]].corr().iloc[0, 1],
    })


def read_process_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read the repository's legacy CSV, whose header is wrapped in single quotes."""
    return pd.read_csv(csv_path, quotechar="'")


def plot_diagnostics(sensors: pd.DataFrame, output_dir: str | Path = "virtual_sensor_plots", sample_id: object | None = None) -> None:
    """Save correlation plots, one stream plot, and normal/failure comparison."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = sensors.copy()
    sample = frame["sample_id"].iloc[0] if sample_id is None else sample_id
    subset = frame[frame["sample_id"] == sample]
    plots = [
        ("laser_vs_temperature.png", "P_laser_sensor", "temperature_sensor", "Laser Power Sensor", "Temperature Sensor"),
        ("energy_vs_optical_width.png", "linear_energy", "optical_width", "Linear Energy", "Optical Melt Pool Width"),
        ("aspect_ratio_vs_acoustic.png", "true_R", "acoustic_rms", "Melt Pool Aspect Ratio", "Acoustic RMS"),
    ]
    for filename, x, y, xlabel, ylabel in plots:
        if x not in frame:
            continue
        plt.figure(figsize=(6, 4)); plt.scatter(frame[x], frame[y], s=8, alpha=0.25); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(output / filename, dpi=150); plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(subset["time"], subset["temperature_sensor"]); axes[0].set_ylabel("Temperature")
    axes[1].plot(subset["time"], subset["optical_intensity"]); axes[1].set_ylabel("Optical")
    axes[2].plot(subset["time"], subset["acoustic_rms"]); axes[2].set_ylabel("Acoustic RMS"); axes[2].set_xlabel("Time [s]")
    fig.tight_layout(); fig.savefig(output / "sensor_stream.png", dpi=150); plt.close(fig)


def load_and_generate(csv_path: str | Path, output_csv: str | Path = "synthetic_sensor_data.csv", seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience API for CSV input and sensor CSV output."""
    sensors, states = VirtualSensorSimulator(seed=seed).generate(read_process_csv(csv_path))
    sensors.to_csv(output_csv, index=False)
    return sensors, states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate correlated synthetic LPBF sensor streams")
    parser.add_argument("csv", nargs="?", default="data/316L_meltpool.csv")
    parser.add_argument("--output", default="synthetic_sensor_data.csv")
    parser.add_argument("--plots", default="virtual_sensor_plots")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = SimulatorConfig(sequence_length=args.length)
    sensor_data, physical_states = VirtualSensorSimulator(config, seed=args.seed).generate(read_process_csv(args.csv))
    sensor_data.to_csv(args.output, index=False)
    print(f"Generated {len(sensor_data):,} sensor observations for {sensor_data['sample_id'].nunique():,} samples")
    print(correlation_analysis(sensor_data, physical_states).round(4).to_string())
    plot_diagnostics(sensor_data, args.plots)
    print(f"Saved {args.output} and plots to {args.plots}")