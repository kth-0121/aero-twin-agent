import json
from pathlib import Path

import joblib
import torch

from .config import INPUT_FEATURES, N_LOOPS, PROCESS_FEATURES, TARGETS, WINDOW_EVENTS
from .windows import SEQUENCE_FEATURES


def save_artifacts(model, x_scaler, y_scaler, eta, output_dir="physical_ai_lt_artifacts"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(SEQUENCE_FEATURES),
            "hidden": 64,
            "n_loops": N_LOOPS,
            "max_delta": model.max_delta,
            "targets": TARGETS,
            "input_features": INPUT_FEATURES,
            "seq_features": SEQUENCE_FEATURES,
            "window_events": WINDOW_EVENTS,
            "eta_effective": float(eta),
        },
        output_dir / "v5.pt",
    )
    joblib.dump(x_scaler, output_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "y_scaler.joblib")
    config = {
        "process_features": PROCESS_FEATURES,
        "targets": TARGETS,
        "input_features": INPUT_FEATURES,
        "seq_features": SEQUENCE_FEATURES,
        "window_events": WINDOW_EVENTS,
        "eta_effective": float(eta),
        "model_updated_online": False,
        "data_origin": "simulation_from_static_dataset",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as output:
        json.dump(config, output, ensure_ascii=False, indent=2)
    return output_dir