from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, INPUT_FEATURES, PROCESS_FEATURES, TARGETS


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the source CSV used by the V5 replay experiment."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    columns = ["source_index", *PROCESS_FEATURES, *TARGETS, "balling"]
    raw = pd.read_csv(path, skiprows=1, header=None, names=columns)
    required = PROCESS_FEATURES + TARGETS
    for column in required:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    data = raw[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    data = data[(data[TARGETS] > 0).all(axis=1)].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No valid positive geometry rows found in {path}")
    return data


def transform_stream_splits(stream: pd.DataFrame):
    """Split chronologically and fit scalers on the train partition only."""
    n_rows = len(stream)
    train_end = int(n_rows * 0.60)
    val_end = int(n_rows * 0.75)
    train = stream.iloc[:train_end].copy()
    validation = stream.iloc[train_end:val_end].copy()
    test = stream.iloc[val_end:].copy()

    x_scaler = StandardScaler().fit(train[INPUT_FEATURES])
    y_scaler = StandardScaler().fit(np.log1p(train[TARGETS]))

    def transform(part):
        out = part.copy()
        out[[f"z_{name}" for name in INPUT_FEATURES]] = x_scaler.transform(part[INPUT_FEATURES])
        out[[f"z_{name}" for name in TARGETS]] = y_scaler.transform(np.log1p(part[TARGETS]))
        return out

    return transform(train), transform(validation), transform(test), x_scaler, y_scaler


def add_scaled_anchors(part, y_scaler):
    out = part.copy()
    anchor = out[["anchor_W", "anchor_D", "anchor_A"]].to_numpy(float)
    out[["z_anchor_W", "z_anchor_D", "z_anchor_A"]] = y_scaler.transform(np.log1p(anchor))
    return out