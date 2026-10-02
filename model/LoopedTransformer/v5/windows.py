import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import INPUT_FEATURES, TARGETS, WINDOW_EVENTS


SEQUENCE_FEATURES = [f"z_{name}" for name in INPUT_FEATURES]
ANCHOR_FEATURES = ["z_anchor_W", "z_anchor_D", "z_anchor_A"]
TARGET_Z = [f"z_{name}" for name in TARGETS]


def make_windows(part, window=WINDOW_EVENTS):
    samples = []
    for job_id, group in part.groupby("job_id", sort=False):
        group = group.sort_values("global_index").reset_index(drop=True)
        for index in range(len(group)):
            start = max(0, index - window + 1)
            history = group.iloc[start:index + 1]
            current = group.iloc[index]
            samples.append({
                "job_id": str(job_id),
                "global_index": int(current["global_index"]),
                "x": history[SEQUENCE_FEATURES].to_numpy(np.float32),
                "anchor": current[ANCHOR_FEATURES].to_numpy(np.float32),
                "y": current[TARGET_Z].to_numpy(np.float32),
                "y_raw": current[TARGETS].to_numpy(np.float32),
                "drift": bool(current["drift_injected"]),
            })
    return samples


def collate_windows(batch):
    max_length = max(len(sample["x"]) for sample in batch)
    batch_size = len(batch)
    feature_count = batch[0]["x"].shape[1]
    x = torch.zeros(batch_size, max_length, feature_count)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for index, sample in enumerate(batch):
        length = len(sample["x"])
        x[index, :length] = torch.from_numpy(sample["x"])
        mask[index, :length] = True
    return {
        "x": x,
        "mask": mask,
        "anchor": torch.tensor(np.stack([sample["anchor"] for sample in batch]), dtype=torch.float32),
        "y": torch.tensor(np.stack([sample["y"] for sample in batch]), dtype=torch.float32),
        "y_raw": torch.tensor(np.stack([sample["y_raw"] for sample in batch]), dtype=torch.float32),
        "drift": torch.tensor([sample["drift"] for sample in batch], dtype=torch.bool),
    }


def make_loaders(train_samples, validation_samples, test_samples, batch_size=64):
    return (
        DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=collate_windows),
        DataLoader(validation_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_windows),
        DataLoader(test_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_windows),
    )