import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import DEVICE, TARGETS


def gaussian_nll(prediction, log_variance, target):
    return 0.5 * (log_variance + (target - prediction).pow(2) * torch.exp(-log_variance)).mean()


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        x = batch["x"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        anchor = batch["anchor"].to(DEVICE)
        target = batch["y"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        prediction, log_variance = model(x, mask, anchor)
        loss = gaussian_nll(prediction, log_variance, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(target)
        sample_count += len(target)
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    values = []
    for batch in loader:
        prediction, log_variance = model(
            batch["x"].to(DEVICE), batch["mask"].to(DEVICE), batch["anchor"].to(DEVICE)
        )
        values.append(gaussian_nll(prediction, log_variance, batch["y"].to(DEVICE)).item())
    return float(np.mean(values)) if values else np.nan


def fit_model(model, train_loader, validation_loader, epochs=60, learning_rate=1e-3, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = {"train": [], "val": []}
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        validation_loss = evaluate_loss(model, validation_loader)
        history["train"].append(train_loss)
        history["val"].append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}: train={train_loss:.4f}, val={validation_loss:.4f}")
        if stale_epochs >= patience:
            print("Early stopping at", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    predictions, standard_deviations, targets, raw_targets, drift_flags = [], [], [], [], []
    for batch in loader:
        prediction, log_variance = model(
            batch["x"].to(DEVICE), batch["mask"].to(DEVICE), batch["anchor"].to(DEVICE)
        )
        predictions.append(prediction.cpu().numpy())
        standard_deviations.append(torch.exp(0.5 * log_variance).cpu().numpy())
        targets.append(batch["y"].numpy())
        raw_targets.append(batch["y_raw"].numpy())
        drift_flags.append(batch["drift"].numpy())
    return tuple(np.concatenate(values) for values in (
        predictions, standard_deviations, targets, raw_targets, drift_flags
    ))


def metric_rows(name, y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    return [
        {
            "model": name,
            "target": target,
            "MAE": mean_absolute_error(y_true[:, index], y_pred[:, index]),
            "RMSE": mean_squared_error(y_true[:, index], y_pred[:, index]) ** 0.5,
            "R2": r2_score(y_true[:, index], y_pred[:, index]) if len(y_true) > 1 else np.nan,
        }
        for index, target in enumerate(TARGETS)
    ]