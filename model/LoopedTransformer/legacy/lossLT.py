"""
601Solutions
"""

from __future__ import annotations

import torch
from torch import nn


class DeepSupervisionLoss(nn.Module): # param 가중치 조절
    """
    L = L_data + alpha * L_physics
    """

    def __init__(self, gamma: float = 0.8, state_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        weights = torch.as_tensor(state_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != 3:
            raise ValueError("state_weights must contain exactly 3 values: W, R, A")
        if torch.any(weights <= 0):
            raise ValueError("state_weights must be positive")
        self.gamma = gamma
        self.register_buffer("state_weights", weights)

    def forward(self, predictions, target):
        if target.size(-1) != self.state_weights.numel():
            raise ValueError(
                f"Expected target dimension {self.state_weights.numel()}, got {target.size(-1)}"
            )

        total = target.new_tensor(0.0)
        weight_sum = 0.0
        for k, pred in enumerate(predictions):
            loop_weight = self.gamma ** (len(predictions) - 1 - k)
            per_state = (pred - target).pow(2).mean(dim=0)
            state_loss = (per_state * self.state_weights.to(pred)).sum() / self.state_weights.sum()
            total = total + loop_weight * state_loss
            weight_sum += loop_weight
        return total / weight_sum

class PhysicsConsistencyLoss(nn.Module):
    """Mean squared physics residual"""

    def forward(self, feedbacks, confidences=None):
        if not feedbacks:
            return torch.tensor(0.0)

        terms = []
        for i, feedback in enumerate(feedbacks):
            if confidences is not None and confidences[i] is not None:
                mask = confidences[i]
                denom = mask.sum().clamp_min(1.0)
                terms.append((feedback.pow(2) * mask).sum() / denom)
            else:
                terms.append(feedback.pow(2).mean())
        return torch.stack(terms).mean()
