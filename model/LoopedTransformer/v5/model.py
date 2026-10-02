import torch
from torch import nn


class AdaptivePhysicsResidualLT(nn.Module):
    """GRU history encoder with bounded, target-wise physics residual loops."""

    def __init__(self, input_dim, hidden=64, n_loops=4, dropout=0.1, max_delta=0.50):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.loop_emb = nn.Embedding(n_loops, 8)
        self.residual = nn.Sequential(
            nn.Linear(hidden + 3 + 3 + 8, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden + 6, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.n_loops = n_loops
        self.max_delta = float(max_delta)

    def forward(self, x, mask, anchor, return_loop_preds=False):
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError("x must be [B,T,F] and mask must be [B,T]")
        if x.shape[:2] != mask.shape:
            raise ValueError("x and mask sequence dimensions must match")

        projected = self.input_proj(x)
        lengths = mask.sum(1).clamp(min=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        context = hidden[-1]
        prediction = anchor
        loop_predictions = []

        for loop_index in range(self.n_loops):
            indices = torch.full((len(prediction),), loop_index, dtype=torch.long, device=prediction.device)
            embedding = self.loop_emb(indices)
            residual = self.residual(torch.cat([context, anchor, prediction, embedding], dim=-1))
            delta = torch.tanh(residual[:, :3]) * self.max_delta
            gate = torch.sigmoid(residual[:, 3:])
            prediction = prediction + gate * delta
            loop_predictions.append(prediction)

        log_variance = self.uncertainty(torch.cat([context, anchor, prediction], dim=-1)).clamp(-7, 4)
        if return_loop_preds:
            return prediction, log_variance, torch.stack(loop_predictions, dim=1)
        return prediction, log_variance