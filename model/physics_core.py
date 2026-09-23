"""Physics-aware looped reasoning core (Looped Transformer, arXiv:2311.12424 style).

Closed-loop reasoning: AI -> Physics -> AI -> Physics -> ...
    y0 = f(x, 0, 0)
    r0 = P(x, y0)          <- physics evaluator (governing equations / DT)
    y1 = f(x, y0, r0)
    r1 = P(x, y1)
    ...
Each iteration feeds the previous prediction AND its physics residual back
into the model, so the network can correct itself toward a physically
consistent answer instead of emitting one unchecked forward pass.
"""
import torch
import torch.nn as nn


class LoopedPhysicsCore(nn.Module):
    def __init__(self, input_dim, state_dim, hidden_dim=64, n_loops=4, n_heads=4):
        super().__init__()
        self.n_loops = n_loops
        self.state_dim = state_dim
        self.in_proj = nn.Linear(input_dim + 2 * state_dim, hidden_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, x, physics_fn):
        """
        x: (batch, seq, input_dim) process conditions (e.g. temp, alpha, dt)
        physics_fn(x, y) -> residual (batch, seq, state_dim), the evaluator P.
        Returns (y_final, residual_norms) where residual_norms has one entry
        per loop -- should trend down once the core is trained, and is the
        thing to plot for the "does the loop converge" validation check.
        """
        batch, seq, _ = x.shape
        y = torch.zeros(batch, seq, self.state_dim)
        r = torch.zeros(batch, seq, self.state_dim)

        residual_norms = []
        for _ in range(self.n_loops):
            # detach the carried-over state: each loop step is trained on its
            # own prediction, not backprop'd through the whole unroll (matches
            # the reference dummy -- keeps training stable for a shared-weight loop)
            h = self.in_proj(torch.cat([x, y.detach(), r.detach()], dim=-1))
            h = self.block(h)
            y = self.out_proj(h)
            r = physics_fn(x, y)
            residual_norms.append(r.norm().item())

        return y, residual_norms


def demo():
    """Structural self-check: wires an untrained core to the real cure-kinetics
    residual from sim.cure. Convergence itself only shows up after training --
    this just proves the AI<->Physics loop is actually wired correctly."""
    from sim.cure import cure_rate

    torch.manual_seed(0)
    # x = [current_temp_K, current_alpha, dt]; y = predicted next_alpha
    core = LoopedPhysicsCore(input_dim=3, state_dim=1, hidden_dim=32, n_loops=3)

    temp_k = torch.full((2, 5, 1), 453.15)   # 180C hold
    alpha = torch.rand(2, 5, 1) * 0.3
    dt = torch.full((2, 5, 1), 2.0)
    x = torch.cat([temp_k, alpha, dt], dim=-1)

    def physics_fn(x, y_next_alpha):
        temp_k, alpha, dt = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        expected_next = alpha + cure_rate(alpha.numpy(), temp_k.numpy()) * dt.numpy()
        return y_next_alpha - torch.as_tensor(expected_next, dtype=y_next_alpha.dtype)

    y_final, residual_norms = core(x, physics_fn)
    assert y_final.shape == (2, 5, 1)
    assert len(residual_norms) == 3
    print(f"demo ok: final pred shape={tuple(y_final.shape)}, residual per loop={residual_norms}")


if __name__ == "__main__":
    demo()
