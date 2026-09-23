"""Train baseline vs physics-aware looped core on one-step cure prediction,
then compare them on an out-of-distribution temperature regime -- this is the
concrete test of the "제한된 데이터/미지 환경에서도 물리적 일관성 유지" claim.

Task: (surface_temp_K, current_alpha, dt) -> next_alpha.
physics_fn is the exact Arrhenius cure-rate equation (sim.cure.cure_rate),
so the physics-aware core gets told precisely how wrong each guess is at
every loop step, even outside the training distribution -- the baseline has
no such signal and can only interpolate what it saw during training.
"""
import numpy as np
import torch
import torch.nn as nn

from sim.cure import cure_rate
from model.physics_core import LoopedPhysicsCore
from model.baseline import BaselineMLP


def sample_dataset(n, temp_c_range, alpha_range=(0.0, 0.9), dt=2.0, seed=0):
    rng = np.random.default_rng(seed)
    temp_k = rng.uniform(*temp_c_range, size=n) + 273.15
    alpha = rng.uniform(*alpha_range, size=n)
    alpha_next = np.clip(alpha + cure_rate(alpha, temp_k) * dt, 0.0, 1.0)

    x = np.stack([temp_k, alpha, np.full(n, dt)], axis=-1).astype(np.float32)
    y = alpha_next.reshape(-1, 1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def physics_fn(x, y_next_alpha):
    temp_k, alpha, dt = x[..., 0:1], x[..., 1:2], x[..., 2:3]
    expected = alpha + torch.as_tensor(
        cure_rate(alpha.detach().numpy(), temp_k.detach().numpy()), dtype=x.dtype
    ) * dt
    return y_next_alpha - expected


def train(model, forward_fn, x, y, epochs=1500, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(forward_fn(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def fit_core(temp_c_range=(140, 190), n=2000, dt=2.0, seed=1, epochs=1500, lr=3e-3, **core_kwargs):
    """Train a LoopedPhysicsCore on the cure task and return it in eval mode.
    Reused by app.py so the dashboard doesn't duplicate the training setup."""
    x_train, y_train = sample_dataset(n, temp_c_range=temp_c_range, dt=dt, seed=seed)
    core = LoopedPhysicsCore(input_dim=3, state_dim=1, **core_kwargs)
    core_forward = lambda x: core(x.unsqueeze(1), physics_fn)[0].squeeze(1)
    train(core, core_forward, x_train, y_train, epochs=epochs, lr=lr)
    core.eval()
    return core


def main():
    torch.manual_seed(0)
    dt = 2.0

    x_train, y_train = sample_dataset(2000, temp_c_range=(140, 190), dt=dt, seed=1)
    x_ood, y_ood = sample_dataset(500, temp_c_range=(200, 230), dt=dt, seed=2)  # unseen regime

    baseline = BaselineMLP(input_dim=3, output_dim=1)
    train(baseline, baseline, x_train, y_train)

    core = fit_core(hidden_dim=32, n_loops=3)
    core_forward = lambda x: core(x.unsqueeze(1), physics_fn)[0].squeeze(1)

    with torch.no_grad():
        mse = nn.functional.mse_loss
        in_baseline = mse(baseline(x_train), y_train).item()
        in_physics = mse(core_forward(x_train), y_train).item()
        ood_baseline = mse(baseline(x_ood), y_ood).item()
        ood_physics = mse(core_forward(x_ood), y_ood).item()

    print(f"in-distribution  MSE  baseline={in_baseline:.5f}  physics-aware={in_physics:.5f}")
    print(f"out-of-distrib.  MSE  baseline={ood_baseline:.5f}  physics-aware={ood_physics:.5f}")
    print(f"OOD/in-dist error blowup  baseline={ood_baseline / in_baseline:.1f}x"
          f"  physics-aware={ood_physics / in_physics:.1f}x")

    assert ood_physics < ood_baseline, (
        "physics-aware core should generalize better than a physics-agnostic "
        "baseline on the unseen temperature regime -- if this fails, the loop "
        "isn't actually using the physics residual usefully"
    )
    print("check ok: physics-aware core generalizes better on unseen conditions")


if __name__ == "__main__":
    main()
