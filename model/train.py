"""Train baseline vs physics-aware looped core on real 316L LPBF melt pool
data (data/316L_meltpool.csv), then compare them on an unseen high-power
regime -- the concrete test of "제한된 데이터/미지 환경에서도 물리적 일관성
유지". Unlike the earlier cure experiment, the training LABELS here are real
measured depth/width, not equation-generated -- physics_fn (the Rosenthal ROM)
is only an approximate guide, exactly as the team's scenario doc expects.

Task: (laser_power_W, scan_speed_mm_s, layer_thickness_um) -> (depth_um, width_um).
"""
import csv

import numpy as np
import torch
import torch.nn as nn

from sim.lpbf import melt_pool_width, melt_pool_depth
from model.physics_core import LoopedPhysicsCore
from model.baseline import BaselineMLP

DATA_PATH = "data/316L_meltpool.csv"


def load_dataset(power_range, path=DATA_PATH):
    """Real 316L single-track rows within a laser-power window."""
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            power = float(row["Laser Power"])
            speed = float(row["Scan Speed"])
            layer_t = float(row["Layer Thickness"])
            depth = float(row["Depth"])
            width = float(row["Width"])
            if not (power_range[0] < power <= power_range[1]):
                continue
            if speed <= 0 or depth <= 0 or width <= 0:
                continue  # no-melt / missing rows aren't useful regression targets
            xs.append([power, speed, layer_t])
            ys.append([depth, width])
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def physics_fn(x, y_pred):
    """ROM residual: predicted (depth,width) in um minus the Rosenthal-ROM
    estimate in um. x = [power_W, speed_mm_s, layer_thickness_um]."""
    power, speed_mm_s = x[..., 0:1], x[..., 1:2]
    speed_m_s = (speed_mm_s / 1000.0).detach().numpy()
    power_np = power.detach().numpy()

    w_rom_um = melt_pool_width(power_np, speed_m_s) * 1e6
    d_rom_um = melt_pool_depth(power_np, speed_m_s) * 1e6
    rom = np.concatenate([d_rom_um, w_rom_um], axis=-1)
    return y_pred - torch.as_tensor(rom, dtype=y_pred.dtype)


def train(model, forward_fn, x, y, epochs=2000, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(forward_fn(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def fit_core(x_train, y_train, epochs=2000, lr=3e-3, **core_kwargs):
    """Train a LoopedPhysicsCore on the LPBF task and return it in eval mode.
    Reused by app.py so the dashboard doesn't duplicate the training setup."""
    core = LoopedPhysicsCore(input_dim=3, state_dim=2, **core_kwargs)
    core_forward = lambda x: core(x.unsqueeze(1), physics_fn)[0].squeeze(1)
    train(core, core_forward, x_train, y_train, epochs=epochs, lr=lr)
    core.eval()
    return core


def main():
    torch.manual_seed(0)

    x_train, y_train = load_dataset(power_range=(0, 400))
    x_ood, y_ood = load_dataset(power_range=(600, 10_000))  # unseen high-power regime
    print(f"train rows={len(x_train)}  ood rows={len(x_ood)}")

    baseline = BaselineMLP(input_dim=3, output_dim=2)
    train(baseline, baseline, x_train, y_train)

    core = fit_core(x_train, y_train, hidden_dim=32, n_loops=3)
    core_forward = lambda x: core(x.unsqueeze(1), physics_fn)[0].squeeze(1)

    with torch.no_grad():
        mse = nn.functional.mse_loss
        in_baseline = mse(baseline(x_train), y_train).item()
        in_physics = mse(core_forward(x_train), y_train).item()
        ood_baseline = mse(baseline(x_ood), y_ood).item()
        ood_physics = mse(core_forward(x_ood), y_ood).item()

    print(f"in-distribution  MSE(um^2)  baseline={in_baseline:.1f}  physics-aware={in_physics:.1f}")
    print(f"out-of-distrib.  MSE(um^2)  baseline={ood_baseline:.1f}  physics-aware={ood_physics:.1f}")
    print(f"OOD/in-dist error blowup  baseline={ood_baseline / in_baseline:.1f}x"
          f"  physics-aware={ood_physics / in_physics:.1f}x")

    if ood_physics < ood_baseline:
        print("check ok: physics-aware core generalizes better on the unseen high-power regime")
    else:
        print("check FAILED: physics-aware core did not generalize better -- "
              "ROM may be too inaccurate at this power range to help, investigate before pitching this result")


if __name__ == "__main__":
    main()
