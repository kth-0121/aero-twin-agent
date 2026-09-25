"""Train + ablate on real 316L LPBF melt pool data (data/316L_meltpool.csv,
Zenodo 16979848, CC BY 4.0, 677 tracks / 623 unique conditions).

Train/test split is physics-motivated, not arbitrary: conduction-regime rows
(D/W < 0.5, where the team confirmed the Rosenthal ROM is roughly valid) are
the training distribution; transition/keyhole rows (D/W >= 0.5, where the
team measured the ROM breaking down) are the held-out "unseen regime" test.
This directly tests whether physics feedback still helps once the physics
model itself gets less reliable -- the exact risk the team's scenario doc
flags as the biggest technical risk.

Ablation (team's experiment #2): MLP / ROM-only / loop-without-residual /
loop-with-residual (physics-aware), to separate "does iterating help" from
"does the physics residual specifically help".
"""
import numpy as np
import torch
import torch.nn as nn

from sim.lpbf import melt_pool_width, melt_pool_depth
from model.physics_core import LoopedPhysicsCore
from model.baseline import BaselineMLP

DATA_PATH = "data/316L_meltpool.csv"
_COLS = ["idx", "power_w", "speed_mm_s", "spot_mm", "layer_t_um", "width_um", "depth_um", "area_um2", "balling"]


def _read_rows(path=DATA_PATH):
    """The Zenodo CSV's header row is malformed (double-quoted as one cell),
    but every data row is plain comma-separated -- skip the header and name
    columns manually instead of fighting csv.DictReader over it."""
    with open(path) as f:
        lines = f.readlines()
    rows = []
    for line in lines[1:]:
        vals = line.strip().split(",")
        if len(vals) == len(_COLS):
            rows.append(dict(zip(_COLS, vals)))
    return rows


def load_dataset(regime="all", path=DATA_PATH):
    """regime: 'conduction', 'keyhole_or_transition', or 'all'."""
    xs, ys = [], []
    for row in _read_rows(path):
        power, speed = float(row["power_w"]), float(row["speed_mm_s"])
        spot_um = float(row["spot_mm"]) * 1000
        layer_t = float(row["layer_t_um"])
        width, depth = float(row["width_um"]), float(row["depth_um"])
        if width <= 0 or depth <= 0 or spot_um <= 0:
            continue
        row_regime = "keyhole_or_transition" if (depth / width) >= 0.5 else "conduction"
        if regime != "all" and row_regime != regime:
            continue
        xs.append([power, speed, spot_um, layer_t])
        ys.append([depth, width])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def physics_fn(x, y_pred):
    """ROM residual: predicted (depth,width) in um minus the Rosenthal-ROM
    estimate in um. x = [power_W, speed_mm_s, spot_um, layer_thickness_um]."""
    power, speed_mm_s = x[..., 0:1], x[..., 1:2]
    speed_m_s = (speed_mm_s / 1000.0).detach().numpy()
    power_np = power.detach().numpy()

    w_rom_um = melt_pool_width(power_np, speed_m_s) * 1e6
    d_rom_um = melt_pool_depth(power_np, speed_m_s) * 1e6
    rom = np.concatenate([d_rom_um, w_rom_um], axis=-1)
    return y_pred - torch.as_tensor(rom, dtype=y_pred.dtype)


def zero_physics_fn(x, y_pred):
    """Ablation C: same loop architecture, but no physics grounding --
    isolates whether iterating alone helps vs. the residual specifically."""
    return torch.zeros_like(y_pred)


def rom_predict(x):
    """Ablation B: the physics ROM alone, no learning at all."""
    power, speed_mm_s = x[..., 0:1].numpy(), x[..., 1:2].numpy()
    speed_m_s = speed_mm_s / 1000.0
    w_um = melt_pool_width(power, speed_m_s) * 1e6
    d_um = melt_pool_depth(power, speed_m_s) * 1e6
    return torch.as_tensor(np.concatenate([d_um, w_um], axis=-1), dtype=torch.float32)


def train(model, forward_fn, x, y, epochs=2000, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(forward_fn(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def fit_core(x_train, y_train, pfn=physics_fn, epochs=2000, lr=3e-3, **core_kwargs):
    """Train a LoopedPhysicsCore and return it in eval mode. pfn swaps in
    zero_physics_fn for the no-feedback ablation. Reused by app.py."""
    core = LoopedPhysicsCore(input_dim=4, state_dim=2, **core_kwargs)
    core_forward = lambda x: core(x.unsqueeze(1), pfn)[0].squeeze(1)
    train(core, core_forward, x_train, y_train, epochs=epochs, lr=lr)
    core.eval()
    return core


def monotonicity_check(x, y):
    """Team's Test 1: holding speed/spot/layer fixed, higher power should
    mean deeper melt pool (real data should already show this). Group on
    all three of speed+spot+layer -- grouping on speed alone confounds
    different spot sizes and layer thicknesses into the same comparison."""
    from collections import defaultdict
    groups = defaultdict(list)
    for (power, speed, spot, layer_t), (depth, _) in zip(x.tolist(), y.tolist()):
        groups[(speed, spot, layer_t)].append((power, depth))

    violations, pairs = 0, 0
    for speed, pts in groups.items():
        pts.sort()
        for (p1, d1), (p2, d2) in zip(pts, pts[1:]):
            if p2 > p1:
                pairs += 1
                if d2 < d1:
                    violations += 1
    rate = violations / pairs if pairs else float("nan")
    print(f"monotonicity (P up -> D up, speed held fixed): {violations}/{pairs} violations ({rate:.1%})")
    return rate


def _run_ablation(x_train, y_train, x_test, y_test, label):
    mse = nn.functional.mse_loss

    baseline = BaselineMLP(input_dim=4, output_dim=2)
    train(baseline, baseline, x_train, y_train)

    rom_in = mse(rom_predict(x_train), y_train).item()
    rom_test = mse(rom_predict(x_test), y_test).item()

    loop_nr = fit_core(x_train, y_train, pfn=zero_physics_fn, hidden_dim=32, n_loops=3)
    loop_nr_forward = lambda x: loop_nr(x.unsqueeze(1), zero_physics_fn)[0].squeeze(1)

    core = fit_core(x_train, y_train, hidden_dim=32, n_loops=3)
    core_forward = lambda x: core(x.unsqueeze(1), physics_fn)[0].squeeze(1)

    with torch.no_grad():
        results = {
            "A MLP":               (mse(baseline(x_train), y_train).item(), mse(baseline(x_test), y_test).item()),
            "B ROM only":          (rom_in, rom_test),
            "C loop, no residual": (mse(loop_nr_forward(x_train), y_train).item(), mse(loop_nr_forward(x_test), y_test).item()),
            "D loop + residual":   (mse(core_forward(x_train), y_train).item(), mse(core_forward(x_test), y_test).item()),
        }

    print(f"\n=== {label} ===")
    print(f"{'variant':<22}{'train MSE':>12}{'test MSE':>12}{'test/train':>12}")
    for name, (i, o) in results.items():
        print(f"{name:<22}{i:>12.1f}{o:>12.1f}{o/i:>11.1f}x")
    return results, core


def main():
    torch.manual_seed(0)

    monotonicity_check(*load_dataset(regime="all"))

    # Experiment 1: fair within-regime generalization (random holdout inside conduction)
    x_cond, y_cond = load_dataset(regime="conduction")
    n = len(x_cond)
    perm = torch.randperm(n)
    n_test = n // 5
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    _run_ablation(x_cond[train_idx], y_cond[train_idx], x_cond[test_idx], y_cond[test_idx],
                  "Experiment 1: random holdout WITHIN conduction regime (fair generalization test)")

    # Experiment 2: adversarial regime-shift (team's documented risk -- ROM validity breaks down here)
    x_ood, y_ood = load_dataset(regime="keyhole_or_transition")
    results2, core = _run_ablation(x_cond, y_cond, x_ood, y_ood,
                                    "Experiment 2: conduction -> keyhole/transition regime shift")

    d_test, a_test = results2["D loop + residual"][1], results2["A MLP"][1]
    verdict = "beats" if d_test < a_test else "does NOT beat"
    print(f"\nExperiment 2 verdict: physics-aware (D) {verdict} plain MLP (A) across the regime shift "
          f"({d_test:.0f} vs {a_test:.0f}). This is exactly the risk the team's scenario doc flagged -- "
          f"the ROM itself is only validated in conduction, so its residual signal is less trustworthy "
          f"once real conditions cross into keyhole/transition. Argues for the regime gate: trust the "
          f"physics residual in conduction, fall back to a flagged warning (not a numeric correction) "
          f"in keyhole/transition.")


if __name__ == "__main__":
    main()
