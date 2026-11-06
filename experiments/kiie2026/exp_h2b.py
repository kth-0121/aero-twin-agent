"""H2 (revised): physical consistency on the unseen keyhole set (energy-density split, as in train_v9).
Usage: python exp_h2b.py <MLP|LT-NoPhys|LT-Phys> <energy|dw> <n_synthetic> [seeds]
(exp_h2.py with a split choice and a synthetic-row count; real+syn only)

H2-1 detection : does the evaluator's physics residual flag samples with large prediction error?
                 per-sample |residual| (W, R, A dims) vs per-sample std squared error:
                 Spearman rho, and AUROC for "error in the top 25%".
                 Reference: the same AUROC using raw energy density P/(v d) as the score.
                 The residual is computed by the same evaluator for every model, so MLP is scored too.
H2-2 correction: physics residual and independent constraint violations, feedback vs no feedback.
  range    : any of W, R, A outside the min-max of all 677 measured rows
  nonpos   : any of W, R, A <= 0
  monotone : P +5 % should not decrease depth (W*R) or area; v +5 % should not increase them.
             violation rate over 4 checks per sample (derived features recomputed).
"""
import sys, pathlib, os
R = pathlib.Path(os.environ.get("REPO", pathlib.Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(R / "model" / "Baseline"), str(R / "model" / "LoopedTransformer" / "legacy"), str(R / "model" / "LoopedTransformer")]
import numpy as np, pandas as pd, torch
from types import SimpleNamespace
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import common, dataLT
from common import set_seed, train_nn
from modelLT import LoopedTransformer, StateMLP
from physicsLT import FluidThermalPhysicsEvaluator
from lossLT import PhysicsConsistencyLoss
from train_v9_synthetic_augmentation import generate_synthetic_keyhole_rows

MODEL = sys.argv[1]
SPLIT, NSYN = sys.argv[2], int(sys.argv[3])
SEED0 = int(os.environ.get("SEED0", 0))
SEEDS = range(SEED0, SEED0 + (int(sys.argv[4]) if len(sys.argv) > 4 else 5))
torch.set_num_threads(1)
OUT = pathlib.Path(__file__).parent / "exp_results"
if os.environ.get("SMOKE"): common.EPOCHS = 1; OUT = OUT.parent / "exp_smoke"; OUT.mkdir(exist_ok=True); SEEDS = range(1)


def derive(df):
    """Recompute derived features from raw inputs (checked against the data below)."""
    d = df.copy(); v = d.v_scan / 1000.0; dm = d.d_laser / 1000.0; a, hs = 5.0e-6, 7800.0 * 460.0 * (1678.0 - 293.15)
    d["linear_energy"] = d.P_laser / d.v_scan; d["peclet_number"] = v * dm / a
    d["normalized_enthalpy"] = 0.34 * d.P_laser / (hs * np.sqrt(a * dm ** 3 * v)); return d


full = dataLT.load_316l_csv(common.DATA_PATH)
ed = full.P_laser / (full.v_scan / 1000 * full.d_laser / 1000)
ev_mask = ed > 4.4e6 if SPLIT == "energy" else full.aspect_ratio >= 0.5
tr, evdf = full[~ev_mask], full[ev_mask]
s = dataLT.fit_scalers(tr)
X = lambda d: torch.from_numpy(s.x.transform(d[s.features]).astype(np.float32))
Y = lambda d: s.y.transform(d[s.targets]).astype(np.float32)
assert np.allclose(derive(evdf)[s.features].to_numpy(), evdf[s.features].to_numpy())
syn = generate_synthetic_keyhole_rows(NSYN, seed=42)
base = SimpleNamespace(x_train=X(tr).numpy(), y_train=Y(tr))
aug = SimpleNamespace(x_train=np.vstack([base.x_train, X(syn).numpy()]), y_train=np.vstack([base.y_train, Y(syn)]))
y_true = Y(evdf)
lo, hi = full[s.targets].min().to_numpy(), full[s.targets].max().to_numpy()
evaluator = lambda: FluidThermalPhysicsEvaluator(s.x.mean_, s.x.scale_, s.y.mean_, s.y.scale_)
probe = evaluator()   # residual is always measured with a fresh evaluator, whatever the model


def train(data, seed):
    set_seed(seed)
    if MODEL == "MLP":
        m = StateMLP(n_features=7, state_dim=3); train_nn(m, data, lambda m, x: ([m(x)], 0.0))
        return lambda x: m(x)
    phys = MODEL == "LT-Phys"
    e = evaluator() if phys else None; pl = PhysicsConsistencyLoss()
    m = LoopedTransformer(n_features=7, feedback_dim=7 if phys else 3, loop_steps=10)
    def fwd(m, x):
        o = m(x, physics_evaluator=e); return o.predictions, 0.1 * pl(o.feedbacks, o.confidences)
    train_nn(m, data, fwd)
    return lambda x: m(x, physics_evaluator=e, loop_steps=10).predictions[-1]


def depth_area(p):
    W, Rr, A = s.y.inverse_transform(p).T
    return W * Rr, A


rows = []
for dname, data in ((f"real+syn{NSYN}", aug),):
    for seed in SEEDS:
        f = train(data, seed)
        with torch.no_grad():
            x = X(evdf); p = f(x).numpy()
            res = probe.evaluate(x, torch.from_numpy(p)).residual[:, :3].abs().mean(1).numpy()
            mono = []
            for c, sign in (("P_laser", 1), ("v_scan", -1)):
                d = evdf.copy(); d[c] = d[c] * 1.05
                D1, A1 = depth_area(f(X(derive(d))).numpy()); D0, A0 = depth_area(p)
                mono += [sign * (D1 - D0) < 0, sign * (A1 - A0) < 0]
        err = ((p - y_true) ** 2).mean(1)
        yv = s.y.inverse_transform(p)
        rows.append({"model": MODEL, "split": SPLIT, "data": dname, "seed": seed,
                     "std_MSE": float(err.mean()),
                     "residual": float(res.mean()),
                     "spearman": float(spearmanr(res, err)[0]),
                     "auroc_top25": float(roc_auc_score(err >= np.quantile(err, 0.75), res)),
                     # non-physics reference detector: how far into keyhole the input is (energy density)
                     "auroc_top25_energy": float(roc_auc_score(err >= np.quantile(err, 0.75), ed[evdf.index])),
                     "range_%": float(((yv < lo) | (yv > hi)).any(1).mean() * 100),
                     "nonpos_%": float((yv <= 0).any(1).mean() * 100),
                     "monotone_viol_%": float(np.mean(mono) * 100)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"h2b_{MODEL}_{SPLIT}_{NSYN}{'_s%d' % SEED0 if SEED0 else ''}.csv", index=False)
print("done")
