"""H1/H2/H3 with 10-member ensembles (2k synthetic rows, seeds 800-829, three disjoint ensembles).
  train  : python exp_ens3.py train <n_procs>  -> ens_preds/<split>_<model>_<seed>.npz (std-space predictions)
           each model predicts: eval rows, eval rows with P x1.05 and v x1.05, and the 31x31 (P, v) grid of every eval row
  judge  : python exp_ens3.py judge             -> prints H1/H2/H3 tables (ensemble = mean of 10 members' predictions)
Metrics follow exp_h2b.py (H2) and exp_risk.py (H3)."""
import os, sys, json, itertools, pathlib
import numpy as np, torch
import exp_lt as E
from exp_risk import derive, cls, GP, GV, PR, VR

F = E.FULL
OUTD = E.OUT.parent / "ens_preds"; OUTD.mkdir(exist_ok=True)
SEEDS = range(800, 830); BLOCKS = (800, 810, 820); BS = 10
MLP = {"model": "MLP", "lr": 1e-3}
LT = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3, "arch": "orig", "loops": 10}
MODELS = {"MLP-A": {**MLP, "hidden": 128, "wd": 1e-2, "epochs": 300},     # best MLP ensemble in rounds 12 and 13
          "MLP-B": {**MLP, "hidden": 64, "wd": 1e-4, "epochs": 100},      # default MLP
          "LT-fb": {**LT, "phys": 1}, "LT-nofb": {**LT, "phys": 0}}
evrows = lambda s: F[E.ED > 4.4e6] if s == "energy" else F[F.aspect_ratio >= 0.5]


def grid_frame(ev):
    gp, gv = np.meshgrid(GP, GV); gp, gv = gp.ravel(), gv.ravel()
    c = ev.loc[ev.index.repeat(len(gp))].copy()
    c["P_laser"], c["v_scan"] = np.tile(gp, len(ev)), np.tile(gv, len(ev))
    return derive(c)


def train_one(job):
    split, mname, seed = job
    out = OUTD / f"{split}_{mname}_{seed}.npz"
    if out.exists(): return str(out)
    f, d = E.fit({"cfg": MODELS[mname], "split": split, "syn": 2000, "stage": "final", "seed": seed})
    s, ev = d.s, evrows(split)
    def pred(df):
        with torch.no_grad():
            return np.concatenate([f(torch.from_numpy(s.x.transform(df[s.features][i:i + 8192]).astype(np.float32))).numpy()
                                   for i in range(0, len(df), 8192)])
    pp = {}
    for c in ("P_laser", "v_scan"):
        q = ev.copy(); q[c] = q[c] * 1.05; pp[c] = pred(derive(q))
    np.savez_compressed(out, ev=pred(ev), P=pp["P_laser"], v=pp["v_scan"],
                        grid=pred(grid_frame(ev)).astype(np.float16).reshape(len(ev), -1, 3))
    return str(out)


def judge():
    from sklearn.metrics import f1_score, precision_score, recall_score
    gp, gv = np.meshgrid(GP, GV); gp, gv = gp.ravel(), gv.ravel()
    res = {}
    for split in ("energy", "dw"):
        d = E.make_data(split, "final", 0); s = d.s; ev = evrows(split); y = d.y_eval
        lo, hi = F[s.targets].min().to_numpy(), F[s.targets].max().to_numpy()
        evl = E.FluidThermalPhysicsEvaluator(s.x.mean_, s.x.scale_, s.y.mean_, s.y.scale_)
        xev = torch.from_numpy(d.x_eval)
        WD = lambda p: (lambda w, r, a: (w, w * r, a))(*s.y.inverse_transform(p.reshape(-1, 3)).T)
        yt = cls((ev.melt_depth / ev.melt_width).to_numpy())
        recs = {}
        for m in MODELS:
            L = {k: np.load(OUTD / f"{split}_{m}_{k}.npz") for k in SEEDS}
            single = np.mean([((L[k]["ev"] - y) ** 2).mean() for k in SEEDS])
            for b in BLOCKS:
                mem = range(b, b + BS)
                pe = np.mean([L[k]["ev"] for k in mem], 0)
                w, D, A = WD(pe)
                mono = []
                for c, sign in (("P", 1), ("v", -1)):
                    w1, D1, A1 = WD(np.mean([L[k][c] for k in mem], 0))
                    mono += [sign * (D1 - D) < 0, sign * (A1 - A) < 0]
                yv = s.y.inverse_transform(pe)
                yp = cls(D / w)
                g = np.mean([L[k]["grid"].astype(np.float32) for k in mem], 0)       # (n_eval, 961, 3)
                gw, gD, _ = WD(g); gw, gD = gw.reshape(len(ev), -1), gD.reshape(len(ev), -1)
                ok = (gD / gw < 1.0) & (gD >= ev.t_powder.to_numpy()[:, None])
                dist = np.sqrt(((gp[None] - ev.P_laser.to_numpy()[:, None]) / (PR[1] - PR[0])) ** 2 +
                               ((gv[None] - ev.v_scan.to_numpy()[:, None]) / (VR[1] - VR[0])) ** 2)
                dist = np.where(ok, dist, np.inf); flag = yp == 2
                found = np.isfinite(dist.min(1)) & flag
                recs[(m, b)] = {i: int(a) for i, a, fo in zip(range(len(ev)), dist.argmin(1), found) if fo}
                with torch.no_grad():
                    resid = evl.evaluate(xev, torch.from_numpy(pe.astype(np.float32))).residual[:, :3].abs().mean().item()
                res[(split, m, b)] = {
                    "std_MSE": float(((pe - y) ** 2).mean()), "single_mean": float(single),
                    "monotone_viol_%": float(np.mean(mono) * 100), "range_%": float(((yv < lo) | (yv > hi)).any(1).mean() * 100),
                    "residual": resid,
                    "macro_f1": float(f1_score(yt, yp, average="macro")), "keyhole_precision": float(precision_score(yt == 2, flag, zero_division=0)),
                    "keyhole_recall": float(recall_score(yt == 2, flag)), "n_flagged": int(flag.sum()),
                    "found_rate": float(found.sum() / max(flag.sum(), 1)), "mean_change": float(dist.min(1)[found].mean()) if found.any() else None}
        for m in ("LT-fb", "LT-nofb"):
            for b in BLOCKS:
                for mm in ("MLP-A", "MLP-B"):
                    a, c = recs[(m, b)], recs[(mm, b)]; com = set(a) & set(c)
                    res[(split, m, b)][f"same_rec_vs_{mm}_%"] = float(np.mean([a[i] == c[i] for i in com]) * 100) if com else None
    import pandas as pd
    df = pd.DataFrame([{"split": k[0], "model": k[1], "block": k[2], **v} for k, v in res.items()])
    df.to_csv(E.OUT / "ens3_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["split", "model"]).mean(numeric_only=True).drop(columns="block").round(3).to_string())
    for split in ("energy", "dw"):
        q = df[df.split == split].set_index(["model", "block"]).std_MSE.unstack("block")
        best = q.loc[["MLP-A", "MLP-B"]].mean(1).idxmin()
        for m in ("LT-fb", "LT-nofb"):
            print(f"{split}: {m} {q.loc[m].mean():.3f} vs {best} {q.loc[best].mean():.3f}  blocks won {(q.loc[m] < q.loc[best]).sum()}/3")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        judge(); sys.exit()
    import multiprocessing as mp
    todo = [(s, m, k) for s, m, k in itertools.product(("energy", "dw"), MODELS, SEEDS)]
    if os.environ.get("SMOKE"):
        E.ROUND = 0; MODELS = {k: {**v, "epochs": 1} for k, v in MODELS.items()}
        print(train_one(("dw", "LT-fb", 800)), train_one(("dw", "MLP-A", 800))); sys.exit()
    todo.sort(key=lambda j: j[1].startswith("MLP"))       # LTs first
    with mp.Pool(int(sys.argv[2])) as pool:
        for i, r in enumerate(pool.imap_unordered(train_one, todo), 1):
            if i % 10 == 0: print(i, r, flush=True)
    print("done")
