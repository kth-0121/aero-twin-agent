"""Auxiliary recommendation experiment, re-run with the paper's models (replaces the v5-checkpoint numbers):
60 scenarios = 20 measured process conditions from each energy bin (ED = P / (v d): under < 1.5e6, normal 1.5e6..4.4e6,
over > 4.4e6 = the paper's keyhole threshold), unique (P, v, d, t) only, drawn with seed 0.
For every scenario the agent's grid search (feasible = predicted D/W < 1 and predicted D >= t, smallest normalised change)
is run at 15x15, 31x31 and 61x61 grids with the LT-fb ensembles (2k synthetic, seeds 800-829, three 10-seed blocks).
  train : python exp_scen.py train <n_procs>  -> scen_preds/<split>_<seed>.npz
  judge : python exp_scen.py judge            -> exp_results/scen_summary.csv (+ scen_stability.csv)"""
import os, sys, itertools
import numpy as np, pandas as pd, torch
import exp_lt as E
from exp_risk import derive, cls, PR, VR
from exp_ens3 import MODELS, SEEDS, BLOCKS, BS

F = E.FULL
OUTD = E.OUT.parent / "scen_preds"; OUTD.mkdir(exist_ok=True)
RES = (15, 31, 61)
BINS = ("under", "normal", "over")
BIN = pd.Series(np.digitize(E.ED, [1.5e6, 4.4e6]), index=F.index)


def scenarios():
    u = F.drop_duplicates(["P_laser", "v_scan", "d_laser", "t_powder"])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(u.index[BIN.loc[u.index] == b], 20, replace=False) for b in range(3)])
    return F.loc[idx]


def grid(n):
    gp, gv = np.meshgrid(np.linspace(*PR, n), np.linspace(*VR, n)); return gp.ravel(), gv.ravel()


def grid_frame(sc, n):
    gp, gv = grid(n)
    c = sc.loc[sc.index.repeat(len(gp))].copy()
    c["P_laser"], c["v_scan"] = np.tile(gp, len(sc)), np.tile(gv, len(sc))
    return derive(c)


def train_one(job):
    split, seed = job
    out = OUTD / f"{split}_{seed}.npz"
    if out.exists(): return str(out)
    f, d = E.fit({"cfg": MODELS["LT-fb"], "split": split, "syn": 2000, "stage": "final", "seed": seed})
    s, sc = d.s, scenarios()
    def pred(df):
        with torch.no_grad():
            return np.concatenate([f(torch.from_numpy(s.x.transform(df[s.features][i:i + 8192]).astype(np.float32))).numpy()
                                   for i in range(0, len(df), 8192)])
    np.savez_compressed(out, sc=pred(derive(sc)), **{f"g{n}": pred(grid_frame(sc, n)).reshape(len(sc), -1, 3) for n in RES})
    return str(out)


def judge():
    sc = scenarios(); b = BIN.loc[sc.index].to_numpy(); t = sc.t_powder.to_numpy()
    rows, stab = [], []
    for split in ("energy", "dw"):
        s = E.make_data(split, "final", 0).s
        WD = lambda p: (lambda w, r, a: (w, w * r))(*s.y.inverse_transform(p.reshape(-1, 3)).T)
        for blk in BLOCKS:
            L = [np.load(OUTD / f"{split}_{k}.npz") for k in range(blk, blk + BS)]
            w0, D0 = WD(np.mean([l["sc"] for l in L], 0)); flag = cls(D0 / w0) == 2
            rec = {}
            for n in RES:
                gp, gv = grid(n)
                gw, gD = WD(np.mean([l[f"g{n}"] for l in L], 0)); gw, gD = gw.reshape(len(sc), -1), gD.reshape(len(sc), -1)
                ok = (gD / gw < 1.0) & (gD >= t[:, None])
                dist = np.sqrt(((gp[None] - sc.P_laser.to_numpy()[:, None]) / (PR[1] - PR[0])) ** 2 +
                               ((gv[None] - sc.v_scan.to_numpy()[:, None]) / (VR[1] - VR[0])) ** 2)
                dist = np.where(ok, dist, np.inf); best = dist.argmin(1); chg = dist.min(1); found = np.isfinite(chg)
                rec[n] = (gp[best], gv[best], chg, found)
                for bi, name in enumerate(BINS):
                    m = b == bi
                    rows.append({"split": split, "block": blk, "grid": n, "bin": name, "n": int(m.sum()), "found": int(found[m].sum()),
                                 "flagged_keyhole": int(flag[m].sum()), "mean_change": float(chg[m & found].mean()) if (m & found).any() else None})
            for a, c in ((15, 31), (31, 61)):
                pa, va, ca, fa = rec[a]; pc, vc, cc, fc = rec[c]; both = fa & fc
                stab.append({"split": split, "block": blk, "from": a, "to": c, "same_rec_%": float(np.mean(((np.abs(pa - pc) < 1e-6) & (np.abs(va - vc) < 1e-6))[both]) * 100) if both.any() else None,
                             "mean_change_from": float(ca[both].mean()), "mean_change_to": float(cc[both].mean())})
    df, ds = pd.DataFrame(rows), pd.DataFrame(stab)
    df.to_csv(E.OUT / "scen_summary.csv", index=False); ds.to_csv(E.OUT / "scen_stability.csv", index=False)
    sc.assign(bin=[BINS[i] for i in b]).to_csv(E.OUT / "scen_rows.csv")
    pd.set_option("display.width", 250)
    print(df.groupby(["split", "grid", "bin"]).agg(n=("n", "first"), found=("found", "mean"), flagged=("flagged_keyhole", "mean"),
                                                   change=("mean_change", "mean"), change_min=("mean_change", "min"), change_max=("mean_change", "max")).round(4).to_string())
    print(ds.groupby(["split", "from", "to"]).mean(numeric_only=True).drop(columns="block").round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        judge(); sys.exit()
    import multiprocessing as mp
    todo = list(itertools.product(("dw", "energy"), SEEDS))
    if os.environ.get("SMOKE"):
        MODELS["LT-fb"] = {**MODELS["LT-fb"], "epochs": 1}; OUTD = E.OUT.parent / "scen_smoke"; OUTD.mkdir(exist_ok=True)
        print(train_one(("dw", 800))); sys.exit()
    with mp.Pool(int(sys.argv[2])) as pool:
        for i, r in enumerate(pool.imap_unordered(train_one, todo), 1):
            if i % 10 == 0: print(i, r, flush=True)
    print("done")
