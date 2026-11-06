"""Exp 4: does the model choice change the agent's decisions on the unseen region?
1. risk assessment, scored against measurements: class by D/W (<0.5 conduction, 0.5-1 transition, >=1 keyhole)
   from predicted vs measured W, D on the eval rows.
2. recommendation: for every eval row the model calls keyhole, 31x31 grid over (P, v) within the measured
   ranges (d, t fixed); feasible = predicted D/W < 1 and predicted D >= t; pick the smallest normalised change.
   No measured truth exists for new conditions, so this part is reported as found-rate / change size /
   agreement between models, not as accuracy.
Usage: python exp_risk.py <n_procs>   -> exp_results/risk.jsonl
"""
import os, sys, json, itertools, pathlib
import numpy as np, torch
import exp_lt as E

SET2 = bool(os.environ.get("RISK2"))     # paper confirmation: fresh seeds, feedback-off LT added
LOG = E.OUT / ("risk2.jsonl" if SET2 else "risk.jsonl")
F = E.FULL
PR, VR = (F.P_laser.min(), F.P_laser.max()), (F.v_scan.min(), F.v_scan.max())
GP, GV = np.linspace(*PR, 31), np.linspace(*VR, 31)
cls = lambda r: np.digitize(r, [0.5, 1.0])          # 0 conduction, 1 transition, 2 keyhole


def derive(df):
    d = df.copy(); v = d.v_scan / 1000.0; dm = d.d_laser / 1000.0; a, hs = 5.0e-6, 7800.0 * 460.0 * (1678.0 - 293.15)
    d["linear_energy"] = d.P_laser / d.v_scan; d["peclet_number"] = v * dm / a
    d["normalized_enthalpy"] = 0.34 * d.P_laser / (hs * np.sqrt(a * dm ** 3 * v)); return d


def predict(f, s, df):
    with torch.no_grad():
        p = np.concatenate([f(torch.from_numpy(s.x.transform(df[s.features][i:i + 4096]).astype(np.float32))).numpy()
                            for i in range(0, len(df), 4096)])
    W, R, A = s.y.inverse_transform(p).T
    return W, W * R


def run(job):
    f, d = E.fit({**job, "stage": "final"})
    s = d.s
    ev = F[E.ED > 4.4e6] if job["split"] == "energy" else F[F.aspect_ratio >= 0.5]
    W, D = predict(f, s, ev)
    yt, yp = cls((ev.melt_depth / ev.melt_width).to_numpy()), cls(D / W)
    from sklearn.metrics import f1_score, recall_score, precision_score
    out = {"acc": float((yt == yp).mean()), "macro_f1": float(f1_score(yt, yp, average="macro")),
           "keyhole_recall": float(recall_score(yt == 2, yp == 2)), "keyhole_precision": float(precision_score(yt == 2, yp == 2, zero_division=0))}
    # agent: rows called keyhole -> nearest feasible grid point
    rows = ev[yp == 2]
    if not len(rows):
        return {**job, **out, "n_flagged": 0, "found_rate": None, "mean_change": None, "rec": {}}
    gp, gv = np.meshgrid(GP, GV); gp, gv = gp.ravel(), gv.ravel()
    cand = rows.loc[rows.index.repeat(len(gp))].copy()
    cand["P_laser"], cand["v_scan"] = np.tile(gp, len(rows)), np.tile(gv, len(rows))
    cW, cD = predict(f, s, derive(cand))
    ok = ((cD / cW < 1.0) & (cD >= cand.t_powder.to_numpy())).reshape(len(rows), -1)
    dist = np.sqrt(((gp[None] - rows.P_laser.to_numpy()[:, None]) / (PR[1] - PR[0])) ** 2 +
                   ((gv[None] - rows.v_scan.to_numpy()[:, None]) / (VR[1] - VR[0])) ** 2)
    dist = np.where(ok, dist, np.inf); best = dist.argmin(1); found = np.isfinite(dist.min(1))
    rec = {int(i): [float(gp[b]), float(gv[b])] for i, b, fo in zip(rows.index, best, found) if fo}
    out |= {"n_flagged": int(len(rows)), "found_rate": float(found.mean()) if len(rows) else None,
            "mean_change": float(dist.min(1)[found].mean()) if found.any() else None, "rec": rec}
    return {**job, **out}


def jobs():
    mlp = {"model": "MLP", "hidden": 64, "wd": 1e-4, "epochs": 100, "lr": 1e-3}
    lt = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3}
    cfgs = [("MLP", mlp), ("LT orig fb", {**lt, "arch": "orig", "loops": 10, "phys": 1}),
            ("LT mlpres fb", {**lt, "arch": "mlpres", "loops": 1, "phys": 1})]
    if SET2:
        cfgs = cfgs[:2] + [("LT orig nofb", {**lt, "arch": "orig", "loops": 10, "phys": 0})]
    for (name, cfg), split, syn, seed in itertools.product(cfgs, ("energy", "dw"), (200, 2000), range(500, 510) if SET2 else range(400, 405)):
        yield {"name": name, "cfg": cfg, "split": split, "syn": syn, "seed": seed}


if __name__ == "__main__":
    import multiprocessing as mp
    todo = list(jobs())
    if os.environ.get("SMOKE"):
        todo = [{**j, "cfg": {**j["cfg"], "epochs": 1}} for j in todo[:1] + todo[25:26]]
        for r in map(run, todo): print({k: v for k, v in r.items() if k != "rec"})
        sys.exit()
    todo.sort(key=lambda j: -(j["cfg"].get("loops", 0) + 1) * j["syn"])
    with mp.Pool(int(sys.argv[1])) as pool, LOG.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(i, r["name"], r["split"], r["syn"], round(r["macro_f1"], 3), r["found_rate"], flush=True)
    print("done")
