"""Can an LT beat MLP on the unseen keyhole set?  Fair protocol:
  - configs are selected on an INNER validation set (top-20 % energy density rows of the training set),
    never on the keyhole eval set; MLP gets the same tuning budget.
  - every config is also trained on the full training set (5 seeds) and scored on keyhole, so the
    selected config's keyhole score can be read off afterwards.
Hypothesis: LayerNorm in the LT output head bounds the output scale, so LT cannot extrapolate.
  arch: orig | lin (pred = head(h) + Linear(x)) | noln (head without LayerNorm) | linnoln
Usage: python exp_lt.py <n_procs>      (results appended to exp_results/lt_jobs.jsonl, resumable)
"""
import sys, pathlib, json, itertools, os, functools, random
R = pathlib.Path(os.environ.get("REPO", pathlib.Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(R / "model" / "Baseline"), str(R / "model" / "LoopedTransformer" / "legacy"), str(R / "model" / "LoopedTransformer")]
import numpy as np, torch
from torch import nn
from types import SimpleNamespace
import common, dataLT
from common import set_seed
from modelLT import LoopedTransformer, StateMLP, LoopOutput
from physicsLT import FluidThermalPhysicsEvaluator
from lossLT import PhysicsConsistencyLoss, DeepSupervisionLoss
from train_v9_synthetic_augmentation import generate_synthetic_keyhole_rows
from torch.utils.data import DataLoader, TensorDataset

OUT = pathlib.Path(__file__).parent / "exp_results"; OUT.mkdir(exist_ok=True)
ROUND = int(os.environ.get("ROUND", "2" if os.environ.get("ROUND2") else "1"))
LOG = OUT / ("lt_jobs.jsonl" if ROUND == 1 else f"lt_jobs{ROUND}.jsonl")
FULL = dataLT.load_316l_csv(common.DATA_PATH)
ED = FULL.P_laser / (FULL.v_scan / 1000 * FULL.d_laser / 1000)
SYN = generate_synthetic_keyhole_rows(200, seed=42)
synth = functools.lru_cache(None)(lambda n: generate_synthetic_keyhole_rows(n, seed=42))
_RHO, _CP, _ALPHA, _TM, _T0, _AB = 7800.0, 460.0, 5.0e-6, 1678.0, 293.15, 0.34


def _phys(P, v_mm_s, d_mm):
    v, r0, dT = v_mm_s / 1000, d_mm / 2000, _TM - _T0
    w = np.sqrt((8 / (np.pi * np.e)) * _AB * P / (_RHO * _CP * v * dT)) * 1e6
    Ke = _AB * P / (dT * np.pi * _RHO * _CP * np.sqrt(_ALPHA * v * r0 ** 3))
    return w, Ke, r0 * 1e6


def calib(tr):
    """fit the generator's width scale, e* = a (Ke - b) and the area factor on REAL training rows only"""
    from scipy.optimize import curve_fit
    w, Ke, r0 = _phys(tr.P_laser.to_numpy(), tr.v_scan.to_numpy(), tr.d_laser.to_numpy())
    W, D, A = tr.melt_width.to_numpy(), tr.melt_depth.to_numpy(), tr.melt_area.to_numpy()
    (a, b), _ = curve_fit(lambda k, a, b: a * np.clip(k[0] - b, 0, None) * k[1], (Ke, r0), D, p0=(0.4, 1.4))
    return float(np.median(W / w)), a, b, float(np.median(A / (W * D)))


class Anchor(nn.Module):
    """calibrated Rosenthal width + Gan depth as a fixed prior, in standardized target space (no trainable params)"""
    def __init__(self, s, cal):
        super().__init__()
        self.cal = cal
        for k, v in (("xm", s.x.mean_), ("xs", s.x.scale_), ("ym", s.y.mean_), ("ys", s.y.scale_)):
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))

    def forward(self, x):
        kw, a, b, ka = self.cal
        z = x * self.xs + self.xm
        P, v, dm = z[:, 0].clamp(min=1.0), z[:, 1].clamp(min=1.0) / 1000, z[:, 2].clamp(min=1e-3) / 1000
        r0, dT = dm / 2, _TM - _T0
        W = kw * torch.sqrt((8 / (np.pi * np.e)) * _AB * P / (_RHO * _CP * v * dT)) * 1e6
        Ke = _AB * P / (dT * np.pi * _RHO * _CP * torch.sqrt(_ALPHA * v * r0 ** 3))
        D = (a * (Ke - b).clamp(min=0) * r0 * 1e6).clamp(min=1.0)
        return (torch.stack([W, D / W, ka * W * D], 1) - self.ym) / self.ys


class AnchorEval:
    """physics feedback = gap between the loop's prediction and the train-calibrated physics anchor"""
    def __init__(self, anc): self.anc = anc

    def evaluate(self, x, p):
        from physicsLT import PhysicsFeedback
        return PhysicsFeedback(residual=p - self.anc(x))


@functools.lru_cache(None)
def synth_cal(n, cal, seed=42, noise=0.08):
    """same sampling as generate_synthetic_keyhole_rows, labels from the calibrated relations"""
    kw, a, b, ka = cal
    rng = np.random.default_rng(seed)
    P, v, d = rng.uniform(50, 500, n), rng.uniform(225, 1500, n), rng.uniform(0.05, 0.14, n)
    t = rng.choice([0.0, 30.0, 60.0], n)
    w, Ke, r0 = _phys(P, v, d); m = Ke > 1.4
    P, v, d, t, w, Ke, r0 = P[m], v[m], d[m], t[m], w[m], Ke[m], r0[m]
    W = kw * w * rng.normal(1, noise, m.sum())
    D = np.clip(a * np.clip(Ke - b, 0, None) * r0 * rng.normal(1, noise, m.sum()), 1.0, None)
    import pandas as pd
    hs = _RHO * _CP * (_TM - _T0)
    return pd.DataFrame({"P_laser": P, "v_scan": v, "d_laser": d, "t_powder": t, "linear_energy": P / v,
                         "peclet_number": v / 1000 * d / 1000 / _ALPHA,
                         "normalized_enthalpy": _AB * P / (hs * np.sqrt(_ALPHA * (d / 1000) ** 3 * v / 1000)),
                         "melt_width": W, "aspect_ratio": D / W, "melt_area": ka * W * D})


def make_data(split, stage, syn, unl=0):
    """stage 'inner': train on train-rows minus their top-20 % energy density, score on that 20 %.
       stage 'final': train on all train rows, score on the keyhole eval set."""
    ev_mask = ED > 4.4e6 if split == "energy" else FULL.aspect_ratio >= 0.5
    tr, ev = FULL[~ev_mask], FULL[ev_mask]
    s = dataLT.fit_scalers(tr)                       # scalers always from the real outer training rows
    if stage == "inner":
        cut = ED[tr.index].quantile(0.8)
        tr, ev = tr[ED[tr.index] <= cut], tr[ED[tr.index] > cut]
    X = lambda d: s.x.transform(d[s.features]).astype(np.float32)
    Y = lambda d: s.y.transform(d[s.targets]).astype(np.float32)
    xt, yt = X(tr), Y(tr)
    if syn:                                          # 1 = the usual 200 requested rows, else the requested count
        if isinstance(syn, str) and syn.startswith("gap"):   # "gap5000": raw generator rows only beyond the training rows' max energy density
            rows = synth(int(syn[3:])); red = rows.P_laser / (rows.v_scan / 1000 * rows.d_laser / 1000)
            rows = rows[red > ED[tr.index].max()]
        elif isinstance(syn, str) and syn.startswith("cal"):   # "cal2000": generator calibrated on the real training rows
            rows = synth_cal(int(syn[3:]), calib(tr))
        elif isinstance(syn, str):                     # "200x10": 200 distinct rows repeated 10 times
            n, k = map(int, syn.split("x")); rows = synth(n).loc[np.repeat(synth(n).index, k)]
        else:
            rows = SYN if syn == 1 else synth(syn)
        xt, yt = np.vstack([xt, X(rows)]), np.vstack([yt, Y(rows)])
    xu = None
    if unl:                                          # unlabeled process conditions beyond the training energy range
        rows = synth(unl * 8); red = rows.P_laser / (rows.v_scan / 1000 * rows.d_laser / 1000)
        xu = X(rows[red > 4.4e6].head(unl))
    return SimpleNamespace(x_train=xt, y_train=yt, x_eval=X(ev), y_eval=Y(ev), s=s, tr_real=FULL[~ev_mask], x_unl=xu)


class EvalOnly:
    def __init__(self, ev): self.evaluate = ev.evaluate


class LTVar(LoopedTransformer):
    """Same LT; optional linear skip from inputs to every loop's prediction, optional LN-free head."""
    def __init__(self, lin, noln, base=None, **kw):
        super().__init__(**kw)
        self.lin = nn.Linear(7, 3) if lin else None
        if noln:
            self.state_head = nn.Linear(self.d_model, self.state_dim)
        self.step = 0.25
        self.base = base            # frozen pre-trained MLP: LT only learns a correction on top of it
        if base is not None:
            for q in base.parameters(): q.requires_grad_(False)
            last = self.state_head if isinstance(self.state_head, nn.Linear) else self.state_head[-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, x, physics_evaluator=None, loop_steps=None):
        steps = loop_steps or self.loop_steps
        proc = self.tokenizer(x); state = self.state_token.expand(x.size(0), -1, -1)
        preds, fbs, confs = [], [], []
        for i in range(steps):
            h = self.transformer(torch.cat([proc, state], 1))[:, -1, :]
            p = self.state_head(h) + (self.lin(x) if self.lin is not None else 0) + (self.base(x) if self.base is not None else 0)
            preds.append(p)
            if physics_evaluator is None:
                fb, cf = torch.zeros_like(p), None
            else:
                r = (physics_evaluator.evaluate_loop_wise(x, p, loop_idx=i) if hasattr(physics_evaluator, "evaluate_loop_wise")
                     else physics_evaluator.evaluate(x, p))
                fb, cf = r.residual, r.confidence
            fbs.append(fb); confs.append(cf)
            state = state + self.step * self.state_update(torch.cat([h, self.feedback_encoder(fb)], -1)).unsqueeze(1)
        return LoopOutput(preds, fbs, confs)


class LoopedMLP(nn.Module):
    """attention-free looped refiner: y <- y + alpha * f([x, y, physics residual]), f weight-tied (or one f per loop), y0 = anchor(x)"""
    def __init__(self, hidden, loops, alpha, fb_dim, tied=True, sn=False, base=None):
        super().__init__()
        def mk():
            l1, l2 = nn.Linear(7 + 3 + fb_dim, hidden), nn.Linear(hidden, 3)
            nn.init.zeros_(l2.bias)
            if sn:                      # spectral norm can't normalise an all-zero weight: start small instead
                nn.init.normal_(l2.weight, std=1e-3); l1, l2 = (nn.utils.parametrizations.spectral_norm(l) for l in (l1, l2))
            else:
                nn.init.zeros_(l2.weight)
            return nn.Sequential(l1, nn.GELU(), l2)
        self.f = nn.ModuleList([mk() for _ in range(1 if tied else loops)])
        self.alpha, self.fb_dim, self.base, self.loop_steps = alpha, fb_dim, base, loops

    def forward(self, x, physics_evaluator=None, loop_steps=None):
        y = self.base(x) if self.base is not None else torch.zeros(x.size(0), 3)
        preds, fbs, confs = [], [], []
        for i in range(loop_steps or self.loop_steps):
            if physics_evaluator is None:
                fb, cf = torch.zeros(x.size(0), self.fb_dim), None
            else:
                r = physics_evaluator.evaluate(x, y); fb, cf = r.residual, r.confidence
            y = y + self.alpha * self.f[min(i, len(self.f) - 1)](torch.cat([x, y, fb], -1))
            preds.append(y); fbs.append(fb); confs.append(cf)
        return LoopOutput(preds, fbs, confs)


def train(model, d, fwd, lr, wd, epochs):
    if epochs == "match":
        epochs = max(1, round(100 * 550 / len(d.x_train)))
    dl = DataLoader(TensorDataset(torch.from_numpy(d.x_train), torch.from_numpy(d.y_train)), batch_size=64, shuffle=True)
    lf, opt = DeepSupervisionLoss(gamma=0.8), torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xu = None if getattr(d, "x_unl", None) is None else torch.from_numpy(d.x_unl[np.random.randint(0, len(d.x_unl), len(xb))])
            opt.zero_grad(); p, extra = fwd(model, xb, xu); (lf(p, yb) + extra).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()


def run(job):
    f, d = fit(job)
    with torch.no_grad():
        p = f(torch.from_numpy(d.x_eval)).numpy()
    if ROUND in (7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        job = {**job, "pred": p.round(5).tolist()}   # kept for ensembling
    if ROUND in (20, 21, 22, 23, 24, 25, 26, 27, 28) and hasattr(f, "at"):         # exploratory: eval at other loop counts
        with torch.no_grad():
            job["pred_at"] = {L: f.at(torch.from_numpy(d.x_eval), L).numpy().round(5).tolist() for L in ((1, 2, 3, 5, 10, 20, 40) if ROUND == 20 else (10, 20, 40, 80, 160, 320, 640) if ROUND == 21 else (10, 40, 80, 160) if ROUND == 22 else (10, 20, 40, 80, 160, 320) if ROUND == 24 else (10, 20, 40, 80) if ROUND == 25 else (20, 40, 80, 160) if ROUND == 26 else (1, 3, 10, 20, 40, 80) if ROUND == 27 else (3, 10, 20, 40))}
    return {**job, "std_MSE": float(((p - d.y_eval) ** 2).mean()),
            "per_target": ((p - d.y_eval) ** 2).mean(0).round(4).tolist()}


def fit(job):
    """train one model for a job; returns (predict function on std inputs, data)"""
    torch.set_num_threads(1)
    c = job["cfg"]; d = make_data(job["split"], job["stage"], job["syn"], c.get("unl", 0)); set_seed(job["seed"]); random.seed(job["seed"])
    anc = Anchor(d.s, calib(d.tr_real)) if c.get("anchor") else None
    if c["model"] == "MLP":
        m = StateMLP(n_features=7, state_dim=3, hidden=c["hidden"])
        g = (lambda x: m(x) + anc(x)) if anc is not None else (lambda x: m(x))
        pe = FluidThermalPhysicsEvaluator(d.s.x.mean_, d.s.x.scale_, d.s.y.mean_, d.s.y.scale_)
        def fwd(m, x, xu=None):                     # control: same physics-consistency loss on the same unlabeled inputs
            extra = c["uphys"] * (pe.evaluate(xu, g(xu)).residual ** 2).mean() if c.get("uphys") and xu is not None else 0.0
            return [g(x)], extra
        train(m, d, fwd, c["lr"], c["wd"], c["epochs"])
        f = g
    else:
        ev = None
        if c["phys"] == 2:
            ev = AnchorEval(anc)
        elif c["phys"]:
            e = FluidThermalPhysicsEvaluator(d.s.x.mean_, d.s.x.scale_, d.s.y.mean_, d.s.y.scale_)
            ev = EvalOnly(e) if c["loops"] == 1 else e
        base = anc
        if c["arch"] == "mlpres":
            base = StateMLP(n_features=7, state_dim=3, hidden=c.get("base_h", 64))
            if anc is not None:             # anchored MLP base: anc(x) + mlp(x)
                base = AncPlus(anc, base)
                train(base, d, lambda m, x, xu=None: ([m(x)], 0.0), 1e-3, c.get("base_wd", 1e-4), c.get("base_ep", 100))
            else:
                train(base, d, lambda m, x, xu=None: ([m(x)], 0.0), 1e-3, 1e-4, c.get("base_ep", 100))
            set_seed(job["seed"])
            if c.get("corr_real"):          # base saw the synthetic rows; the correction only sees real ones
                d = SimpleNamespace(**{**vars(d), **{k: getattr(make_data(job["split"], job["stage"], 0), k) for k in ("x_train", "y_train")}})
        if c["model"] == "LMLP":
            m = LoopedMLP(c["hidden"], c["loops"], c.get("alpha", 0.25), 7 if c["phys"] == 1 else 3, tied=not c.get("untied"), sn=c.get("sn", False), base=base)
        else:
            m = LTVar(c["arch"] in ("lin", "linnoln"), c["arch"] in ("noln", "linnoln", "mlpres"), base=base,
                      n_features=7, feedback_dim=7 if c["phys"] == 1 else 3, loop_steps=c["loops"], d_model=c.get("dm", 64))
        pl = PhysicsConsistencyLoss()
        contr = lambda ps: sum(((ps[i + 1] - ps[i]) ** 2).mean() for i in range(len(ps) - 1)) / max(1, len(ps) - 1)
        def fwd(m, x, xu=None):
            L = random.randint(1, c["loops"]) if c.get("rloop") else None   # random loop count per batch
            o = m(x, physics_evaluator=ev, loop_steps=L)
            extra0 = 0.0
            if xu is not None and (c.get("uconv") or c.get("uphys")):   # unlabeled eval-region inputs: loops must settle / agree with physics
                ou = m(xu, physics_evaluator=ev, loop_steps=c.get("uloops", c["loops"]))
                extra0 = c.get("uconv", 0) * contr(ou.predictions) + (c.get("uphys", 0) * pl(ou.feedbacks, ou.confidences) if c["phys"] else 0)
            extra = extra0 + (c.get("plw", 0.1) * pl(o.feedbacks, o.confidences) if c["phys"] else 0.0)
            if c.get("lam"):                # shrink the correction towards the MLP base
                b = m.base(x); extra = extra + c["lam"] * sum(((p - b) ** 2).mean() for p in o.predictions) / len(o.predictions)
            if c.get("conv"):               # contraction: successive loop predictions should settle
                extra = extra + c["conv"] * contr(o.predictions)
            return o.predictions, extra
        m.step = c.get("step", 0.25)
        train(m, d, fwd, c["lr"], c["wd"], c["epochs"])
        f = lambda x: m(x, physics_evaluator=ev, loop_steps=c["loops"]).predictions[-1]
        f.at = lambda x, L: m(x, physics_evaluator=ev, loop_steps=L).predictions[-1]
    return f, d


class AncPlus(nn.Module):
    def __init__(self, anc, m):
        super().__init__(); self.anc, self.m = anc, m

    def forward(self, x):
        return self.anc(x) + self.m(x)


def configs():
    for h, wd, ep in itertools.product((64, 128), (1e-4, 1e-2), (100, 300)):
        yield {"model": "MLP", "hidden": h, "wd": wd, "epochs": ep, "lr": 1e-3}
    for arch, loops, phys in itertools.product(("orig", "lin", "noln", "linnoln"), (1, 10), (0, 1)):
        yield {"model": "LT", "arch": arch, "loops": loops, "phys": phys, "wd": 1e-4, "epochs": 100, "lr": 1e-3}


def configs2():
    for loops, phys, lr in itertools.product((1, 10), (0, 1), (1e-3, 3e-4)):
        yield {"model": "LT", "arch": "mlpres", "loops": loops, "phys": phys, "wd": 1e-4, "epochs": 100, "lr": lr}


def configs3():
    """tune the extrapolating LT (linear skip + LN-free head), feedback on/off kept paired"""
    for loops, phys, lr, ep in itertools.product((3, 10), (0, 1), (1e-3, 3e-4), (100, 300)):
        yield {"model": "LT", "arch": "linnoln", "loops": loops, "phys": phys, "wd": 1e-4, "epochs": ep, "lr": lr}


def configs4():
    """physics-loss weight and width for the two LT forms that extrapolate (feedback on)"""
    for arch, plw, loops, dm in itertools.product(("linnoln", "mlpres"), (0.1, 0.5, 1.0), (1, 10), (32, 64)):
        yield {"model": "LT", "arch": arch, "loops": loops, "phys": 1, "wd": 1e-4, "epochs": 100, "lr": 1e-3, "plw": plw, "dm": dm}


def configs5():
    """MLP base + physics-feedback correction: base tuning, correction data, correction shrinkage"""
    for be, cr, lam, loops in itertools.product((100, 300), (0, 1), (0.0, 0.1, 1.0), (1, 10)):
        yield {"model": "LT", "arch": "mlpres", "loops": loops, "phys": 1, "wd": 1e-4, "epochs": 100, "lr": 1e-3,
               "base_ep": be, "corr_real": cr, "lam": lam}


def configs6():
    """confirmation: the top-ranked LT config of rounds 1-5 (+ its feedback-off twin) vs every MLP config, fresh seeds"""
    yield from (c for c in configs() if c["model"] == "MLP")
    for phys in (1, 0):
        yield {"model": "LT", "arch": "mlpres", "loops": 1, "phys": phys, "wd": 1e-4, "epochs": 100, "lr": 1e-3}


def configs7():
    """ensembles: candidates fixed from rounds 1-6, no new search"""
    yield from (c for c in configs() if c["model"] == "MLP")
    lt = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3}
    yield {**lt, "arch": "orig", "loops": 10, "phys": 1}
    for phys in (1, 0):
        yield {**lt, "arch": "linnoln", "loops": 10, "phys": phys}
        yield {**lt, "arch": "mlpres", "loops": 1, "phys": phys}


def configs8():
    """more synthetic rows: same fixed candidates as round 7"""
    yield from configs7()


def configs9():
    """synthetic scale-up check: fixed candidates, fresh seeds, plus step-matched MLPs"""
    mlp = {"model": "MLP", "lr": 1e-3}
    for h, wd, ep in ((64, 1e-4, 100), (64, 1e-2, 100), (128, 1e-2, 300), (64, 1e-4, "match"), (128, 1e-2, "match")):
        yield {**mlp, "hidden": h, "wd": wd, "epochs": ep}
    lt = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3}
    yield {**lt, "arch": "orig", "loops": 10, "phys": 1}
    yield {**lt, "arch": "orig", "loops": 10, "phys": 0}
    yield {**lt, "arch": "linnoln", "loops": 10, "phys": 1}
    yield {**lt, "arch": "mlpres", "loops": 1, "phys": 1}


def configs11():
    """paper confirmation (H1), fixed in advance: 5 MLPs vs LT orig with/without feedback"""
    yield from (c for c in configs9() if c["model"] == "MLP")
    lt = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3, "arch": "orig", "loops": 10}
    yield {**lt, "phys": 1}
    yield {**lt, "phys": 0}


def configs14():
    """energy-split push: calibrated synthetic labels; MLP gets the same data and tuning budget"""
    yield from (c for c in configs9() if c["model"] == "MLP" and c["epochs"] != "match")
    lt = {"model": "LT", "wd": 1e-4, "epochs": 100, "lr": 1e-3, "loops": 10}
    yield {**lt, "arch": "orig", "phys": 1}
    yield {**lt, "arch": "orig", "phys": 0}
    yield {**lt, "arch": "linnoln", "phys": 1}


def configs15():
    """physics anchor (train-calibrated Rosenthal/Gan) + learned correction, same anchor for MLP and LT"""
    for c in configs14():
        yield {**c, "anchor": 1}


def configs17():
    """energy split, real rows + physics anchor: LT tuning grid (incl. anchor feedback) vs a wider MLP grid"""
    mlp = {"model": "MLP", "lr": 1e-3, "anchor": 1}
    for h, wd, ep in ((64, 1e-2, 100), (64, 1e-4, 300), (64, 1e-2, 300), (128, 1e-4, 100), (128, 1e-2, 300), (32, 1e-2, 100)):
        yield {**mlp, "hidden": h, "wd": wd, "epochs": ep}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "loops": 10, "wd": 1e-4, "epochs": 100}
    yield {**lt, "arch": "linnoln", "phys": 1}
    yield {**lt, "arch": "linnoln", "phys": 1, "epochs": 300}
    yield {**lt, "arch": "linnoln", "phys": 1, "wd": 1e-2}
    yield {**lt, "arch": "linnoln", "phys": 0}
    yield {**lt, "arch": "linnoln", "phys": 2}
    yield {**lt, "arch": "linnoln", "phys": 2, "epochs": 300}
    yield {**lt, "arch": "linnoln", "phys": 2, "plw": 0.0}
    yield {**lt, "arch": "orig", "phys": 2}
    yield {**lt, "arch": "linnoln", "phys": 1, "loops": 3}


def configs18():
    """energy split, real rows + anchor: smaller capacity for both (round 17 trend: fewer loops / more wd / smaller MLP help)"""
    mlp = {"model": "MLP", "lr": 1e-3, "anchor": 1}
    for h, wd, ep in ((32, 1e-2, 100), (16, 1e-2, 100), (32, 1e-1, 100), (16, 1e-2, 300), (32, 1e-2, 300)):
        yield {**mlp, "hidden": h, "wd": wd, "epochs": ep}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2}
    for dm, loops in ((32, 3), (16, 3), (32, 10), (16, 10)):
        yield {**lt, "dm": dm, "loops": loops}
    yield {**lt, "dm": 32, "loops": 3, "wd": 1e-1}
    yield {**lt, "dm": 32, "loops": 3, "phys": 0}


def configs19():
    """energy split, real rows + anchor: LT training budget, and LT refinement on an anchored h32 MLP base"""
    yield {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 3}
    yield {**lt, "epochs": 30}; yield {**lt, "epochs": 300}; yield {**lt, "lr": 3e-4}; yield {**lt, "dm": 8}
    res = {**lt, "arch": "mlpres", "base_h": 32, "base_wd": 1e-2}
    for lam in (0, 1.0, 10.0):
        yield {**res, "lam": lam}
    yield {**res, "lam": 1.0, "epochs": 30}


def configs20():
    """energy split, real rows + anchor: train with a random loop count per batch (looped-transformer depth generalisation)"""
    yield {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32}
    yield {**lt, "loops": 3}; yield {**lt, "loops": 10}
    yield {**lt, "loops": 10, "rloop": 1}; yield {**lt, "loops": 20, "rloop": 1}; yield {**lt, "loops": 10, "rloop": 1, "phys": 0}


def configs21():
    """round 20 follow-up: more test-time loops kept lowering energy-OOD error; how far does it go"""
    yield {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 10}
    yield lt; yield {**lt, "phys": 0}; yield {**lt, "loops": 20, "rloop": 1}


def configs25():
    """unlabeled eval-region inputs (1000, generator sampler, no labels): LT loop-consistency and/or physics loss on them; MLP physics-loss control"""
    mlp = {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    yield mlp; yield {**mlp, "unl": 1000, "uphys": 0.1}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 10}
    yield lt
    for uc, up in ((1.0, 0), (10.0, 0), (0, 0.1), (1.0, 0.1), (10.0, 0.1)):
        yield {**lt, "unl": 1000, "uconv": uc, "uphys": up}
    yield {**lt, "unl": 1000, "uconv": 10.0, "uloops": 40}


def configs26():
    """round 25 follow-up: train at the loop count we evaluate at (40), unlabeled loop-consistency strength between 1 and 10"""
    yield {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "unl": 1000, "uphys": 0.1}
    for loops, uc in ((40, 0.3), (40, 1.0), (40, 3.0), (20, 1.0), (20, 3.0)):
        yield {**lt, "loops": loops, "uconv": uc}
    yield {**lt, "loops": 40, "uconv": 1.0, "uphys": 0}


def configs27():
    """attention-free looped MLP (weight-tied, physics-residual feedback, damped update) vs plain MLP and an untied unrolled control"""
    yield {"model": "MLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100}
    lm = {"model": "LMLP", "lr": 1e-3, "anchor": 1, "hidden": 32, "wd": 1e-2, "epochs": 100, "phys": 1, "loops": 10, "alpha": 0.25, "arch": "lmlp"}
    yield lm; yield {**lm, "alpha": 0.1}; yield {**lm, "loops": 3}; yield {**lm, "phys": 0}
    yield {**lm, "untied": 1}; yield {**lm, "sn": 1}; yield {**lm, "unl": 1000, "uconv": 1.0, "uphys": 0.1}


def jobs():
    if ROUND == 31:                 # energy split in the paper's setting (no anchor, +2000 synthetic): does the looped MLP still win there?
        lm = {"model": "LMLP", "lr": 1e-3, "hidden": 32, "wd": 1e-2, "epochs": 100, "phys": 0, "loops": 10, "alpha": 0.25, "arch": "lmlp"}
        cfgs = [{"model": "MLP", "lr": 1e-3, "hidden": 64, "wd": 1e-2, "epochs": 100}, {"model": "MLP", "lr": 1e-3, "hidden": 128, "wd": 1e-2, "epochs": 300},
                lm, {**lm, "phys": 1}, {"model": "LT", "lr": 1e-3, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 10}]
        for cfg, sd in itertools.product(cfgs, range(2900, 2910)):
            yield {"cfg": cfg, "split": "energy", "syn": 2000, "stage": "final", "seed": sd}
        return
    if ROUND == 30:                 # D/W in the paper's setting: no anchor, +2000 synthetic
        lm = {"model": "LMLP", "lr": 1e-3, "hidden": 32, "wd": 1e-2, "epochs": 100, "phys": 0, "loops": 10, "alpha": 0.25, "arch": "lmlp"}
        cfgs = [{"model": "MLP", "lr": 1e-3, "hidden": 64, "wd": 1e-2, "epochs": 100}, lm, {**lm, "phys": 1},
                {"model": "LT", "lr": 1e-3, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 10}]
        for cfg, sd in itertools.product(cfgs, range(2800, 2810)):
            yield {"cfg": cfg, "split": "dw", "syn": 2000, "stage": "final", "seed": sd}
        return
    if ROUND == 29:                 # D/W split: looped MLP vs MLP vs transformer-LT, real-only and +2000 synthetic
        cs = list(configs27())
        lt = {"model": "LT", "lr": 1e-3, "anchor": 1, "arch": "linnoln", "phys": 1, "epochs": 100, "wd": 1e-2, "dm": 32, "loops": 10}
        cfgs = [cs[0], {**cs[1], "phys": 0}, cs[1], lt]
        for cfg, syn, sd in itertools.product(cfgs, (0, 2000), range(2700, 2710)):
            yield {"cfg": cfg, "split": "dw", "syn": syn, "stage": "final", "seed": sd}
        return
    if ROUND == 28:                 # pre-registered confirmation of round 27's pick (looped MLP, no feedback) on fresh seeds; phys=1 logged as control
        cs = list(configs27()); cfgs = [cs[0], {**cs[1], "phys": 0}, cs[1]]
        for cfg, sd in itertools.product(cfgs, range(2600, 2630)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 27:
        for cfg, sd in itertools.product(list(configs27()), range(2500, 2510)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 26:
        for cfg, sd in itertools.product(list(configs26()), range(2400, 2410)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 25:
        for cfg, sd in itertools.product(list(configs25()), range(2200, 2210)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 24:                 # make looping converge: contraction penalty / smaller state step
        lt = [c for c in configs21() if c["model"] == "LT" and c["phys"] == 1 and not c.get("rloop")][0]
        cfgs = [c for c in configs21() if c["model"] == "MLP"] + [{**lt, "conv": mu} for mu in (0.1, 1.0, 10.0)] +                [{**lt, "loops": 20, "conv": 1.0}, {**lt, "step": 0.1}, {**lt, "step": 0.1, "conv": 1.0}]
        for cfg, sd in itertools.product(cfgs, range(2000, 2010)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 23:                 # train with many loops so the gain from deep looping is learned, not drift
        lt = [c for c in configs21() if c["model"] == "LT" and c["phys"] == 1 and not c.get("rloop")][0]
        cfgs = [c for c in configs21() if c["model"] == "MLP"] + [{**lt, "loops": L} for L in (20, 40, 80)]
        for cfg, sd in itertools.product(cfgs, range(1800, 1810)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 22:                 # confirmation of LT L10 fb evaluated at 80 loops (pre-registered), fresh seeds
        for cfg, sd in itertools.product([c for c in configs21() if c["model"] == "MLP" or (c["phys"] == 1 and not c.get("rloop"))], range(1600, 1630)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 21:
        for cfg, sd in itertools.product(list(configs21()), range(1700, 1710)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 20:
        for cfg, sd in itertools.product(list(configs20()), range(1500, 1510)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 19:
        for cfg, sd in itertools.product(list(configs19()), range(1400, 1410)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 18:
        for cfg, sd in itertools.product(list(configs18()), range(1300, 1310)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 17:
        for cfg, sd in itertools.product(list(configs17()), range(1200, 1210)):
            yield {"cfg": cfg, "split": "energy", "syn": 0, "stage": "final", "seed": sd}
        return
    if ROUND == 16:
        for cfg, split, syn, sd in itertools.product(list(configs14()), ("energy", "dw"), ("gap2000", "gap5000"), range(1100, 1110)):
            yield {"cfg": cfg, "split": split, "syn": syn, "stage": "final", "seed": sd}
        return
    if ROUND == 15:
        for cfg, split, syn, sd in itertools.product(list(configs15()), ("energy", "dw"), (0, 2000), range(1000, 1010)):
            yield {"cfg": cfg, "split": split, "syn": syn, "stage": "final", "seed": sd}
        return
    if ROUND == 14:
        for cfg, split, syn, sd in itertools.product(list(configs14()), ("energy", "dw"), (0, 2000, "cal2000"), range(900, 910)):
            if not (syn == 0 and cfg["model"] == "LT" and cfg["arch"] == "orig"):
                yield {"cfg": cfg, "split": split, "syn": syn, "stage": "final", "seed": sd}
        return
    syns = (0, 200, 500, 1000, 2000) if ROUND == 8 else (2000, 5000, 10000) if ROUND == 9 else ("200x10", "500x4", "1000x2") if ROUND == 10 else (0, 200, 2000) if ROUND == 11 else (2000,) if ROUND in (12, 13) else (0, 1)
    for cfg, split, syn in itertools.product(list({1: configs, 2: configs2, 3: configs3, 4: configs4, 5: configs5, 6: configs6, 7: configs7, 8: configs8, 9: configs9, 10: configs9, 11: configs11, 12: configs11, 13: configs11}[ROUND]()), ("energy", "dw"), syns):
        stages = (("final", range(5, 15)),) if ROUND == 6 else (("final", range(100, 130)),) if ROUND == 7 else (("final", range(200, 205)),) if ROUND == 8 else (("final", range(300, 310)),) if ROUND in (9, 10) else (("final", range(500, 510)),) if ROUND == 11 else (("final", range(600, 630)),) if ROUND == 12 else (("final", range(700, 730)),) if ROUND == 13 else (("inner", range(3)), ("final", range(5)))
        for stage, seeds in stages:
            for sd in seeds:
                yield {"cfg": cfg, "split": split, "syn": syn, "stage": stage, "seed": sd}


if __name__ == "__main__":
    import multiprocessing as mp
    key = lambda j: json.dumps({k: j[k] for k in ("cfg", "split", "syn", "stage", "seed")}, sort_keys=True)
    done = {key(json.loads(l)) for l in LOG.open()} if LOG.exists() else set()
    todo = [j for j in jobs() if key(j) not in done]
    if os.environ.get("SMOKE"):
        todo = todo[:2]
        for j in todo: j["cfg"] = {**j["cfg"], "epochs": 1}
    # longest jobs first (10-loop LTs) so the pool tail is short
    todo.sort(key=lambda j: -(j["cfg"].get("loops", 0) * (1 + j["cfg"].get("phys", 0)) + 1) * (int(j["syn"][3:]) if str(j["syn"])[:3] in ("cal", "gap") else eval(j["syn"].replace("x", "*")) if isinstance(j["syn"], str) else j["syn"] if j["syn"] > 1 else 200))
    print(f"{len(todo)} jobs ({len(done)} already done)", flush=True)
    with mp.Pool(int(sys.argv[1])) as pool, LOG.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            if not os.environ.get("SMOKE"):
                fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 20 == 0 or os.environ.get("SMOKE"):
                print(i, r["cfg"], r["split"], r["syn"], r["stage"], round(r["std_MSE"], 3), flush=True)
    print("done")
