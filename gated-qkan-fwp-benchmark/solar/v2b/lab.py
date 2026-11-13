"""Experiment lab of solar v2 approach B: ridge skeleton + QKAN (DARUAN) residual core.

The forecast of one window x (528 scaled months) is

    s      = mean(x) + 1e-3                                  (44-year activity level)
    z      = 12-month means of the last 132 months / s       (11 features)
    c(x)   = W z + b  +  alpha * Q(u(x))                     (K decoder coefficients)
    yhat   = s * c(x) @ B                                    (B: K x 132 DCT-II basis)

``W, b`` is a closed-form ridge solution on the fit windows (the linear skip
path) and ``Q`` is a QKAN built from ``daruan.DaruanLayer`` (the only learned
nonlinearity), trained by mini-batch AdamW on the scaled-space MSE.

    python lab.py --sweep NAME [--seeds 0 1] [--folds cv1 ... val] [--part i/n]

appends one JSON line per (config, fold, seed) to runs/solar/v2b/cv/NAME.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import features as F
from daruan import DaruanLayer

torch.set_num_threads(1)

_CACHE = {}


def problem(K=12):
    """Features / targets shared by all experiments (float64 numpy)."""
    key = ("problem", K)
    if key not in _CACHE:
        d = C.data()
        x, y = d.X.astype(np.float64), d.Y.astype(np.float64)
        s = x.mean(axis=1, keepdims=True) + 1e-3
        cyc = F.cycle_descriptors(x)
        dm = F.demod(x, 132.0, 132, (1,))
        feats = {
            "ann": F.patch_means(x, 12, 132) / s,
            "ann6": F.patch_means(x, 24, 144) / s,
            "tmin": (2 * np.pi * cyc[:, 0] / 132.0)[:, None],
            "tmax": (2 * np.pi * cyc[:, 1] / 132.0)[:, None],
            "dphase": np.arctan2(dm[:, 1], dm[:, 0])[:, None],
            "cycval": cyc[:, [2, 3, 5, 6, 7]] / s,
            "trail": F.trailing_means(x, (6, 12, 24, 48, 96, 132, 264)) / s,
            "lvl": x[:, -12:].mean(axis=1, keepdims=True) / s,
            "slope": (x[:, -12:].mean(axis=1, keepdims=True) - x[:, -24:-12].mean(axis=1, keepdims=True)) / s,
            "cycmax": cyc[:, [3]] / s,
            "cycmin": cyc[:, [2]] / s,
            "cyclen": (2 * np.pi * cyc[:, 4] / 132.0)[:, None],
        }
        for sig in (5.0, 12.0):
            cs = F.cycle_descriptors(x, sigma=sig)
            feats[f"tmin_s{sig:g}"] = (2 * np.pi * cs[:, 0] / 132.0)[:, None]
        cnc = F.cycle_descriptors(x, cap=1000.0)
        feats["tmin_nocap"] = (2 * np.pi * cnc[:, 0] / 132.0)[:, None]
        B = C.dct_basis(C.HORIZON, K)
        _CACHE[key] = dict(x=x, y=y, s=s, feats=feats, B=B, T=(y / s) @ B.T, Y32=d.Y)
    return _CACHE[key]


def ridge(Ftr, Ttr, lam):
    """Ridge with unpenalised intercept on standardised features -> (mu, sd, W, b)."""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Z = (Ftr - mu) / sd
    n, d = Z.shape
    b = Ttr.mean(0)
    W = np.linalg.solve(Z.T @ Z + lam * n * np.eye(d), Z.T @ (Ttr - b))
    return mu, sd, W, b


ANGLE_FEATS = ("tmin", "tmax", "dphase", "cyclen", "tmin_s5", "tmin_s12", "tmin_nocap")


class QkanCore(nn.Module):
    """u -> K coefficient residuals through DARUAN layers.

    arch "add":   DaruanLayer(d, K)                      (additive / GAM)
    arch "addlr": DaruanLayer(d, m) -> Linear(m, K)      (additive, rank m)
    arch "deep":  DaruanLayer(d, m) -> DaruanLayer(m, K) (two-layer KAN)
    arch "linear" / "mlp": the ablations (no QKAN): Linear(d, K) / Linear-tanh-Linear.
    The output is multiplied by a per-coefficient gain ``alpha`` initialised at zero,
    so every model starts exactly at the ridge skeleton.
    """

    def __init__(self, d, K, arch="addlr", m=4, reps=1, observable="z", preact=True, in_gain=1.0,
                 alpha_init=0.0):
        super().__init__()
        self.arch, self.in_gain = arch, in_gain
        if arch == "add":
            self.l1 = DaruanLayer(d, K, reps, preact, False, observable)
            self.out = None
        elif arch == "addlr":
            self.l1 = DaruanLayer(d, m, reps, preact, False, observable)
            self.out = nn.Linear(m, K)
        elif arch == "deep":
            self.l1 = DaruanLayer(d, m, reps, preact, False, observable)
            self.l2 = DaruanLayer(m, K, reps, preact, False, observable)
            self.out = None
        elif arch == "linear":
            self.l1 = nn.Linear(d, K)
            self.out = None
        elif arch == "mlp":
            self.l1 = nn.Sequential(nn.Linear(d, m), nn.Tanh())
            self.out = nn.Linear(m, K)
        else:
            raise ValueError(arch)
        self.alpha = nn.Parameter(torch.full((K,), float(alpha_init)))

    def forward(self, u):
        h = self.l1(u * self.in_gain)
        if self.arch == "deep":
            h = self.l2(h)
        elif self.out is not None:
            h = self.out(h)
        return h * self.alpha

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_inputs(P, names, fit):
    """Stack QKAN input features; value features are standardised with fit statistics,
    angle features are passed as raw angles (they enter RZ gates directly)."""
    cols = []
    for nme in names:
        f = P["feats"][nme]
        if nme in ANGLE_FEATS:
            cols.append(f)
        else:
            cols.append((f - f[fit].mean(0)) / (f[fit].std(0) + 1e-9))
    return np.hstack(cols)


DEFAULT = dict(K=12, lam=1e-2, inputs=("ann",), arch="addlr", m=4, reps=1, observable="z", preact=True,
               in_gain=1.0, epochs=60, batch=64, lr=3e-3, wd=1e-2, loss="raw", alpha_init=0.0, noise=0.0,
               ema=0.0, stride=1)


def run_one(cfg, fold, seed, verbose=False):
    cfg = {**DEFAULT, **cfg}
    P = problem(cfg["K"])
    fit, ev = C.folds((fold,))[fold]
    mu, sd, W, b = ridge(P["feats"]["ann"][fit], P["T"][fit], cfg["lam"])
    skel = ((P["feats"]["ann"] - mu) / sd) @ W + b  # (n, K) ridge coefficients for all windows
    U = build_inputs(P, cfg["inputs"], fit)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    B, S, Yt, Sk, Ut = t(P["B"]), t(P["s"]), t(P["y"]), t(skel), t(U)
    torch.manual_seed(seed)
    np.random.seed(seed)
    core = QkanCore(U.shape[1], cfg["K"], cfg["arch"], cfg["m"], cfg["reps"], cfg["observable"], cfg["preact"],
                    cfg["in_gain"], cfg["alpha_init"])
    decay = [p for n, p in core.named_parameters() if n.startswith("out") or n == "alpha"]
    nodecay = [p for n, p in core.named_parameters() if not (n.startswith("out") or n == "alpha")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg["lr"])
    fit_t = torch.tensor(fit[::cfg["stride"]])
    steps_per_epoch = math.ceil(len(fit_t) / cfg["batch"])
    total = cfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.15)
    ema = None
    if cfg["ema"] > 0:
        ema = {k: v.detach().clone() for k, v in core.state_dict().items()}

    def predict(idx, state=None):
        if state is not None:
            keep = {k: v.clone() for k, v in core.state_dict().items()}
            core.load_state_dict(state)
        core.eval()
        with torch.no_grad():
            c = Sk[idx] + core(Ut[idx])
            out = (S[idx] * (c @ B)).numpy()
        if state is not None:
            core.load_state_dict(keep)
        return out

    gen = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        core.train()
        perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            u = Ut[idx]
            if cfg["noise"] > 0:
                u = u + cfg["noise"] * torch.randn(u.shape, generator=gen)
            c = Sk[idx] + core(u)
            if cfg["loss"] == "raw":
                loss = ((S[idx] * (c @ B) - Yt[idx]) ** 2).mean()
            else:  # normalised space, like the ridge skeleton
                loss = ((c @ B - Yt[idx] / S[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in core.state_dict().items():
                        ema[k].mul_(cfg["ema"]).add_(v, alpha=1 - cfg["ema"])
        if verbose or ep == cfg["epochs"] - 1 or (ep + 1) % 10 == 0:
            m = C.score(predict(ev, ema), P["Y32"][ev])
            curve.append((ep + 1, m["scaled_mse"]))
            if verbose:
                print(ep + 1, C.short(m), flush=True)
    final = C.score(predict(ev, ema), P["Y32"][ev])
    base = C.score((P["s"][ev] * (skel[ev] @ P["B"])), P["Y32"][ev])
    fitm = C.score(predict(fit, ema), P["Y32"][fit])
    return dict(fold=fold, seed=seed, params=core.n_params(), mse=final["scaled_mse"], pae=final["pae"],
                pte=final["pte"], base_mse=base["scaled_mse"], fit_mse=fitm["scaled_mse"],
                best_mse=min(c[1] for c in curve), curve=curve, sec=round(time.time() - t0, 1))


SWEEPS = {}


def sweep(name):
    def deco(fn):
        SWEEPS[name] = fn
        return fn
    return deco


@sweep("s1")
def _s1():
    """First look: architecture x inputs, default regularisation."""
    out = []
    for inputs in (("ann",), ("ann", "tmin"), ("tmin", "tmax", "cycval"), ("ann", "tmin", "tmax")):
        for arch, m in (("addlr", 4), ("add", 0), ("deep", 4), ("linear", 0), ("mlp", 8)):
            out.append((f"{arch}{m}-{'+'.join(inputs)}", dict(inputs=inputs, arch=arch, m=m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", nargs="+", default=list(C.ALL_NAMES))
    ap.add_argument("--part", default="0/1")
    args = ap.parse_args()
    i, n = (int(v) for v in args.part.split("/"))
    configs = SWEEPS[args.sweep]()[i::n]
    out_dir = C.RUNS_DIR / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.sweep}.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["name"], r["fold"], r["seed"]))
    for name, cfg in configs:
        for seed in args.seeds:
            for fold in args.folds:
                if (name, fold, seed) in done:
                    continue
                r = run_one(cfg, fold, seed)
                r.update(name=name, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
                with open(path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"{name:40s} {fold} s{seed} mse {r['mse']:.4f} (ridge {r['base_mse']:.4f}) "
                      f"fit {r['fit_mse']:.4f} p {r['params']} {r['sec']}s", flush=True)


if __name__ == "__main__":
    main()
