"""QKAN features trained on a leave-cycles-out objective, closed-form readout.

Model (per window, see lab.py for the notation):

    c(x) = W_lin z + W_q h(u) + b,       h = DaruanLayer(d -> m)(u)     (QKAN layer)
    yhat = s * c(x) @ B

``W_lin, W_q, b`` are a closed-form generalised ridge solution (penalty
``lam_lin`` on the standardised linear features, ``lam_q`` on the standardised
QKAN features, unpenalised intercept).  The DARUAN angles, pre-activation
frequencies / phases and ``log lam_q`` are trained by full-batch Adam on a
BLOCKED CROSS-VALIDATION objective inside the fit windows: for each of
``n_blocks`` contiguous blocks, the readout is solved on the fit windows at
least 132 windows away from the block and scored on the block.  The QKAN layer
is therefore optimised to produce features that generalise across solar cycles,
not to fit the training targets (SGD on the training loss overfits: sweep s1).

One optimisation step = one pass over the fit windows; ``steps <= 100``.

    python vp.py --sweep NAME [--seeds 0 1 2] [--folds ...] [--part i/n]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import calib
import common as C
import lab
from daruan import DaruanLayer

torch.set_num_threads(1)

DEFAULT = dict(K=12, lam_lin=1e-2, lam_q=0.3, inputs=("ann", "tmin"), m=3, reps=1, observable="z", preact=True,
               in_gain=1.0, steps=60, lr=2e-2, n_blocks=8, train_lam=True, theta_std=None, arch="daruan")


class LinearFeat(nn.Module):
    """Ablation: the QKAN layer replaced by a linear map (d -> m)."""

    def __init__(self, d, m):
        super().__init__()
        self.lin = nn.Linear(d, m)

    def forward(self, u):
        return self.lin(u)


class TanhFeat(nn.Module):
    """Ablation: the QKAN layer replaced by a classical tanh layer (d -> m)."""

    def __init__(self, d, m):
        super().__init__()
        self.lin = nn.Linear(d, m)

    def forward(self, u):
        return torch.tanh(self.lin(u))


def solve_readout(Fz, T, pen):
    """Generalised ridge with unpenalised intercept. Fz (n, p) torch, T (n, K), pen (p,)."""
    n = Fz.shape[0]
    mu_f, b = Fz.mean(0), T.mean(0)
    Fc = Fz - mu_f
    A = Fc.T @ Fc + n * torch.diag(pen)
    W = torch.linalg.solve(A, Fc.T @ (T - b))
    return W, b - mu_f @ W


class VPModel:
    """A fitted model: predict coefficients for any window index of the problem, or for new windows."""

    def __init__(self, cfg, layer, stats, W, b0, n_params):
        self.cfg, self.layer, self.stats, self.W, self.b0, self.n_params = cfg, layer, stats, W, b0, n_params

    def features(self, Zlin, U):
        with torch.no_grad():
            if self.layer is None:
                return Zlin
            H = self.layer(U * self.cfg["in_gain"])
            Hs = (H - self.stats["h_mu"]) / self.stats["h_sd"]
            return torch.cat([Zlin, Hs], dim=1)

    def coef(self, Zlin, U):
        return (self.features(Zlin, U) @ self.W + self.b0).numpy()


def fit_vp(P, fit, cfg, seed, verbose=False):
    cfg = {**DEFAULT, **cfg}
    t64 = lambda a: torch.tensor(a, dtype=torch.float64)
    ann = P["feats"]["ann"]
    mu, sd = ann[fit].mean(0), ann[fit].std(0) + 1e-9
    Zlin = t64((ann - mu) / sd)
    T = t64(P["T"])
    fit_t = torch.tensor(fit)
    d_lin = Zlin.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg["m"] == 0:
        W, b0 = solve_readout(Zlin[fit_t], T[fit_t], torch.full((d_lin,), cfg["lam_lin"], dtype=torch.float64))
        return VPModel(cfg, None, {}, W, b0, W.numel() + b0.numel()), t64(np.zeros((len(ann), 0))), Zlin, []
    U = t64(lab.build_inputs(P, cfg["inputs"], fit))
    d = U.shape[1]
    if cfg["arch"] == "daruan":
        layer = DaruanLayer(d, cfg["m"], cfg["reps"], cfg["preact"], False, cfg["observable"],
                            theta_std=cfg["theta_std"]).double()
    elif cfg["arch"] == "linear":
        layer = LinearFeat(d, cfg["m"]).double()
    else:
        layer = TanhFeat(d, cfg["m"]).double()
    loglam = nn.Parameter(torch.tensor(float(np.log(cfg["lam_q"])), dtype=torch.float64),
                          requires_grad=cfg["train_lam"])
    params = [p for p in layer.parameters() if p.requires_grad] + ([loglam] if cfg["train_lam"] else [])
    splits = [(torch.tensor(tr), torch.tensor(held)) for tr, held in calib.crossfit_blocks(fit, cfg["n_blocks"])]
    pen_lin = torch.full((d_lin,), cfg["lam_lin"], dtype=torch.float64)

    def pen():
        return torch.cat([pen_lin, torch.exp(loglam).expand(cfg["m"])])

    def cv_loss():
        H = layer(U * cfg["in_gain"])
        tot, cnt = 0.0, 0
        for tr, held in splits:
            h_mu, h_sd = H[tr].mean(0), H[tr].std(0) + 1e-6
            Fz = torch.cat([Zlin, (H - h_mu) / h_sd], dim=1)
            W, b0 = solve_readout(Fz[tr], T[tr], pen())
            err = Fz[held] @ W + b0 - T[held]
            tot = tot + (err ** 2).sum()
            cnt += err.numel()
        return tot / cnt

    hist = []
    if cfg["steps"] > 0:
        opt = torch.optim.Adam(params, lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.05)
        best = (float("inf"), None)
        for step in range(cfg["steps"]):
            loss = cv_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(loss))
            if verbose and step % 10 == 0:
                print(step, f"{float(loss):.6f}", f"lam_q {float(torch.exp(loglam)):.3f}", flush=True)
    with torch.no_grad():
        H = layer(U * cfg["in_gain"])
        stats = dict(h_mu=H[fit_t].mean(0), h_sd=H[fit_t].std(0) + 1e-6)
        Fz = torch.cat([Zlin, (H - stats["h_mu"]) / stats["h_sd"]], dim=1)
        W, b0 = solve_readout(Fz[fit_t], T[fit_t], pen())
    n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad) + W.numel() + b0.numel()
    cfg = dict(cfg, lam_q_final=float(torch.exp(loglam.detach())))
    return VPModel(cfg, layer, stats, W, b0, n_params), U, Zlin, hist


def run_one(cfg, fold, seed):
    full = {**DEFAULT, **cfg}
    P = lab.problem(full["K"])
    fit, ev = C.folds((fold,))[fold]
    t0 = time.time()
    model, U, Zlin, hist = fit_vp(P, fit, cfg, seed)
    ev_t, fit_t = torch.tensor(ev), torch.tensor(fit)
    pred = P["s"][ev] * (model.coef(Zlin[ev_t], U[ev_t]) @ P["B"])
    m = C.score(pred, P["Y32"][ev])
    mu, sd, W, b = lab.ridge(P["feats"]["ann"][fit], P["T"][fit], full["lam_lin"])
    base = C.score(P["s"][ev] * ((((P["feats"]["ann"][ev] - mu) / sd) @ W + b) @ P["B"]), P["Y32"][ev])
    pf = P["s"][fit] * (model.coef(Zlin[fit_t], U[fit_t]) @ P["B"])
    return dict(fold=fold, seed=seed, params=model.n_params, mse=m["scaled_mse"], pae=m["pae"], pte=m["pte"],
                base_mse=base["scaled_mse"], fit_mse=float(((pf - P["y"][fit]) ** 2).mean()),
                cv_first=hist[0] if hist else None, cv_last=hist[-1] if hist else None,
                lam_q=model.cfg.get("lam_q_final"), sec=round(time.time() - t0, 1))


SWEEPS = {}


def sweep(name):
    def deco(fn):
        SWEEPS[name] = fn
        return fn
    return deco


@sweep("v1")
def _v1():
    out = []
    for inputs in (("tmin",), ("ann",), ("ann", "tmin"), ("tmin", "tmax")):
        for m in (2, 4):
            for steps in (0, 60):
                out.append((f"{'+'.join(inputs)} m{m} steps{steps}", dict(inputs=inputs, m=m, steps=steps)))
    return out


@sweep("v3")
def _v3():
    """Variants of the phase-clock QKAN (single input: cycle phase angle)."""
    out = []
    base = dict(inputs=("tmin",), m=4, reps=1, steps=60)
    out.append(("clock base", dict(base)))
    for nm in ("tmin_s5", "tmin_s12", "tmin_nocap", "tmax"):
        out.append((f"clock {nm}", dict(base, inputs=(nm,))))
    for K in (6, 8):
        out.append((f"clock K{K}", dict(base, K=K)))
    for nb in (4, 6, 12):
        out.append((f"clock blocks{nb}", dict(base, n_blocks=nb)))
    for lam in (0.1, 1.0):
        out.append((f"clock lamq{lam:g}", dict(base, lam_q=lam)))
    out.append(("clock fixed-lam", dict(base, train_lam=False)))
    out.append(("clock no-preact", dict(base, preact=False)))
    out.append(("clock fastmeasure", dict(base, observable="fast")))
    out.append(("clock m2", dict(base, m=2)))
    out.append(("clock m3", dict(base, m=3)))
    out.append(("clock m6", dict(base, m=6)))
    out.append(("ABL linear-of-angle", dict(base, arch="linear")))
    out.append(("ABL tanh layer", dict(base, arch="tanh")))
    out.append(("ABL no QKAN (ridge)", dict(base, m=0)))
    return out


@sweep("v2")
def _v2():
    """Low-dimensional physically meaningful QKAN inputs (cycle clock + state)."""
    out = []
    for inputs in (("tmin",), ("tmin", "lvl"), ("tmin", "lvl", "slope"), ("tmin", "cycmax"), ("tmin", "tmax", "lvl"),
                   ("dphase",), ("tmin", "dphase"), ("tmin", "cyclen"), ("tmin", "lvl", "cycmax", "cycmin")):
        for m, reps in ((4, 1), (8, 1), (4, 2), (4, 3)):
            out.append((f"{'+'.join(inputs)} m{m} r{reps}", dict(inputs=inputs, m=m, reps=reps, steps=60)))
    for lr in (5e-3, 5e-2):
        for steps in (30, 100):
            out.append((f"tmin m4 r1 lr{lr:g} st{steps}", dict(inputs=("tmin",), m=4, lr=lr, steps=steps)))
            out.append((f"tmin m4 r2 lr{lr:g} st{steps}", dict(inputs=("tmin",), m=4, reps=2, lr=lr, steps=steps)))
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
    path = out_dir / f"vp_{args.sweep}.jsonl"
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
                print(f"{name:36s} {fold} s{seed} mse {r['mse']:.4f} (ridge {r['base_mse']:.4f}) fit {r['fit_mse']:.4f} "
                      f"cvobj {r['cv_first'] or 0:.5f}->{r['cv_last'] or 0:.5f} lam_q {r['lam_q']:.3g} "
                      f"p {r['params']} {r['sec']}s", flush=True)


if __name__ == "__main__":
    main()
