"""Generic nonlinear yardstick (no QKAN): RBF kernel ridge on smoothed patch features.

    python solar/v2a/nonlinear_probe.py -> runs/solar/v2a/nonlinear_probe.json

Tells how much a generic nonlinear smoother gains over the linear ridge probes
on the rolling-origin folds.  Reference point only, not a candidate.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

import common as C
from classical_probe import patch_means


def krr(Ftr, Ttr, Fev, gamma, lam):
    mu_t = Ttr.mean(0)
    sd = Ftr.std() + 1e-9
    a, b = Ftr / sd, Fev / sd
    d_tr = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    d_ev = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    scale = gamma / a.shape[1]
    K = np.exp(-scale * d_tr)
    alpha = np.linalg.solve(K + lam * len(a) * np.eye(len(a)), Ttr - mu_t)
    return np.exp(-scale * d_ev) @ alpha + mu_t


def main():
    d = C.data()
    folds = C.folds()
    x64, y64 = d.X.astype(np.float64), d.Y.astype(np.float64)
    rows = []
    grid = itertools.product((12,), (132, 264), (0.3, 1.0, 3.0), (1e-4, 1e-3, 1e-2), (False, True))
    for patch, months, gamma, lam, norm in grid:
        scale = x64.mean(axis=1, keepdims=True) + 1e-3 if norm else np.ones((len(x64), 1))
        feats = patch_means(x64, patch, months) / scale
        targ = y64 / scale
        res = {}
        for f, (fit, ev) in folds.items():
            pred = krr(feats[fit], targ[fit], feats[ev], gamma, lam) * scale[ev]
            res[f] = C.score(pred, d.Y[ev])
        name = f"krr p{patch} m{months} g{gamma:g} lam{lam:g} norm{int(norm)}"
        ms = [res[f]["scaled_mse"] for f in folds]
        rows.append({"name": name, "folds": res, "cv_mean_mse": float(np.mean(ms[:4]))})
    (C.RUNS_DIR / "nonlinear_probe.json").write_text(json.dumps(rows, indent=1))
    print(f"{'name':40s} cvmean  " + "  ".join(f"{f:>16s}" for f in folds))
    for r in sorted(rows, key=lambda r: r["cv_mean_mse"])[:14]:
        print(f"{r['name']:40s} {r['cv_mean_mse']:.4f}  " + "  ".join(C.short(r["folds"][f]) for f in folds))


if __name__ == "__main__":
    main()
