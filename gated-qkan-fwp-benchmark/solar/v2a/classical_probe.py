"""Closed-form probes (no QKAN): which featurisation / baseline / decoder matter.

    python solar/v2a/classical_probe.py  -> runs/solar/v2a/classical_probe.json

Ridge regressions from smoothed patch features of the input window to the
132-month target, scored on the four rolling-origin folds and the official
validation split.  These are reference points and the "all-linear" ablation
anchors for the QKAN-FWP models; they are not candidates themselves.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

import common as C


def patch_means(x, patch, n_months):
    """Means of the last ``n_months`` of the window in patches of ``patch`` months, oldest first."""
    seg = x[:, C.SEQ_LEN - n_months:]
    return seg.reshape(x.shape[0], n_months // patch, patch).mean(axis=2)


def ridge_fit(F, T, lam):
    """Ridge with unpenalised intercept; returns predict(Fnew)."""
    mu_f, mu_t = F.mean(0), T.mean(0)
    Fc, Tc = F - mu_f, T - mu_t
    n, d = Fc.shape
    A = Fc.T @ Fc + lam * n * np.eye(d)
    W = np.linalg.solve(A, Fc.T @ Tc)
    return lambda G: (G - mu_f) @ W + mu_t


def run(config, x, y, fit, ev):
    patch, months, lam, resid, norm, rank = (config[k] for k in ("patch", "months", "lam", "resid", "norm", "rank"))
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    scale = x64.mean(axis=1, keepdims=True) + 1e-3 if norm else np.ones((len(x64), 1))
    feats = patch_means(x64, patch, months) / scale
    base = C.block_mean_baseline(x64) if resid else np.zeros_like(y64)
    targ = (y64 - base) / scale
    Ttr = targ[fit]
    if rank:
        mu = Ttr.mean(0)
        _, _, vt = np.linalg.svd(Ttr - mu, full_matrices=False)
        basis = vt[:rank]
        coef = (Ttr - mu) @ basis.T
        predict = ridge_fit(feats[fit], coef, lam)
        out = predict(feats[ev]) @ basis + mu
    else:
        out = ridge_fit(feats[fit], Ttr, lam)(feats[ev])
    pred = out * scale[ev] + base[ev]
    return C.score(pred, y[ev])


def main():
    d = C.data()
    folds = C.folds()
    rows = []
    # simple baselines
    for name, fn in (("block4", lambda x: C.block_mean_baseline(x, 4)),
                     ("block3", lambda x: C.block_mean_baseline(x, 3)),
                     ("block2", lambda x: C.block_mean_baseline(x, 2)),
                     ("block1", lambda x: C.block_mean_baseline(x, 1))):
        res = {f: C.score(fn(d.X[ev]), d.Y[ev]) for f, (fit, ev) in folds.items()}
        rows.append({"name": name, "config": {}, "folds": res})
    grid = itertools.product((12, 6), (132, 264, 528), (1e-3, 1e-2, 1e-1, 1.0), (False, True), (False, True), (0, 4, 8))
    for patch, months, lam, resid, norm, rank in grid:
        cfg = dict(patch=patch, months=months, lam=lam, resid=resid, norm=norm, rank=rank)
        res = {f: run(cfg, d.X, d.Y, fit, ev) for f, (fit, ev) in folds.items()}
        name = f"ridge p{patch} m{months} lam{lam:g} resid{int(resid)} norm{int(norm)} rank{rank}"
        rows.append({"name": name, "config": cfg, "folds": res})
    for r in rows:
        ms = [r["folds"][f]["scaled_mse"] for f in folds]
        r["mean_mse"] = float(np.mean(ms))
        r["cv_mean_mse"] = float(np.mean(ms[:4]))
        r["max_mse"] = float(np.max(ms))
    out = C.RUNS_DIR / "classical_probe.json"
    out.write_text(json.dumps(rows, indent=1))
    rows_sorted = sorted(rows, key=lambda r: r["cv_mean_mse"])
    print(f"{'name':58s} cvmean  " + "  ".join(f"{f:>16s}" for f in folds))
    show = [r for r in rows if r["name"].startswith("block")] + rows_sorted[:25]
    for r in show:
        print(f"{r['name']:58s} {r['cv_mean_mse']:.4f}  " + "  ".join(C.short(r["folds"][f]) for f in folds))


if __name__ == "__main__":
    main()
