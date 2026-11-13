"""Probe 1 (no QKAN): which features carry signal, and does ANY generic nonlinear
learner beat ridge on the rolling-origin folds?

    python solar/v2b/probe1.py  -> runs/solar/v2b/probe1.json (+ table on stdout)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import common as C
import features as F


def ridge_fit(Ftr, Ttr, lam):
    mu_f, mu_t = Ftr.mean(0), Ttr.mean(0)
    sd = Ftr.std(0) + 1e-9
    Z = (Ftr - mu_f) / sd
    n, d = Z.shape
    W = np.linalg.solve(Z.T @ Z + lam * n * np.eye(d), Z.T @ (Ttr - mu_t))
    return lambda G: ((G - mu_f) / sd) @ W + mu_t


def krr_fit(Ftr, Ttr, lam, gamma):
    mu_f, mu_t = Ftr.mean(0), Ttr.mean(0)
    sd = Ftr.std(0) + 1e-9
    Z = (Ftr - mu_f) / sd
    d = Z.shape[1]

    def kern(A, B):
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
        return np.exp(-gamma * d2 / d) + 0.0

    K = kern(Z, Z)
    alpha = np.linalg.solve(K + lam * len(Z) * np.eye(len(Z)), Ttr - mu_t)
    return lambda G: kern((G - mu_f) / sd, Z) @ alpha + mu_t


def gbm_fit(Ftr, Ttr, depth, n_est, lr):
    from sklearn.ensemble import GradientBoostingRegressor
    models = []
    for k in range(Ttr.shape[1]):
        m = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, learning_rate=lr, subsample=0.7,
                                      random_state=0)
        m.fit(Ftr, Ttr[:, k])
        models.append(m)
    return lambda G: np.stack([m.predict(G) for m in models], axis=1)


def rf_fit(Ftr, Ttr, leaf):
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=leaf, max_features=0.5, random_state=0, n_jobs=1)
    m.fit(Ftr, Ttr)
    return lambda G: m.predict(G)


FEATS = {}


def feature_sets(x):
    if FEATS:
        return FEATS
    ann = F.patch_means(x, 12, 132)
    ann22 = F.patch_means(x, 12, 264)
    q = F.patch_means(x, 6, 132)
    tr = F.trailing_means(x)
    dm = F.demod(x, 132.0, 264, (1, 2))
    cyc = F.cycle_descriptors(x)
    FEATS.update({
        "ann11": ann,
        "ann22": ann22,
        "semi22": q,
        "trail13": tr,
        "ann11+trail": np.hstack([ann, tr]),
        "ann11+demod": np.hstack([ann, dm]),
        "ann11+cyc": np.hstack([ann, cyc]),
        "cyc8": cyc,
        "cyc+trail": np.hstack([cyc, tr]),
        "ann11+cyc+trail": np.hstack([ann, cyc, tr]),
    })
    return FEATS


def main():
    d = C.data()
    x, y = d.X.astype(np.float64), d.Y.astype(np.float64)
    folds = C.folds()
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    B = C.dct_basis(C.HORIZON, K)
    feats = feature_sets(x)
    rows = []

    def run(name, fs, norm, fit_fn):
        f = feats[fs]
        scale = x.mean(axis=1, keepdims=True) + 1e-3 if norm else np.ones((len(x), 1))
        res = {}
        for fold, (fit, ev) in folds.items():
            fz = f.copy()
            # value-type features are divided by the scale; cycle timing columns are not
            if norm:
                if fs.startswith("cyc"):
                    fz[:, [2, 3, 5, 6, 7]] /= scale
                    if fs == "cyc+trail":
                        fz[:, 8:] /= scale
                elif fs == "ann11+cyc" or fs == "ann11+cyc+trail":
                    fz[:, :11] /= scale
                    fz[:, [13, 14, 16, 17, 18]] /= scale
                    fz[:, 19:] /= scale
                else:
                    fz = fz / scale
            T = (y / scale) @ B.T
            pred = fit_fn(fz[fit], T[fit])(fz[ev]) @ B * scale[ev]
            res[fold] = C.score(pred, d.Y[ev])
        ms = [res[k]["scaled_mse"] for k in C.ALL_NAMES]
        row = {"name": name, "folds": res, "cv_mean_mse": float(np.mean(ms[:4])), "val_mse": ms[4]}
        rows.append(row)
        print(f"{name:58s} " + " ".join(C.short(res[k]) for k in C.ALL_NAMES) + f"  cv {row['cv_mean_mse']:.4f}",
              flush=True)

    t0 = time.time()
    # oracle projection floor: project the truth onto the basis
    res = {}
    for fold, (fit, ev) in folds.items():
        res[fold] = C.score(y[ev] @ B.T @ B, d.Y[ev])
    print(f"{'ORACLE projection onto DCT-' + str(K):58s} " + " ".join(C.short(res[k]) for k in C.ALL_NAMES))

    for fs in feats:
        for norm in (False, True):
            for lam in (1e-3, 1e-2, 1e-1, 1.0):
                run(f"ridge {fs} norm{int(norm)} lam{lam:g}", fs, norm, lambda a, b, lam=lam: ridge_fit(a, b, lam))
    for fs in ("ann11", "cyc8", "ann11+cyc", "cyc+trail", "ann11+demod"):
        for norm in (False, True):
            for lam in (1e-3, 1e-2, 1e-1):
                for gamma in (0.1, 0.3, 1.0):
                    run(f"krr {fs} norm{int(norm)} lam{lam:g} g{gamma:g}", fs, norm,
                        lambda a, b, lam=lam, gamma=gamma: krr_fit(a, b, lam, gamma))
    for fs in ("ann11", "cyc8", "ann11+cyc", "cyc+trail"):
        for norm in (False, True):
            for leaf in (10, 40):
                run(f"extratrees {fs} norm{int(norm)} leaf{leaf}", fs, norm,
                    lambda a, b, leaf=leaf: rf_fit(a, b, leaf))
    print(f"elapsed {time.time() - t0:.0f}s")
    json.dump(rows, open(C.RUNS_DIR / f"probe1_K{K}.json", "w"), indent=0)


if __name__ == "__main__":
    main()
