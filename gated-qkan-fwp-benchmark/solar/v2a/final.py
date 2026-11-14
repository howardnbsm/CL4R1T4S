"""Solar v2a: candidates, the final-fit entry point and its evaluation harness.

ENTRY POINT FOR THE SEALED EVALUATION
-------------------------------------
    import sys; sys.path.insert(0, "<project>/solar/v2a")
    import final
    predict_fn = final.fit(seed)            # recommended candidate, peak-calibrated
    # predict_fn: scaled windows (n, 528) float32 -> scaled forecasts (n, 132) float32

``fit(seed)`` uses the 2084 TRAINING windows only.  The validation split is
not used at all (no checkpoint selection is needed: the read-out is a closed
form and the programmer is trained for a fixed number of full-batch steps),
and the test split is never touched (``protocol.load()`` is sealed).

A candidate is an ensemble of ``members`` independently initialised CVVP
models (``cvvp.py``): member seed = 1000 * seed + m, so each reported seed is
an independent repetition of the whole procedure.  The ensemble forecast is
the mean of the member forecasts, clipped at zero (sunspot numbers are
non-negative).

Peak calibration (optional, reported with and without)
------------------------------------------------------
An MSE-optimal forecast is a conditional mean: it is smooth and damped, while
PAE compares its maximum with the maximum of the NOISY monthly truth.  The
calibrated forecast multiplies the curve by a smooth bump centred on the
forecast's own maximum:

    y_cal(tau) = y(tau) * (1 + a * exp(-(tau - tau*)^2 / (2 sigma^2))),  tau* = argmax y

    sigma = 3.75 months, fixed: the standard deviation of the 13-month boxcar that defines the
            smoothed sunspot number (13 / sqrt(12)); the bump has the width of that smoothing kernel;
    a     = one scalar per fit, chosen on the TRAINING windows only: the value in [0, 1.5] that
            minimises mean |max(y) - (1 + a) * max(y_oob)| where y_oob are out-of-block forecasts of
            the training windows (two-sided blocks with a 132-window embargo, read-out re-solved per
            block), i.e. the gain is estimated on forecasts whose targets the read-out has not seen.

The position of the maximum (PTE) is unchanged by construction; the MSE cost
is reported next to every calibrated number.

CLI (results appended to runs/solar/v2a/final/results.jsonl, restart-safe)
    python solar/v2a/final.py run --candidates rec --folds cv1 cv2 cv3 cv4 val --seeds 0 1 2 3 4
    python solar/v2a/final.py table
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import common as C
from cvvp import CVVP_DEFAULTS, RidgeSummaryForecaster, fit_cvvp, out_of_block_prediction

SIGMA_MONTHS = 13.0 / math.sqrt(12.0)       # 3.75: std of the 13-month smoothing boxcar
GAIN_GRID = np.linspace(0.0, 1.5, 151)

_BASE = dict(O=16, I=8, theta0=True)
_FWD = dict(steps=60, scheme="forward")

# name -> (model overrides, cvvp overrides, ensemble members)
CANDIDATES = {
    # recommended: QKAN slow programmer -> gated fast weights -> programmed QKAN fast programmer -> ridge read-out
    "rec": (dict(**_BASE), dict(**_FWD), 2),
    # variants of the training of the programmer
    "rec_rand": (dict(**_BASE), dict(steps=0), 2),
    "rec_two_sided": (dict(**_BASE), dict(steps=60), 2),
    "rec_single": (dict(**_BASE), dict(**_FWD), 1),
    # QKAN ablations (same procedure, QKAN layers replaced / removed)
    "abl_linear_both": (dict(**_BASE, slow="linear", fast="linear"), dict(**_FWD), 2),
    "abl_fast_linear": (dict(**_BASE, fast="linear"), dict(**_FWD), 2),
    "abl_slow_linear": (dict(**_BASE, slow="linear"), dict(**_FWD), 2),
    "abl_no_program": (dict(**_BASE, program=False), dict(**_FWD), 2),
    # no QKAN and no fast weights at all: closed-form ridge on the 11 recent 12-month means
    "ridge11": (dict(), dict(lam=0.03), 1),
}
RECOMMENDED = "rec"


def bump_calibrate(pred: np.ndarray, gain: float, sigma: float = SIGMA_MONTHS) -> np.ndarray:
    """Multiply every forecast by ``1 + gain * Gaussian(tau - argmax)``; (n, 132) -> (n, 132)."""
    pred = np.asarray(pred, dtype=np.float64)
    tau = np.arange(pred.shape[1])[None, :]
    centre = pred.argmax(axis=1)[:, None]
    return pred * (1.0 + gain * np.exp(-0.5 * ((tau - centre) / sigma) ** 2))


def fit_gain(pred_oob: np.ndarray, y: np.ndarray) -> float:
    """"full" gain: the scalar minimising mean |max(y) - (1 + a) max(pred_oob)| on the fit windows."""
    top_pred, top_true = pred_oob.max(axis=1), np.asarray(y, dtype=np.float64).max(axis=1)
    errors = [np.abs(top_true - (1.0 + a) * top_pred).mean() for a in GAIN_GRID]
    return float(GAIN_GRID[int(np.argmin(errors))])


def fit_balanced_gain(pred_oob: np.ndarray, y: np.ndarray, gain_full: float) -> float:
    """"balanced" gain: argmin over a in [0, gain_full] of  MSE(a) / MSE(0) + PAE(a) / PAE(0)  on the
    out-of-block forecasts of the fit windows (the two reported metrics the bump moves, equally weighted,
    each relative to the uncalibrated forecast)."""
    y = np.asarray(y, dtype=np.float64)
    top_true = y.max(axis=1)
    grid = GAIN_GRID[GAIN_GRID <= gain_full + 1e-12]

    def both(a):
        cal = bump_calibrate(pred_oob, a)
        return np.mean((cal - y) ** 2), np.abs(top_true - cal.max(axis=1)).mean()

    mse0, pae0 = both(0.0)
    scores = [m / mse0 + p / pae0 for m, p in map(both, grid)]
    return float(grid[int(np.argmin(scores))])


CALIBRATIONS = ("none", "balanced", "full")


class Ensemble:
    """A fitted candidate: member forecasters + the peak-calibration gains."""

    def __init__(self, members, gains, info):
        self.members, self.gains, self.info = members, dict(gains, none=0.0), info

    def predict(self, x: np.ndarray, calibration: str = "balanced") -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean([m.predict(x).astype(np.float64) for m in self.members], axis=0)
        mean = np.maximum(mean, 0.0)
        if self.gains[calibration]:
            mean = bump_calibrate(mean, self.gains[calibration])
        return mean.astype(np.float32)

    def n_parameters(self) -> dict:
        per = self.members[0].n_parameters()
        return {"members": len(self.members), "per_member": per,
                "total": per["total"] * len(self.members) + 1}      # + 1: the calibration gain


@torch.no_grad()
def _out_of_block_forecast(member, x_fit, y_fit, lam, blocks=6) -> np.ndarray:
    """Out-of-block (two-sided, embargoed) forecasts of the fit windows from a member's final features."""
    z, scale = member.features(x_fit)
    target = torch.from_numpy(np.asarray(y_fit, dtype=np.float64)) / scale
    pred, first = out_of_block_prediction(z, target, lam, blocks, "two_sided")
    assert first == 0
    return (pred * scale).numpy()


def fit_candidate(name: str, x_fit, y_fit, seed: int) -> Ensemble:
    """Fit one candidate on the given fit windows (one independent repetition for ``seed``)."""
    mcfg, tcfg, n_members = CANDIDATES[name]
    t0 = time.time()
    members, oob = [], []
    lam = tcfg.get("lam", CVVP_DEFAULTS["lam"])
    for m in range(n_members):
        if name == "ridge11":
            member = RidgeSummaryForecaster(x_fit, y_fit, lam=lam, mcfg=mcfg)
        else:
            member = fit_cvvp(mcfg, tcfg, x_fit, y_fit, seed=1000 * seed + m)
        members.append(member)
        oob.append(_out_of_block_forecast(member, x_fit, y_fit, lam))
    pred_oob = np.maximum(np.mean(oob, axis=0), 0.0)
    gain_full = fit_gain(pred_oob, y_fit)
    gains = {"full": gain_full, "balanced": fit_balanced_gain(pred_oob, y_fit, gain_full)}
    info = {"candidate": name, "seed": seed, "gains": gains, "seconds": time.time() - t0,
            "oob_scaled_mse": float(np.mean((pred_oob - np.asarray(y_fit, dtype=np.float64)) ** 2))}
    return Ensemble(members, gains, info)


def fit(seed: int, candidate: str = RECOMMENDED, calibration: str = "balanced"):
    """Final-fit entry point: train on the TRAINING split only, return ``predict_fn``.

    ``predict_fn(x)`` maps scaled windows ``(n, 528)`` to scaled forecasts ``(n, 132)`` (float32).
    ``calibration``: "none" | "balanced" | "full" (see the module docstring).
    The validation split is not used; the test split is sealed.
    """
    assert calibration in CALIBRATIONS
    d = C.data()
    x_train, y_train = d.split("train")
    ensemble = fit_candidate(candidate, x_train, y_train, seed)

    def predict_fn(x):
        return ensemble.predict(x, calibration=calibration)

    predict_fn.ensemble = ensemble
    return predict_fn


# ---------------------------------------------------------------------------------------------------
# evaluation harness (rolling-origin folds + official validation), restart-safe
# ---------------------------------------------------------------------------------------------------

RESULTS = C.RUNS_DIR / "final" / "results.jsonl"
KEYS = ("scaled_mse", "pae", "pte", "peak_aware_mse")
ALL_FOLDS = ("cv1", "cv2", "cv3", "cv4", "far1", "far2", "far3", "far4", "val")


def _load_results():
    if not RESULTS.is_file():
        return []
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    for r in rows:                      # rows written before the balanced calibration existed
        if "none" not in r:
            r["none"], r["full"], r["gains"] = r.pop("raw"), r.pop("cal"), {"full": r.pop("gain")}
    return rows


def run(candidates, fold_names, seeds, part=(0, 1)):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["candidate"], r["fold"], r["seed"]) for r in _load_results() if "balanced" in r}
    d = C.data()
    folds = C.folds(tuple(fold_names))
    jobs = [(c, f, s) for c in candidates for f in fold_names for s in seeds]
    for j, (cand, f, seed) in enumerate(jobs):
        if j % part[1] != part[0] or (cand, f, seed) in done:
            continue
        fit_idx, ev = folds[f]
        ens = fit_candidate(cand, d.X[fit_idx], d.Y[fit_idx], seed)
        rec = {"candidate": cand, "fold": f, "seed": seed, "gains": ens.gains, "params": ens.n_parameters(),
               "oob_scaled_mse": ens.info["oob_scaled_mse"], "seconds": round(ens.info["seconds"], 1)}
        for cal in CALIBRATIONS:
            m = C.score(ens.predict(d.X[ev], calibration=cal), d.Y[ev])
            rec[cal] = {k: m[k] for k in KEYS}
        with open(RESULTS, "a") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(f"{cand:16s} {f:4s} s{seed} gains {ens.gains['balanced']:.2f}/{ens.gains['full']:.2f} none {C.short(rec['none'])} "
              f"bal {C.short(rec['balanced'])} full {C.short(rec['full'])} params {rec['params']['total']} "
              f"{rec['seconds']:.0f}s", flush=True)


def table(which=CALIBRATIONS, names=None):
    rows = _load_results()
    order = []
    for r in rows:
        if r["candidate"] not in order:
            order.append(r["candidate"])
    names = [n for n in order if names is None or n in names]
    summary = {}
    for kind in which:
        print(f"== calibration {kind}: scaled MSE / PAE / PTE, mean over seeds (n); val: mean +- population std")
        for name in names:
            cells, groups = [], {"cv": [], "far": []}
            for f in ALL_FOLDS:
                rs = [r for r in rows if r["candidate"] == name and r["fold"] == f and kind in r]
                if not rs:
                    cells.append(f"{f} -")
                    continue
                vals = np.array([[r[kind][k] for k in ("scaled_mse", "pae", "pte")] for r in rs])
                mean, std = vals.mean(axis=0), vals.std(axis=0)
                entry = summary.setdefault(name, {}).setdefault(kind, {})
                entry[f] = {"n_seeds": len(rs), "mse": mean[0], "pae": mean[1], "pte": mean[2],
                            "mse_std": std[0], "pae_std": std[1], "pte_std": std[2],
                            "gain": float(np.mean([r["gains"].get(kind, 0.0) for r in rs]))}
                if f == "val":
                    cells.append(f"val {mean[0]:.4f}+-{std[0]:.4f}/{mean[1]:.1f}+-{std[1]:.1f}/{mean[2]:.1f}+-{std[2]:.1f}(n{len(rs)})")
                else:
                    groups[f[:-1]].append(mean)
                    cells.append(f"{mean[0]:.4f}/{mean[1]:.0f}/{mean[2]:.0f}(n{len(rs)})")
            means = []
            for g in ("cv", "far"):
                if len(groups[g]) == 4:
                    gm = np.mean(groups[g], axis=0)
                    summary[name][kind][g + "_mean"] = {"mse": gm[0], "pae": gm[1], "pte": gm[2]}
                    means.append(f"{g}mean {gm[0]:.4f}/{gm[1]:.1f}/{gm[2]:.1f}")
                else:
                    means.append(f"{g}mean -")
            par = next(r["params"]["total"] for r in rows if r["candidate"] == name)
            print(f"{name:18s} par {par:6d} | " + " | ".join(means) + " | " + "  ".join(cells))
    (RESULTS.parent / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", nargs="+", default=[RECOMMENDED])
    r.add_argument("--folds", nargs="+", default=list(ALL_FOLDS))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--part", default="0/1")
    t = sub.add_parser("table")
    t.add_argument("--names", nargs="*", default=None)
    cli = ap.parse_args()
    if cli.cmd == "run":
        run(cli.candidates, cli.folds, cli.seeds, tuple(int(v) for v in cli.part.split("/")))
    else:
        table(names=cli.names)
