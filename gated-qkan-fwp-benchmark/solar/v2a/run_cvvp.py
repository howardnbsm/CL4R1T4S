"""Run CVVP configurations over folds / seeds; one JSON line per fit (same layout as run_cv.py).

    python solar/v2a/run_cvvp.py --sweep <name> [--folds ...] [--seeds ...] [--part i/n]

Sweeps: ``sweeps.CVVP_SWEEPS[name] = [(config_name, model_overrides, cvvp_overrides, n_members), ...]``.
``n_members > 1`` averages the forecasts of that many independently initialised models
(member seeds ``1000 * seed + m``); the reported seed is then one repetition of the whole ensemble.
The eval block of a fold is never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C
from cvvp import fit_cvvp
from sweeps import CVVP_SWEEPS

KEYS = ("scaled_mse", "pae", "pte", "peak_aware_mse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--folds", nargs="+", default=["cv1", "cv2", "cv3", "cv4", "val"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--part", default="0/1")
    cli = ap.parse_args()
    part_i, part_n = (int(v) for v in cli.part.split("/"))
    out = C.RUNS_DIR / "cv" / f"{cli.sweep}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["name"], r["fold"], r["seed"]))
    d = C.data()
    folds = C.folds(tuple(cli.folds))
    jobs = [(name, mc, tc, nm, f, s) for (name, mc, tc, nm) in CVVP_SWEEPS[cli.sweep] for f in cli.folds for s in cli.seeds]
    for j, (name, mc, tc, n_members, f, seed) in enumerate(jobs):
        if j % part_n != part_i or (name, f, seed) in done:
            continue
        fit, ev = folds[f]
        t0 = time.time()
        preds, fit_preds, oob, params = [], [], [], None
        for m in range(n_members):
            fc = fit_cvvp(mc, tc, d.X[fit], d.Y[fit], seed=(seed if n_members == 1 else 1000 * seed + m))
            preds.append(fc.predict(d.X[ev]))
            fit_preds.append(fc.predict(d.X[fit]))
            oob.append(fc.info["oob_mse_final"])
            params = fc.info["params"]
        last = C.score(np.mean(preds, axis=0), d.Y[ev])
        fit_m = C.score(np.mean(fit_preds, axis=0), d.Y[fit])
        rec = {"name": name, "fold": f, "seed": seed, "model": mc, "train": tc, "members": n_members,
               "params": params["total"] * n_members, "params_detail": params,
               "last": {k: last[k] for k in KEYS}, "fit": {k: fit_m[k] for k in KEYS},
               "oob_mse": float(np.mean(oob)), "best_epoch": 0, "best_epoch_mse": float(np.mean(oob)),
               "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(f"{name:26s} {f:4s} s{seed} params {rec['params']:5d} eval {C.short(last)} fit {fit_m['scaled_mse']:.4f} "
              f"oob {rec['oob_mse']:.4f} {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
