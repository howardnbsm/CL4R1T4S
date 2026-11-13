"""Run named configurations over rolling-origin folds / seeds; append one JSON line per fit.

    python solar/v2a/run_cv.py --sweep <name> [--folds cv1 cv2 cv3 cv4 val] [--seeds 0 1] [--part i/n]

Sweeps are defined in ``sweeps.py`` (``SWEEPS[name]`` = list of (config_name,
model overrides, train overrides)).  Results go to
``runs/solar/v2a/cv/<sweep>.jsonl``; finished (config, fold, seed) triples are
skipped, so an interrupted sweep can simply be restarted.

In the CV folds the eval block is NEVER used for selection: the reported
``last`` metrics are those of the final-epoch weights.  ``best_epoch`` (the
epoch with the lowest eval MSE) is recorded for diagnosis only.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C
from model import predict_numpy
from sweeps import SWEEPS
from train import fit_model

KEYS = ("scaled_mse", "pae", "pte", "peak_aware_mse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--folds", nargs="+", default=["cv1", "cv2", "cv3", "cv4", "val"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--part", default="0/1", help="i/n: run only jobs with index % n == i")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these config names")
    cli = ap.parse_args()
    part_i, part_n = (int(v) for v in cli.part.split("/"))

    out_dir = C.RUNS_DIR / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cli.sweep}.jsonl"
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["name"], r["fold"], r["seed"]))

    d = C.data()
    folds = C.folds(tuple(cli.folds))
    jobs = [(name, mc, tc, f, s) for (name, mc, tc) in SWEEPS[cli.sweep]
            if (cli.only is None or name in cli.only) for f in cli.folds for s in cli.seeds]
    for j, (name, mc, tc, f, seed) in enumerate(jobs):
        if j % part_n != part_i or (name, f, seed) in done:
            continue
        fit, ev = folds[f]
        t0 = time.time()
        model, info = fit_model(mc, tc, d.X[fit], d.Y[fit], d.X[ev], d.Y[ev], seed=seed)
        last = C.score(predict_numpy(model, d.X[ev]), d.Y[ev])
        fit_m = C.score(predict_numpy(model, d.X[fit]), d.Y[fit])
        hist = info["history"]
        best_ep = int(np.argmin([h["scaled_mse"] for h in hist]))
        rec = {"name": name, "fold": f, "seed": seed, "model": mc, "train": tc,
               "params": info["params_trainable"], "params_data": info["params_data_derived"],
               "last": {k: last[k] for k in KEYS}, "fit": {k: fit_m[k] for k in KEYS},
               "best_epoch": hist[best_ep]["epoch"], "best_epoch_mse": hist[best_ep]["scaled_mse"],
               "curve": [round(h["scaled_mse"], 5) for h in hist[4::5]],
               "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(f"{name:28s} {f:4s} s{seed} params {rec['params']:5d} last {C.short(last)} "
              f"fit {fit_m['scaled_mse']:.4f} best@{rec['best_epoch']} {rec['best_epoch_mse']:.4f} "
              f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
