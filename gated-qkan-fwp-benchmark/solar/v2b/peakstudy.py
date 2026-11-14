"""Peak-calibration study on all folds (cv1..cv4, val) for the clock-QKAN model.

    python peakstudy.py [--seed 0] -> runs/solar/v2b/peakstudy_seed{seed}.json (+ printed table)

For every fold: fit on the fold's fit windows only, calibration parameters from the
out-of-block forecasts of the fit windows only, score on the fold's evaluation windows.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C
import model as M


def all_feats():
    d = C.data()
    return M.featurize(d.X), d.X.astype(np.float64), d.Y.astype(np.float64), d.Y


def run(seed):
    feats, X, Y, Y32 = all_feats()
    out = {}
    variants = [
        ("none", dict()),
        ("alpha1", dict(alpha=1.0)),
        ("alpha4", dict(alpha=4.0)),
        ("alpha16", dict(alpha=16.0)),
        ("gain", dict(calib="gain", strength="auto")),
        ("head", dict(calib="head", strength="auto")),
        ("head-joint", dict(calib="head", strength="auto", head_w=1.0)),
        ("head-ridge", dict(calib="head", strength="auto", arch="none")),
        ("gain-ridge", dict(calib="gain", strength="auto", arch="none")),
    ]
    for name, cfg in variants:
        out[name] = {}
        for fold, (fit, ev) in C.folds().items():
            t0 = time.time()
            mdl = M.ClockQKAN(**cfg).fit(None, Y[fit], seed=seed, feats=M.subset(feats, fit))
            fe = M.subset(feats, ev)
            row = dict(params=mdl.n_params, g_l1=mdl.g_l1, strength_auto=mdl.strength)
            strengths = (0.0, 0.25, 0.5, 0.75, 1.0) if cfg.get("calib", "none") != "none" else (0.0,)
            for lam in strengths:
                m = C.score(mdl.predict(feats=fe, strength=lam), Y32[ev])
                row[f"s{lam:g}"] = [m["scaled_mse"], m["pae"], m["pte"]]
            m = C.score(mdl.predict(feats=fe), Y32[ev])
            row["auto"] = [m["scaled_mse"], m["pae"], m["pte"]]
            if mdl.oof_head is not None:
                # quality of the peak-height head itself (MAE of the predicted monthly maximum, SSN units)
                _, curve, head = mdl.predict(feats=fe, return_parts=True)
                scale = float(C.data().inverse(np.array([[1.0]]))[0, 0] - C.data().inverse(np.array([[0.0]]))[0, 0])
                row["head_mae_ssn"] = float(np.abs(head - Y[ev].max(1)).mean() * scale)
            row["sec"] = round(time.time() - t0, 1)
            out[name][fold] = row
            print(f"{name:11s} {fold} g_l1 {row['g_l1']:.2f} auto-strength {row['strength_auto']:.2f} | "
                  + " | ".join(f"{k} {v[0]:.4f}/{v[1]:5.1f}/{v[2]:4.1f}" for k, v in row.items()
                               if isinstance(v, list)), flush=True)
    (C.RUNS_DIR / f"peakstudy_seed{seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
