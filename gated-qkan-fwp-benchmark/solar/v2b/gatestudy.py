"""Gated clock-QKAN probe: QKAN features additionally multiplied by an amplitude feature (three pre-declared choices).

    python gatestudy.py -> runs/solar/v2b/gatestudy.json
"""
import json
import warnings

import numpy as np

import common as C
import model as M

warnings.filterwarnings("ignore")
d = C.data()
feats, Y, Y32 = M.featurize(d.X), d.Y.astype(np.float64), d.Y
out = {}
for gate in ("none", "lvl", "cyc", "max"):
    for m in (4, 2):
        name = f"gate-{gate}-m{m}"
        out[name] = {}
        for fold, (fit, ev) in C.folds().items():
            v = []
            for seed in (0, 1, 2):
                mdl = M.ClockQKAN(gate=gate, m=m).fit(None, Y[fit], seed=seed, feats=M.subset(feats, fit))
                sc = C.score(mdl.predict(feats=M.subset(feats, ev)), Y32[ev])
                v.append([sc["scaled_mse"], sc["pae"], sc["pte"]])
            out[name][fold] = np.mean(v, axis=0).tolist()
        cv = np.mean([out[name][f] for f in C.CV_NAMES], axis=0)
        print(f"{name:14s} " + " ".join(f"{out[name][f][0]:.4f}/{out[name][f][1]:5.1f}/{out[name][f][2]:4.1f}" for f in C.ALL_NAMES)
              + f" | CV-mean {cv[0]:.4f}/{cv[1]:5.1f}/{cv[2]:4.1f} p {mdl.n_params}", flush=True)
(C.RUNS_DIR / "gatestudy.json").write_text(json.dumps(out, indent=1))
