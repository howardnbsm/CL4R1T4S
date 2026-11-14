"""Final-fit entry point of solar v2 approach B and its evaluation runner.

THE entry point for the later sealed evaluation::

    import final                        # solar/v2b on sys.path
    predict_fn = final.fit(seed)        # the recommended candidate (RECOMMENDED below)
    forecasts  = predict_fn(windows)    # (n, 528) scaled windows -> (n, 132) scaled forecasts, float32

``fit`` reads the TRAINING split only (through the sealed ``protocol.load()``); the
validation split is not used by the procedure at all (no checkpoint selection is
needed: the read-out is closed form and the QKAN layer is trained for a fixed 60
full-batch steps on a blocked cross-validation objective inside the training split;
the calibration gain is fitted on out-of-block forecasts of the training windows).
Each seed is an independent repetition of the whole procedure (the seed sets the
DARUAN initialisation).

Runner::

    python final.py --run cv     # seeds 0-4 x folds cv1..cv4,val x architectures -> runs/solar/v2b/final/cv.jsonl
    python final.py --run val    # official protocol through fit(seed) + protocol.evaluate_split -> final/val.json
    python final.py --table      # print / write final/table.txt
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

import common as C
import model as M

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")

# variant -> (architecture of the nonlinear feature layer, calibration strength)
#   architecture: "daruan" (the model), "none" (QKAN removed = ridge skeleton),
#                 "linear" (QKAN replaced by a linear map of the clock angle), "tanh" (classical tanh layer)
#   strength: 0 = uncalibrated curve, "auto" = joint MSE/PAE criterion, 1 = full L1-optimal gain
VARIANTS = {
    "clock": ("daruan", 0.0), "clock+cal": ("daruan", "auto"), "clock+calfull": ("daruan", 1.0),
    "ridge": ("none", 0.0), "ridge+cal": ("none", "auto"), "ridge+calfull": ("none", 1.0),
    "linear": ("linear", 0.0), "linear+cal": ("linear", "auto"),
    "tanh": ("tanh", 0.0), "tanh+cal": ("tanh", "auto"),
}
RECOMMENDED = "clock+cal"
SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = C.RUNS_DIR / "final"

_TRAIN_FEATS = None


def _train_split():
    global _TRAIN_FEATS
    d = C.data()
    xtr, ytr = d.split("train")
    if _TRAIN_FEATS is None:
        _TRAIN_FEATS = M.featurize(xtr)
    return _TRAIN_FEATS, ytr.astype(np.float64)


def fit_model(seed, variant=RECOMMENDED, **overrides):
    """Fit one independent repetition on the training split; returns the ``ClockQKAN`` object."""
    arch, strength = VARIANTS[variant]
    feats, ytr = _train_split()
    cfg = dict(arch=arch, calib="none" if strength == 0 else "gain", strength=strength if strength != 0 else 1.0)
    cfg.update(overrides)
    return M.ClockQKAN(**cfg).fit(None, ytr, seed=seed, feats=feats)


def fit(seed, variant=RECOMMENDED):
    """fit(seed) -> predict_fn mapping scaled windows (n, 528) to scaled forecasts (n, 132)."""
    mdl = fit_model(seed, variant)

    def predict_fn(x):
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2 and x.shape[1] == C.SEQ_LEN, x.shape
        return mdl.predict(x).astype(np.float32)

    predict_fn.model = mdl
    predict_fn.variant = variant
    return predict_fn


# ----------------------------------------------------------------------------------- runners
def run_cv(seeds=SEEDS):
    """All folds, all architectures; one fit per (arch, fold, seed), scored at strengths 0 / auto / 1."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "cv.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["arch"], r["fold"], r["seed"]))
    d = C.data()
    feats, Y, Y32 = M.featurize(d.X), d.Y.astype(np.float64), d.Y
    for arch in ("daruan", "none", "linear", "tanh"):
        for fold, (fit_idx, ev) in C.folds().items():
            for seed in seeds:
                if (arch, fold, seed) in done or (arch == "none" and seed != seeds[0]):
                    continue  # the ridge skeleton is deterministic
                t0 = time.time()
                mdl = M.ClockQKAN(arch=arch, calib="gain", strength="auto").fit(
                    None, Y[fit_idx], seed=seed, feats=M.subset(feats, fit_idx))
                fe = M.subset(feats, ev)
                row = dict(arch=arch, fold=fold, seed=seed, params=mdl.n_params, g_l1=mdl.g_l1,
                           strength_auto=mdl.strength, gain_auto=1 + mdl.strength * (mdl.g_l1 - 1))
                for key, lam in (("uncal", 0.0), ("auto", None), ("full", 1.0)):
                    m = C.score(mdl.predict(feats=fe, strength=lam), Y32[ev])
                    row[key] = [m["scaled_mse"], m["pae"], m["pte"]]
                row["sec"] = round(time.time() - t0, 1)
                with open(path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{arch:7s} {fold} s{seed} p {row['params']} g {row['g_l1']:.2f} str {row['strength_auto']:.2f} "
                      + " ".join(f"{k} {row[k][0]:.4f}/{row[k][1]:5.1f}/{row[k][2]:4.1f}" for k in ("uncal", "auto", "full")),
                      flush=True)


def run_val(seeds=SEEDS, variants=tuple(VARIANTS)):
    """Official protocol: fit(seed) on the training split, score the validation split through the protocol door."""
    import protocol
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for variant in variants:
        rows = []
        for seed in seeds:
            t0 = time.time()
            predict_fn = fit(seed, variant)
            res = protocol.evaluate_split(predict_fn, "val", tag=f"v2b-{variant}-seed{seed}")
            mdl = predict_fn.model
            rows.append(dict(seed=seed, params=mdl.n_params, g_l1=mdl.g_l1, strength=mdl.strength,
                             gain=1 + mdl.strength * (mdl.g_l1 - 1), sec=round(time.time() - t0, 1),
                             **{k: res[k] for k in ("scaled_mse", "scaled_mae", "scaled_r2", "pae", "pte", "peak_aware_mse")}))
            print(f"{variant:14s} s{seed} val {res['scaled_mse']:.5f}/{res['pae']:.2f}/{res['pte']:.2f} "
                  f"gain {rows[-1]['gain']:.3f} p {mdl.n_params}", flush=True)
        agg = {}
        for k in ("scaled_mse", "pae", "pte", "peak_aware_mse"):
            v = np.array([r[k] for r in rows])
            agg[k] = dict(mean=float(v.mean()), std_population=float(v.std()), std_sample=float(v.std(ddof=1)))
        out[variant] = dict(seeds=rows, aggregate=agg, params=rows[0]["params"])
    (OUT_DIR / "val.json").write_text(json.dumps(out, indent=1))


def table():
    lines = []
    rows = [json.loads(line) for line in open(OUT_DIR / "cv.jsonl")]
    lines.append("Rolling-origin CV + official validation, mean over seeds 0-4 (scaled MSE / PAE / PTE)")
    lines.append(f"{'model':26s} " + " ".join(f"{f:>17s}" for f in C.ALL_NAMES) + "   CV-mean MSE/PAE/PTE   params")
    names = {"daruan": "clock-QKAN", "none": "ridge (QKAN removed)", "linear": "QKAN -> linear map", "tanh": "QKAN -> tanh layer"}
    for arch in ("daruan", "none", "linear", "tanh"):
        for key, lab in (("uncal", ""), ("auto", " +cal(auto)"), ("full", " +cal(full)")):
            cells, cv = [], []
            for f in C.ALL_NAMES:
                v = np.array([r[key] for r in rows if r["arch"] == arch and r["fold"] == f])
                if len(v) == 0:
                    cells.append(" " * 17)
                    continue
                mu = v.mean(0)
                cells.append(f"{mu[0]:.4f}/{mu[1]:5.1f}/{mu[2]:4.1f}")
                if f != "val":
                    cv.append(mu)
            cvm = np.mean(cv, axis=0)
            p = [r["params"] for r in rows if r["arch"] == arch][0]
            lines.append(f"{names[arch] + lab:26s} " + " ".join(f"{c:>17s}" for c in cells)
                         + f"   {cvm[0]:.4f}/{cvm[1]:5.1f}/{cvm[2]:4.1f}   {p}")
    g = {f: np.mean([r["gain_auto"] for r in rows if r["arch"] == "daruan" and r["fold"] == f]) for f in C.ALL_NAMES}
    lines.append("auto gain (clock-QKAN): " + " ".join(f"{f} {g[f]:.3f}" for f in C.ALL_NAMES))
    vp = OUT_DIR / "val.json"
    if vp.exists():
        val = json.loads(vp.read_text())
        lines.append("")
        lines.append("Official validation split through fit(seed) + protocol.evaluate_split, seeds 0-4: mean +- sample std (population std)")
        for variant, r in val.items():
            a = r["aggregate"]
            lines.append(f"{variant:14s} MSE {a['scaled_mse']['mean']:.5f} +- {a['scaled_mse']['std_sample']:.5f} ({a['scaled_mse']['std_population']:.5f})"
                         f"  PAE {a['pae']['mean']:6.2f} +- {a['pae']['std_sample']:.2f} ({a['pae']['std_population']:.2f})"
                         f"  PTE {a['pte']['mean']:5.2f} +- {a['pte']['std_sample']:.2f} ({a['pte']['std_population']:.2f})"
                         f"  params {r['params']}")
    text = "\n".join(lines)
    (OUT_DIR / "table.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["cv", "val"])
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.run == "cv":
        run_cv()
    elif args.run == "val":
        run_val()
    if args.table:
        table()
