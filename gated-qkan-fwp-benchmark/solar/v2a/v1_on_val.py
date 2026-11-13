"""v1 reproduction scored on the VALIDATION split (reference for all v2a candidates).

    python solar/v2a/v1_on_val.py -> runs/solar/v2a/v1_on_val.json

Loads the best-validation checkpoints of the 5-seed CPU reproduction of v1
(runs/solar/reference/seed*/best_model.pth, trained by the authors' trainer)
and scores them with protocol.metrics on the validation windows (and, for
context, on the training windows).  The test split is never touched.
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import analyze_reference as AR  # solar/analyze_reference.py (read-only helper)


def main():
    d = C.data()
    xtr, ytr = d.split("train")
    xva, yva = d.split("val")
    rows, preds = {}, []
    for seed in range(5):
        model = AR.load_reference_model(seed, AR.DEFAULT_OUT_ROOT)
        pva = AR.predict(model, xva)
        ptr = AR.predict(model, xtr)
        preds.append(pva)
        rows[f"seed{seed}"] = {"val": C.score(pva, yva), "train": C.score(ptr, ytr),
                               "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
        print(seed, "val", C.short(rows[f"seed{seed}"]["val"]), "train", C.short(rows[f"seed{seed}"]["train"]))
    agg = {}
    for k in ("scaled_mse", "pae", "pte", "peak_aware_mse"):
        v = np.array([rows[f"seed{s}"]["val"][k] for s in range(5)])
        agg[k] = {"mean": float(v.mean()), "std_population": float(v.std()), "std_sample": float(v.std(ddof=1))}
    rows["aggregate_val"] = agg
    rows["seed_ensemble_val"] = C.score(np.mean(preds, axis=0), yva)
    print("aggregate val:", {k: (round(v["mean"], 4), round(v["std_population"], 4)) for k, v in agg.items()})
    print("5-seed mean forecast on val:", C.short(rows["seed_ensemble_val"]))
    (C.RUNS_DIR / "v1_on_val.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
