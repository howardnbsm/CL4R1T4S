"""Ratio-to-ridge table of CVVP sweeps: mean over seeds of scaled MSE / ridge11 MSE per fold.

    python solar/v2a/summarize_ratio.py <sweep> [<sweep> ...]
"""
import collections
import json
import sys

import numpy as np

import common as C

FOLDS = ("cv1", "cv2", "cv3", "cv4", "far1", "far2", "far3", "far4", "val")
# ridge11 (lam 0.03) scaled MSE per fold, from final.py run --candidates ridge11
RIDGE = {"cv1": 0.008852, "cv2": 0.007560, "cv3": 0.016038, "cv4": 0.021053, "far1": 0.008698, "far2": 0.007628,
         "far3": 0.015998, "far4": 0.021869, "val": 0.012930}

if __name__ == "__main__":
    print(f"{'config (MSE ratio to ridge11)':30s} n  par  " + " ".join(f"{f:>6s}" for f in FOLDS) + "   cv    far   | val MSE/PAE/PTE")
    for sweep in sys.argv[1:]:
        rows = [json.loads(line) for line in (C.RUNS_DIR / "cv" / f"{sweep}.jsonl").read_text().splitlines() if line.strip()]
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[r["name"]][r["fold"]].append(r["last"])
        for name, v in by.items():
            ratio = {f: np.mean([m["scaled_mse"] for m in v[f]]) / RIDGE[f] for f in FOLDS if v.get(f)}
            cv = np.mean([ratio[f] for f in FOLDS[:4]]) if all(f in ratio for f in FOLDS[:4]) else float("nan")
            far = np.mean([ratio[f] for f in FOLDS[4:8]]) if all(f in ratio for f in FOLDS[4:8]) else float("nan")
            val = v.get("val")
            vs = (f"{np.mean([m['scaled_mse'] for m in val]):.4f}/{np.mean([m['pae'] for m in val]):.1f}/"
                  f"{np.mean([m['pte'] for m in val]):.1f}") if val else "-"
            par = next(r["params"] for r in rows if r["name"] == name)
            print(f"{name:30s} {max(len(x) for x in v.values())} {par:5d} " +
                  " ".join(f"{ratio[f]:6.3f}" if f in ratio else "     -" for f in FOLDS) + f"  {cv:5.3f} {far:5.3f} | {vs}")
