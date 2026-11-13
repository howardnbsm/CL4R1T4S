"""Summarise a lab sweep: per-fold mean MSE over seeds and the ratio to the ridge skeleton.

    python summarize.py SWEEP [--top N]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--key", default="mse")
    args = ap.parse_args()
    rows = defaultdict(lambda: defaultdict(list))
    base = {}
    params = {}
    for line in open(C.RUNS_DIR / "cv" / f"{args.sweep}.jsonl"):
        r = json.loads(line)
        rows[r["name"]][r["fold"]].append(r[args.key])
        base[r["fold"]] = r["base_mse"]
        params[r["name"]] = r["params"]
    table = []
    for name, by_fold in rows.items():
        if not all(f in by_fold for f in C.ALL_NAMES):
            continue
        means = [float(np.mean(by_fold[f])) for f in C.ALL_NAMES]
        ratio = [m / base[f] for m, f in zip(means, C.ALL_NAMES)]
        nseed = min(len(by_fold[f]) for f in C.ALL_NAMES)
        table.append((float(np.mean(ratio)), name, means, ratio, nseed))
    table.sort()
    if args.top:
        table = table[:args.top]
    print(f"{'ridge skeleton':44s} " + " ".join(f"{base[f]:.4f}" for f in C.ALL_NAMES)
          + f"  cv {np.mean([base[f] for f in C.CV_NAMES]):.4f}")
    print(f"{'name':44s} " + " ".join(f"{f:>6s}" for f in C.ALL_NAMES) + "  cv-mean  mean-ratio  worst-ratio  n  params")
    for mr, name, means, ratio, nseed in table:
        print(f"{name:44s} " + " ".join(f"{m:.4f}" for m in means)
              + f"  {np.mean(means[:4]):.4f}   {mr:.3f}      {max(ratio):.3f}     {nseed}  {params[name]}")


if __name__ == "__main__":
    main()
