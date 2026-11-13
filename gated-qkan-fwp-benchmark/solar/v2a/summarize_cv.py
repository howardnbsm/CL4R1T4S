"""Compact table of a CV sweep: per fold mean over seeds of the LAST-epoch metrics.

    python solar/v2a/summarize_cv.py <sweep> [<sweep> ...] [--best]

Columns: config, params, mean MSE over the 4 CV folds, worst CV fold MSE, then
MSE/PAE/PTE per fold (mean over the seeds available; n = number of seeds).
``--best`` additionally shows the oracle best-epoch MSE (diagnostic only).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

import common as C

FOLDS = ("cv1", "cv2", "cv3", "cv4", "val")


def load(sweep):
    path = C.RUNS_DIR / "cv" / f"{sweep}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def table(rows, show_best=False):
    by = defaultdict(lambda: defaultdict(list))
    params = {}
    order = []
    for r in rows:
        if r["name"] not in params:
            order.append(r["name"])
        by[r["name"]][r["fold"]].append(r)
        params[r["name"]] = r["params"]
    out = []
    for name in order:
        cells, cv_mse, cv_pae, cv_pte = [], [], [], []
        nseed = 0
        for f in FOLDS:
            rs = by[name].get(f, [])
            if not rs:
                cells.append(" " * 17)
                continue
            nseed = max(nseed, len(rs))
            mse = np.mean([r["last"]["scaled_mse"] for r in rs])
            pae = np.mean([r["last"]["pae"] for r in rs])
            pte = np.mean([r["last"]["pte"] for r in rs])
            if f != "val":
                cv_mse.append(mse), cv_pae.append(pae), cv_pte.append(pte)
            cell = f"{mse:.4f}/{pae:5.1f}/{pte:4.1f}"
            if show_best:
                cell += f"[{np.mean([r['best_epoch_mse'] for r in rs]):.4f}]"
            cells.append(cell)
        head = f"{name:30s} {params[name]:5d} n{nseed} "
        if cv_mse:
            head += f"cv {np.mean(cv_mse):.4f}/{np.mean(cv_pae):5.1f}/{np.mean(cv_pte):4.1f} max {np.max(cv_mse):.4f} | "
        else:
            head += " " * 44
        out.append(head + "  ".join(cells))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    best = "--best" in sys.argv
    print(f"{'config':30s} {'par':>5s} n  {'cv mean mse/pae/pte':22s} {'max':10s} | " + "  ".join(f"{f:^17s}" for f in FOLDS))
    for sweep in args:
        for line in table(load(sweep), best):
            print(line)
