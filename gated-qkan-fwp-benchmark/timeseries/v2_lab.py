"""Development tool for v2: compare configurations on the TRAINING hold-out only.

Every number printed by this script is the MSE on the last 20% of the training split of a
model fitted on the first 80% of the training split.  The test split is never loaded into a
model or a metric here, so anything decided with this tool is free of test-set selection.

Usage
    python v2_lab.py --tag probe1 --configs '{"h16": {"hidden": 16}, "h32": {"hidden": 32}}' \\
        --datasets narma_5 narma_10 --windows 8 16 --seeds 0 1 2

Output: a table (rows = configurations, columns = cells, entries = mean hold-out MSE over
seeds) and ``runs/timeseries/v2_lab/<tag>.json`` with the per-seed values.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import protocol
import v2

OUT_DIR = protocol.RUNS_ROOT / "v2_lab"
HOLDOUT_FRACTION = 0.2


def holdout_mse(config: v2.V2Config, dataset: str, window: int, seed: int) -> dict:
    x_train, y_train, _, _ = protocol.load(dataset, window)  # test arrays deliberately unused
    n_fit = v2.holdout_split(len(y_train), HOLDOUT_FRACTION)
    model, info = v2.fit_auto(config, x_train[:n_fit], y_train[:n_fit], seed)
    prediction = v2.predict(model, x_train[n_fit:])
    error = float(np.mean((prediction - y_train[n_fit:].astype(np.float64)) ** 2))
    return {"holdout_mse": error, "fit_mse": info["train_mse"], "evaluations": info["evaluations"],
            "params": info["params"]["total"], "selected": info.get("selected")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--configs", required=True, help="JSON dict name -> V2Config overrides")
    parser.add_argument("--base", default="{}", help="JSON dict of overrides applied to every configuration")
    parser.add_argument("--datasets", nargs="+", default=list(protocol.PAPER_DATASETS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(protocol.WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args(argv)
    base = json.loads(args.base)
    configs = {name: v2.V2Config(**{**base, **overrides}) for name, overrides in json.loads(args.configs).items()}
    names = list(configs)
    records, start = [], time.perf_counter()
    cells = [(d, w) for d in args.datasets for w in args.windows]
    short = {"bessel_j2": "bes", "damped_shm": "shm", "narma_5": "n5", "narma_10": "n10",
             "delayed_quantum_control": "dqc", "jaynes_cummings": "jc", "transmon": "tr"}
    print("hold-out MSE, mean over seeds %s (rows = configurations, columns = cells)" % args.seeds)
    print("| config | " + " | ".join(f"{short[d]}{w}" for d, w in cells) + " | geo-mean |")
    print("|---|" + "---|" * (len(cells) + 1))
    for name in names:
        row = []
        for dataset, window in cells:
            runs = [holdout_mse(configs[name], dataset, window, seed) for seed in args.seeds]
            values = [r["holdout_mse"] for r in runs]
            records.append({"dataset": dataset, "window": window, "config": name,
                            "config_values": configs[name].to_dict(), "runs": runs,
                            "mean": float(np.mean(values)), "median": float(np.median(values))})
            row.append(float(np.mean(values)))
        print(f"| {name} | " + " | ".join("%.1e" % v for v in row) + " | %.1e |" % np.exp(np.mean(np.log(row))),
              flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(records, indent=1))
    print("seconds %.1f" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
