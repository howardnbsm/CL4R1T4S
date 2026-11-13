"""Aggregate the CPU reproduction of the authors' model next to paper Table 7.

    /home/user/venvs/gqkan/bin/python solar/collect_reference.py

Reads ``runs/solar/reference/seed<k>/`` (written by ``reference_cpu.py``),
prints the per-seed final test metrics produced by the reference trainer,
their mean +- std over the finished seeds (population std, ddof = 0, and
sample std, ddof = 1) and the paper's numbers, and lists seeds that are still
running.  The aggregate is also written to ``runs/solar/reference/summary.json``.

Std convention: the released reference code contains no aggregation script and
neither the paper nor the README states which estimator was used, so the
paper's convention cannot be determined from the release; both are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_DIR / "runs" / "solar" / "reference"
PAPER_TARGETS = PROJECT_DIR / "paper_targets.json"
METRICS = (("scaled_mse", "scaled MSE", 4), ("pae", "PAE", 2), ("pte", "PTE", 2))


def paper_table7() -> dict:
    """GQKAN-QKANFWP row of Table 7 as ``{metric: (mean, std)}`` plus ``params``."""
    table = json.loads(PAPER_TARGETS.read_text())["solar_table7"]
    row = dict(zip(table["columns"], table["GQKAN-QKANFWP"]))
    return {
        "params": int(row["params"]),
        "scaled_mse": (row["scaled_mse"], row["scaled_mse_std"]),
        "pae": (row["pae"], row["pae_std"]),
        "pte": (row["pte"], row["pte_std"]),
    }


def seed_status(seed_dir: Path, expected_epochs: int) -> dict:
    """Describe one seed directory: finished (with metrics), running or missing."""
    if not seed_dir.is_dir():
        return {"state": "missing", "epochs_logged": 0}
    epochs_logged = 0
    last = None
    train_log = seed_dir / "train_log.csv"
    if train_log.is_file():
        with open(train_log, newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs_logged = len(rows)
        last = rows[-1] if rows else None
    info = {"state": "running", "epochs_logged": epochs_logged}
    if last is not None:
        info["last_train_loss"] = float(last["train_loss"])
        info["last_val_loss"] = float(last["test_loss"])  # the trainer's column name; it is the val loss
    final = seed_dir / "final_metrics.json"
    if final.is_file():
        summary = json.loads(final.read_text())
        if summary.get("final_eval_split", "test") != "test":
            info["state"] = "probe run (final evaluation not on the test split)"
        elif summary["epochs"] != expected_epochs:
            info["state"] = f"finished with {summary['epochs']} epochs (expected {expected_epochs})"
        else:
            info["state"] = "done"
        info["summary"] = summary
    return info


def aggregate(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "std_sample": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    cli = parser.parse_args()

    paper = paper_table7()
    statuses = {seed: seed_status(cli.root / f"seed{seed}", cli.epochs) for seed in cli.seeds}
    done = [seed for seed, info in statuses.items() if info["state"] == "done"]
    pending = [seed for seed in cli.seeds if seed not in done]

    print(f"Reproduction of the authors' GQKAN-QKANFWP on CPU ({cli.root})")
    print(f"{'seed':>4}  {'state':<10} {'epochs':>6}  {'params':>6}  {'best val':>9} {'@ep':>4}  "
          f"{'scaled MSE':>10}  {'PAE':>7}  {'PTE':>7}  {'s/epoch':>7}")
    for seed in cli.seeds:
        info = statuses[seed]
        if info["state"] == "done":
            s = info["summary"]
            f = s["final_eval"]
            print(f"{seed:>4}  {'done':<10} {s['epochs']:>6}  {s['trainable_parameters']:>6}  "
                  f"{s['best_val_loss']:>9.6f} {s['best_val_epoch']:>4}  {f['scaled_mse']:>10.6f}  "
                  f"{f['pae']:>7.2f}  {f['pte']:>7.2f}  {s['median_seconds_per_epoch']:>7.2f}")
        else:
            extra = ""
            if "last_val_loss" in info:
                extra = f"  last train {info['last_train_loss']:.5f} val {info['last_val_loss']:.5f}"
            print(f"{seed:>4}  {info['state']:<10} {info['epochs_logged']:>6}/{cli.epochs}{extra}")

    result = {"root": str(cli.root), "seeds_done": done, "seeds_pending": pending, "paper": paper}
    if done:
        print()
        print(f"Aggregate over {len(done)} finished seed(s) {done}  (paper Table 7: 5 seeds)")
        print(f"{'metric':<11} {'mean':>9}  {'std (pop, ddof=0)':>18}  {'std (sample, ddof=1)':>20}  "
              f"{'paper mean +- std':>20}  {'ours - paper':>12}")
        result["aggregate"] = {}
        for key, label, digits in METRICS:
            agg = aggregate([statuses[seed]["summary"]["final_eval"][key] for seed in done])
            result["aggregate"][key] = agg
            p_mean, p_std = paper[key]
            print(f"{label:<11} {agg['mean']:>9.{digits}f}  {agg['std_population']:>18.{digits}f}  "
                  f"{agg['std_sample']:>20.{digits}f}  {p_mean:>12.{digits}f} +- {p_std:<6.{digits}f} "
                  f"{agg['mean'] - p_mean:>+12.{digits}f}")
        params = sorted({statuses[seed]["summary"]["trainable_parameters"] for seed in done})
        result["trainable_parameters"] = params
        print(f"trainable parameters: {params} (paper: {paper['params']})")
        print("std convention of the paper: not determinable from the release "
              "(no aggregation script); both estimators are shown.")
    if pending:
        print()
        print(f"Seeds not finished yet: {pending} -> re-run this script later.")
    else:
        print()
        print("All requested seeds are finished.")
    cli.root.mkdir(parents=True, exist_ok=True)
    (cli.root / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
