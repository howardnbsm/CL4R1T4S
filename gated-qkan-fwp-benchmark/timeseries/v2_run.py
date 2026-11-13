"""Final evaluation of the v2 time-series models on the paper's 24 cells, and the report.

Protocol (unchanged from v1): ``protocol.load`` arrays (same data, windows, chronological
80/20 split, scaling), metric = ``protocol.test_mse`` of the finished model, seeds 0-4, mean
+- population std.  For every (cell, seed) the whole procedure ``v2.fit_auto`` is repeated
from scratch on the TRAINING split; the test split is touched exactly once, by ``predict``
after training has finished.

Sub-commands
    run      python v2_run.py run --preset v2_main [--datasets ..] [--windows ..] [--seeds ..] [--jobs 2]
    report   python v2_run.py report      -> timeseries/RESULTS.md, runs/timeseries/v2_final/summary.json

Results: ``runs/timeseries/v2_final/<preset>/cells/<dataset>_w<window>_s<seed>.json``.
Finished cells are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import protocol
import v2

FINAL_ROOT = protocol.RUNS_ROOT / "v2_final"
RESULTS_MD = Path(__file__).resolve().parent / "RESULTS.md"
FOLLOWUP_PATH = protocol.PROJECT_ROOT / "followup_targets.json"
V1_TAG = "gqkan_qkanfwp_paper_recipe"
SHORT = {"bessel_j2": "Bessel J2", "damped_shm": "Damped SHM", "narma_5": "NARMA-5", "narma_10": "NARMA-10",
         "delayed_quantum_control": "Delayed quantum control", "jaynes_cummings": "Jaynes-Cummings"}


def cell_path(preset: str, dataset: str, window: int, seed: int) -> Path:
    return FINAL_ROOT / preset / "cells" / f"{dataset}_w{window}_s{seed}.json"


def run_cell(job) -> str:
    preset, dataset, window, seed = job
    config = v2.PRESETS[preset]
    x_train, y_train, x_test, y_test = protocol.load(dataset, window)
    start = time.perf_counter()
    model, info = v2.fit_auto(config, x_train, y_train, seed)  # training split only
    seconds = time.perf_counter() - start
    test = protocol.test_mse(v2.predict(model, x_test), y_test)  # the paper's metric, evaluated once
    record = {"preset": preset, "dataset": dataset, "window": window, "seed": seed, "config": config.to_dict(),
              "final_test_mse": test, "train_mse": info["train_mse"], "evaluations": info["evaluations"],
              "params": info["params"], "selected": info.get("selected"), "n_train": len(y_train),
              "n_test": len(y_test), "seconds": seconds}
    path = cell_path(preset, dataset, window, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return (f"  {preset:18s} {dataset:24s} w{window:<3d} seed {seed}  test MSE {test:.3e}  evals {info['evaluations']:2d}"
            f"  params {info['params']['total']}  {seconds:.1f}s")


def run(preset: str, datasets, windows, seeds, jobs: int, force: bool) -> None:
    todo = [(preset, d, w, s) for d in datasets for w in windows for s in seeds
            if force or not cell_path(preset, d, w, s).exists()]
    print(f"[{preset}] {len(todo)} cell-seeds to run", flush=True)
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(min(jobs, 2)) as pool:
            for line in pool.imap(run_cell, todo):
                print(line, flush=True)
    else:
        for job in todo:
            print(run_cell(job), flush=True)


# ------------------------------------------------------------------------------ report
def load_preset(preset: str) -> dict:
    """{(dataset, window): {"values": [...], "mean", "std", "params", ...}} for one preset."""
    groups: dict = {}
    for path in sorted((FINAL_ROOT / preset / "cells").glob("*.json")):
        record = json.loads(path.read_text())
        groups.setdefault((record["dataset"], record["window"]), []).append(record)
    table = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r["seed"])
        values = np.array([r["final_test_mse"] for r in records], dtype=float)
        table[key] = {"seeds": [r["seed"] for r in records], "values": values.tolist(),
                      "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                      "params": records[0]["params"]["total"],
                      "params_inner": records[0]["params"]["inner"],
                      "evaluations_max": max(r["evaluations"] for r in records),
                      "selected_ridge": [r["selected"]["ridge"] if r.get("selected") else None for r in records]}
    return table


def followup_best(dataset: str, window: int) -> float:
    data = json.loads(FOLLOWUP_PATH.read_text())
    return float(data["best_per_cell"][dataset][data["windows"].index(int(window))])


def load_v1() -> dict:
    summary = json.loads((protocol.RUNS_ROOT / V1_TAG / "summary.json").read_text())
    return {(row["dataset"], row["window"]): row for row in summary["rows"]}


def load_classical() -> dict:
    records = json.loads((protocol.RUNS_ROOT / "classical_baselines" / "results.json").read_text())
    return {(r["dataset"], r["window"]): r for r in records}


def fmt(mean: float, std: float | None = None) -> str:
    return f"{mean:.2e}" if std is None else f"{mean:.2e} ± {std:.1e}"


def build_summary() -> dict:
    presets = {name: load_preset(name) for name in v2.PRESETS if (FINAL_ROOT / name / "cells").exists()}
    v1, classical = load_v1(), load_classical()
    cells = []
    for dataset, window in protocol.PAPER_CELLS:
        key = (dataset, window)
        entry = {"dataset": dataset, "window": window, "paper_best": protocol.paper_best(dataset, window),
                 "paper_best_model": protocol.paper_best_model(dataset, window),
                 "followup_best": followup_best(dataset, window)}
        if key in v1:
            entry["v1_reproduced"] = {"mean": v1[key]["mean"], "std": v1[key]["std"],
                                      "params": v1[key]["params_effective"], "seeds": v1[key]["seeds"]}
        if key in classical:
            entry["classical_closed_form"] = {"test_mse": classical[key]["test_mse"],
                                              "family": classical[key]["selected_family"],
                                              "params": classical[key]["n_parameters"]}
        for name, table in presets.items():
            if key in table:
                entry[name] = table[key]
        if "v2_main" in entry:
            mean = entry["v2_main"]["mean"]
            worst = max(entry["v2_main"]["values"])
            entry["win_factor_vs_paper_best"] = entry["paper_best"] / mean
            entry["win_factor_vs_followup_best"] = entry["followup_best"] / mean
            entry["beats_paper_best"] = mean < entry["paper_best"]
            entry["beats_followup_best"] = mean < entry["followup_best"]
            entry["worst_seed_beats_both"] = worst < min(entry["paper_best"], entry["followup_best"])
        cells.append(entry)
    done = [c for c in cells if "v2_main" in c]
    summary = {
        "protocol": "arXiv:2605.06734 time-series benchmark: final test MSE, seeds 0-4, mean and population std",
        "presets": {name: v2.PRESETS[name].to_dict() for name in presets},
        "cells": cells,
        "v2_main_cells_evaluated": len(done),
        "v2_main_wins_vs_paper_best": sum(c["beats_paper_best"] for c in done),
        "v2_main_wins_vs_followup_best": sum(c["beats_followup_best"] for c in done),
        "v2_main_cells_where_every_seed_beats_both": sum(c["worst_seed_beats_both"] for c in done),
    }
    return summary


def results_tables(summary: dict) -> str:
    """The per-dataset tables of RESULTS.md (the prose sections are kept in RESULTS_TEMPLATE)."""
    lines = []
    by_dataset: dict = {}
    for cell in summary["cells"]:
        by_dataset.setdefault(cell["dataset"], []).append(cell)
    for dataset in protocol.PAPER_DATASETS:
        lines.append(f"### {SHORT[dataset]} (`{dataset}`)\n")
        lines.append("| N | paper best (model) | follow-up best | v1 reproduced (159 p) | **v2** (params) | "
                     "v2 without QKAN (params) | classical closed form (params) | paper best / v2 | follow-up / v2 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cell in by_dataset[dataset]:
            v1 = cell.get("v1_reproduced")
            main, ablation, classical = cell.get("v2_main"), cell.get("v2_no_qkan"), cell.get("classical_closed_form")
            lines.append("| {n} | {pb:.1e} ({pm}) | {fb:.2e} | {v1} | {main} | {abl} | {cl} | {w1} | {w2} |".format(
                n=cell["window"], pb=cell["paper_best"], pm=cell["paper_best_model"], fb=cell["followup_best"],
                v1=fmt(v1["mean"], v1["std"]) if v1 else "-",
                main=f"**{fmt(main['mean'], main['std'])}** ({main['params']})" if main else "-",
                abl=f"{fmt(ablation['mean'], ablation['std'])} ({ablation['params']})" if ablation else "-",
                cl=f"{classical['test_mse']:.2e} ({classical['params']})" if classical else "-",
                w1=f"{cell['win_factor_vs_paper_best']:.1e}x" if main else "-",
                w2=f"{cell['win_factor_vs_followup_best']:.1e}x" if main else "-"))
        lines.append("")
    return "\n".join(lines)


def ablation_table(summary: dict) -> str:
    names = [n for n in ("v2_main", "v2_no_qkan", "v2_linear_edges", "v2_random_qkan", "v2_tanh_edges",
                         "v2_fit_objective", "v2_fast_measure", "v2_wide", "v2_fwp159", "v2_fwp159_no_program")
             if n in summary["presets"]]
    lines = ["| cell | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for cell in summary["cells"]:
        row = [f"{cell[n]['mean']:.1e} ± {cell[n]['std']:.0e}" if n in cell else "-" for n in names]
        lines.append(f"| {cell['dataset']} N={cell['window']} | " + " | ".join(row) + " |")
    geo = []
    for n in names:
        values = [cell[n]["mean"] for cell in summary["cells"] if n in cell]
        geo.append(f"{np.exp(np.mean(np.log(values))):.2e} ({len(values)} cells)" if values else "-")
    lines.append("| **geometric mean** | " + " | ".join(geo) + " |")
    params = []
    for n in names:
        counts = sorted({cell[n]["params"] for cell in summary["cells"] if n in cell})
        params.append(f"{counts[0]}-{counts[-1]}" if counts else "-")
    lines.append("| parameters (min-max over N) | " + " | ".join(params) + " |")
    return "\n".join(lines)


def report() -> None:
    summary = build_summary()
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    (FINAL_ROOT / "tables.md").write_text(results_tables(summary) + "\n\n" + ablation_table(summary) + "\n")
    template = RESULTS_MD.with_name("RESULTS_TEMPLATE.md")
    if template.exists():
        text = template.read_text()
        text = text.replace("{{TABLES}}", results_tables(summary)).replace("{{ABLATIONS}}", ablation_table(summary))
        text = text.replace("{{WINS_PAPER}}", str(summary["v2_main_wins_vs_paper_best"]))
        text = text.replace("{{WINS_FOLLOWUP}}", str(summary["v2_main_wins_vs_followup_best"]))
        text = text.replace("{{WINS_EVERY_SEED}}", str(summary["v2_main_cells_where_every_seed_beats_both"]))
        text = text.replace("{{CELLS}}", str(summary["v2_main_cells_evaluated"]))
        RESULTS_MD.write_text(text)
    print(f"v2_main: {summary['v2_main_cells_evaluated']} cells; beats paper best in "
          f"{summary['v2_main_wins_vs_paper_best']}, follow-up best in {summary['v2_main_wins_vs_followup_best']}, "
          f"every single seed beats both in {summary['v2_main_cells_where_every_seed_beats_both']}")
    print(f"wrote {FINAL_ROOT / 'summary.json'}, {FINAL_ROOT / 'tables.md'}"
          + (f", {RESULTS_MD}" if template.exists() else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--preset", required=True, choices=sorted(v2.PRESETS))
    run_parser.add_argument("--datasets", nargs="+", default=list(protocol.PAPER_DATASETS))
    run_parser.add_argument("--windows", nargs="+", type=int, default=list(protocol.WINDOWS))
    run_parser.add_argument("--seeds", nargs="+", type=int, default=list(protocol.PAPER_SEEDS))
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.preset, args.datasets, args.windows, args.seeds, args.jobs, args.force)
    else:
        report()


if __name__ == "__main__":
    main()
