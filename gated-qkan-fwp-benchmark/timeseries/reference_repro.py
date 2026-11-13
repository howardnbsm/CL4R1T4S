"""Reproduce the paper's time-series numbers by running the AUTHORS' train.py unchanged.

The reference checkout is read-only, but the authors' ``train.py`` roots every
output directory at its own location (``Path(__file__).resolve().parent``) and
rejects absolute ``--save_dir`` values.  To keep the reference tree untouched we
execute the unmodified source text of their ``train.py`` in a child process with
``__file__`` pointing into ``runs/timeseries/reference/`` (no file is copied, and
``PYTHONPATH`` makes ``import src...`` resolve to the reference package).  Their
code therefore runs byte-for-byte as published and writes below our runs dir.

Sub-commands
    param-counts   effective trainable parameters of the four reference models
    run            run a grid of (dataset, window, seed) cells, resumable
    time-epoch     measure the cost of one epoch for a cell
    report         final-epoch test MSE, mean +- std, next to the paper
    compare        seed-for-seed comparison of the authors' runs with a train.py tag

Examples
    python reference_repro.py param-counts
    python reference_repro.py run --datasets bessel_j2 narma_5 --windows 16 --seeds 0 1 2 3 4
    python reference_repro.py report
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

import protocol

REFERENCE_TRAIN = protocol.REFERENCE_ROOT / "train.py"
OUT_ROOT = protocol.RUNS_ROOT / "reference"
SAVE_DIR = "results"  # the authors' --save_dir, relative to OUT_ROOT

# --model flag -> name used in the paper's tables
PAPER_NAME = {
    "gqkan_qkanfwp": "GQKAN-QKANFWP",
    "gqkanfwp": "G-QKANFWP",
    "gqkan_fwp": "GQKAN-FWP",
    "gqkan_qfwp": "GQKAN-QFWP",
}

# The configuration that reproduces Table 2's parameter counts (see param-counts).
PAPER_HIDDEN_SIZE = 8
PAPER_QNN_DEPTH = 2

# Executes the unmodified reference source with __file__ redirected to our runs dir.
_LAUNCHER = (
    "import sys\n"
    "source_path, fake_file = sys.argv[1], sys.argv[2]\n"
    "sys.argv = [fake_file] + sys.argv[3:]\n"
    "with open(source_path) as f:\n"
    "    code = compile(f.read(), fake_file, 'exec')\n"
    "exec(code, {'__name__': '__main__', '__file__': fake_file})\n"
)


# ------------------------------------------------------------------ running cells
def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(protocol.REFERENCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # never write __pycache__ into the reference tree
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def reference_command(model, dataset, window, seed, epochs, hidden_size, qnn_depth,
                      batch_size=protocol.PAPER_BATCH_SIZE, lr=protocol.PAPER_LR, nice=True):
    """Command line that runs the authors' train.py for one cell."""
    cmd = ["nice", "-n", "19"] if nice else []
    cmd += [
        sys.executable, "-c", _LAUNCHER, str(REFERENCE_TRAIN), str(OUT_ROOT / "train.py"),
        "--epochs", str(epochs), "--lr", str(lr), "--model", model, "--dataset", dataset,
        "--window_len", str(window), "--input_size", "1", "--hidden_size", str(hidden_size),
        "--qnn_depth", str(qnn_depth), "--batch_size", str(batch_size), "--seed", str(seed),
        "--device", "cpu", "--save_dir", SAVE_DIR,
        "--exp_name", f"repro_{dataset}_{model}_w{window}_s{seed}_e{epochs}",
    ]
    return cmd


def cell_dir(model, dataset, window, seed, hidden_size, qnn_depth) -> Path:
    """Directory the authors' build_result_path uses for a cell (without RUN_<timestamp>)."""
    return (OUT_ROOT / SAVE_DIR / f"DATASET_{dataset}" / f"MODEL_{model}" / f"HIDDEN_SIZE_{hidden_size}"
            / f"QNN_DEPTH_{qnn_depth}" / f"SEQ_LEN_{window}" / f"SEED_{seed}")


def read_train_log(run_path: Path):
    """Rows of the authors' train_log.csv as (epoch, train_loss, test_loss)."""
    path = run_path / "train_log.csv"
    if not path.is_file():
        return []
    with open(path) as f:
        return [(int(r["epoch"]), float(r["train_loss"]), float(r["test_loss"])) for r in csv.DictReader(f)]


def _run_args(run_path: Path) -> dict:
    with open(run_path / "args.json") as f:
        return json.load(f)


def find_completed_run(model, dataset, window, seed, epochs, hidden_size, qnn_depth):
    """Latest run directory of this cell that was launched with ``epochs`` and finished."""
    base = cell_dir(model, dataset, window, seed, hidden_size, qnn_depth)
    if not base.is_dir():
        return None
    for run_path in sorted(base.glob("RUN_*"), reverse=True):
        try:
            run_args = _run_args(run_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows = read_train_log(run_path)
        if run_args.get("epochs") == epochs and len(rows) == epochs:
            return run_path
    return None


def trainable_params_from_log(run_path: Path):
    """The 'Trainable parameters' line the authors' trainer writes after epoch 1."""
    log = run_path / "log.txt"
    if log.is_file():
        match = re.search(r"Trainable parameters: ([\d,]+)", log.read_text())
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def epoch_seconds_from_log(run_path: Path):
    """Seconds between consecutive 'Epoch k' lines of the authors' log.txt."""
    stamps = []
    for line in (run_path / "log.txt").read_text().splitlines():
        match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) - Epoch \d+:", line)
        if match:
            stamps.append(time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")) + int(match.group(2)) / 1000)
    return [b - a for a, b in zip(stamps[:-1], stamps[1:])]


def _allowed_jobs(default: int) -> int:
    """Concurrency limit; ``runs/timeseries/reference/jobs.txt`` overrides it while running.

    Lets a long detached sweep be throttled (e.g. ``echo 1 > jobs.txt``) without a restart.
    """
    try:
        return max(0, int((OUT_ROOT / "jobs.txt").read_text().strip()))
    except (OSError, ValueError):
        return default


def cmd_run(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "console").mkdir(exist_ok=True)
    todo = []
    for dataset in args.datasets:
        for window in args.windows:
            for seed in args.seeds:
                done = find_completed_run(args.model, dataset, window, seed, args.epochs,
                                          args.hidden_size, args.qnn_depth)
                if done is None or args.force:
                    todo.append((dataset, window, seed))
    print(f"[run] {len(todo)} cells to run with {args.jobs} job(s); outputs under {OUT_ROOT}", flush=True)
    running = []  # (Popen, cell, start time, console file)
    while todo or running:
        while todo and len(running) < _allowed_jobs(args.jobs):
            dataset, window, seed = todo.pop(0)
            cmd = reference_command(args.model, dataset, window, seed, args.epochs,
                                    args.hidden_size, args.qnn_depth)
            console = open(OUT_ROOT / "console" / f"{dataset}_{args.model}_w{window}_s{seed}_e{args.epochs}.log", "w")
            proc = subprocess.Popen(cmd, cwd=OUT_ROOT, env=child_env(), stdout=console, stderr=subprocess.STDOUT)
            running.append((proc, (dataset, window, seed), time.time(), console))
            print(f"[run] started {dataset} w{window} seed{seed} (pid {proc.pid})", flush=True)
        time.sleep(2.0)
        for item in list(running):
            proc, (dataset, window, seed), start, console = item
            if proc.poll() is None:
                continue
            running.remove(item)
            console.close()
            done = find_completed_run(args.model, dataset, window, seed, args.epochs,
                                      args.hidden_size, args.qnn_depth)
            final = read_train_log(done)[-1][2] if done else float("nan")
            print(f"[run] finished {dataset} w{window} seed{seed} rc={proc.returncode} "
                  f"in {time.time() - start:.0f}s final test MSE {final:.3e}", flush=True)
    print("[run] all done", flush=True)


# ------------------------------------------------------------------- param counts
def load_reference_train_module():
    """Import the authors' train.py as a module (its ``main`` is not executed)."""
    protocol.add_reference_to_path()
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("reference_train", REFERENCE_TRAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_reference_model(model, hidden_size, qnn_depth, reference_train=None):
    """Build a reference model exactly as the authors' ``make_model`` does."""
    reference_train = reference_train or load_reference_train_module()
    args = types.SimpleNamespace(model=model, input_size=1, hidden_size=hidden_size, output_size=1,
                                 qnn_depth=qnn_depth, device="cpu")
    return reference_train.make_model(args)


def count_parameters(model, window=8):
    """(total, requires_grad, effective): effective uses the authors' own counter.

    The authors log "Trainable parameters" as the number of parameters that
    received a gradient (``p.grad is not None``) after a training step; that is
    the count that matches Table 2.
    """
    import torch
    from src.trainers.train_loading import count_effective_trainable  # the authors' code

    out = model(torch.randn(4, window, 1))[-1]
    out.sum().backward()
    total, effective = count_effective_trainable(model)
    requires_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, requires_grad, effective


def cmd_param_counts(args):
    import torch

    torch.set_num_threads(1)
    reference_train = load_reference_train_module()
    table2 = protocol.paper_param_counts()
    rows = []
    for model in PAPER_NAME:
        for hidden_size in args.hidden_sizes:
            for qnn_depth in (args.qnn_depths if model == "gqkan_qfwp" else args.qnn_depths[:1]):
                try:
                    net = make_reference_model(model, hidden_size, qnn_depth, reference_train)
                    total, requires_grad, effective = count_parameters(net)
                except Exception as error:  # e.g. gqkan_qfwp needs hidden_size >= 4 qubits
                    rows.append((model, hidden_size, qnn_depth, None, None, None, type(error).__name__))
                    continue
                rows.append((model, hidden_size, qnn_depth, total, requires_grad, effective, ""))
    print(f"{'--model':14s} {'paper name':14s} {'hidden':>6s} {'depth':>5s} {'total':>6s} {'req_grad':>8s} "
          f"{'effective':>9s} {'Table 2':>7s}  match")
    for model, hidden_size, qnn_depth, total, requires_grad, effective, error in rows:
        target = table2[PAPER_NAME[model]]
        if total is None:
            print(f"{model:14s} {PAPER_NAME[model]:14s} {hidden_size:6d} {qnn_depth:5d}   failed: {error}")
            continue
        depth = str(qnn_depth) if model == "gqkan_qfwp" else "(any)"
        print(f"{model:14s} {PAPER_NAME[model]:14s} {hidden_size:6d} {depth:>5s} {total:6d} {requires_grad:8d} "
              f"{effective:9d} {target:7d}  {'YES' if effective == target else 'no'}")
    print("\nSettings (hidden_size, qnn_depth) for which ALL four models match Table 2:")
    for hidden_size in args.hidden_sizes:
        for qnn_depth in args.qnn_depths:
            ok = True
            for model in PAPER_NAME:
                hits = [r for r in rows if r[0] == model and r[1] == hidden_size
                        and (r[2] == qnn_depth or model != "gqkan_qfwp")]
                ok = ok and bool(hits) and hits[0][5] == table2[PAPER_NAME[model]]
            if ok:
                print(f"  --hidden_size {hidden_size} --qnn_depth {qnn_depth}")


# --------------------------------------------------------------------- time-epoch
def clean_epoch_seconds(run_path: Path) -> float:
    """Median duration of an epoch (training + test evaluation) from the authors' log.

    The interval that follows a prediction dump (after epochs 1, 15, 30, 50) also contains
    that dump, so those intervals are left out.
    """
    intervals = epoch_seconds_from_log(run_path)  # intervals[k] = time from epoch k+1 to epoch k+2
    clean = [seconds for k, seconds in enumerate(intervals) if (k + 1) not in (1, 15, 30, 50, 100)]
    return float(np.median(clean)) if clean else float("nan")


def cmd_time_epoch(args):
    if args.epochs < 3:
        raise SystemExit("time-epoch needs --epochs >= 3 (the interval after epoch 1 contains a prediction dump)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for window in args.windows:
            cmd = reference_command(args.model, dataset, window, 0, args.epochs, args.hidden_size,
                                    args.qnn_depth, nice=not args.no_nice)
            start = time.time()
            subprocess.run(cmd, cwd=OUT_ROOT, env=child_env(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            wall = time.time() - start
            run_path = find_completed_run(args.model, dataset, window, 0, args.epochs,
                                          args.hidden_size, args.qnn_depth)
            epoch = clean_epoch_seconds(run_path)
            dump = epoch_seconds_from_log(run_path)[0] - epoch  # extra time of the epoch-1 prediction dump
            print(f"{dataset} w{window}: {args.epochs} epochs took {wall:.1f}s wall; one epoch (train + test eval) "
                  f"{epoch:.1f}s, one prediction dump {dump:.1f}s; estimated 50-epoch run "
                  f"{(50 * epoch + 4 * dump) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------- report
def collect(model, datasets, windows, seeds, epochs, hidden_size, qnn_depth):
    """{(dataset, window): {seed: final-epoch test MSE}} for all finished runs."""
    out = {}
    for dataset in datasets:
        for window in windows:
            finals = {}
            for seed in seeds:
                run_path = find_completed_run(model, dataset, window, seed, epochs, hidden_size, qnn_depth)
                if run_path is not None:
                    finals[seed] = read_train_log(run_path)[-1][2]
            if finals:
                out[(dataset, window)] = finals
    return out


def cmd_report(args):
    results = collect(args.model, args.datasets, args.windows, args.seeds, args.epochs,
                      args.hidden_size, args.qnn_depth)
    paper_name = PAPER_NAME[args.model]
    print(f"Authors' train.py, model {args.model} ({paper_name}), hidden_size {args.hidden_size}, "
          f"qnn_depth {args.qnn_depth}, epochs {args.epochs}, batch 4, lr 1e-3; FINAL-epoch test MSE")
    print(f"{'dataset':24s} {'N':>3s} {'seeds':>5s} {'ours mean':>10s} {'std(pop)':>10s} {'std(n-1)':>10s} "
          f"{'paper mean':>10s} {'paper std':>10s} {'ours/paper':>10s}  per-seed")
    summary = {}
    for (dataset, window), finals in results.items():
        values = np.array([finals[s] for s in sorted(finals)])
        mean = float(values.mean())
        std_pop = float(values.std(ddof=0))
        std_sample = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        if protocol.is_paper_cell(dataset, window) and paper_name in protocol.paper_results(dataset, window):
            paper_mean, paper_std = protocol.paper_model(paper_name, dataset, window)
            ratio = f"{mean / paper_mean:10.2f}"
        else:
            paper_mean = paper_std = float("nan")
            ratio = f"{'-':>10s}"
        per_seed = " ".join(f"{v:.2e}" for v in values)
        print(f"{dataset:24s} {window:3d} {len(values):5d} {mean:10.6f} {std_pop:10.6f} {std_sample:10.6f} "
              f"{paper_mean:10.6f} {paper_std:10.6f} {ratio}  {per_seed}")
        run_path = find_completed_run(args.model, dataset, window, sorted(finals)[0], args.epochs,
                                      args.hidden_size, args.qnn_depth)
        summary[f"{dataset}|{window}"] = {
            "seeds": sorted(finals), "final_test_mse": values.tolist(), "mean": mean,
            "std_population": std_pop, "std_sample": std_sample,
            "paper_mean": paper_mean, "paper_std": paper_std,
            "trainable_params_logged_by_authors_code": trainable_params_from_log(run_path),
        }
    out_path = OUT_ROOT / f"summary_{args.model}_e{args.epochs}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")


def cmd_compare(args):
    """Seed-for-seed comparison of the authors' runs with a harness tag (train.py results)."""
    import train  # the harness (imported here: it is not needed by the other sub-commands)

    cells = {(c["dataset"], c["window"], c["seed"]): c for c in train.load_cells(args.tag)}
    print(f"Authors' train.py ({args.model}) vs harness tag {args.tag!r}: test MSE per epoch, same seed")
    print(f"{'dataset':24s} {'N':>3s} {'seed':>4s} {'epochs':>6s} {'authors final':>14s} {'harness final':>14s} "
          f"{'rel.diff final':>14s} {'max rel.diff, epochs 1-5':>25s} {'max rel.diff, all epochs':>25s}")
    worst_final = 0.0
    compared = 0
    for (dataset, window, seed), cell in sorted(cells.items()):
        epochs = cell["config"]["epochs"]
        run_path = find_completed_run(args.model, dataset, window, seed, epochs, args.hidden_size, args.qnn_depth)
        if run_path is None:
            continue
        theirs = np.array([row[2] for row in read_train_log(run_path)])
        ours = np.array(cell["curve"]["test_mse"])
        rel = np.abs(ours - theirs) / theirs
        worst_final = max(worst_final, rel[-1])
        compared += 1
        print(f"{dataset:24s} {window:3d} {seed:4d} {epochs:6d} {theirs[-1]:14.6e} {ours[-1]:14.6e} "
              f"{rel[-1]:14.2e} {rel[:5].max():25.2e} {rel.max():25.2e}")
    print(f"\n{compared} runs compared; largest relative difference of the final-epoch test MSE: {worst_final:.2e}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cell_arguments(p, datasets, windows, seeds, epochs):
        p.add_argument("--model", default="gqkan_qkanfwp", choices=list(PAPER_NAME))
        p.add_argument("--datasets", nargs="+", default=datasets, choices=list(protocol.ALL_DATASETS))
        p.add_argument("--windows", nargs="+", type=int, default=windows)
        p.add_argument("--seeds", nargs="+", type=int, default=seeds)
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--hidden_size", type=int, default=PAPER_HIDDEN_SIZE)
        p.add_argument("--qnn_depth", type=int, default=PAPER_QNN_DEPTH)

    p = sub.add_parser("param-counts")
    p.add_argument("--hidden_sizes", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9, 16])
    p.add_argument("--qnn_depths", nargs="+", type=int, default=[2, 5])
    p.set_defaults(func=cmd_param_counts)

    p = sub.add_parser("run")
    add_cell_arguments(p, ["bessel_j2", "damped_shm", "narma_5", "narma_10"], [16], [0, 1, 2, 3, 4], 50)
    p.add_argument("--jobs", type=int, default=1, help="concurrent child processes (machine rule: at most 2)")
    p.add_argument("--force", action="store_true", help="re-run cells that already finished")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("time-epoch")
    add_cell_arguments(p, ["delayed_quantum_control", "jaynes_cummings"], [16], [0], 3)
    p.add_argument("--no_nice", action="store_true")
    p.set_defaults(func=cmd_time_epoch)

    p = sub.add_parser("compare")
    p.add_argument("--tag", required=True, help="harness tag (runs/timeseries/<tag>) trained with model gqkan_qkanfwp")
    p.add_argument("--model", default="gqkan_qkanfwp", choices=list(PAPER_NAME))
    p.add_argument("--hidden_size", type=int, default=PAPER_HIDDEN_SIZE)
    p.add_argument("--qnn_depth", type=int, default=PAPER_QNN_DEPTH)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report")
    add_cell_arguments(p, list(protocol.ALL_DATASETS), list(protocol.WINDOWS), [0, 1, 2, 3, 4], 50)
    p.set_defaults(func=cmd_report)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
