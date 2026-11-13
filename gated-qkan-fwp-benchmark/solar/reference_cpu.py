"""Train the AUTHORS' GQKAN-QKANFWP on CPU through the AUTHORS' own trainer.

This is the reference ``train.py`` driver with two differences, both forced by
the absence of a GPU and neither touching the model definition or the trainer:

    * ``--device cpu`` and ``--fast_solver cutile`` (run.sh uses cuda / flash);
    * the two solver strings are resolved to CPU code (``cpu_backend.py``).

Everything else is imported from the pinned reference checkout: the dataset and
loaders (``make_datasets`` / ``make_loaders``), the model (``make_model``), and
``run_training`` (peak-aware loss alpha = 1, Adam lr 2.5e-3, keras_decay per
batch, batch 32, best-validation checkpoint, final denormalised test metrics).
Hyper-parameters are those of the reference ``run.sh``.

The reference driver writes its results inside the reference checkout, which
is read-only for us, so this script replicates its ``main()`` (same seeding
sequence, same call order) and writes to ``runs/solar/reference/`` instead.

The final test evaluation performed by the reference trainer is the authors'
protocol applied to the authors' model.  It is recorded in the test-access
audit log for transparency.  Models of ours must never be routed through this
script; they use ``protocol.evaluate_split``.

Examples
    # full protocol, one seed (about 10 minutes on one core with --impl fast)
    python solar/reference_cpu.py --seed 0 --epochs 100
    # one-epoch sanity anchor with the minimal-patch implementation (slow)
    python solar/reference_cpu.py --seed 0 --epochs 1 --impl reference \
        --out_dir runs/solar/anchor/reference_impl_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import cpu_backend  # noqa: E402
import protocol  # noqa: E402

DEFAULT_OUT_ROOT = protocol.PROJECT_DIR / "runs" / "solar" / "reference"
MODEL_KEY = "gqkan_qkanfwp"


def reference_args(seed: int, epochs: int = 100) -> argparse.Namespace:
    """The reference ``run.sh`` configuration (plus ``train.py`` defaults).

    Only ``device`` ("cuda" in run.sh) and ``fast_solver`` ("flash" by
    default) differ; see the module docstring.
    """
    return argparse.Namespace(
        # run.sh
        epochs=epochs, lr=2.5e-3, lr_schedule="keras_decay", loss="peak_aware_mse",
        model=MODEL_KEY, dataset="sunspots", exp_name=f"gqkan_qkanfwp_seed{seed}",
        save_dir="results", window_len=528, horizon=132, input_size=1, output_size=132,
        hidden_size=48, batch_size=32, seed=seed, alpha=1.0, qnn_depth=2,
        in_resize=10, out_resize=20, qkan_s_dim_1=25, qkan_s_dim_2=25, fast_in=16, fast_out=16,
        # train.py defaults that run.sh leaves untouched
        output_relu=False, streaming_fwp="off",
        # the two CPU substitutions
        device="cpu", fast_solver="cutile",
    )


def seed_everything(seed: int) -> None:
    """Seeding sequence of the reference ``train.py`` (verbatim)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _PrivateStreamDropout(torch.nn.Dropout):
    """Dropout whose masks come from a private generator (anchor probe only).

    Used to measure how much of the one-epoch loss is explained by the dropout
    masks alone: parameter initialisation and the epoch-1 batch order stay
    exactly those of the default run, only the masks change.
    """

    def __init__(self, p: float, stream: int):
        super().__init__(p)
        self.generator = torch.Generator().manual_seed(stream)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep), generator=self.generator)
        return x * mask / keep


def _epoch_seconds(log_path: Path) -> list:
    """Seconds between consecutive 'Epoch k:' lines of the trainer's log.txt."""
    stamps = []
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - Epoch \d+:")
    for line in log_path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            stamps.append(stamp + int(match.group(2)) / 1000.0)
    return [b - a for a, b in zip(stamps, stamps[1:])]


def run(seed: int, epochs: int, impl: str, out_dir: Path, threads: int = 1,
        dropout_stream: int | None = None, overwrite: bool = False) -> dict:
    torch.set_num_threads(threads)
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (the trainer appends to its CSV logs); "
                                  "choose another --out_dir or pass --overwrite")
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cpu_backend.reference()
    args = reference_args(seed=seed, epochs=epochs)

    # ---- reference train.py main(), same order ----
    seed_everything(args.seed)
    (out_dir / "args.json").write_text(json.dumps(
        {**vars(args), "impl": impl, "threads": threads, "dropout_stream": dropout_stream},
        indent=4, sort_keys=True))
    logger = ref.setup_logger(out_dir / "log.txt")
    dataset_bundle = ref.make_datasets(args)
    loaders = ref.make_loaders(args, dataset_bundle)
    model = cpu_backend.build_model(args, impl=impl)
    if dropout_stream is not None:
        model.dropout = _PrivateStreamDropout(model.dropout.p, dropout_stream)

    print(f"[reference_cpu] seed={seed} epochs={epochs} impl={impl} threads={torch.get_num_threads()} "
          f"model={type(model).__name__} out_dir={out_dir}", flush=True)
    (out_dir / "STATUS").write_text("running\n")
    t0 = time.time()
    ref.run_training(args=args, model=model, loaders=loaders, result_path=out_dir, logger=logger)
    wall = time.time() - t0

    # ---- collect what the reference trainer wrote ----
    import pandas as pd

    row = pd.read_csv(out_dir / f"{MODEL_KEY}_final_metrics_summary.csv", float_precision="round_trip").iloc[0]
    train_log = pd.read_csv(out_dir / "train_log.csv", float_precision="round_trip")
    best = train_log.loc[train_log["test_loss"].idxmin()]  # column name is the trainer's; it is the VAL loss
    epoch_seconds = _epoch_seconds(out_dir / "log.txt")
    summary = {
        "seed": seed, "epochs": epochs, "impl": impl, "threads": threads, "dropout_stream": dropout_stream,
        "trainable_parameters": int(row["trainable_parameters"]),
        "test": {
            "scaled_mse": float(row["scaled_mse"]), "scaled_mae": float(row["scaled_mae"]),
            "scaled_rmse": float(row["scaled_rmse"]), "scaled_r2": float(row["scaled_r2"]),
            "pae": float(row["peak_error"]), "pte": float(row["timing_error"]),
        },
        "epoch1_train_loss": float(train_log["train_loss"].iloc[0]),
        "epoch1_val_loss": float(train_log["test_loss"].iloc[0]),
        "best_val_loss": float(best["test_loss"]), "best_val_epoch": int(best["epoch"]),
        "final_train_loss": float(train_log["train_loss"].iloc[-1]),
        "final_val_loss": float(train_log["test_loss"].iloc[-1]),
        "trainer_wall_seconds": float(row["train_wall_time_seconds"]),
        "trainer_mean_batch_ms": float(row["mean_batch_time_ms"]),
        "total_wall_seconds": wall,
        "median_seconds_per_epoch": float(np.median(epoch_seconds)) if epoch_seconds else None,
        "torch": torch.__version__, "python": platform.python_version(), "machine": platform.processor(),
        "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (out_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "STATUS").write_text("done\n")

    protocol.log_external_test_evaluation(
        tag=f"AUTHORS-MODEL reference trainer: seed={seed} epochs={epochs} impl={impl}"
            + (f" dropout_stream={dropout_stream}" if dropout_stream is not None else ""),
        metrics=summary["test"],
        note=f"final evaluation of the authors' model by the reference run_training; out_dir={out_dir}",
    )
    print(f"[reference_cpu] done in {wall:.1f}s  best val {summary['best_val_loss']:.6f} "
          f"@ epoch {summary['best_val_epoch']}  test scaled MSE {summary['test']['scaled_mse']:.6f} "
          f"PAE {summary['test']['pae']:.2f} PTE {summary['test']['pte']:.2f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--impl", choices=["fast", "reference"], default="fast",
                        help="fast: equivalent closed-form/deduplicated forward; "
                             "reference: untouched FWP.forward, solver strings patched only (slow)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help=f"default: {DEFAULT_OUT_ROOT}/seed<seed>")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dropout_stream", type=int, default=None,
                        help="anchor probe only: draw dropout masks from a private generator with this seed")
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = cli.out_dir if cli.out_dir is not None else DEFAULT_OUT_ROOT / f"seed{cli.seed}"
    run(cli.seed, cli.epochs, cli.impl, out_dir, threads=cli.threads,
        dropout_stream=cli.dropout_stream, overwrite=cli.overwrite)


if __name__ == "__main__":
    main()
