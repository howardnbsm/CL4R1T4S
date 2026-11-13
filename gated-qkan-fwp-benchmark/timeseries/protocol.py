"""Evaluation protocol for the time-series benchmark of arXiv:2605.06734.

Everything here is derived from the authors' reference code (read-only checkout,
commit 2cf0457) so that numbers produced on top of this module are directly
comparable with the paper's Tables 3-6:

* ``load(dataset, window)`` returns the exact train / test arrays the authors'
  ``registry.make_datasets`` produces (same values, same order, float32).
* ``test_mse(pred, y)`` is the authors' test loss: their loop accumulates
  ``batch_mse * batch_size`` over batches of 4 and divides by the number of test
  samples, which is the plain mean squared error over all test samples.
* ``PAPER_CELLS`` lists the 24 (dataset, window) cells of Tables 4-6 and the
  ``paper_*`` helpers read the transcribed targets from ``paper_targets.json``.

Protocol facts (verified by ``test_protocol.py``):

* An item is a window ``x`` of shape (N, 1) and a scalar target ``y``.
* A series of length L yields ``L - N - 1`` items (the reference loop stops one
  item early); the first ``int(0.8 * n)`` items are train, the rest are test.
* bessel_j2 / damped_shm / delayed_quantum_control / jaynes_cummings / transmon:
  ``x`` is a window of the series and ``y`` its next value; the series is
  MinMax-scaled to [-1, 1] using the FULL series (train and test).
* narma_5 / narma_10: ``x`` is a window of the INPUT sequence ``u`` and ``y`` is
  the NARMA output at the next index; neither is rescaled.
* ``transmon`` ships with the reference code but is not in the paper, so it is
  supported but is not a target cell.
"""

from __future__ import annotations

import functools
import json
import sys
import types
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
REFERENCE_ROOT = Path("/home/user/matthewpeng57/gated-qkan-fwp/times_series_benchmark")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_TARGETS_PATH = PROJECT_ROOT / "paper_targets.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "timeseries"

# ----------------------------------------------------------------------- the cells
PAPER_DATASETS = (
    "bessel_j2",
    "damped_shm",
    "narma_5",
    "narma_10",
    "delayed_quantum_control",
    "jaynes_cummings",
)
NON_TARGET_DATASETS = ("transmon",)  # in the repository, not in the paper
ALL_DATASETS = PAPER_DATASETS + NON_TARGET_DATASETS
NARMA_DATASETS = ("narma_5", "narma_10")
WINDOWS = (8, 16, 32, 64)
PAPER_CELLS = tuple((d, w) for d in PAPER_DATASETS for w in WINDOWS)  # 24 cells

# The paper's training protocol (Section 6.1).
PAPER_EPOCHS = 50
PAPER_BATCH_SIZE = 4
PAPER_LR = 1e-3
PAPER_SEEDS = (0, 1, 2, 3, 4)
TRAIN_FRACTION = 0.8


def is_paper_cell(dataset: str, window: int) -> bool:
    """True if (dataset, window) is one of the 24 cells of the paper's Tables 4-6."""
    return (dataset, int(window)) in PAPER_CELLS


# ------------------------------------------------------------------ reference code
def add_reference_to_path() -> None:
    """Make the authors' ``src`` package importable (``import src.datasets...``).

    Bytecode caching is switched off first so that importing the reference modules
    never writes ``__pycache__`` files into the read-only reference checkout.
    """
    sys.dont_write_bytecode = True
    root = str(REFERENCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def reference_bundle(dataset: str, window: int):
    """Call the authors' ``make_datasets`` and return their DatasetBundle unchanged."""
    if dataset not in ALL_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {ALL_DATASETS}")
    add_reference_to_path()
    from src.datasets.registry import make_datasets  # the authors' code

    args = types.SimpleNamespace(dataset=dataset, window_len=int(window))
    return make_datasets(args)


def _stack_items(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a torch Dataset item by item, preserving the item order."""
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


@functools.lru_cache(maxsize=None)
def _load_cached(dataset: str, window: int):
    bundle = reference_bundle(dataset, window)
    x_train, y_train = _stack_items(bundle.train_ds)
    x_test, y_test = _stack_items(bundle.test_ds)
    return x_train, y_train, x_test, y_test


def load(dataset: str, window: int):
    """Return ``(Xtr, ytr, Xte, yte)`` exactly as the reference code builds them.

    Shapes: ``Xtr`` (n_train, window, 1), ``ytr`` (n_train,), ``Xte``
    (n_test, window, 1), ``yte`` (n_test,); all float32, chronological order.
    Fresh copies are returned, so callers may modify them freely.
    """
    return tuple(a.copy() for a in _load_cached(dataset, int(window)))


def load_torch(dataset: str, window: int):
    """Same as ``load`` but as torch tensors."""
    import torch

    return tuple(torch.from_numpy(a) for a in load(dataset, window))


# -------------------------------------------------------------------------- metric
def test_mse(pred, y) -> float:
    """The paper's metric: mean squared error over ALL test samples.

    The reference loop computes a float32 MSE per batch of 4, multiplies it by the
    batch size, sums over batches and divides by the number of test samples; that
    is the plain mean over samples.  Squared errors are formed in float32 (as the
    reference does) and averaged in float64, which agrees with the reference loop
    to float32 rounding (relative difference ~1e-7, see ``test_protocol.py``).
    """
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs y {y.shape}")
    if not np.all(np.isfinite(pred)):
        return float("nan")
    diff = pred - y
    return float(np.mean((diff * diff).astype(np.float64)))


test_mse.__test__ = False  # not a unit test, despite the name (for test collectors)


# ------------------------------------------------------------------- paper targets
@functools.lru_cache(maxsize=None)
def paper_targets() -> dict:
    """The transcribed paper numbers (``paper_targets.json``)."""
    with open(PAPER_TARGETS_PATH) as f:
        return json.load(f)


def paper_best(dataset: str, window: int) -> float:
    """Best (lowest) mean test MSE any model of the paper reports for this cell."""
    if not is_paper_cell(dataset, window):
        raise KeyError(f"({dataset}, {window}) is not a paper cell")
    best = paper_targets()["time_series"]["best_per_cell_any_paper_model"]
    return float(best[dataset][best["windows"].index(int(window))])


def paper_results(dataset: str, window: int) -> dict:
    """All paper results for one cell: ``{model_name: (mean, std)}``.

    Window 16 has all eight models (Table 3); the other windows have the four
    models the paper carried to Stage II (Tables 4-6).
    """
    if not is_paper_cell(dataset, window):
        raise KeyError(f"({dataset}, {window}) is not a paper cell")
    ts = paper_targets()["time_series"]
    out = {}
    by_window = ts["tables4_5_6_by_window"]
    idx = by_window["windows"].index(int(window))
    for model, rows in by_window[dataset].items():
        out[model] = tuple(rows[idx])
    if int(window) == 16:
        table3 = ts["table3_window16_all_models"]
        col = table3["columns"].index(dataset)
        for model, rows in table3.items():
            if model != "columns":
                out[model] = tuple(rows[col])
    return out


def paper_model(model: str, dataset: str, window: int) -> tuple:
    """(mean, std) the paper reports for one model on one cell."""
    return paper_results(dataset, window)[model]


def paper_best_model(dataset: str, window: int) -> str:
    """Name of the paper model that attains ``paper_best`` for this cell."""
    results = paper_results(dataset, window)
    return min(results, key=lambda m: results[m][0])


def paper_param_counts() -> dict:
    """Table 2 trainable-parameter counts of the paper's time-series models."""
    return dict(paper_targets()["param_counts_table2"]["time_series"])


if __name__ == "__main__":
    print(f"{'dataset':26s} {'N':>3s} {'n_train':>7s} {'n_test':>6s} {'paper best':>11s}  best model")
    for dataset in ALL_DATASETS:
        for window in WINDOWS:
            xtr, ytr, xte, yte = load(dataset, window)
            if is_paper_cell(dataset, window):
                target = f"{paper_best(dataset, window):.6f}"
                who = paper_best_model(dataset, window)
            else:
                target, who = "non-target", "-"
            print(f"{dataset:26s} {window:3d} {len(ytr):7d} {len(yte):6d} {target:>11s}  {who}")
