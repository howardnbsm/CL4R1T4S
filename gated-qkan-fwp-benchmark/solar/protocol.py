"""Solar-cycle forecasting protocol: data access, metrics and a sealed test split.

This module is the single entry point every model of ours must use to obtain
the sunspot data and to compute metrics.  It does NOT re-implement the data
pipeline: windows, scaling and the chronological split come from the authors'
reference code (``make_sunspot_split`` in ``Solar_cycle_forecasting/src/data/
sunspot.py`` of the pinned reference checkout), which is loaded read-only by
file path.  The sunspot CSV is read from the reference checkout and is never
copied into this project (it is CC BY-NC 4.0, source: WDC-SILSO, Royal
Observatory of Belgium, Brussels).

Protocol (paper Table 7 / reference README):
    * 528-month input window -> 132-month direct forecast, stride-1 windows,
      N = 2606, chronological window-level split 2084 / 260 / 262.
    * MinMax[0, 1] scaling fit on the FULL series (the authors' frozen choice).
    * Test metrics: scaled MSE over all flattened elements, PAE (mean over
      windows of |max(true) - max(pred)| on denormalised values) and PTE (mean
      |argmax(true) - argmax(pred)|), exactly as in the reference trainer's
      final evaluation (``run_training`` in ``src/trainers/train_loading.py``).

Test sealing
------------
The test split cannot be touched by accident:

    * ``load()`` returns train+validation windows only.  Test windows, test
      targets and the months of the series that only test windows can see are
      absent from the returned object.
    * ``load(unseal=True, tag=...)``, ``evaluate_split(..., "test", tag,
      unseal=True)`` are the only ways to reach test data / test metrics.
    * Every such access appends one JSON line (UTC timestamp, tag, metrics)
      to ``runs/solar/TEST_ACCESS_LOG.jsonl`` so that the number of test
      evaluations is auditable.

Typical use::

    from solar import protocol            # or: import protocol
    data = protocol.load()                # sealed: train + val only
    Xtr, Ytr = data.split("train")
    ...
    val = protocol.evaluate_split(predict_fn, "val", tag="my-model-v3")
    # once, at the very end, for the final model only:
    test = protocol.evaluate_split(predict_fn, "test", tag="my-model-final",
                                   unseal=True)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_TASK_DIR = Path(
    os.environ.get(
        "GQKAN_SOLAR_REFERENCE_DIR",
        "/home/user/matthewpeng57/gated-qkan-fwp/Solar_cycle_forecasting",
    )
)
TEST_ACCESS_LOG = PROJECT_DIR / "runs" / "solar" / "TEST_ACCESS_LOG.jsonl"

# Frozen protocol constants (asserted against the reference pipeline on load).
SEQ_LEN = 528
HORIZON = 132
N_TOTAL, N_TRAIN, N_VAL, N_TEST = 2606, 2084, 260, 262
CSV_SHA256 = "15c3d116ad6c5a5427837ae4cec39aa9b4b2e4a0d8e374d501a4ac760fc50b35"

SPLITS = ("train", "val", "test")
METRIC_KEYS = ("scaled_mse", "scaled_mae", "scaled_rmse", "scaled_r2", "pae", "pte")


class TestSealedError(PermissionError):
    """Raised when test data / test metrics are requested without unseal=True."""


# --------------------------------------------------------------------------
# Reference pipeline (imported, not re-implemented)
# --------------------------------------------------------------------------

_REF_MODULE = None
_REF_SPLIT = None  # the reference SunspotSplit; holds ALL windows -> keep private


def _reference_module():
    """Load the reference ``src/data/sunspot.py`` by file path (read-only).

    Loading by path (instead of ``import src...``) keeps ``sys.path`` clean and
    avoids clashes with other packages that are also called ``src``.
    """
    global _REF_MODULE
    if _REF_MODULE is None:
        path = REFERENCE_TASK_DIR / "src" / "data" / "sunspot.py"
        if not path.is_file():
            raise FileNotFoundError(f"reference data pipeline not found: {path}")
        old_flag = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # never write __pycache__ into the reference
        try:
            spec = importlib.util.spec_from_file_location(
                "gqkan_solar_reference_sunspot", path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # needed by @dataclass in the module
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = old_flag
        _REF_MODULE = module
    return _REF_MODULE


def csv_sha256() -> str:
    """sha256 of the sunspot CSV shipped with the reference checkout."""
    digest = hashlib.sha256()
    with open(_reference_module().DEFAULT_CSV, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_split():
    """Build (once) the reference split and check it against the frozen protocol."""
    global _REF_SPLIT
    if _REF_SPLIT is None:
        got_sha = csv_sha256()
        if got_sha != CSV_SHA256:
            raise RuntimeError(f"sunspot CSV sha256 mismatch: {got_sha} != {CSV_SHA256}")
        split = _reference_module().make_sunspot_split(seq_len=SEQ_LEN, horizon=HORIZON)
        counts = (split.n_total, split.n_train, split.n_val, split.n_test)
        if counts != (N_TOTAL, N_TRAIN, N_VAL, N_TEST):
            raise RuntimeError(f"unexpected split sizes from the reference pipeline: {counts}")
        _REF_SPLIT = split
    return _REF_SPLIT


def _index_ranges() -> dict:
    return {
        "train": range(0, N_TRAIN),
        "val": range(N_TRAIN, N_TRAIN + N_VAL),
        "test": range(N_TRAIN + N_VAL, N_TOTAL),
    }


def inverse_transform(scaled) -> np.ndarray:
    """Invert the MinMax scaling (scaled -> sunspot number), preserving shape.

    Delegates to the reference dataset's ``inverse_transform`` so the dtype
    path is identical to the reference evaluation (float32 in -> float32 out).
    """
    return _reference_split().dataset.inverse_transform(scaled)


# --------------------------------------------------------------------------
# Test-access audit log
# --------------------------------------------------------------------------


def _append_test_access(event: str, tag: str, metrics: Optional[dict], extra: Optional[dict] = None) -> dict:
    record = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "tag": tag,
        "metrics": metrics,
        "pid": os.getpid(),
        "argv0": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "",
    }
    if extra:
        record.update(extra)
    TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(TEST_ACCESS_LOG, "a", encoding="utf-8") as handle:  # O_APPEND: one atomic line
        handle.write(line)
    return record


def _check_tag(tag) -> str:
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("a non-empty string tag is required for every test access")
    return tag.strip()


def log_external_test_evaluation(tag: str, metrics: dict, note: str = "") -> dict:
    """Record a test evaluation that happened OUTSIDE this module.

    The only legitimate use is the authors' own trainer evaluating the authors'
    own model (``reference_cpu.py``); recording it keeps the audit log complete.
    Models of ours must go through ``evaluate_split`` instead.
    """
    return _append_test_access("external_test_evaluation", _check_tag(tag), dict(metrics), {"note": note})


def read_test_access_log() -> list:
    """Return all audit records (empty list when the log does not exist yet)."""
    if not TEST_ACCESS_LOG.is_file():
        return []
    with open(TEST_ACCESS_LOG, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SolarData:
    """Scaled sunspot windows.

    ``X`` is ``(n, 528)`` float32 and ``Y`` is ``(n, 132)`` float32, both in
    MinMax[0, 1] units.  When sealed (the default) ``n = 2344`` (train + val)
    and when unsealed ``n = 2606`` (all windows).  ``train`` / ``val`` /
    ``test`` are index ranges into the full, chronologically ordered window
    list, so ``X[data.train]`` and ``X[data.val]`` always work, while
    ``X[data.test]`` raises ``IndexError`` on a sealed object.

    ``series`` is the scaled monthly series restricted to the months that the
    visible windows cover (window ``i`` covers months ``i .. i+659``).
    """

    X: np.ndarray
    Y: np.ndarray
    train: range
    val: range
    test: range
    series: np.ndarray
    data_min: float
    data_max: float
    unsealed: bool

    @property
    def n_total(self) -> int:
        return N_TOTAL

    def split(self, name: str):
        """Return ``(X, Y)`` of one split (``"test"`` only on an unsealed object)."""
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        if name == "test" and not self.unsealed:
            raise TestSealedError(
                "the test split is sealed; use load(unseal=True, tag=...) or "
                "evaluate_split(..., 'test', tag, unseal=True)"
            )
        index = getattr(self, name)
        return self.X[index.start : index.stop], self.Y[index.start : index.stop]

    @staticmethod
    def inverse(scaled) -> np.ndarray:
        """Scaled values -> sunspot numbers (reference ``inverse_transform``)."""
        return inverse_transform(scaled)


def load(unseal: bool = False, tag: Optional[str] = None) -> SolarData:
    """Load the scaled windows through the reference pipeline.

    Args:
        unseal: ``False`` (default) returns train + validation windows only.
            ``True`` additionally returns the test windows and is recorded in
            the test-access log.
        tag: required (non-empty) when ``unseal=True``; says who looked and why.
    """
    if unseal:
        tag = _check_tag(tag)
    ref = _reference_split()
    ranges = _index_ranges()
    n_visible = N_TOTAL if unseal else N_TRAIN + N_VAL
    X = ref.dataset.x[:n_visible, :, 0].numpy().copy()
    Y = ref.dataset.y[:n_visible].numpy().copy()
    assert X.shape == (n_visible, SEQ_LEN) and X.dtype == np.float32
    assert Y.shape == (n_visible, HORIZON) and Y.dtype == np.float32
    n_months = n_visible + SEQ_LEN + HORIZON - 1  # months covered by the visible windows
    series = np.asarray(ref.dataset.full_scaled_data[:n_months], dtype=np.float64).copy()
    scaler = ref.dataset.scaler
    data = SolarData(
        X=X,
        Y=Y,
        train=ranges["train"],
        val=ranges["val"],
        test=ranges["test"],
        series=series,
        data_min=float(scaler.data_min_[0]),
        data_max=float(scaler.data_max_[0]),
        unsealed=bool(unseal),
    )
    if unseal:
        _append_test_access("load_unsealed", tag, None)
    return data


# --------------------------------------------------------------------------
# Metrics (replicates the reference final evaluation arithmetic)
# --------------------------------------------------------------------------


def _as_float32_2d(values, name: str) -> np.ndarray:
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[0] == 1:  # reference models return (1, B, H)
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (windows, horizon); got shape {array.shape}")
    # The reference evaluates float32 model outputs against float32 targets.
    return np.ascontiguousarray(array, dtype=np.float32)


def metrics(pred_scaled, y_scaled) -> dict:
    """Final-evaluation metrics, arithmetic identical to the reference trainer.

    Mirrors the "FINAL EVALUATION" block of ``run_training`` in the reference
    ``src/trainers/train_loading.py``:

        * scaled MAE / MSE / RMSE / R^2 over ALL flattened elements, computed
          in float64 from the float32 values;
        * PAE = mean over windows of ``|max(true) - max(pred)|`` and
          PTE = mean over windows of ``|argmax(true) - argmax(pred)|``, both on
          DENORMALISED float32 values (reference ``inverse_transform``).

    Args:
        pred_scaled: ``(n, 132)`` predictions in scaled units (cast to float32,
            which is what a float32 model emits).
        y_scaled: ``(n, 132)`` targets in scaled units.

    Returns:
        dict with keys ``scaled_mse, scaled_mae, scaled_rmse, scaled_r2, pae, pte``.
    """
    pred = _as_float32_2d(pred_scaled, "pred_scaled")
    true = _as_float32_2d(y_scaled, "y_scaled")
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {true.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN or inf")

    # Scaled metrics.  The reference collects python floats (float64) of the
    # float32 values, window after window, and reduces the flat float64 array.
    true_scaled_arr = true.astype(np.float64).reshape(-1)
    pred_scaled_arr = pred.astype(np.float64).reshape(-1)
    mae_scaled = float(np.mean(np.abs(pred_scaled_arr - true_scaled_arr)))
    mse_scaled = float(np.mean((pred_scaled_arr - true_scaled_arr) ** 2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    ss_res_scaled = float(np.sum((true_scaled_arr - pred_scaled_arr) ** 2))
    ss_tot_scaled = float(np.sum((true_scaled_arr - np.mean(true_scaled_arr)) ** 2))
    r2_scaled = 1.0 - ss_res_scaled / ss_tot_scaled if ss_tot_scaled > 0.0 else float("nan")

    # Peak metrics on denormalised values (float32, as in the reference).
    pred_real = inverse_transform(pred)
    true_real = inverse_transform(true)
    peak_amplitude_errors = []
    peak_timing_errors = []
    for b in range(true_real.shape[0]):
        true_seq = true_real[b]
        pred_seq = pred_real[b]
        peak_amplitude_errors.append(abs(np.max(true_seq) - np.max(pred_seq)))
        peak_timing_errors.append(abs(np.argmax(true_seq) - np.argmax(pred_seq)))

    return {
        "scaled_mse": mse_scaled,
        "scaled_mae": mae_scaled,
        "scaled_rmse": rmse_scaled,
        "scaled_r2": r2_scaled,
        "pae": float(np.mean(peak_amplitude_errors)),
        "pte": float(np.mean(peak_timing_errors)),
    }


def peak_aware_mse(pred_scaled, y_scaled, alpha: float = 1.0) -> float:
    """The training / checkpoint-selection loss, ``mean((p - y)^2 * (1 + alpha*y))``.

    Equals the reference per-epoch "val loss" (a size-weighted mean of per-batch
    means) up to float32 rounding; computed here in float64.
    """
    pred = _as_float32_2d(pred_scaled, "pred_scaled").astype(np.float64)
    true = _as_float32_2d(y_scaled, "y_scaled").astype(np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {true.shape}")
    return float(np.mean((pred - true) ** 2 * (1.0 + alpha * true)))


# --------------------------------------------------------------------------
# Split evaluation (the only door to test metrics)
# --------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], object]


def evaluate_split(
    predict_fn_or_array: Union[PredictFn, np.ndarray],
    split: str,
    tag: str,
    unseal: bool = False,
    batch_size: Optional[int] = None,
) -> dict:
    """Evaluate predictions on one split with the reference metrics.

    Args:
        predict_fn_or_array: either a callable mapping scaled inputs
            ``(n, 528)`` float32 (numpy) to scaled forecasts ``(n, 132)``
            (numpy array or torch tensor), or an array of forecasts for the
            whole split, in chronological window order.
        split: ``"train"``, ``"val"`` or ``"test"``.
        tag: free-text label of what is being evaluated (mandatory for test).
        unseal: must be ``True`` to evaluate on ``"test"``; each test
            evaluation appends one line to ``TEST_ACCESS_LOG``.
        batch_size: if given, a callable is fed chunks of at most this many
            windows (in order) instead of the whole split at once.

    Returns:
        the ``metrics`` dict plus ``peak_aware_mse`` (alpha = 1), ``split``
        and ``n_windows``.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test":
        if not unseal:
            raise TestSealedError(
                "test metrics are sealed; pass unseal=True (the access will be logged)"
            )
        tag = _check_tag(tag)

    ref = _reference_split()
    index = _index_ranges()[split]
    X = ref.dataset.x[index.start : index.stop, :, 0].numpy()
    Y = ref.dataset.y[index.start : index.stop].numpy()

    if callable(predict_fn_or_array):
        if batch_size is None:
            pred = _as_float32_2d(predict_fn_or_array(X.copy()), "predictions")
        else:
            chunks = [
                _as_float32_2d(predict_fn_or_array(X[i : i + batch_size].copy()), "predictions")
                for i in range(0, len(X), batch_size)
            ]
            pred = np.concatenate(chunks, axis=0)
    else:
        pred = _as_float32_2d(predict_fn_or_array, "predictions")
    if pred.shape != Y.shape:
        raise ValueError(f"predictions have shape {pred.shape}, expected {Y.shape} for split {split!r}")

    result = metrics(pred, Y)
    result["peak_aware_mse"] = peak_aware_mse(pred, Y, alpha=1.0)
    result["split"] = split
    result["n_windows"] = int(len(Y))
    if split == "test":
        _append_test_access("evaluate_split", tag, {k: result[k] for k in METRIC_KEYS + ("peak_aware_mse",)})
    return result


if __name__ == "__main__":
    info = load()
    print(f"reference task dir : {REFERENCE_TASK_DIR}")
    print(f"csv sha256         : {csv_sha256()}")
    print(f"windows (sealed)   : X {info.X.shape} {info.X.dtype}, Y {info.Y.shape} {info.Y.dtype}")
    print(f"split ranges       : train {info.train}, val {info.val}, test {info.test}")
    print(f"scaler             : data_min {info.data_min}, data_max {info.data_max}")
    print(f"test-access records: {len(read_test_access_log())} in {TEST_ACCESS_LOG}")
