"""Tests for solar/protocol.py.  Plain asserts; run as a script:

    /home/user/venvs/gqkan/bin/python solar/test_protocol.py

What is checked
    1. data gate: CSV sha256 (the value documented in the reference
       sunspot.py), N = 2606, split 2084 / 260 / 262, shapes, dtypes, scaling;
    2. metrics(): equal (to 1e-9) to the reference final-evaluation arithmetic.
       The oracle is the reference trainer itself: ``run_training`` is called
       with ``epochs=0`` and a replay "model" that emits fixed random
       predictions, and the numbers it writes to its metrics CSV are compared
       with ``protocol.metrics``.  A literal transcription of the reference
       block is used as a second oracle;
    3. test sealing: no test data / metrics without ``unseal=True`` and one
       audit line per unsealed access.

The metric oracles run on train / validation windows only, so this test never
reads test targets, except for the sealing test, which evaluates a constant
forecast against a temporary audit log and discards the numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # never write __pycache__ into the reference checkout
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import protocol  # noqa: E402

TOL = 1e-9
_TMP_ROOT = protocol.PROJECT_DIR / "runs" / "solar" / "_tmp"


# --------------------------------------------------------------------------
# 1. data gate
# --------------------------------------------------------------------------


def test_csv_sha256_matches_reference_documentation():
    source = (protocol.REFERENCE_TASK_DIR / "src" / "data" / "sunspot.py").read_text()
    documented = re.findall(r"\b[0-9a-f]{64}\b", source)
    assert documented == [protocol.CSV_SHA256], documented
    assert protocol.csv_sha256() == protocol.CSV_SHA256


def test_window_counts_and_split():
    data = protocol.load()
    assert (protocol.N_TOTAL, protocol.N_TRAIN, protocol.N_VAL, protocol.N_TEST) == (2606, 2084, 260, 262)
    assert data.n_total == 2606
    assert (len(data.train), len(data.val), len(data.test)) == (2084, 260, 262)
    assert data.train == range(0, 2084)
    assert data.val == range(2084, 2344)
    assert data.test == range(2344, 2606)
    # The reference split object itself reports the same numbers.
    ref = protocol._reference_split()
    assert (ref.n_total, ref.n_train, ref.n_val, ref.n_test) == (2606, 2084, 260, 262)
    assert (len(ref.train), len(ref.val), len(ref.test)) == (2084, 260, 262)
    assert list(ref.test.indices) == list(range(2344, 2606))


def test_shapes_dtypes_scaling():
    data = protocol.load()
    assert data.X.shape == (2344, 528) and data.X.dtype == np.float32
    assert data.Y.shape == (2344, 132) and data.Y.dtype == np.float32
    assert np.isfinite(data.X).all() and np.isfinite(data.Y).all()
    assert data.series.shape == (2344 + 528 + 132 - 1,)
    # Stride-1 windows of the scaled series.
    for i in (0, 1, 1000, 2083, 2084, 2343):
        assert np.array_equal(data.X[i], data.series[i : i + 528].astype(np.float32))
        assert np.array_equal(data.Y[i], data.series[i + 528 : i + 660].astype(np.float32))
    # MinMax[0, 1] fit on the full series: the global extremes lie in the visible part.
    assert abs(float(data.series.min()) - 0.0) < 1e-12
    assert abs(float(data.series.max()) - 1.0) < 1e-12
    assert data.data_min == 0.0 and abs(data.data_max - 398.2) < 1e-9
    # Inverse scaling: shape-preserving, float32 path, x * (max - min) + min.
    real = data.inverse(data.Y[:3])
    assert real.shape == (3, 132) and real.dtype == np.float32
    assert np.allclose(real, data.Y[:3].astype(np.float64) * 398.2, rtol=0, atol=1e-3)
    xtr, ytr = data.split("train")
    xva, yva = data.split("val")
    assert xtr.shape == (2084, 528) and ytr.shape == (2084, 132)
    assert xva.shape == (260, 528) and yva.shape == (260, 132)
    assert np.array_equal(xva, data.X[data.val]) and np.array_equal(yva, data.Y[data.val])


# --------------------------------------------------------------------------
# 2. metrics == reference arithmetic
# --------------------------------------------------------------------------


def _random_predictions(rng: np.random.Generator, y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "uniform":
        pred = rng.uniform(-0.3, 1.3, size=y.shape)
    elif kind == "noisy_truth":
        pred = y + rng.normal(0.0, 0.08, size=y.shape)
    elif kind == "smooth":
        t = np.linspace(0.0, 1.0, y.shape[1])[None, :]
        phase = rng.uniform(0, 2 * np.pi, size=(y.shape[0], 1))
        pred = 0.3 + 0.25 * np.sin(2 * np.pi * t + phase) + rng.normal(0, 0.01, size=y.shape)
    else:
        raise ValueError(kind)
    return pred.astype(np.float32)


def _reference_trainer_metrics(pred: np.ndarray, index: range, batch_size: int) -> dict:
    """Run the reference ``run_training`` final evaluation on fixed predictions.

    ``epochs=0`` skips training entirely, so only the reference
    "FINAL EVALUATION (DENORMALIZED METRICS)" block runs, on a loader over the
    windows ``index`` of the reference dataset; the replay model returns the
    given predictions batch by batch (with the leading singleton axis the
    reference model emits).
    """
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset

    ref_dir = str(protocol.REFERENCE_TASK_DIR)
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    from src.trainers.train_loading import LoaderBundle, run_training  # reference code

    class Replay(torch.nn.Module):
        def __init__(self, values: np.ndarray):
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(1))  # Adam needs one parameter
            self.values = torch.from_numpy(values)
            self.cursor = 0

        def forward(self, x):
            n = x.shape[0]
            out = self.values[self.cursor : self.cursor + n]
            self.cursor += n
            return out.unsqueeze(0)

    dataset = protocol._reference_split().dataset
    loader = DataLoader(Subset(dataset, index), batch_size=batch_size, shuffle=False)
    loaders = LoaderBundle(
        train_loader=loader, test_loader=loader, simulation_loader=loader,
        train_len=len(index), val_loader=loader,
    )
    args = argparse.Namespace(
        epochs=0, lr=1e-3, alpha=1.0, loss="peak_aware_mse", model="gqkan_qkanfwp",
        device="cpu", dataset="sunspots", window_len=528, horizon=132, seed=0,
    )
    logger = logging.getLogger("test_protocol_reference_oracle")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmp:
        model = Replay(pred)
        run_training(args=args, model=model, loaders=loaders, result_path=Path(tmp), logger=logger)
        assert model.cursor == len(index)
        # round_trip: parse the floats the trainer wrote without 1-ulp parser error
        row = pd.read_csv(
            Path(tmp) / "gqkan_qkanfwp_final_metrics_summary.csv", float_precision="round_trip"
        ).iloc[0]
    return {
        "scaled_mse": float(row["scaled_mse"]),
        "scaled_mae": float(row["scaled_mae"]),
        "scaled_rmse": float(row["scaled_rmse"]),
        "scaled_r2": float(row["scaled_r2"]),
        "pae": float(row["peak_error"]),
        "pte": float(row["timing_error"]),
    }


def _transcribed_reference_metrics(pred: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    """Literal transcription of the reference final-evaluation block (second oracle)."""
    base_dataset = protocol._reference_split().dataset
    all_true_scaled, all_pred_scaled = [], []
    peak_amplitude_errors, peak_timing_errors = [], []
    for start in range(0, len(y), batch_size):
        y_pred_np = pred[start : start + batch_size]
        y_true_np = y[start : start + batch_size]
        all_true_scaled.extend(y_true_np.flatten().tolist())
        all_pred_scaled.extend(y_pred_np.flatten().tolist())
        y_pred_np = base_dataset.inverse_transform(y_pred_np)
        y_true_np = base_dataset.inverse_transform(y_true_np)
        for b in range(y_true_np.shape[0]):
            true_seq = y_true_np[b]
            pred_seq = y_pred_np[b]
            peak_amplitude_errors.append(abs(np.max(true_seq) - np.max(pred_seq)))
            peak_timing_errors.append(abs(np.argmax(true_seq) - np.argmax(pred_seq)))
    true_scaled_arr = np.array(all_true_scaled)
    pred_scaled_arr = np.array(all_pred_scaled)
    mae_scaled = float(np.mean(np.abs(pred_scaled_arr - true_scaled_arr)))
    mse_scaled = float(np.mean((pred_scaled_arr - true_scaled_arr) ** 2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    ss_res_scaled = float(np.sum((true_scaled_arr - pred_scaled_arr) ** 2))
    ss_tot_scaled = float(np.sum((true_scaled_arr - np.mean(true_scaled_arr)) ** 2))
    r2_scaled = 1.0 - ss_res_scaled / ss_tot_scaled if ss_tot_scaled > 0.0 else float("nan")
    return {
        "scaled_mse": mse_scaled,
        "scaled_mae": mae_scaled,
        "scaled_rmse": rmse_scaled,
        "scaled_r2": r2_scaled,
        "pae": float(np.mean(peak_amplitude_errors)),
        "pte": float(np.mean(peak_timing_errors)),
    }


def test_metrics_equal_reference_arithmetic():
    data = protocol.load()
    rng = np.random.default_rng(20260513)
    cases = [
        ("val", data.val, "uniform", 32),
        ("val", data.val, "noisy_truth", 32),
        ("val", data.val, "smooth", 7),
        ("train[500:762]", range(500, 762), "noisy_truth", 32),
        ("train[0:262]", range(0, 262), "uniform", 64),
    ]
    worst = 0.0
    for name, index, kind, batch_size in cases:
        y = data.Y[index.start : index.stop]
        pred = _random_predictions(rng, y, kind)
        ours = protocol.metrics(pred, y)
        assert tuple(ours) == protocol.METRIC_KEYS
        oracle_trainer = _reference_trainer_metrics(pred, index, batch_size)
        oracle_literal = _transcribed_reference_metrics(pred, y, batch_size)
        for key in protocol.METRIC_KEYS:
            d1 = abs(ours[key] - oracle_trainer[key])
            d2 = abs(ours[key] - oracle_literal[key])
            worst = max(worst, d1, d2)
            assert d1 <= TOL, (name, kind, key, ours[key], oracle_trainer[key])
            assert d2 <= TOL, (name, kind, key, ours[key], oracle_literal[key])
        print(f"  metrics == reference on {name:15s} {kind:12s} batch {batch_size:3d}: "
              f"mse {ours['scaled_mse']:.6f} pae {ours['pae']:.3f} pte {ours['pte']:.3f}")
    print(f"  max |ours - reference| over all cases and metrics: {worst:.3e}")
    assert worst == 0.0, "metrics() is expected to be bit-identical to the reference arithmetic"


def test_metrics_input_handling():
    data = protocol.load()
    y = data.Y[data.val.start : data.val.stop]
    rng = np.random.default_rng(1)
    pred = _random_predictions(rng, y, "noisy_truth")
    base = protocol.metrics(pred, y)
    # float64 input is cast to float32 (what a float32 model would emit).
    assert protocol.metrics(pred.astype(np.float64), y) == base
    # Reference-style (1, B, H) output is accepted.
    assert protocol.metrics(pred[None], y) == base
    import torch
    assert protocol.metrics(torch.from_numpy(pred), torch.from_numpy(y)) == base
    # A perfect forecast scores zero everywhere and R^2 = 1.
    perfect = protocol.metrics(y, y)
    assert perfect["scaled_mse"] == 0.0 and perfect["pae"] == 0.0 and perfect["pte"] == 0.0
    assert perfect["scaled_r2"] == 1.0
    for bad in (pred[:-1], pred[:, :-1]):
        try:
            protocol.metrics(bad, y)
        except ValueError:
            pass
        else:
            raise AssertionError("shape mismatch not detected")
    nan_pred = pred.copy()
    nan_pred[0, 0] = np.nan
    try:
        protocol.metrics(nan_pred, y)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN prediction not detected")


def test_evaluate_split_train_val():
    data = protocol.load()
    xva, yva = data.split("val")
    rng = np.random.default_rng(2)
    pred = _random_predictions(rng, yva, "noisy_truth")
    lookup = {xva[i].tobytes(): pred[i] for i in range(len(xva))}

    def predict(x):
        assert x.dtype == np.float32 and x.shape[1] == 528
        return np.stack([lookup[row.tobytes()] for row in x])

    from_array = protocol.evaluate_split(pred, "val", tag="unit-test")
    from_fn = protocol.evaluate_split(predict, "val", tag="unit-test")
    from_fn_batched = protocol.evaluate_split(predict, "val", tag="unit-test", batch_size=37)
    assert from_array == from_fn == from_fn_batched
    assert from_array["split"] == "val" and from_array["n_windows"] == 260
    expected = protocol.metrics(pred, yva)
    for key in protocol.METRIC_KEYS:
        assert from_array[key] == expected[key]
    # peak_aware_mse equals the reference per-epoch validation loss definition.
    import torch
    ref_dir = str(protocol.REFERENCE_TASK_DIR)
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    from src.trainers.train_loading import peak_aware_loss
    total = 0.0
    for start in range(0, 260, 32):
        p = torch.from_numpy(pred[start : start + 32])
        t = torch.from_numpy(yva[start : start + 32])
        total += peak_aware_loss(p, t, alpha=1.0).item() * p.shape[0]
    assert abs(from_array["peak_aware_mse"] - total / 260) < 1e-7
    train_result = protocol.evaluate_split(np.zeros((2084, 132), np.float32), "train", tag="unit-test")
    assert train_result["n_windows"] == 2084


# --------------------------------------------------------------------------
# 3. test sealing
# --------------------------------------------------------------------------


def _expect(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"{exc_type.__name__} not raised")


def test_sealing_and_audit_log():
    real_log = protocol.TEST_ACCESS_LOG
    before = real_log.read_bytes() if real_log.is_file() else None
    zeros = np.zeros((262, 132), np.float32)

    # Sealed by default: none of these may succeed or write to the audit log.
    data = protocol.load()
    assert not data.unsealed and data.X.shape[0] == 2344 and data.series.shape[0] == 3003
    _expect(protocol.TestSealedError, lambda: data.split("test"))
    _expect(IndexError, lambda: data.X[data.test])
    _expect(IndexError, lambda: data.Y[data.test])
    _expect(protocol.TestSealedError, lambda: protocol.evaluate_split(zeros, "test", tag="x"))
    _expect(protocol.TestSealedError, lambda: protocol.evaluate_split(lambda x: zeros, "test", tag="x"))
    _expect(ValueError, lambda: protocol.evaluate_split(zeros, "test", tag="", unseal=True))
    _expect(ValueError, lambda: protocol.evaluate_split(zeros, "test", tag=None, unseal=True))
    _expect(ValueError, lambda: protocol.load(unseal=True))
    _expect(ValueError, lambda: protocol.evaluate_split(zeros, "nope", tag="x"))
    after = real_log.read_bytes() if real_log.is_file() else None
    assert before == after, "sealed calls must not touch the audit log"

    # Unsealed access is logged: exercised against a temporary log file so the
    # real audit log only ever contains genuine evaluations.
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmp:
        protocol.TEST_ACCESS_LOG = Path(tmp) / "TEST_ACCESS_LOG.jsonl"
        try:
            assert protocol.read_test_access_log() == []
            result = protocol.evaluate_split(zeros, "test", tag="sealing self-test", unseal=True)
            assert result["split"] == "test" and result["n_windows"] == 262
            records = protocol.read_test_access_log()
            assert len(records) == 1
            rec = records[0]
            assert rec["event"] == "evaluate_split" and rec["tag"] == "sealing self-test"
            assert rec["utc"].endswith("+00:00")
            assert set(protocol.METRIC_KEYS) <= set(rec["metrics"])
            assert rec["metrics"]["scaled_mse"] == result["scaled_mse"]

            protocol.evaluate_split(lambda x: np.zeros((len(x), 132), np.float32), "test",
                                    tag="sealing self-test 2", unseal=True, batch_size=50)
            assert len(protocol.read_test_access_log()) == 2

            full = protocol.load(unseal=True, tag="sealing self-test 3")
            assert full.unsealed and full.X.shape == (2606, 528) and full.Y.shape == (2606, 132)
            assert full.series.shape == (3265,)
            assert np.array_equal(full.X[:2344], data.X) and np.array_equal(full.Y[:2344], data.Y)
            xte, yte = full.split("test")
            assert xte.shape == (262, 528) and yte.shape == (262, 132)
            records = protocol.read_test_access_log()
            assert len(records) == 3 and records[2]["event"] == "load_unsealed"

            protocol.log_external_test_evaluation("external self-test", {"scaled_mse": 1.0}, note="n")
            records = protocol.read_test_access_log()
            assert len(records) == 4 and records[3]["event"] == "external_test_evaluation"
            # every line is valid JSON with the mandatory fields
            for line in protocol.TEST_ACCESS_LOG.read_text().splitlines():
                parsed = json.loads(line)
                assert {"utc", "tag", "metrics", "event"} <= set(parsed)
        finally:
            protocol.TEST_ACCESS_LOG = real_log
    after = real_log.read_bytes() if real_log.is_file() else None
    assert before == after, "the self-test must not write to the real audit log"


def main():
    tests = [
        test_csv_sha256_matches_reference_documentation,
        test_window_counts_and_split,
        test_shapes_dtypes_scaling,
        test_metrics_equal_reference_arithmetic,
        test_metrics_input_handling,
        test_evaluate_split_train_val,
        test_sealing_and_audit_log,
    ]
    for test in tests:
        print(f"[run ] {test.__name__}")
        test()
        print(f"[pass] {test.__name__}")
    print(f"ALL {len(tests)} PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
