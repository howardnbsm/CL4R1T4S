"""Tests that ``protocol.py`` is faithful to the authors' reference code.

Run with:  python test_protocol.py        (plain asserts, no test framework needed)
"""

from __future__ import annotations

import csv
import logging
import shutil
import types

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import protocol

torch.set_num_threads(1)
protocol.add_reference_to_path()

from src.datasets import narma_generator  # noqa: E402  (the authors' code)
from src.trainers.train_loading import make_loaders, run_training  # noqa: E402


def reference_eval_loop(pred: torch.Tensor, y: torch.Tensor, batch_size: int = 4) -> float:
    """Literal transcription of the test-loss loop in the authors' run_training."""
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(pred, y), batch_size=batch_size, shuffle=False)
    test_loss = 0.0
    with torch.no_grad():
        for out, target in loader:
            loss = criterion(out.squeeze(), target.float())
            test_loss += loss.item() * out.size(0)
    return test_loss / len(loader.dataset)


def test_arrays_equal_reference_items():
    """load() returns exactly the reference items, in order, as float32."""
    for dataset in protocol.ALL_DATASETS:
        for window in protocol.WINDOWS:
            xtr, ytr, xte, yte = protocol.load(dataset, window)
            bundle = protocol.reference_bundle(dataset, window)
            for arr in (xtr, ytr, xte, yte):
                assert arr.dtype == np.float32
            assert xtr.shape == (len(bundle.train_ds), window, 1) and ytr.shape == (len(bundle.train_ds),)
            assert xte.shape == (len(bundle.test_ds), window, 1) and yte.shape == (len(bundle.test_ds),)
            # item by item, straight from the reference Dataset objects
            for i in range(len(bundle.train_ds)):
                x, y = bundle.train_ds[i]
                assert x.shape == (window, 1) and y.shape == ()
                assert np.array_equal(xtr[i], x.numpy()) and ytr[i] == y.item()
            for i in range(len(bundle.test_ds)):
                x, y = bundle.test_ds[i]
                assert np.array_equal(xte[i], x.numpy()) and yte[i] == y.item()
            # and through the loaders the authors' trainer actually iterates
            args = types.SimpleNamespace(batch_size=4)
            loaders = make_loaders(args, bundle)
            x_batches, y_batches = zip(*[(x.numpy(), y.numpy()) for x, y in loaders.test_loader])
            assert np.array_equal(np.concatenate(x_batches), xte)
            assert np.array_equal(np.concatenate(y_batches), yte)
            # chronological 80/20 split of the reference's full item list
            n_total = len(bundle.simulation_ds)
            assert len(ytr) == int(0.8 * n_total) and len(ytr) + len(yte) == n_total
            x_all, y_all = bundle.simulation_ds.x.numpy(), bundle.simulation_ds.y.numpy()
            assert np.array_equal(np.concatenate([xtr, xte])[:, :, 0], x_all)
            assert np.array_equal(np.concatenate([ytr, yte]), y_all)


def test_next_step_structure_and_full_series_scaling():
    """Non-NARMA sets: y is the next value of the windowed series, scaled on the FULL series."""
    for dataset in protocol.ALL_DATASETS:
        if dataset in protocol.NARMA_DATASETS:
            continue
        for window in protocol.WINDOWS:
            xtr, ytr, xte, yte = protocol.load(dataset, window)
            x = np.concatenate([xtr, xte])[:, :, 0]
            y = np.concatenate([ytr, yte])
            assert np.array_equal(x[1:, :-1], x[:-1, 1:])  # windows slide by one step
            assert np.array_equal(y[:-1], x[1:, -1])  # the target is the next value of the series
            # MinMax to [-1, 1] over the whole series: the extremes -1 and +1 are both present
            # in the union of all windows (the series maximum may fall in the test part)
            series = np.concatenate([x[0], y])
            assert abs(series.min() + 1.0) < 1e-6 and abs(series.max() - 1.0) < 1e-6, dataset
    # explicit check on one dataset: the test part alone would not span [-1, 1], and re-deriving
    # the series with a full-series MinMax scaler reproduces the arrays
    import scipy.special

    raw = scipy.special.jv(2, np.linspace(2, 100, 256))
    scaled = 2.0 * (raw - raw.min()) / (raw.max() - raw.min()) - 1.0
    xtr, ytr, xte, yte = protocol.load("bessel_j2", 16)
    n = len(scaled) - 16 - 1  # the reference loop drops the last possible item
    assert len(ytr) + len(yte) == n
    expected_x = np.stack([scaled[i:i + 16] for i in range(n)]).astype(np.float32)
    expected_y = np.array([scaled[i + 16] for i in range(n)], dtype=np.float32)
    assert np.allclose(np.concatenate([xtr, xte])[:, :, 0], expected_x, atol=1e-6)
    assert np.allclose(np.concatenate([ytr, yte]), expected_y, atol=1e-6)
    assert yte.max() < 0.5 and yte.min() > -0.5  # the test targets are far inside [-1, 1]


def test_narma_maps_input_window_to_next_output_unscaled():
    """NARMA sets: x is a window of the input u, y the NARMA output at the next index, no scaling."""
    for dataset, order in (("narma_5", 5), ("narma_10", 10)):
        u, y = narma_generator.NARMA_n_Generator(n_0=order, T=300)
        assert len(u) == len(y) == 300
        for window in protocol.WINDOWS:
            xtr, ytr, xte, yte = protocol.load(dataset, window)
            x_all = np.concatenate([xtr, xte])[:, :, 0]
            y_all = np.concatenate([ytr, yte])
            n = 300 - window - 1
            assert len(y_all) == n
            for i in range(n):
                assert np.array_equal(x_all[i], u[i:i + window].astype(np.float32))
                assert y_all[i] == np.float32(y[i + window])
            assert 0.0 < x_all.min() and x_all.max() < 0.2  # raw u in (0, 0.2): not MinMax-scaled
            assert 0.1 < y_all.min() and y_all.max() < 0.3  # raw NARMA output: not MinMax-scaled


def test_mse_equals_reference_loop_on_random_predictions():
    """test_mse equals the authors' batch-4, sample-weighted loop (incl. ragged last batches)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    remainders = set()
    for dataset in protocol.ALL_DATASETS:
        for window in protocol.WINDOWS:
            _, _, _, yte = protocol.load(dataset, window)
            remainders.add(len(yte) % 4)
            for scale in (1.0, 1e-3):
                pred = (yte + scale * rng.standard_normal(len(yte))).astype(np.float32)
                ours = protocol.test_mse(pred, yte)
                # the model output has shape (B, 1) in the authors' loop
                theirs = reference_eval_loop(torch.from_numpy(pred).unsqueeze(-1), torch.from_numpy(yte))
                rel = abs(ours - theirs) / theirs
                worst = max(worst, rel)
                assert rel < 1e-6, (dataset, window, ours, theirs)
                assert protocol.test_mse(pred.reshape(-1, 1), yte) == ours  # (n, 1) predictions accepted
    assert remainders == {0, 1, 2, 3}, remainders  # every ragged-last-batch case was exercised
    return worst


class _StandIn(nn.Module):
    """Tiny model with the reference output convention: forward -> (T, B, 1), last step is used."""

    def __init__(self, window):
        super().__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        out = torch.tanh(self.linear(x.squeeze(-1)))  # (B, 1)
        return out.unsqueeze(0).expand(2, -1, -1)


def test_mse_equals_authors_run_training():
    """End to end: the test loss the authors' run_training logs equals test_mse of the same model."""
    tmp = protocol.RUNS_ROOT / "tmp_test_protocol"
    worst = 0.0
    for dataset, window in (("bessel_j2", 16), ("narma_5", 16), ("damped_shm", 8)):
        if tmp.exists():
            shutil.rmtree(tmp)
        torch.manual_seed(0)
        args = types.SimpleNamespace(model="gqkan_qkanfwp", dataset=dataset, window_len=window, device="cpu",
                                     lr=1e-2, epochs=2, batch_size=4, horizon=1)
        bundle = protocol.reference_bundle(dataset, window)
        loaders = make_loaders(args, bundle)
        model = _StandIn(window)
        logger = logging.getLogger("test_protocol")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        run_training(args=args, model=model, loaders=loaders, result_path=tmp, logger=logger)
        with open(tmp / "train_log.csv") as f:
            logged = float(list(csv.DictReader(f))[-1]["test_loss"])
        _, _, xte, yte = protocol.load(dataset, window)
        with torch.no_grad():
            pred = model(torch.from_numpy(xte))[-1].squeeze(-1).numpy()
        ours = protocol.test_mse(pred, yte)
        rel = abs(ours - logged) / logged
        worst = max(worst, rel)
        assert rel < 1e-5, (dataset, window, ours, logged)
    shutil.rmtree(tmp)
    return worst


def test_paper_targets():
    assert len(protocol.PAPER_CELLS) == 24 and len(set(protocol.PAPER_CELLS)) == 24
    for dataset, window in protocol.PAPER_CELLS:
        results = protocol.paper_results(dataset, window)
        assert len(results) == (8 if window == 16 else 4)
        best = protocol.paper_best(dataset, window)
        assert best == min(mean for mean, _ in results.values()), (dataset, window)
        assert results[protocol.paper_best_model(dataset, window)][0] == best
        assert best > 0
    # spot checks against the paper text (Tables 3-6)
    assert protocol.paper_model("GQKAN-QKANFWP", "bessel_j2", 16) == (0.000011, 0.000012)
    assert protocol.paper_model("QFWP", "narma_10", 16) == (0.000002, 0.000002)
    assert protocol.paper_model("G-QKANFWP", "narma_5", 8) == (0.000005, 0.000004)
    assert protocol.paper_model("GQKAN-FWP", "jaynes_cummings", 64) == (0.000037, 0.000044)
    assert protocol.paper_best("delayed_quantum_control", 64) == 0.000117
    counts = protocol.paper_param_counts()
    assert (counts["GQKAN-QKANFWP"], counts["GQKAN-FWP"], counts["G-QKANFWP"], counts["GQKAN-QFWP"]) == (159, 113, 116, 100)
    # transmon is supported but is not a target
    assert not protocol.is_paper_cell("transmon", 16)
    assert protocol.load("transmon", 16)[0].shape == (2386, 16, 1)
    try:
        protocol.paper_best("transmon", 16)
    except KeyError:
        pass
    else:
        raise AssertionError("transmon must not have a paper target")


def test_mse_basic_properties():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert protocol.test_mse(y, y) == 0.0
    assert abs(protocol.test_mse(y + 0.5, y) - 0.25) < 1e-7
    assert np.isnan(protocol.test_mse(np.full(5, np.nan), y))
    try:
        protocol.test_mse(y[:4], y)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="Using a target size")  # ragged batch of 1, as in the reference
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            result = test()
            extra = f" (max relative difference {result:.2e})" if isinstance(result, float) else ""
            print(f"PASS {name}{extra}")
    print("all protocol tests passed")
