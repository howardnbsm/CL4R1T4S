"""Tests for the training harness (``train.py``).

Run with:  python test_train.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import types

import numpy as np
import torch

import models
import protocol
import train
from reference_repro import make_reference_model

torch.set_num_threads(1)
protocol.add_reference_to_path()

from src.trainers.train_loading import make_loaders, run_training  # noqa: E402  (the authors' code)

TMP_TAG = "tmp_test_train"


def test_reference_batch_order_matches_authors_dataloaders():
    """BatchOrder('reference') yields the batches the authors' DataLoaders yield, epoch after epoch."""
    dataset, window, seed = "bessel_j2", 8, 3
    x_train, y_train, _, _ = protocol.load_torch(dataset, window)
    bundle = protocol.reference_bundle(dataset, window)
    loaders = make_loaders(types.SimpleNamespace(batch_size=4), bundle)

    torch.manual_seed(seed)
    expected = []
    for epoch in range(1, 17):  # covers the extra prediction-dump loaders at epochs 1 and 15
        expected.append(torch.cat([y for _, y in loaders.train_loader]))
        for _ in loaders.test_loader:
            pass
        if epoch in {1, 15, 30, 50, 100}:
            for _ in loaders.simulation_loader:
                pass
    state_after_reference = torch.rand(1).item()

    torch.manual_seed(seed)
    order = train.BatchOrder("reference", len(y_train), seed)
    for epoch in range(1, 17):
        permutation = order.train_permutation()
        assert torch.equal(y_train[permutation], expected[epoch - 1]), epoch
        order.after_epoch(epoch)
    assert torch.rand(1).item() == state_after_reference  # global RNG stream stays aligned


def test_harness_follows_authors_training_trajectory():
    """Same seed -> the harness (vectorised model) reproduces the per-epoch train and test
    losses that the authors' run_training logs for their own model."""
    dataset, window, seed, epochs = "bessel_j2", 8, 1, 3
    tmp = protocol.RUNS_ROOT / TMP_TAG
    if tmp.exists():
        shutil.rmtree(tmp)
    # the authors' pipeline, in process: seed -> datasets -> loaders -> model -> run_training
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = types.SimpleNamespace(model="gqkan_qkanfwp", dataset=dataset, window_len=window, device="cpu",
                                 lr=1e-3, epochs=epochs, batch_size=4, horizon=1)
    loaders = make_loaders(args, protocol.reference_bundle(dataset, window))
    reference = make_reference_model("gqkan_qkanfwp", 8, 2)
    logger = logging.getLogger("test_train")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    run_training(args=args, model=reference, loaders=loaders, result_path=tmp, logger=logger)
    with open(tmp / "train_log.csv") as f:
        logged = [(float(r["train_loss"]), float(r["test_loss"])) for r in csv.DictReader(f)]
    shutil.rmtree(tmp)

    result = train.train_cell(train.TrainConfig(model="gqkan_qkanfwp", epochs=epochs), dataset, window, seed)
    assert result["params_effective"] == 159 and result["follows_paper_recipe"] is False  # 3 epochs only
    worst = 0.0
    for epoch in range(epochs):
        for ours, theirs in ((result["curve"]["train_loss"][epoch], logged[epoch][0]),
                             (result["curve"]["test_mse"][epoch], logged[epoch][1])):
            rel = abs(ours - theirs) / theirs
            worst = max(worst, rel)
            assert rel < 1e-4, (epoch, ours, theirs)
    return worst


def test_authors_modules_run_in_the_harness():
    """The harness driving the authors' own module follows the same curve as our vectorised model,
    and all four reference models are usable through the ``ref_*`` registry entries."""
    worst = 0.0
    ours = train.train_cell(train.TrainConfig(model="gqkan_qkanfwp", epochs=1), "bessel_j2", 8, 2)
    theirs = train.train_cell(train.TrainConfig(model="ref_gqkan_qkanfwp", epochs=1), "bessel_j2", 8, 2)
    assert theirs["params_effective"] == 159 and theirs["params_trainable"] == 195  # 36 unused fast-layer angles
    for key in ("train_loss", "test_mse"):
        rel = abs(ours["curve"][key][0] - theirs["curve"][key][0]) / theirs["curve"][key][0]
        worst = max(worst, rel)
        assert rel < 1e-4, (key, rel)
    table2 = protocol.paper_param_counts()
    x = torch.rand(4, 8, 1)
    for name, paper_name in (("ref_gqkanfwp", "G-QKANFWP"), ("ref_gqkan_fwp", "GQKAN-FWP"), ("ref_gqkan_qfwp", "GQKAN-QFWP")):
        model = models.build_model(name, window=8)
        out = model(x)
        assert out.shape == (4,)
        out.sum().backward()
        assert train.count_parameters(model)[1] == table2[paper_name], name
    return worst


def test_full_batch_eval_equals_batch4_eval():
    """Evaluating the test set in one batch gives the same metric as the authors' batches of 4."""
    worst = 0.0
    for name in ("gqkan_qkanfwp", "linear"):
        for dataset, window in (("narma_5", 16), ("bessel_j2", 8)):
            torch.manual_seed(0)
            model = models.build_model(name, window)
            _, _, x_test, y_test = protocol.load_torch(dataset, window)
            full = protocol.test_mse(train.predict(model, x_test), y_test.numpy())
            batched = protocol.test_mse(train.predict(model, x_test, batch_size=4), y_test.numpy())
            rel = abs(full - batched) / batched
            worst = max(worst, rel)
            assert rel < 1e-5, (name, dataset, rel)
    return worst


def test_lr_schedules():
    steps_per_epoch, total = 10, 100
    none = train.TrainConfig()
    assert all(train.lr_factor(none, s, total, steps_per_epoch) == 1.0 for s in range(total))
    cosine = train.TrainConfig(scheduler="cosine", final_lr_ratio=0.01)
    factors = [train.lr_factor(cosine, s, total, steps_per_epoch) for s in range(total)]
    assert factors[0] == 1.0 and abs(factors[-1] - 0.01) < 1e-12
    assert all(a >= b for a, b in zip(factors[:-1], factors[1:]))
    warm = train.TrainConfig(scheduler="linear", warmup_epochs=1.0, final_lr_ratio=0.0)
    factors = [train.lr_factor(warm, s, total, steps_per_epoch) for s in range(total)]
    assert abs(factors[0] - 0.1) < 1e-12 and factors[9] == 1.0 and factors[10] == 1.0 and factors[-1] == 0.0
    expo = train.TrainConfig(scheduler="exponential", final_lr_ratio=1e-2)
    assert abs(train.lr_factor(expo, total - 1, total, steps_per_epoch) - 1e-2) < 1e-12


def test_options_train_and_are_recorded():
    """Non-default optimiser / schedule / full batch run and are flagged as not the paper recipe."""
    config = train.TrainConfig(model="linear", epochs=3, batch_size=16, lr=3e-2, optimizer="adamw",
                               weight_decay=1e-4, scheduler="cosine", warmup_epochs=0.5, grad_clip=1.0,
                               shuffle="simple")
    result = train.train_cell(config, "damped_shm", 8, 0)
    assert not result["follows_paper_recipe"] and result["params_effective"] == 9
    assert len(result["curve"]["epoch"]) == 3 and math.isfinite(result["final_test_mse"])
    again = train.train_cell(config, "damped_shm", 8, 0)
    assert again["final_test_mse"] == result["final_test_mse"]  # deterministic given the seed
    other = train.train_cell(config, "damped_shm", 8, 1)
    assert other["final_test_mse"] != result["final_test_mse"]
    lbfgs = train.TrainConfig(model="linear", epochs=3, batch_size=0, lr=1.0, optimizer="lbfgs", shuffle="none")
    result = train.train_cell(lbfgs, "damped_shm", 8, 0)
    assert result["final_test_mse"] < 1e-3  # full-batch L-BFGS on a linear model converges at once
    assert train.TrainConfig().follows_paper_recipe()


def test_sweep_and_aggregate():
    directory = train.tag_dir(TMP_TAG)
    if directory.exists():
        shutil.rmtree(directory)
    config = train.TrainConfig(model="linear", epochs=2)
    train.run_sweep(TMP_TAG, config, ["narma_5", "transmon"], [8], [0, 1], jobs=1)
    assert len(train.load_cells(TMP_TAG)) == 4
    train.run_sweep(TMP_TAG, config, ["narma_5", "transmon"], [8], [0, 1, 2], jobs=2)  # resumes: 2 new cells
    cells = train.load_cells(TMP_TAG)
    assert len(cells) == 6
    summary = train.aggregate(TMP_TAG)
    rows = {(r["dataset"], r["window"]): r for r in summary["rows"]}
    values = [c["final_test_mse"] for c in cells if c["dataset"] == "narma_5"]
    row = rows[("narma_5", 8)]
    assert abs(row["mean"] - np.mean(values)) < 1e-15 and abs(row["std"] - np.std(values)) < 1e-15
    assert row["paper_best"] == 0.000005 and row["win"] == (round(row["mean"], 6) < 0.000005)
    assert "paper_best" not in rows[("transmon", 8)]  # non-target dataset
    assert summary["paper_cells_evaluated"] == 1
    assert json.loads((directory / "summary.json").read_text())["tag"] == TMP_TAG
    try:  # a tag cannot silently mix two configurations
        train.run_sweep(TMP_TAG, train.TrainConfig(model="linear", epochs=5), ["narma_5"], [8], [0])
    except SystemExit:
        pass
    else:
        raise AssertionError("config mismatch must be rejected")
    shutil.rmtree(directory)


def test_vmap_engine_matches_loop_engine():
    """Training all seeds at once gives the same per-seed curves as training them one by one."""
    worst = 0.0
    cases = (
        (train.TrainConfig(model="gqkan_qkanfwp", epochs=3), "narma_5", 8),  # 232 samples: ragged last batch
        (train.TrainConfig(model="linear", epochs=4, batch_size=16, lr=1e-2, optimizer="adamw", weight_decay=1e-3,
                           scheduler="cosine", warmup_epochs=1.0, grad_clip=0.05, shuffle="simple"), "damped_shm", 16),
    )
    for config, dataset, window in cases:
        seeds = [0, 1, 2]
        together = train.train_cells_vmap(config, dataset, window, seeds)
        for seed, batched in zip(seeds, together):
            single = train.train_cell(config, dataset, window, seed)
            assert batched["seed"] == seed and batched["params_effective"] == single["params_effective"]
            for key in ("train_loss", "test_mse"):
                for a, b in zip(batched["curve"][key], single["curve"][key]):
                    rel = abs(a - b) / b
                    worst = max(worst, rel)
                    assert rel < 1e-3, (config.model, seed, key, a, b)
    return worst


def test_win_criterion_uses_paper_precision():
    assert protocol.compare_with_paper(1.062e-05, 0.000011) == "tie"  # same number after rounding
    assert protocol.compare_with_paper(1.04e-05, 0.000011) == "WIN"
    assert protocol.compare_with_paper(1.16e-05, 0.000011) == "lose"
    assert protocol.compare_with_paper(1.6e-06, 0.000002) == "tie"
    assert protocol.compare_with_paper(1.4e-06, 0.000002) == "WIN"
    assert protocol.compare_with_paper(float("nan"), 0.000002) == "lose"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="Using a target size")
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            result = test()
            extra = f" (max relative difference {result:.2e})" if isinstance(result, float) else ""
            print(f"PASS {name}{extra}")
    print("all harness tests passed")
