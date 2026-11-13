"""Tests for ``classical_baselines.py``.

Run with:  python test_classical_baselines.py
"""

from __future__ import annotations

from math import comb

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

import classical_baselines as cb
import protocol


def test_ridge_path_matches_sklearn():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((80, 6)) * np.array([1, 10, 0.1, 1, 5, 2.0]) + 3.0
    y = features @ rng.standard_normal(6) + 0.1 * rng.standard_normal(80) + 2.0
    new = rng.standard_normal((20, 6)) + 3.0
    path = cb.RidgePath(features, y)
    for lam in (1e-6, 1e-2, 1.0):
        z = (features - features.mean(0)) / features.std(0)
        reference = Ridge(alpha=lam * len(y), fit_intercept=True).fit(z, y)
        expected = reference.predict((new - features.mean(0)) / features.std(0))
        assert np.allclose(path.predict(new, lam), expected, atol=1e-9)


def test_kernel_ridge_path_matches_sklearn():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 4))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2
    new = rng.standard_normal((15, 4))
    length, lam = 1.5, 1e-3
    path = cb.KernelRidgePath(x, y, length)
    z, z_new = (x - x.mean(0)) / x.std(0), (new - x.mean(0)) / x.std(0)
    gamma = 1.0 / (2.0 * (length * np.sqrt(4)) ** 2)
    reference = KernelRidge(alpha=lam * len(y), kernel="rbf", gamma=gamma).fit(z, y - y.mean())
    assert np.allclose(path.predict(new, lam), reference.predict(z_new) + y.mean(), atol=1e-8)


def test_monomial_features():
    x = np.arange(12, dtype=float).reshape(2, 6) + 1.0
    for lags, degree in ((3, 1), (3, 2), (4, 3)):
        features = cb.monomial_features(x, lags, degree)
        assert features.shape == (2, comb(lags + degree, degree) - 1)
    features = cb.monomial_features(x, 2, 2)  # last two entries a, b -> a, b, a^2, ab, b^2
    a, b = x[:, -2], x[:, -1]
    assert np.array_equal(features, np.stack([a, b, a * a, a * b, b * b], axis=1))


def test_holdout_is_the_end_of_the_training_split():
    assert cb.holdout_split(100) == 80 and cb.holdout_split(226) == 181 and cb.holdout_split(3) == 2


def test_selection_never_uses_test_targets():
    """Corrupting the test targets changes the reported error but not a single model choice."""
    original_load = protocol.load

    def corrupted_load(dataset, window):
        x_train, y_train, x_test, y_test = original_load(dataset, window)
        return x_train, y_train, x_test, y_test[::-1].copy() + 1.0

    for dataset, window in (("narma_10", 16), ("bessel_j2", 8)):
        clean = cb.run_cell(dataset, window)
        protocol.load = corrupted_load
        try:
            corrupted = cb.run_cell(dataset, window)
        finally:
            protocol.load = original_load
        assert clean["selected_family"] == corrupted["selected_family"]
        for name in clean["families"]:
            assert clean["families"][name]["hyperparameters"] == corrupted["families"][name]["hyperparameters"]
            assert clean["families"][name]["holdout_mse"] == corrupted["families"][name]["holdout_mse"]
            assert corrupted["families"][name]["test_mse"] > 0.1 > clean["families"][name]["test_mse"]


def test_ridge_window_beats_paper_on_a_smooth_cell():
    record = cb.run_cell("damped_shm", 16)
    ridge = record["families"]["ridge_window"]
    assert ridge["n_parameters"] == 17 and ridge["hyperparameters"]["degree"] == 1
    assert ridge["test_mse"] < 1e-9 < protocol.paper_best("damped_shm", 16)
    assert record["std_over_seeds"] == 0.0


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all classical-baseline tests passed")
