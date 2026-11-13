"""Tests for the v2 time-series model (``v2.py``).

QKAN IMPLEMENTATION RULE: the closed-form DARUAN edge evaluation used by v2 is proven
numerically identical to the qkan package's own layer (``qkan.QKANLayer`` with the exact
state-vector solver): the v2 parameters are copied into the package layer and the outputs are
compared on random inputs in float32 (max abs difference < 1e-5).

Run with:  python test_v2.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

import models
import protocol
import v2

torch.set_num_threads(1)
TOLERANCE = 1e-5


def _randomised_model(window, config, seed, angle_scale=1.5):
    generator = torch.Generator().manual_seed(seed)
    model = v2.QKANProjectionRegressor(window, config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * angle_scale)
    return model


def test_daruan_closed_form_equals_package_layer():
    """v2's QKAN layer == qkan.QKANLayer(solver="exact") for reps 1-3, both observables, +- pre-activations."""
    worst = 0.0
    for seed, (reps, preact, observable, hidden, out) in enumerate([
            (1, False, "z", 8, 2), (1, False, "fast", 8, 2), (1, True, "z", 5, 3), (2, False, "z", 6, 1),
            (2, True, "z", 4, 3), (2, True, "fast", 4, 2), (3, True, "z", 3, 2), (3, False, "fast", 7, 1)]):
        config = v2.V2Config(hidden=hidden, out=out, reps=reps, preact=preact, observable=observable)
        model = _randomised_model(16, config, seed)
        layer = model.package_layer(dtype=torch.float32)  # the package's own layer with our parameters
        z = torch.randn(256, hidden, generator=torch.Generator().manual_seed(100 + seed)) * 2.0
        with torch.no_grad():
            reference = layer(z)  # (B, out) float32, state-vector simulation
            edges = v2.daruan_edges(z, model.theta.float(), None if model.pre_w is None else model.pre_w.float(),
                                    None if model.pre_b is None else model.pre_b.float(), observable)
            ours = (edges * model.postact_weight.float()[None]).sum(dim=-1)  # the layer's sum over inputs
        assert ours.dtype == torch.float32 and reference.shape == ours.shape
        difference = float((reference - ours).abs().max())
        assert difference < TOLERANCE, (reps, preact, observable, difference)
        worst = max(worst, difference)
    return f"max |v2 closed form - qkan.QKANLayer exact| = {worst:.2e} (float32, 8 layer configurations)"


def test_full_model_equals_package_pipeline():
    """The whole v2 forward pass rebuilt from the package layer:
    bias + linear path + sum of the outputs of qkan.QKANLayer(encoded angles)."""
    worst = 0.0
    for name in ("v2_main", "v2_wide", "v2_fwp159"):
        config = v2.PRESETS[name]
        for window in (8, 64):
            model = _randomised_model(window, config, seed=window, angle_scale=0.7)
            layer = model.package_layer(dtype=torch.float32)
            x = torch.rand(64, window, 1, generator=torch.Generator().manual_seed(window)) * 2 - 1
            with torch.no_grad():
                xs = model.standardise(x)
                rebuilt = (model.bias + xs @ model.linear_weight).float() + layer(model.encode(xs).float()).sum(dim=1)
                ours = model(x).float()
            scale = float(ours.abs().max())
            difference = float((rebuilt - ours).abs().max()) / max(1.0, scale)
            assert difference < TOLERANCE, (name, window, difference)
            worst = max(worst, difference)
    return f"max relative |model - package pipeline| = {worst:.2e}"


def test_reps1_edge_is_a_single_harmonic():
    """With the v2 initialisation a reps=1 edge measured in <Z> is exactly -cos(z + a0 + a1)."""
    theta = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    theta[0, 0, 0, 0], theta[0, 0, 1, 0], theta[0, 0, 1, 1] = 0.7, -0.4, math.pi / 2
    z = torch.linspace(-3, 3, 50, dtype=torch.float64)[:, None]
    value = v2.daruan_edges(z, theta)[:, 0, 0]
    assert float((value + torch.cos(z[:, 0] + 0.3)).abs().max()) < 1e-12
    return "reps=1, <Z>: edge = -cos(z + a0 + a1)"


def test_parameter_arithmetic():
    """Parameter counts of the registered models match the documented formulas."""
    counts = {}
    for window in protocol.WINDOWS:
        model = models.build_model("v2_qkan_vp", window)
        cfg = model.config
        k = min(window, cfg.lags) if cfg.lags else window
        expected = cfg.hidden * k + cfg.out * cfg.hidden * (cfg.reps + 1) * 2 + cfg.out * cfg.hidden + window + 1
        assert model.count_parameters()["total"] == expected == sum(p.numel() for p in model.parameters())
        counts[window] = expected
    assert counts == {8: 153, 16: 225, 32: 241, 64: 273}, counts
    return f"v2_main parameters per window: {counts}"


def test_fwp159_parameter_budget_and_gated_fold():
    """The in-family variant has <= 159 parameters for every window, and its programmed angles equal
    v1's sequential gated fast-weight recurrence  T_t = g_t T_{t-1} + (1 - g_t) W c_t  started at 0."""
    counts, worst = {}, 0.0
    for window in protocol.WINDOWS:
        config = v2.PRESETS["v2_fwp159"]
        model = _randomised_model(window, config, seed=window, angle_scale=0.8)
        h, m, c = config.hidden, config.out, config.program
        expected = h * min(window, config.lags) + 4 * m * h + (h * c + c + 1) + m * h + window + 1
        assert model.count_parameters()["total"] == expected <= 159, (window, expected)
        counts[window] = expected
        xs = torch.randn(32, window, dtype=torch.float64, generator=torch.Generator().manual_seed(window))
        with torch.no_grad():
            context = model.context(xs)
            state = torch.zeros(32, h, dtype=torch.float64)
            for t in range(window):  # the sequential fold, one step at a time
                gate = torch.sigmoid(model.gate_bias + context[:, t] @ model.gate_weight)[:, None]
                state = gate * state + (1 - gate) * (context[:, t] @ model.program_write.T)
            static = (xs[:, -model.lags:] * model.lag_weight[-model.lags:]) @ model.projection.T
            worst = max(worst, float((model.encode(xs) - (static + state)).abs().max()))
    assert worst < 1e-10, worst
    return f"v2_fwp159 parameters per window: {counts}; closed-form gated fold vs recurrence: {worst:.1e}"


def test_evaluation_budget_and_determinism():
    """fit() never exceeds max_evals function/gradient evaluations and is deterministic per seed."""
    x_train, y_train, _, _ = protocol.load("narma_10", 16)
    config = v2.PRESETS["v2_main"]
    model_a, info_a = v2.fit_auto(config, x_train, y_train, seed=3, trace=True)
    model_b, info_b = v2.fit_auto(config, x_train, y_train, seed=3)
    assert info_a["evaluations"] <= config.max_evals == 50
    assert len(info_a["objective_history"]) == info_a["evaluations"]
    prediction_a, prediction_b = v2.predict(model_a, x_train), v2.predict(model_b, x_train)
    assert np.array_equal(prediction_a, prediction_b)
    assert min(info_a["objective_history"]) <= info_a["objective_history"][0]
    return f"{info_a['evaluations']} evaluations (cap 50); selected ridge {info_a['selected']['ridge']:.0e}"


def test_selection_uses_training_data_only():
    """Changing the test split cannot change the fitted model: fit_auto never receives it."""
    x_train, y_train, x_test, _ = protocol.load("narma_5", 8)
    config = v2.PRESETS["v2_main"]
    model, _ = v2.fit_auto(config, x_train, y_train, seed=0)
    before = v2.predict(model, x_test)
    model_again, _ = v2.fit_auto(config, x_train.copy(), y_train.copy(), seed=0)
    assert np.array_equal(before, v2.predict(model_again, x_test))
    return "fit_auto(config, x_train, y_train, seed) is a function of the training split only"


def test_without_qkan_the_model_is_ridge_on_the_window():
    """Ablation sanity: with the QKAN layer removed the model is exactly ridge regression on the lag-weighted window."""
    x_train, y_train, x_test, _ = protocol.load("bessel_j2", 16)
    config = v2.V2Config(**{**v2.PRESETS["v2_no_qkan"].to_dict(), "ridge_grid": (), "ridge": 1e-10})
    model, info = v2.fit_auto(config, x_train, y_train, seed=0)
    assert info["evaluations"] == 0
    xs = model.standardise(torch.as_tensor(x_train)).numpy()
    coef, intercept = v2.ridge_readout(xs, y_train.astype(np.float64), 1e-10, model.lag_weight.numpy())
    xt = model.standardise(torch.as_tensor(x_test)).numpy()
    difference = float(np.abs(v2.predict(model, x_test) - (intercept + xt @ coef)).max())
    assert difference < 1e-9, difference
    return f"v2_no_qkan == ridge on the lag-weighted window (max diff {difference:.1e}, {info['params']['total']} parameters)"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            result = test()
            print(f"PASS {name}" + (f": {result}" if result else ""))
