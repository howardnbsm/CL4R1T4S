"""Unit tests: the closed-form DARUAN layers equal the qkan package layer.

    python solar/v2a/test_daruan.py

For every configuration the parameters are COPIED into ``qkan.QKANLayer``
(solver "exact") and both layers are evaluated on random inputs in float32;
the test fails unless the max abs difference is < 1e-5.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import torch
from qkan import QKANLayer

from daruan import StaticQKANLayer, programmed_qkan

TOL = 1e-5
torch.set_num_threads(1)


def package_layer(in_dim, out_dim, reps, preact_trainable, ba_trainable, observable):
    return QKANLayer(in_dim, out_dim, reps=reps, solver="exact", ansatz="pz_encoding",
                     preact_trainable=preact_trainable, ba_trainable=ba_trainable,
                     fast_measure=(observable == "fast"))


def test_static_layer():
    worst = 0.0
    gen = torch.Generator().manual_seed(0)
    for observable in ("z", "fast"):
        for reps in (1, 2, 3):
            for preact in (False, True):
                for ba in (False, True):
                    ours = StaticQKANLayer(5, 7, reps=reps, preact_trainable=preact, ba_trainable=ba,
                                           observable=observable)
                    with torch.no_grad():
                        ours.theta.copy_(torch.randn(ours.theta.shape, generator=gen))
                        if preact:
                            ours.preacts_weight.copy_(1 + 0.5 * torch.randn(ours.preacts_weight.shape, generator=gen))
                            ours.preacts_bias.copy_(0.5 * torch.randn(ours.preacts_bias.shape, generator=gen))
                        if ba:
                            ours.base_weight.copy_(torch.randn(ours.base_weight.shape, generator=gen))
                    ref = package_layer(5, 7, reps, preact, ba, observable)
                    with torch.no_grad():
                        ref.theta.copy_(ours.theta)
                        ref.preacts_weight.copy_(ours.preacts_weight)
                        ref.preacts_bias.copy_(ours.preacts_bias)
                        ref.base_weight.copy_(ours.base_weight)
                    assert ref.theta.shape == ours.theta.shape
                    assert ref.preacts_weight.shape == ours.preacts_weight.shape
                    x = 2.0 * torch.randn(64, 5, generator=gen)
                    with torch.no_grad():
                        diff = float((ours(x) - ref(x)).abs().max())
                    worst = max(worst, diff)
                    assert diff < TOL, (observable, reps, preact, ba, diff)
    return worst


def test_programmed_layer():
    """Per-sample angles: sample b of the programmed layer == a package layer holding theta[b]."""
    worst = 0.0
    gen = torch.Generator().manual_seed(1)
    for observable in ("z", "fast"):
        for reps in (1, 2):
            for use_base in (False, True):
                batch, n_query, in_dim, out_dim = 6, 9, 4, 5
                theta = torch.randn(batch, out_dim, in_dim, reps + 1, 2, generator=gen)
                x = 2.0 * torch.randn(batch, n_query, in_dim, generator=gen)
                base = torch.randn(out_dim, in_dim, generator=gen) if use_base else None
                ours = programmed_qkan(x, theta, observable, base)
                ours_single = programmed_qkan(x[:, 0], theta, observable, base)
                assert torch.allclose(ours[:, 0], ours_single, atol=1e-6)
                for b in range(batch):
                    ref = package_layer(in_dim, out_dim, reps, False, use_base, observable)
                    with torch.no_grad():
                        ref.theta.copy_(theta[b])
                        if use_base:
                            ref.base_weight.copy_(base)
                        diff = float((ours[b] - ref(x[b])).abs().max())
                    worst = max(worst, diff)
                    assert diff < TOL, (observable, reps, use_base, b, diff)
    return worst


def test_gradients():
    """Autograd of the closed form agrees with autograd through the package layer."""
    gen = torch.Generator().manual_seed(2)
    ours = StaticQKANLayer(3, 4, reps=1, preact_trainable=True, ba_trainable=True, observable="z")
    ref = package_layer(3, 4, 1, True, True, "z")
    with torch.no_grad():
        ref.theta.copy_(ours.theta)
        ref.preacts_weight.copy_(ours.preacts_weight)
        ref.preacts_bias.copy_(ours.preacts_bias)
        ref.base_weight.copy_(ours.base_weight)
    x = torch.randn(32, 3, generator=gen)
    weights = torch.randn(32, 4, generator=gen)
    (ours(x) * weights).sum().backward()
    (ref(x) * weights).sum().backward()
    worst = 0.0
    for name in ("theta", "preacts_weight", "preacts_bias", "base_weight"):
        diff = float((getattr(ours, name).grad - getattr(ref, name).grad).abs().max())
        worst = max(worst, diff)
        assert diff < 1e-4, (name, diff)
    return worst


if __name__ == "__main__":
    print(f"static layer      : max abs diff {test_static_layer():.2e}  (< {TOL})")
    print(f"programmed layer  : max abs diff {test_programmed_layer():.2e}  (< {TOL})")
    print(f"gradients         : max abs diff {test_gradients():.2e}  (< 1e-4)")
    print("ALL DARUAN TESTS PASSED")
