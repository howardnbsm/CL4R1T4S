"""Unit test: the closed-form DARUAN layer equals the qkan package layer.

    python solar/v2b/test_daruan.py

For every configuration (observable, reps, trainable pre-activations, SiLU base
path) the parameters are COPIED into ``qkan.QKANLayer`` (solver "exact", ansatz
"pz_encoding") and both layers are evaluated on random inputs in float32.  The
test fails unless the max abs difference of outputs is < 1e-5; gradients with
respect to theta are compared as well.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import torch
from qkan import QKANLayer

from daruan import DaruanLayer

TOL = 1e-5
torch.set_num_threads(1)


def package_layer(in_dim, out_dim, reps, preact, ba, observable):
    return QKANLayer(in_dim, out_dim, reps=reps, solver="exact", ansatz="pz_encoding",
                     preact_trainable=preact, ba_trainable=ba, fast_measure=(observable == "fast"))


def run():
    gen = torch.Generator().manual_seed(0)
    worst_out, worst_grad, n = 0.0, 0.0, 0
    for observable in ("z", "fast"):
        for reps in (1, 2, 3):
            for preact in (False, True):
                for ba in (False, True):
                    ours = DaruanLayer(6, 5, reps=reps, preact_trainable=preact, ba_trainable=ba,
                                       observable=observable)
                    with torch.no_grad():
                        ours.theta.copy_(torch.randn(ours.theta.shape, generator=gen))
                        if preact:
                            ours.preacts_weight.copy_(1 + 0.5 * torch.randn(ours.preacts_weight.shape, generator=gen))
                            ours.preacts_bias.copy_(0.5 * torch.randn(ours.preacts_bias.shape, generator=gen))
                        if ba:
                            ours.base_weight.copy_(torch.randn(ours.base_weight.shape, generator=gen))
                    ref = package_layer(6, 5, reps, preact, ba, observable)
                    assert ref.theta.shape == ours.theta.shape
                    assert ref.preacts_weight.shape == ours.preacts_weight.shape
                    with torch.no_grad():
                        ref.theta.copy_(ours.theta)
                        ref.preacts_weight.copy_(ours.preacts_weight)
                        ref.preacts_bias.copy_(ours.preacts_bias)
                        ref.base_weight.copy_(ours.base_weight)
                    x = 2.0 * torch.randn(64, 6, generator=gen)
                    yo, yr = ours(x), ref(x)
                    assert yo.dtype == torch.float32 and yo.shape == yr.shape == (64, 5)
                    diff = float((yo - yr).detach().abs().max())
                    # gradient of a scalar functional with respect to theta
                    wgt = torch.randn(64, 5, generator=gen)
                    go = torch.autograd.grad((yo * wgt).sum(), ours.theta)[0]
                    gr = torch.autograd.grad((yr * wgt).sum(), ref.theta)[0]
                    gdiff = float((go - gr).abs().max() / (gr.abs().max() + 1e-12))
                    worst_out, worst_grad, n = max(worst_out, diff), max(worst_grad, gdiff), n + 1
                    assert diff < TOL, (observable, reps, preact, ba, diff)
    print(f"{n} configurations: max abs output diff {worst_out:.2e} (< {TOL:g}), "
          f"max relative theta-gradient diff {worst_grad:.2e}")
    assert worst_grad < 1e-3
    print("PASS")


if __name__ == "__main__":
    run()
