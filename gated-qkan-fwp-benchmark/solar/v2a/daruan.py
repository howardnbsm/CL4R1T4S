"""Closed-form DARUAN / QKAN layers in real arithmetic (CPU-fast, package-identical).

The qkan package (0.2.1) evaluates, for every edge (o, i) of a QKAN layer, the
single-qubit data re-uploading circuit of the default ansatz ``pz_encoding``

    |0> - H - [ RZ(theta[l,0]) RY(theta[l,1]) RZ(w_l * x_i + b_l) ] x reps
            -  RZ(theta[reps,0]) RY(theta[reps,1]) - measure Z

and sums the measured activations over the inputs i (plus an optional SiLU
"base" path).  The circuit acts on one qubit, so its state is a Bloch vector
n = (nx, ny, nz) and every gate is a 3-D rotation:

    after H:       n = (1, 0, 0)
    RZ(p):         (nx, ny) -> (nx cos p - ny sin p, nx sin p + ny cos p)
    RY(p):         (nx, nz) -> (nx cos p + nz sin p, -nx sin p + nz cos p)
    <Z> = nz

Observables (both supported, both verified against the package):
    "z"    : the physical expectation value <Z> = |alpha|^2 - |beta|^2
             (qkan ``fast_measure=False``).  THIS IS WHAT THE v2a MODELS USE.
    "fast" : the package default surrogate |alpha| - |beta|
             = sqrt((1+z)/2) - sqrt((1-z)/2)  (qkan ``fast_measure=True``).

``StaticQKANLayer`` has the same parameter tensors (names and shapes) as
``qkan.QKANLayer`` so a state can be copied 1:1; ``programmed_qkan`` evaluates
a layer whose angles ``theta`` differ per sample (the fast programmer of a
fast-weight programmer).  ``test_daruan.py`` proves both numerically identical
to ``qkan.QKANLayer(solver="exact")`` (max abs diff < 1e-5 in float32).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OBSERVABLES = ("z", "fast")


def bloch_z(theta: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
    """<Z> of the pz re-uploading circuit.

    Args:
        theta: ``(..., reps + 1, 2)`` trainable / programmed angles (RZ, RY).
        enc:   ``(..., reps)`` data-encoding angles ``w_l * x + b_l``; must be
               broadcastable against ``theta[..., 0, 0]``.
    Returns:
        ``<Z>`` with the broadcast shape of the leading dimensions.
    """
    reps = theta.shape[-2] - 1
    # layer 0 acts on n = (1, 0, 0): RZ(a) -> (cos a, sin a, 0); RY(b) mixes (nx, nz)
    a, b = theta[..., 0, 0], theta[..., 0, 1]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    nx, ny, nz = ca * cb, sa, -ca * sb
    for layer in range(reps):
        phi = enc[..., layer]
        cp, sp = torch.cos(phi), torch.sin(phi)
        nx, ny = nx * cp - ny * sp, nx * sp + ny * cp          # RZ(data)
        a, b = theta[..., layer + 1, 0], theta[..., layer + 1, 1]
        ca, sa = torch.cos(a), torch.sin(a)
        nx, ny = nx * ca - ny * sa, nx * sa + ny * ca          # RZ(theta)
        cb, sb = torch.cos(b), torch.sin(b)
        nx, nz = nx * cb + nz * sb, -nx * sb + nz * cb         # RY(theta)
    return nz


def measure(z: torch.Tensor, observable: str) -> torch.Tensor:
    """Map <Z> to the chosen observable."""
    if observable == "z":
        return z
    if observable == "fast":
        z = z.clamp(-1.0, 1.0)
        return torch.sqrt((1.0 + z) * 0.5) - torch.sqrt((1.0 - z) * 0.5)
    raise ValueError(f"unknown observable {observable!r}")


class StaticQKANLayer(nn.Module):
    """A QKAN layer with ordinary (static) parameters; mirrors ``qkan.QKANLayer``.

    y[n, o] = sum_i  m(theta[o, i]; w[o, i, l] * x[n, i] + b[o, i, l])
                     + base_weight[o, i] * SiLU(x[n, i])

    Parameter tensors and their initialisation follow the package:
    ``theta (O, I, reps+1, 2)`` Xavier-normal, ``preacts_weight (O, I, reps)``
    = 1 and ``preacts_bias (O, I, reps)`` = 0 (trainable iff
    ``preact_trainable``), ``base_weight (O, I)`` = 0.5 (iff ``ba_trainable``).
    """

    def __init__(self, in_dim, out_dim, reps=1, preact_trainable=True, ba_trainable=True, observable="z"):
        super().__init__()
        assert observable in OBSERVABLES
        self.in_dim, self.out_dim, self.reps, self.observable = in_dim, out_dim, reps, observable
        self.preact_trainable, self.ba_trainable = preact_trainable, ba_trainable
        self.theta = nn.Parameter(torch.empty(out_dim, in_dim, reps + 1, 2))
        nn.init.xavier_normal_(self.theta)
        self.preacts_weight = nn.Parameter(torch.ones(out_dim, in_dim, reps), requires_grad=preact_trainable)
        self.preacts_bias = nn.Parameter(torch.zeros(out_dim, in_dim, reps), requires_grad=preact_trainable)
        self.base_weight = nn.Parameter(
            torch.full((out_dim, in_dim), 0.5 if ba_trainable else 0.0), requires_grad=ba_trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(N, I)`` -> ``(N, O)``."""
        enc = x[:, None, :, None] * self.preacts_weight + self.preacts_bias      # (N, O, I, reps)
        act = measure(bloch_z(self.theta, enc), self.observable).sum(dim=2)      # (N, O)
        if self.ba_trainable:
            act = act + F.linear(F.silu(x), self.base_weight)
        return act


def programmed_qkan(x: torch.Tensor, theta: torch.Tensor, observable: str = "z",
                    base_weight: torch.Tensor | None = None) -> torch.Tensor:
    """A QKAN layer whose angles are programmed per sample (fast programmer).

    Unit pre-activations (the data gate is RZ(x_i), the package default
    ``preact_trainable=False``), exactly as the fast layers of v1.

    Args:
        x:     ``(B, I)`` or ``(B, Q, I)`` queries (Q queries share one sample's angles).
        theta: ``(B, O, I, reps + 1, 2)`` programmed angles.
        base_weight: optional static ``(O, I)`` SiLU base-path weights.
    Returns:
        ``(B, O)`` or ``(B, Q, O)``.
    """
    squeeze = x.dim() == 2
    if squeeze:
        x = x[:, None, :]
    reps = theta.shape[-2] - 1
    enc = x[:, :, None, :, None].expand(-1, -1, -1, -1, reps)                    # (B, Q, 1, I, reps)
    act = measure(bloch_z(theta[:, None], enc), observable).sum(dim=3)            # (B, Q, O)
    if base_weight is not None:
        act = act + F.linear(F.silu(x), base_weight)
    return act[:, 0] if squeeze else act


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def gate_bias_for_horizon(horizon_steps: float) -> float:
    """Bias b with sigmoid(b) = 1 - 1/horizon (a memory of ``horizon`` steps)."""
    horizon_steps = max(float(horizon_steps), 1.0 + 1e-6)
    return math.log(horizon_steps - 1.0)
