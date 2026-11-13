"""Closed-form DARUAN / QKAN layer (pz ansatz) in real arithmetic.

Same circuit, parameter tensors and layer formula as ``qkan.QKANLayer`` with
``ansatz="pz_encoding"`` (qkan 0.2.1), one single-qubit circuit per edge (o, i):

    |0> -H-> [ RZ(theta[o,i,l,0]) RY(theta[o,i,l,1]) RZ(w[o,i,l]*x_i + b[o,i,l]) ]  l = 0..reps-1
         -> RZ(theta[o,i,reps,0]) RY(theta[o,i,reps,1]) -> measure

    y_o = sum_i ( m_oi(x_i) + base_weight[o,i] * SiLU(x_i) )

evaluated by rotating the Bloch vector n = (1, 0, 0) (three real numbers per
circuit) instead of simulating complex amplitudes:

    Rz(p): (nx, ny) -> (nx cos p - ny sin p, nx sin p + ny cos p)
    Ry(p): (nx, nz) -> (nx cos p + nz sin p, -nx sin p + nz cos p)

Observables:
    "z"    the physical Pauli-Z expectation  <Z> = |alpha|^2 - |beta|^2 = n_z
           (``fast_measure=False`` in the package) - a trigonometric polynomial
           of degree <= reps in each encoded angle.
    "fast" the package default surrogate |alpha| - |beta|
           = sqrt((1+z)/2) - sqrt((1-z)/2)  (``fast_measure=True``).

``test_daruan.py`` copies the parameters into the package layer (solver
"exact") and proves max abs difference < 1e-5 in float32 for both observables.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bloch_z(x, theta, w=None, b=None):
    """<Z> of every edge circuit.

    Args:
        x: (..., I) encoded data.
        theta: (O, I, reps+1, 2) rotation angles.
        w, b: optional (O, I, reps) pre-activation weight / bias (None = 1 / 0).
    Returns:
        (..., O, I) expectation values.
    """
    reps = theta.shape[-2] - 1
    xe = x.unsqueeze(-2)  # (..., 1, I)
    ct, st = torch.cos(theta), torch.sin(theta)
    nx = torch.ones_like(xe) + 0.0 * theta[..., 0, 0]
    ny = torch.zeros_like(nx)
    nz = torch.zeros_like(nx)
    for l in range(reps):
        ca, sa = ct[..., l, 0], st[..., l, 0]
        nx, ny = nx * ca - ny * sa, nx * sa + ny * ca
        cb, sb = ct[..., l, 1], st[..., l, 1]
        nx, nz = nx * cb + nz * sb, -nx * sb + nz * cb
        p = xe if w is None else w[..., l] * xe + b[..., l]
        cp, sp = torch.cos(p), torch.sin(p)
        nx, ny = nx * cp - ny * sp, nx * sp + ny * cp
    ca, sa = ct[..., reps, 0], st[..., reps, 0]
    nx = nx * ca - ny * sa
    cb, sb = ct[..., reps, 1], st[..., reps, 1]
    return -nx * sb + nz * cb


def fast_measure(z):
    """|alpha| - |beta| as a function of <Z> (the package's default surrogate)."""
    z = z.clamp(-1.0, 1.0)
    return torch.sqrt((1.0 + z) * 0.5) - torch.sqrt((1.0 - z) * 0.5)


class DaruanLayer(nn.Module):
    """A QKAN layer with DARUAN activations; drop-in equal to ``qkan.QKANLayer`` (pz, exact)."""

    def __init__(self, in_dim, out_dim, reps=2, preact_trainable=True, ba_trainable=False,
                 observable="z", theta_std=None, freq_init=None):
        super().__init__()
        assert observable in ("z", "fast")
        self.in_dim, self.out_dim, self.reps = in_dim, out_dim, reps
        self.observable = observable
        self.preact_trainable = preact_trainable
        self.ba_trainable = ba_trainable
        theta = torch.empty(out_dim, in_dim, reps + 1, 2)
        if theta_std is None:
            nn.init.xavier_normal_(theta)  # the package initialisation
        else:
            nn.init.normal_(theta, std=theta_std)
        self.theta = nn.Parameter(theta)
        w = torch.ones(out_dim, in_dim, reps)
        if freq_init is not None:
            w = w * torch.as_tensor(freq_init, dtype=w.dtype)
        self.preacts_weight = nn.Parameter(w, requires_grad=preact_trainable)
        self.preacts_bias = nn.Parameter(torch.zeros(out_dim, in_dim, reps), requires_grad=preact_trainable)
        self.base_weight = nn.Parameter(
            0.5 * torch.ones(out_dim, in_dim) if ba_trainable else torch.zeros(out_dim, in_dim),
            requires_grad=ba_trainable)

    def edges(self, x):
        """(..., I) -> (..., O, I) measured activations (without the SiLU base path)."""
        if self.preact_trainable:
            z = bloch_z(x, self.theta, self.preacts_weight, self.preacts_bias)
        else:
            z = bloch_z(x, self.theta)
        return z if self.observable == "z" else fast_measure(z)

    def forward(self, x):
        m = self.edges(x)
        if self.ba_trainable:
            m = m + self.base_weight * F.silu(x).unsqueeze(-2)
        return m.sum(dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
