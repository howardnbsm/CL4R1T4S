"""Model registry for the time-series benchmark.

A registered factory is called as ``factory(window=N, **model_kwargs)`` and must
return an ``nn.Module`` that maps a window batch of shape (B, N, 1) to a
prediction of shape (B,).  Register new models with ``@register("name")``.

A model may optionally define ``prepare(x_train, y_train)``; the training harness
calls it once before training with the TRAINING split only (e.g. to fit input
normalisation).  It must never see test data.

The first entry, ``gqkan_qkanfwp``, is an efficient re-implementation of the
authors' GQKAN-QKANFWP time-series model (reference file
``times_series_benchmark/src/models/gqkan_qkanfwp.py``, Apache-2.0).  It computes
the same function with the same 159 trainable parameters and the same initial
weights for a given torch seed, but without the python loop over time steps; see
``GQKANQKANFWP`` and ``test_models.py`` for the equivalence proof.

``ref_*`` entries wrap the authors' own (slow, looped) modules so that they can be
driven by the same harness for validation.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    """Decorator that adds a model factory to ``MODEL_REGISTRY``."""

    def decorator(factory):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def build_model(name: str, window: int, **model_kwargs) -> nn.Module:
    """Instantiate a registered model for a given window size."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](window=window, **model_kwargs)


# --------------------------------------------------------------- QKAN activation
def qkan_edge_activations(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Single-qubit QKAN edge functions (``pz_encoding`` ansatz, reps=1, fast measurement).

    Each edge (output o, input i) simulates the one-qubit circuit

        |0>  --H--  RZ(a0) RY(b0)  --RZ(x_i)--  RZ(a1) RY(b1)  --  measure

    with ``theta[..., o, i, rep, :] = (a_rep, b_rep)`` and returns ``|amp_0| - |amp_1|``
    (the "fast measurement" of the QKAN package).  The reference evaluates this by a
    complex tensor-network contraction (``cutn_solver``); here the same number is
    obtained in real arithmetic by rotating the Bloch vector, which starts at +x
    after the Hadamard:

        after RZ(a0), RY(b0):   (cos a0 cos b0,  sin a0,  -cos a0 sin b0)
        RZ(x + a1) rotates the first two components, RY(b1) the first and third.

    With final Bloch vector (X, Y, Z):  |amp_0|^2 - |amp_1|^2 = Z  and
    2 |amp_0| |amp_1| = sqrt(X^2 + Y^2), hence

        |amp_0| - |amp_1| = Z / sqrt(1 + sqrt(X^2 + Y^2)).

    (``test_models.py`` checks this against a literal complex state-vector simulation
    and against the reference layer.)

    x:     (..., in_dim)
    theta: (out_dim, in_dim, 2, 2) or (..., out_dim, in_dim, 2, 2)
    returns (..., out_dim, in_dim)
    """
    alpha0, beta0 = theta[..., 0, 0], theta[..., 0, 1]
    alpha1, beta1 = theta[..., 1, 0], theta[..., 1, 1]
    cos_alpha0 = torch.cos(alpha0)
    bloch_x0 = cos_alpha0 * torch.cos(beta0)
    bloch_y0 = torch.sin(alpha0)
    minus_bloch_z0 = cos_alpha0 * torch.sin(beta0)
    angle = x.unsqueeze(-2) + alpha1  # data rotation RZ(x) merged with RZ(a1)
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    rotated_x = bloch_x0 * cos_angle - bloch_y0 * sin_angle
    bloch_y = bloch_x0 * sin_angle + bloch_y0 * cos_angle
    cos_beta1, sin_beta1 = torch.cos(beta1), torch.sin(beta1)
    bloch_x = rotated_x * cos_beta1 - minus_bloch_z0 * sin_beta1
    bloch_z = -(rotated_x * sin_beta1 + minus_bloch_z0 * cos_beta1)
    # the tiny constant only keeps the gradient finite exactly at the poles (X = Y = 0)
    return bloch_z / torch.sqrt(1.0 + torch.sqrt(bloch_x * bloch_x + bloch_y * bloch_y + 1e-24))


def qkan_layer(x: torch.Tensor, theta: torch.Tensor, base_weight: torch.Tensor | None = None) -> torch.Tensor:
    """One QKAN layer: sum the edge functions over the inputs (plus the SiLU base branch).

    Equals the reference ``QKANLayer.forward`` for the configuration the paper's
    models use (post-activation weights 1, biases 0, mask 1).  ``base_weight`` is
    the (out_dim, in_dim) weight of the SiLU branch, or None when that branch is off.
    """
    out = qkan_edge_activations(x, theta).sum(dim=-1)
    if base_weight is not None:
        out = out + F.silu(x) @ base_weight.t()
    return out


def gated_memory_weights(gate: torch.Tensor) -> torch.Tensor:
    """Weights of the unrolled gated recursion at the LAST time step.

    The reference loop is ``theta_t = (1 - g_t) * delta_t + g_t * theta_{t-1}`` with
    ``theta_{-1} = 0``; unrolled, ``theta_T = sum_k w_k * delta_k`` with
    ``w_k = (1 - g_k) * prod_{s > k} g_s``.

    gate: (B, T) in (0, 1)   ->   weights (B, T)
    """
    reversed_gate = gate.flip(1)
    # exclusive cumulative product in reversed time: prod_{s > k} g_s
    suffix = torch.cumprod(reversed_gate, dim=1)
    suffix = torch.cat([torch.ones_like(suffix[:, :1]), suffix[:, :-1]], dim=1).flip(1)
    return (1.0 - gate) * suffix


# ------------------------------------------------------------- GQKAN-QKANFWP
class GQKANQKANFWP(nn.Module):
    """Vectorised GQKAN-QKANFWP (QKAN slow programmer, gated low-rank update, QKAN fast programmer).

    Per time step t the reference cell computes, from the scalar input x_t alone,

        r_t      = Linear(QKAN(Linear(x_t)))                 slow programmer, latent
        delta_t  = outer(head_A(r_t), head_B(r_t))           proposed fast parameters
        g_t      = sigmoid(gate(r_t))                        scalar gate
        theta_t  = (1 - g_t) * delta_t + g_t * theta_{t-1}   gated memory, theta_{-1} = 0
        out_t    = post(QKAN_fast(prelinear(x_t); theta_t))

    and the loss only uses ``out_T``.  Since r_t, delta_t and g_t do not depend on
    theta, all time steps are evaluated in one batched pass and theta_T is formed
    with ``gated_memory_weights``; only the last step's fast QKAN is evaluated.

    The sub-modules are created in the same order (and with the same initialisers)
    as the reference constructor, so a given ``torch.manual_seed`` yields exactly
    the reference initial weights.
    """

    def __init__(self, hidden_size: int = 8, input_size: int = 1):
        super().__init__()
        s = int(math.ceil(math.log2(hidden_size)))  # QKAN width ("qkan_s_dim")
        latent = s + 2
        self.qkan_dim = s
        # The reference fast QKAN layer owns a theta parameter that is never used (the
        # generated theta replaces it).  Draw it anyway to keep the RNG stream aligned.
        nn.init.xavier_normal_(torch.empty(s, s, 2, 2))
        # slow programmer: Linear -> QKAN (with trainable SiLU base weights) -> Linear
        self.encoder_in = nn.Linear(input_size, s)
        self.encoder_theta = nn.Parameter(nn.init.xavier_normal_(torch.empty(s, s, 2, 2)))
        self.encoder_base_weight = nn.Parameter(0.5 * torch.ones(s, s))
        self.encoder_out = nn.Linear(s, latent)
        # low-rank generator of the fast QKAN angles, shape (out, in, reps + 1 = 2, 2)
        self.theta_head_A = nn.Linear(latent, s * 2)
        self.theta_head_B = nn.Linear(latent, s * 2)
        self.prelinear = nn.Linear(input_size, s)
        self.post_processing = nn.Linear(s, 1)
        self.fast_gate = nn.Linear(latent, 1)
        self.fast_gate.bias.data.fill_(2.0)

    def slow_programmer(self, x: torch.Tensor):
        """x (B, T, 1) -> proposals A (B, T, s, 2), B (B, T, s, 2) and gates (B, T)."""
        batch, steps, _ = x.shape
        s = self.qkan_dim
        latent = self.encoder_out(qkan_layer(self.encoder_in(x), self.encoder_theta, self.encoder_base_weight))
        head_a = self.theta_head_A(latent).view(batch, steps, s, 2)  # (out index, rep)
        head_b = self.theta_head_B(latent).view(batch, steps, s, 2)  # (in index, angle)
        gate = torch.sigmoid(self.fast_gate(latent)).squeeze(-1)
        return head_a, head_b, gate

    def final_theta(self, x: torch.Tensor) -> torch.Tensor:
        """Fast QKAN angles after the last time step, shape (B, s, s, 2, 2)."""
        head_a, head_b, gate = self.slow_programmer(x)
        weights = gated_memory_weights(gate)
        return torch.einsum("bt,btld,btqe->blqde", weights, head_a, head_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.final_theta(x)
        out = qkan_layer(self.prelinear(x[:, -1, :]), theta)
        return self.post_processing(out).squeeze(-1)

    @torch.no_grad()
    def load_reference_weights(self, reference_model: nn.Module) -> None:
        """Copy the trainable weights of an authors' ``gqkan_qkanfwp`` FWP instance."""
        cell = reference_model.fwp_cell
        fast_layer = cell.qkan_layer.layers[0]
        slow_layer = cell.slow_program_encoder[1].layers[0]
        # constants that this implementation hard-codes
        for layer, base_trainable in ((fast_layer, False), (slow_layer, True)):
            assert layer.reps == 1 and layer.ansatz == "pz_encoding" and layer.fast_measure
            assert not layer.preact_trainable and not layer.is_batchnorm
            assert torch.all(layer.postact_weights == 1) and torch.all(layer.postact_bias == 0)
            assert torch.all(layer.mask == 1)
            assert layer.base_weight.requires_grad == base_trainable
        assert torch.all(fast_layer.base_weight == 0)
        self.encoder_in.load_state_dict(cell.slow_program_encoder[0].state_dict())
        self.encoder_theta.copy_(slow_layer.theta)
        self.encoder_base_weight.copy_(slow_layer.base_weight)
        self.encoder_out.load_state_dict(cell.slow_program_encoder[2].state_dict())
        for name in ("theta_head_A", "theta_head_B", "prelinear", "post_processing", "fast_gate"):
            getattr(self, name).load_state_dict(getattr(cell, name).state_dict())


@register("gqkan_qkanfwp")
def _gqkan_qkanfwp(window: int, hidden_size: int = 8) -> nn.Module:
    """The paper's GQKAN-QKANFWP (159 trainable parameters at hidden_size 8), vectorised."""
    return GQKANQKANFWP(hidden_size=hidden_size)


# ------------------------------------------------- wrappers around the authors' code
class ReferenceModelWrapper(nn.Module):
    """Adapts an authors' FWP module (output (T, B, 1)) to the registry convention ((B,))."""

    def __init__(self, reference_model: nn.Module):
        super().__init__()
        self.reference_model = reference_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference_model(x)[-1].squeeze(-1)


def _reference_factory(flag: str):
    def factory(window: int, hidden_size: int = 8, qnn_depth: int = 2) -> nn.Module:
        from reference_repro import make_reference_model

        return ReferenceModelWrapper(make_reference_model(flag, hidden_size, qnn_depth))

    factory.__doc__ = f"The authors' own {flag} module (slow python loop over time), for validation."
    return factory


for _flag in ("gqkan_qkanfwp", "gqkanfwp", "gqkan_fwp", "gqkan_qfwp"):
    register(f"ref_{_flag}")(_reference_factory(_flag))


# -------------------------------------------------------------- minimal examples
class LinearWindow(nn.Module):
    """Linear read-out of the raw window (N + 1 parameters)."""

    def __init__(self, window: int):
        super().__init__()
        self.readout = nn.Linear(window, 1)

    def forward(self, x):
        return self.readout(x.squeeze(-1)).squeeze(-1)


@register("linear")
def _linear(window: int) -> nn.Module:
    """Linear read-out of the raw window (N + 1 parameters); the simplest possible entry."""
    return LinearWindow(window)


class RidgeInitLinear(LinearWindow):
    """``LinearWindow`` whose weights start at the closed-form ridge solution.

    Shows the ``prepare`` hook: the ridge fit uses the training split only.  Useful to
    measure how well an optimiser HOLDS an (almost) exact solution over 50 epochs.
    """

    def __init__(self, window: int, ridge: float = 1e-10):
        super().__init__(window)
        self.ridge = ridge

    @torch.no_grad()
    def prepare(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        x = x_train.squeeze(-1).double()
        y = y_train.double()
        x_mean, y_mean = x.mean(dim=0), y.mean()
        centred = x - x_mean
        gram = centred.t() @ centred + self.ridge * len(y) * torch.eye(x.shape[1], dtype=torch.float64)
        weight = torch.linalg.solve(gram, centred.t() @ (y - y_mean))
        self.readout.weight.copy_(weight.float().unsqueeze(0))
        self.readout.bias.fill_(float(y_mean - x_mean @ weight))


@register("ridge_init_linear")
def _ridge_init_linear(window: int, ridge: float = 1e-10) -> nn.Module:
    """Linear read-out initialised at the closed-form ridge solution (training split only)."""
    return RidgeInitLinear(window, ridge=ridge)


class WhitenedPolynomialReadout(nn.Module):
    """Gradient-trained linear read-out of whitened monomial features of the last ``lags`` inputs.

    ``prepare`` (training split only) standardises the last ``lags`` window entries, builds all
    monomials of degree <= ``degree`` and PCA-whitens them, dropping directions whose variance
    is below ``eigen_floor`` times the largest.  The read-out then sees features with identity
    covariance, so the least-squares problem that is hopelessly ill-conditioned on raw windows
    (neighbouring samples of a smooth series are almost collinear) becomes isotropic and plain
    SGD converges within a few epochs.  The read-out starts at zero (prediction = training mean).

    Trainable parameters: one weight per monomial + bias.  The standardisation / whitening
    constants are computed in closed form from the training inputs and are not trained.
    """

    def __init__(self, window: int, lags: int | None = None, degree: int = 1, eigen_floor: float = 1e-10):
        super().__init__()
        import itertools

        self.lags = min(lags or window, window)
        self.eigen_floor = eigen_floor
        combos = [c for d in range(1, degree + 1)
                  for c in itertools.combinations_with_replacement(range(self.lags), d)]
        # each monomial is a product of ``degree`` factors; index ``lags`` selects the constant 1
        factors = torch.tensor([list(c) + [self.lags] * (degree - len(c)) for c in combos])
        self.register_buffer("factors", factors)
        n = len(combos)
        self.register_buffer("input_mean", torch.zeros(self.lags))
        self.register_buffer("input_scale", torch.ones(self.lags))
        self.register_buffer("feature_mean", torch.zeros(n))
        self.register_buffer("whitening", torch.eye(n))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_scale", torch.ones(()))
        self.readout = nn.Linear(n, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def monomials(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, 1) window -> (B, n_monomials) products of the standardised last ``lags`` entries."""
        z = (x[:, -self.lags:, 0] - self.input_mean) / self.input_scale
        z = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)
        return z[:, self.factors].prod(dim=-1)

    @torch.no_grad()
    def prepare(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        recent = x_train[:, -self.lags:, 0].double()
        self.input_mean.copy_(recent.mean(dim=0))
        self.input_scale.copy_(recent.std(dim=0).clamp_min(1e-12))
        features = self.monomials(x_train).double()
        mean = features.mean(dim=0)
        covariance = (features - mean).t() @ (features - mean) / len(features)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        keep = eigenvalues > self.eigen_floor * eigenvalues.max()
        whitening = torch.zeros_like(eigenvectors)  # dropped directions map to zero
        whitening[:, keep] = eigenvectors[:, keep] / eigenvalues[keep].sqrt()
        self.feature_mean.copy_(mean)
        self.whitening.copy_(whitening)
        self.target_mean.copy_(y_train.double().mean())
        self.target_scale.copy_(y_train.double().std().clamp_min(1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        white = (self.monomials(x) - self.feature_mean) @ self.whitening
        return self.readout(white).squeeze(-1) * self.target_scale + self.target_mean


@register("whitened_poly")
def _whitened_poly(window: int, lags: int | None = None, degree: int = 1, eigen_floor: float = 1e-10) -> nn.Module:
    """Linear read-out of whitened degree-``degree`` monomials of the last ``lags`` inputs (zero init)."""
    return WhitenedPolynomialReadout(window, lags=lags, degree=degree, eigen_floor=eigen_floor)


# ------------------------------------------------------------------------- v2 models
@register("v2_qkan_vp")
def _v2_qkan_vp(window: int, preset: str = "v2_main", **overrides) -> nn.Module:
    """Gated QKAN-FWP v2 (time series): QKAN regressor with a closed-form read-out (``v2.py``).

        xs    = (x - mu) / sigma                          mu, sigma: scalars from the training windows
        z_j   = sum_{k<K} P_jk g^(age_k) xs_k             h = 8 encoded angles from the last K = min(N, 16) lags,
                                                          fading-memory weights g^age with g = 0.8
        e_oj  = <Z> of  H RZ(a0) RY(b0) RZ(z_j) RZ(a1) RY(b1) |0>     QKAN layer [h -> m], m = 2 (16 circuits)
        y_hat = b + sum_k l_k xs_k + sum_oj c_oj e_oj     read-out = ridge solution on the training windows

    Trainable by gradient (<= 50 full-batch L-BFGS evaluations of a forward-chaining validation
    loss): P (h * K) and the circuit angles (m * h * 4).  Solved in closed form: the QKAN
    post-activation weights c (m * h), the linear path l (N) and the bias b.

    Parameters: h K + 4 m h + m h + N + 1 = 153 / 225 / 241 / 273 for N = 8 / 16 / 32 / 64.

    The returned module is untrained; ``v2.fit_auto(config, x_train, y_train, seed)`` runs the
    whole training procedure (it is not a mini-batch model, so it is not driven by ``train.py``).
    Other presets: see ``v2.PRESETS`` - ablations (``v2_no_qkan``, ``v2_linear_edges``, ``v2_random_qkan``,
    ``v2_tanh_edges``, ``v2_fit_objective``, ``v2_fast_measure``), a wider model (``v2_wide``, h = 16) and the
    in-family variant ``v2_fwp159`` (h = 4, K = 10, plus an input-gated fast-weight programmer on the encoded
    angles: h K + 4 m h + (h C + C + 1) + m h + N + 1 = 92 / 108 / 124 / 156 parameters with C = 2).
    """
    import v2

    config = v2.V2Config(**{**v2.PRESETS[preset].to_dict(), **overrides})
    return v2.QKANProjectionRegressor(window, config)
