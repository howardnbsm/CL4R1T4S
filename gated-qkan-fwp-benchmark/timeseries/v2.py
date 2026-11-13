"""Gated QKAN-FWP v2, time-series track: a QKAN regressor with a closed-form read-out.

Model (one window ``x`` of N values -> next value; preset ``v2_main``: h = 8, m = 2, K = min(N, 16), g = 0.8)

    xs    = (x - mu) / sigma                          scalar mu, sigma from the TRAINING windows
    z_j   = sum_{k in last K lags} P_jk g^age_k xs_k  h learned encoded angles; g^age is a fading-memory prior
                                                      (age 0 = newest lag), i.e. v1's gated kernel with a constant gate
    e_oj  = DARUAN_oj(z_j)                            QKAN layer [h -> m]: one single-qubit data re-uploading
                                                      circuit per edge (o, j), pz ansatz, ``reps`` re-uploads
    y_hat = b + l . xs + sum_oj c_oj e_oj             b, l (linear path over the whole window) and c (the QKAN
                                                      layer's post-activation weights) form the linear read-out

Training (``fit_auto``; uses the training split only)

    1. ``select_on_holdout``: the read-out ridge is chosen from ``ridge_grid`` on the last 20% of the training
       windows, with the QKAN at its seeded initialisation (closed-form solves only, no gradient step).
    2. ``fit``: the inner parameters (P and the circuit angles) are trained by full-batch L-BFGS, hard-capped at
       ``max_evals`` = 50 function/gradient evaluations (one evaluation = one pass over the training windows = one
       epoch).  Objective ``"forward"``: forward-chaining validation loss - the read-out is solved by ridge on the
       windows [0, a) and scored on the following block, for 3 blocks at the end of the training windows,
       differentiated through the ridge solution.  (Objective ``"fit"``: the projected training loss, an ablation.)
    3. The read-out (b, l, c) is never trained by gradient: it is the closed-form ridge solution on all training
       windows for the final inner parameters (variable projection).

DARUAN edge (qkan package, ansatz "pz_encoding"), verified against ``qkan.QKANLayer`` with the exact solver in
``test_v2.py``:

    |0> -H- [RZ(t_l0) RY(t_l1) RZ(w_l z + b_l)] x reps - RZ(t_r0) RY(t_r1) - measure

evaluated here in real arithmetic by rotating the Bloch vector (three numbers per circuit).  Observable: the
physical <Z> (``observable="z"``, used by every preset except ``v2_fast_measure``) or the package's default
"fast measure" |alpha| - |beta| (``observable="fast"``).

``PRESETS`` holds the reported model, its ablations and the <= 159-parameter in-family variant ``v2_fwp159``,
which adds an input-gated fast-weight programmer to the encoded angles (see ``QKANProjectionRegressor.encode``).

Nothing in this module ever touches test data: ``fit`` / ``fit_auto`` receive training arrays only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

torch.set_num_threads(1)
F64 = torch.float64


# ------------------------------------------------------------------ DARUAN closed form
def daruan_edges(z: torch.Tensor, theta: torch.Tensor, pre_w: torch.Tensor | None = None,
                 pre_b: torch.Tensor | None = None, observable: str = "z") -> torch.Tensor:
    """Edge activations of a QKAN layer (ansatz ``pz_encoding``).

    z      (B, I)            layer input
    theta  (O, I, reps+1, 2) circuit angles, [..., l, 0] = RZ angle, [..., l, 1] = RY angle
    pre_w  (O, I, reps)      pre-activation weights (None = 1, the package's untrained default)
    pre_b  (O, I, reps)      pre-activation biases  (None = 0)
    returns (B, O, I): the measurement of the circuit on edge (o, i) for every sample.

    Bloch-vector recursion (verified to 1e-15 against the state-vector solver):
    n = (1, 0, 0) after H; RZ(p): (nx, ny) -> (nx cos p - ny sin p, nx sin p + ny cos p);
    RY(p): (nx, nz) -> (nx cos p + nz sin p, -nx sin p + nz cos p); <Z> = nz.
    """
    reps = theta.shape[2] - 1
    zz = z[:, None, :]
    nx = torch.ones((), dtype=z.dtype)
    ny = torch.zeros((), dtype=z.dtype)
    nz = torch.zeros((), dtype=z.dtype)
    for layer in range(reps + 1):
        a, b = theta[..., layer, 0], theta[..., layer, 1]
        ca, sa = torch.cos(a), torch.sin(a)
        nx, ny = nx * ca - ny * sa, nx * sa + ny * ca
        cb, sb = torch.cos(b), torch.sin(b)
        nx, nz = nx * cb + nz * sb, -nx * sb + nz * cb
        if layer < reps:
            enc = zz if pre_w is None else zz * pre_w[..., layer]
            if pre_b is not None:
                enc = enc + pre_b[..., layer]
            ce, se = torch.cos(enc), torch.sin(enc)
            nx, ny = nx * ce - ny * se, nx * se + ny * ce
    if nz.dim() < 3:  # reps = 0 never happens; keeps the shape contract for degenerate inputs
        nz = nz.expand(z.shape[0], theta.shape[0], theta.shape[1])
    if observable == "z":
        return nz
    if observable == "fast":  # |alpha| - |beta| with |alpha|^2 = (1 + <Z>) / 2
        return torch.sqrt(torch.clamp((1 + nz) / 2, min=0)) - torch.sqrt(torch.clamp((1 - nz) / 2, min=0))
    raise ValueError(f"unknown observable {observable!r}")


# ------------------------------------------------------------------------- configuration
@dataclass
class V2Config:
    hidden: int = 16            # h: learned lag projections = inputs of the QKAN layer
    out: int = 1                # m: outputs of the QKAN layer (h * m edges, each with its own read-out weight)
    reps: int = 1               # data re-uploads per edge
    preact: bool = False        # train the per-re-upload pre-activation weights / biases (w_l, b_l)
    observable: str = "z"       # "z" = <Z>, "fast" = |alpha| - |beta|
    lags: int = 0               # K: lags seen by the QKAN part (0 = whole window)
    scale: float = 1.0          # std of the encoded angle z_j at initialisation (bandwidth)
    ridge: float = 1e-8         # read-out ridge, relative: penalty = ridge * n * ||standardised coefficients||^2
    max_evals: int = 50         # cap on full-batch function/gradient evaluations (= epochs)
    edge: str = "daruan"        # "daruan" | "identity" (ablation: QKAN replaced by a linear map) | "tanh"
    linear_path: bool = True    # include the window itself in the read-out
    decay: float = 1.0          # g: fading-memory prior, lag of age a (0 = newest) is weighted by g ** a
    train: str = "all"          # inner parameters trained by gradient: "all" | "circuit" (projection frozen)
    prox: float = 0.0           # proximal penalty prox * mean((inner - inner_init)^2) added to the objective
    objective: str = "fit"      # "fit": projected training loss | "forward": forward-chaining validation loss
    folds: int = 3              # number of forward-chaining folds of the "forward" objective
    fold_fraction: float = 0.15  # size of each validation block, as a fraction of the training windows
    ridge_grid: tuple = ()      # if non-empty: ridge chosen on the training hold-out (see ``fit_auto``)
    scale_grid: tuple = ()      # if non-empty: scale chosen on the training hold-out (see ``fit_auto``)
    holdout: float = 0.2        # fraction of the training windows (the last ones) used for that choice
    optimizer: str = "lbfgs"    # "lbfgs" | "none" (random features, read-out only)
    program: int = 0            # C > 0: add a gated fast-weight programmer whose slow net reads C recent values
    gate_bias: float = 2.0      # initial gate logit of the programmer (v1 uses +2, i.e. g = 0.881)

    def to_dict(self) -> dict:
        return asdict(self)


RIDGE_GRID = tuple(10.0 ** e for e in range(-14, -1))  # 1e-14 .. 1e-2, chosen per cell on the training hold-out

_MAIN = dict(hidden=8, out=2, reps=1, preact=False, observable="z", lags=16, scale=0.1, decay=0.8,
             objective="forward", folds=3, fold_fraction=0.15, max_evals=50, ridge_grid=RIDGE_GRID, holdout=0.2)

PRESETS = {
    # the v2 model reported in RESULTS.md
    "v2_main": V2Config(**_MAIN),
    # ablations: the same pipeline with the QKAN part removed / replaced / untrained
    "v2_no_qkan": V2Config(**{**_MAIN, "hidden": 0}),                  # QKAN layer removed (linear path only)
    "v2_linear_edges": V2Config(**{**_MAIN, "edge": "identity"}),      # every DARUAN edge replaced by the identity
    "v2_random_qkan": V2Config(**{**_MAIN, "optimizer": "none"}),      # QKAN kept at its random initialisation
    "v2_tanh_edges": V2Config(**{**_MAIN, "edge": "tanh"}),            # classical activation instead of the circuit
    "v2_fit_objective": V2Config(**{**_MAIN, "objective": "fit"}),     # inner parameters trained on the training loss
    "v2_fast_measure": V2Config(**{**_MAIN, "observable": "fast"}),    # package default observable |alpha| - |beta|
    # capacity variant
    "v2_wide": V2Config(**{**_MAIN, "hidden": 16}),
    # in-family variant with <= 159 parameters for every window: 4 encoded angles from the last 10 lags plus a
    # gated fast-weight programmer (slow net reads 2 recent values), QKAN layer [4 -> 2]
    "v2_fwp159": V2Config(**{**_MAIN, "hidden": 4, "lags": 10, "program": 2}),
    "v2_fwp159_no_program": V2Config(**{**_MAIN, "hidden": 4, "lags": 10}),
}


# --------------------------------------------------------------------------------- model
class QKANProjectionRegressor(nn.Module):
    """The fitted v2 model as an ``nn.Module`` mapping windows (B, N, 1) or (B, N) to (B,).

    Parameter arithmetic (N window, K = lags, h = hidden, m = out, r = reps, C = program):
        projection P                     h * K
        circuit angles theta             m * h * (r + 1) * 2
        pre-activation w, b (if trained) 2 * m * h * r
        gated programmer (if C > 0)      h * C (write) + C (gate weights) + 1 (gate bias)
        read-out: QKAN post-activation weights m * h, linear path N, bias 1
    ``v2_main`` (h=8, m=2, r=1, K=min(N,16)): 153 / 225 / 241 / 273 for N = 8 / 16 / 32 / 64.
    """

    def __init__(self, window: int, config: V2Config):
        super().__init__()
        self.window, self.config = int(window), config
        k = self.lags = window if config.lags in (0, None) else min(int(config.lags), window)
        h, m, r = config.hidden, config.out, config.reps
        self.projection = nn.Parameter(torch.zeros(h, k, dtype=F64))
        self.theta = nn.Parameter(torch.zeros(m, h, r + 1, 2, dtype=F64))
        if config.preact:
            self.pre_w = nn.Parameter(torch.ones(m, h, r, dtype=F64))
            self.pre_b = nn.Parameter(torch.zeros(m, h, r, dtype=F64))
        else:
            self.pre_w = self.pre_b = None
        if config.program > 0:  # gated fast-weight programmer: programmed part of the encoded angles
            self.program_write = nn.Parameter(torch.zeros(h, config.program, dtype=F64))  # zero-initialised
            self.gate_weight = nn.Parameter(torch.zeros(config.program, dtype=F64))
            self.gate_bias = nn.Parameter(torch.full((), float(config.gate_bias), dtype=F64))
        else:
            self.program_write = self.gate_weight = self.gate_bias = None
        self.postact_weight = nn.Parameter(torch.zeros(m, h, dtype=F64))  # read-out of the QKAN edges
        if config.linear_path:
            self.linear_weight = nn.Parameter(torch.zeros(window, dtype=F64))
        else:
            self.linear_weight = None
        self.bias = nn.Parameter(torch.zeros((), dtype=F64))
        self.register_buffer("input_mean", torch.zeros((), dtype=F64))
        self.register_buffer("input_scale", torch.ones((), dtype=F64))
        age = torch.arange(window - 1, -1, -1, dtype=F64)
        self.register_buffer("lag_weight", float(config.decay) ** age)  # fading-memory prior scale per lag

    # -- pieces
    def inner_parameters(self):
        """Parameters trained by gradient (the read-out is solved in closed form instead)."""
        params = [self.projection] if self.config.train == "all" else []
        params.append(self.theta)
        if self.pre_w is not None:
            params += [self.pre_w, self.pre_b]
        if self.program_write is not None:
            params += [self.program_write, self.gate_weight, self.gate_bias]
        return params

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(F64)
        if x.dim() == 3:
            x = x[..., 0]
        return (x - self.input_mean) / self.input_scale

    def encode(self, xs: torch.Tensor) -> torch.Tensor:
        """Encoded angles (B, h), the inputs of the QKAN layer.

        Static part: z = P (g ** age * xs[-K:]).  With ``program = C > 0`` a gated fast-weight
        programmer adds a programmed part, exactly v1's gated memory fold with a slow net that
        reads the C most recent values c_t = (xs_t, .., xs_{t-C+1}) at every step t of the window:

            g_t = sigmoid(gate_bias + gate_weight . c_t)            input-dependent scalar gate
            z  += sum_t (1 - g_t) prod_{s > t} g_s * (W c_t)        W = program_write, zero-initialised

        Because RZ(a) RZ(z) = RZ(a + z), the programmed part is a fast weight of every circuit:
        it is added to the static RZ angle that precedes the data-encoding rotation.
        """
        z = (xs[:, -self.lags:] * self.lag_weight[-self.lags:]) @ self.projection.T
        if self.program_write is not None:
            context = self.context(xs)  # (B, N, C)
            log_gate = torch.nn.functional.logsigmoid(self.gate_bias + context @ self.gate_weight)  # (B, N)
            later = torch.flip(torch.cumsum(torch.flip(log_gate, dims=[1]), dim=1), dims=[1]) - log_gate
            kernel = (1 - torch.exp(log_gate)) * torch.exp(later)  # (1 - g_t) prod_{s>t} g_s
            z = z + torch.einsum("bt,btc,hc->bh", kernel, context, self.program_write)
        return z

    def context(self, xs: torch.Tensor) -> torch.Tensor:
        """(B, N, C): the C most recent standardised values at every step (zero before the window start)."""
        columns = [xs]
        for shift in range(1, self.config.program):
            columns.append(torch.cat([torch.zeros_like(xs[:, :shift]), xs[:, :-shift]], dim=1))
        return torch.stack(columns, dim=-1)

    def edges(self, xs: torch.Tensor) -> torch.Tensor:
        """(B, m * h) edge activations of the QKAN layer for standardised windows ``xs`` (B, N)."""
        z = self.encode(xs)  # (B, h)
        kind = self.config.edge
        if kind == "daruan":
            e = daruan_edges(z, self.theta, self.pre_w, self.pre_b, self.config.observable)
        elif kind == "identity":  # ablation: the QKAN layer replaced by a linear map
            e = z[:, None, :].expand(-1, self.config.out, -1)
        elif kind == "tanh":  # ablation: a classical activation with a trainable bias in place of the circuit
            e = torch.tanh(z[:, None, :] + self.theta[None, :, :, 0, 0])
        else:
            raise ValueError(f"unknown edge {kind!r}")
        return e.reshape(e.shape[0], -1)

    def features(self, xs: torch.Tensor) -> torch.Tensor:
        """Read-out design matrix [QKAN edges, window] (the bias is handled by centring)."""
        e = self.edges(xs)
        return torch.cat([e, xs], dim=1) if self.linear_weight is not None else e

    def readout_vector(self) -> torch.Tensor:
        c = self.postact_weight.reshape(-1)
        return torch.cat([c, self.linear_weight]) if self.linear_weight is not None else c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.standardise(x)
        return self.bias + self.features(xs) @ self.readout_vector()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        readout = self.postact_weight.numel() + 1 + (self.linear_weight.numel() if self.linear_weight is not None else 0)
        return {"inner": total - readout, "readout": readout, "total": total}

    # -- the qkan package's own layer with these parameters (used by the equivalence test)
    def package_layer(self, dtype=torch.float32):
        import qkan

        cfg = self.config
        layer = qkan.QKANLayer(cfg.hidden, cfg.out, reps=cfg.reps, solver="exact", ansatz="pz_encoding",
                               preact_trainable=cfg.preact, postact_weight_trainable=True, ba_trainable=False,
                               fast_measure=(cfg.observable == "fast"))
        with torch.no_grad():
            layer.theta.copy_(self.theta.to(dtype).reshape(layer.theta.shape))
            layer.postact_weights.copy_(self.postact_weight.to(dtype).reshape(layer.postact_weights.shape))
            if cfg.preact:
                layer.preacts_weight.copy_(self.pre_w.to(dtype).reshape(layer.preacts_weight.shape))
                layer.preacts_bias.copy_(self.pre_b.to(dtype).reshape(layer.preacts_bias.shape))
        return layer


# ---------------------------------------------------------------------- ridge read-out
def ridge_readout(features: np.ndarray, y: np.ndarray, ridge: float, prior_scale: np.ndarray | None = None):
    """Ridge read-out with an unpenalised intercept (SVD based, stable for collinear columns).

    Minimises ||y - b - F c||^2 + ridge * n * sum_j (c_j / q_j)^2, where q_j is the prior scale
    of column j (``prior_scale``; default 1 / column std, i.e. ridge on standardised columns).
    Returns (c, b) in the coordinates of the raw features.
    """
    mean = features.mean(axis=0)
    if prior_scale is None:
        std = features.std(axis=0)
        prior_scale = 1.0 / np.where(std > 1e-300, std, 1.0)
    design = (features - mean) * prior_scale
    y_mean = y.mean()
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    coef = prior_scale * (vt.T @ ((s / (s * s + ridge * len(y))) * (u.T @ (y - y_mean))))
    return coef, y_mean - mean @ coef


def _initialise(model: QKANProjectionRegressor, xs: torch.Tensor, seed: int) -> None:
    """Seeded initialisation of the inner parameters.

    Projection rows are Gaussian and rescaled so that every encoded angle z_j has standard
    deviation ``config.scale`` over the training windows (data-range matching).  Circuit angles:
    RZ angles uniform in (-pi, pi), RY angles pi/2 on the last rotation and uniform in
    (pi/4, 3pi/4) elsewhere, which gives every edge a full-contrast trigonometric response.
    """
    cfg = model.config
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        if model.projection.numel() == 0:  # QKAN part removed (ablation)
            return
        model.projection.copy_(torch.randn(model.projection.shape, generator=gen, dtype=F64))
        z = model.encode(xs)
        model.projection.mul_((cfg.scale / z.std(dim=0).clamp_min(1e-12))[:, None])
        theta = torch.empty(model.theta.shape, dtype=F64)
        theta[..., 0] = (torch.rand(theta[..., 0].shape, generator=gen, dtype=F64) * 2 - 1) * math.pi
        theta[..., 1] = (0.25 + 0.5 * torch.rand(theta[..., 1].shape, generator=gen, dtype=F64)) * math.pi
        theta[..., -1, 1] = math.pi / 2
        if cfg.reps == 1:
            theta[..., 0, 1] = 0.0  # <Z> = -cos(z + a0 + a1): unit-amplitude single harmonic
        model.theta.copy_(theta)
        if model.pre_w is not None:
            # re-upload l encodes with frequency ~ (l + 1): the edge starts as a Fourier series of z
            freq = torch.arange(1, cfg.reps + 1, dtype=F64)
            model.pre_w.copy_(freq.expand(model.pre_w.shape))
            model.pre_b.zero_()


def fit(config: V2Config, x_train: np.ndarray, y_train: np.ndarray, seed: int = 0, trace: bool = False):
    """Fit the v2 model on TRAINING data only and return ``(model, info)``.

    x_train (n, N, 1) or (n, N); y_train (n,).  ``info`` holds the number of function/gradient
    evaluations actually used and the training objective per evaluation.
    """
    x = torch.as_tensor(np.asarray(x_train), dtype=F64)
    if x.dim() == 3:
        x = x[..., 0]
    y_np = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y = torch.as_tensor(y_np)
    n, window = x.shape
    model = QKANProjectionRegressor(window, config)
    with torch.no_grad():
        model.input_mean.copy_(x.mean())
        model.input_scale.copy_(x.std().clamp_min(1e-12))
    xs = model.standardise(x)
    _initialise(model, xs, seed)
    inner = model.inner_parameters()
    sizes = [p.numel() for p in inner]

    def set_inner(vector: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for p, size in zip(inner, sizes):
                p.copy_(torch.as_tensor(vector[offset:offset + size], dtype=F64).reshape(p.shape))
                offset += size

    n_edges = model.postact_weight.numel()

    def prior_scale(feats: torch.Tensor) -> torch.Tensor:
        """Prior scale of every read-out column: 1 / std for QKAN edges, g ** age for the lags."""
        if n_edges == 0:
            return model.lag_weight
        q = 1.0 / feats[:, :n_edges].std(dim=0, unbiased=False).clamp_min(1e-300)
        return torch.cat([q, model.lag_weight]) if model.linear_weight is not None else q

    def solve_readout():
        with torch.no_grad():
            feats = model.features(xs)
            return ridge_readout(feats.numpy(), y_np, config.ridge, prior_scale(feats).numpy())

    history: list[float] = []
    best = {"objective": math.inf, "vector": None}

    class _Budget(Exception):
        pass

    # forward-chaining folds: read-out solved on windows [0, a), scored on the following block [a, b)
    block = max(1, int(round(config.fold_fraction * n)))
    bounds = [(n - (config.folds - f) * block, n - (config.folds - f - 1) * block) for f in range(config.folds)]

    def forward_chaining_loss(feats: torch.Tensor) -> torch.Tensor:
        """Mean over folds of the MSE on a block of later windows, read-out solved on the earlier ones.

        Differentiable in the inner parameters through the ridge solution (QR of the augmented
        least-squares system, which is well conditioned for every ridge > 0).
        """
        total = 0.0
        width = feats.shape[1]
        for a, b in bounds:
            head = feats[:a]
            mean = head.mean(dim=0)
            q = 1.0 / head[:, :n_edges].std(dim=0, unbiased=False).clamp_min(1e-300)
            if model.linear_weight is not None:
                q = torch.cat([q, model.lag_weight])
            design = torch.cat([(head - mean) * q, math.sqrt(config.ridge * a) * torch.eye(width, dtype=F64)])
            y_mean = y[:a].mean()
            rhs = torch.cat([y[:a] - y_mean, torch.zeros(width, dtype=F64)])
            q_factor, r_factor = torch.linalg.qr(design)
            coef = torch.linalg.solve_triangular(r_factor, (q_factor.T @ rhs)[:, None], upper=True)[:, 0]
            prediction = y_mean + ((feats[a:b] - mean) * q) @ coef
            total = total + ((prediction - y[a:b]) ** 2).mean()
        return total / len(bounds)

    def objective(vector: np.ndarray):
        if len(history) >= config.max_evals:
            raise _Budget()
        set_inner(vector)
        feats = model.features(xs)
        if config.objective == "forward":
            loss = forward_chaining_loss(feats)
        else:
            coef, intercept = solve_readout()  # closed form, no gradient (variable projection)
            coef_t = torch.as_tensor(coef)
            residual = y - (intercept + feats @ coef_t)
            penalty = config.ridge * ((coef_t / prior_scale(feats)) ** 2).sum()
            loss = (residual ** 2).mean() + penalty
        if config.prox > 0:
            loss = loss + config.prox * sum(((p - p0) ** 2).sum() for p, p0 in zip(inner, inner_start)) / len(start)
        grads = torch.autograd.grad(loss, inner)
        value = loss.item()
        history.append(value)
        if value < best["objective"]:
            best["objective"], best["vector"] = value, vector.copy()
        return value, np.concatenate([g.reshape(-1).numpy() for g in grads])

    start = np.concatenate([p.detach().reshape(-1).numpy() for p in inner])
    inner_start = [p.detach().clone() for p in inner]
    trainable = (config.optimizer != "none" and config.max_evals > 0 and config.edge != "identity"
                 and start.size > 0)
    if trainable:
        try:
            minimize(objective, start, jac=True, method="L-BFGS-B",
                     options={"maxfun": config.max_evals, "maxiter": config.max_evals, "maxcor": 20,
                              "ftol": 0.0, "gtol": 0.0})
        except _Budget:
            pass
        set_inner(best["vector"] if best["vector"] is not None else start)
    coef, intercept = solve_readout()
    with torch.no_grad():
        model.postact_weight.copy_(torch.as_tensor(coef[:n_edges]).reshape(model.postact_weight.shape))
        if model.linear_weight is not None:
            model.linear_weight.copy_(torch.as_tensor(coef[n_edges:]))
        model.bias.copy_(torch.as_tensor(intercept))
        train_mse = float(((model(x) - y) ** 2).mean())
    info = {"evaluations": len(history), "train_mse": train_mse, "params": model.count_parameters()}
    if trace:
        info["objective_history"] = history
    return model, info


def holdout_split(n_train: int, fraction: float = 0.2) -> int:
    """Number of leading training windows used for fitting during hyper-parameter selection."""
    return n_train - max(1, int(round(fraction * n_train)))


def select_on_holdout(config: V2Config, x_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> dict:
    """Choose (scale, ridge) from the grids on a chronological hold-out of the TRAINING windows.

    The model is initialised (no gradient step) on the first ``1 - holdout`` of the training
    windows, the read-out is solved for every ridge of the grid (one SVD per scale), and the
    pair with the lowest MSE on the remaining last windows wins.  Costs no gradient evaluation.
    """
    x = torch.as_tensor(np.asarray(x_train), dtype=F64)
    if x.dim() == 3:
        x = x[..., 0]
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n_fit = holdout_split(len(y), config.holdout)
    scales = tuple(config.scale_grid) or (config.scale,)
    ridges = tuple(config.ridge_grid) or (config.ridge,)
    table, best = [], None
    for scale in scales:
        cfg = V2Config(**{**config.to_dict(), "scale": scale})
        model = QKANProjectionRegressor(x.shape[1], cfg)
        with torch.no_grad():
            model.input_mean.copy_(x[:n_fit].mean())
            model.input_scale.copy_(x[:n_fit].std().clamp_min(1e-12))
            xs = model.standardise(x)
            _initialise(model, xs[:n_fit], seed)
            feats = model.features(xs).numpy()
        head, tail = feats[:n_fit], feats[n_fit:]
        n_edges = model.postact_weight.numel()
        q = 1.0 / np.maximum(head[:, :n_edges].std(axis=0), 1e-300) if n_edges else np.zeros(0)
        if model.linear_weight is not None:
            q = np.concatenate([q, model.lag_weight.numpy()])
        mean, y_mean = head.mean(axis=0), y[:n_fit].mean()
        u, sv, vt = np.linalg.svd((head - mean) * q, full_matrices=False)
        projected = u.T @ (y[:n_fit] - y_mean)
        tail_design = ((tail - mean) * q) @ vt.T
        for ridge in ridges:
            prediction = y_mean + tail_design @ ((sv / (sv * sv + ridge * n_fit)) * projected)
            error = float(np.mean((prediction - y[n_fit:]) ** 2))
            table.append({"scale": scale, "ridge": ridge, "holdout_mse": error})
            if best is None or error < best["holdout_mse"]:
                best = table[-1]
    return {"scale": best["scale"], "ridge": best["ridge"], "holdout_mse": best["holdout_mse"], "table": table}


def fit_auto(config: V2Config, x_train: np.ndarray, y_train: np.ndarray, seed: int = 0, trace: bool = False):
    """The complete v2 procedure for one cell and one seed, using TRAINING data only.

    1. ``select_on_holdout`` picks (scale, ridge) on the last ``holdout`` of the training windows.
    2. ``fit`` trains on ALL training windows with the chosen values.
    Returns ``(model, info)``; ``info["selected"]`` records the choice.
    """
    info_selection = None
    if config.ridge_grid or config.scale_grid:
        info_selection = select_on_holdout(config, x_train, y_train, seed)
        config = V2Config(**{**config.to_dict(), "scale": info_selection["scale"], "ridge": info_selection["ridge"]})
    model, info = fit(config, x_train, y_train, seed, trace=trace)
    if info_selection is not None:
        info["selected"] = {k: info_selection[k] for k in ("scale", "ridge", "holdout_mse")}
    return model, info


@torch.no_grad()
def predict(model: QKANProjectionRegressor, x: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(np.asarray(x), dtype=F64)).numpy()
