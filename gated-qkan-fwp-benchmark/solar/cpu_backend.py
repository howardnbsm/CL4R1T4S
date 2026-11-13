"""CPU execution of the authors' GQKAN-QKANFWP model (solar task).

The reference model is GPU-only as shipped:

    * the slow programmer hardcodes ``qkan.QKAN(..., solver="cutile")``, which
      dispatches into the CUDA-only ``cuda-tile`` package, and
    * the fast QKAN layer defaults to ``solver="flash"`` (fused Triton kernels).

Both solvers evaluate the same tiny single-qubit data re-uploading circuit

    H|0>  ->  [Rz(t_l0) Ry(t_l1) Rz(enc_l)] x reps  ->  Rz(t_R0) Ry(t_R1)
          ->  |alpha| - |beta|                               ("fast measure")

with ``enc_l = w_l * x + b_l`` when the pre-activations are trainable and
``enc_l = x`` otherwise.  This module provides two CPU execution paths that
reuse the reference classes (so parameters, initialisation and everything
around the solver call are the authors' code):

``impl="reference"``  (minimal patch, slow)
    Only the solver strings are resolved to CPU code.  The slow layer's
    ``"cutile"`` branch in the installed ``qkan`` package is pointed at the
    reference repository's own ``cutile_batched_solver`` (its pure-PyTorch
    scalar recurrence, a line-by-line mirror of the cuTile kernel), and the
    fast layer is built with the authors' own ``--fast_solver cutile`` option,
    which uses the same function.  ``FWP.forward`` is untouched.

``impl="fast"``  (same function, ~100x faster on CPU)
    ``FastCPUFWP`` subclasses the reference ``FWP`` and overrides ``forward``
    with an algebraically identical evaluation:

    1. the slow programmer is a pure per-time-step function of the input row,
       and a batch of 32 x 528 rows contains only ~2-3k distinct rows, so it is
       evaluated once per distinct row and gathered (no recomputation);
    2. the slow QKAN layer (reps = 1) is evaluated in closed form with an
       analytic backward pass (see ``pz_reps1_coefficients``);
    3. the gated fast-weight fold ``sum_t (1-g_t) prod_{s>t} g_s  A_t (x) B_t``
       is computed with the reference's own decay coefficients but as one
       batched matrix product instead of materialising the
       ``(B, L, 20, 10, 3, 2)`` tensor.

    Parameters, state-dict keys and RNG consumption (dropout, shuffling) are
    identical to the reference model.  ``verify_cpu.py`` proves the numerical
    agreement of both paths with an independent state-vector simulation and
    with each other (outputs and gradients).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REFERENCE_TASK_DIR = Path("/home/user/matthewpeng57/gated-qkan-fwp/Solar_cycle_forecasting")

# The GPU backward kernels (qkan/cutile_ops.py, reference fast_fused_ops.py)
# define d|amp|/d(amp) = 0 when |amp| <= 1e-30; the closed form does the same.
_NORM_GUARD = 1e-30

_REF = None


def reference():
    """Import the reference task code (read-only) and return its entry points."""
    global _REF
    if _REF is None:
        sys.dont_write_bytecode = True  # never write __pycache__ into the reference
        ref_dir = str(REFERENCE_TASK_DIR)
        if ref_dir not in sys.path:
            sys.path.insert(0, ref_dir)
        import qkan.qkan as upstream_qkan
        import train as ref_train
        from src.data.sunspot import make_datasets
        from src.models import gqkan_qkanfwp
        from src.models.utils import fast_solver, qkan_fast
        from src.trainers import train_loading
        from src.trainers.utils import setup_logger

        _REF = SimpleNamespace(
            upstream_qkan=upstream_qkan,
            make_model=ref_train.make_model,
            make_datasets=make_datasets,
            make_loaders=train_loading.make_loaders,
            run_training=train_loading.run_training,
            train_loading=train_loading,
            setup_logger=setup_logger,
            gqkan_qkanfwp=gqkan_qkanfwp,
            fast_solver=fast_solver,
            qkan_fast=qkan_fast,
        )
    return _REF


def install_cpu_solvers() -> None:
    """Resolve the slow layer's hardcoded ``solver="cutile"`` to CPU code.

    ``qkan.qkan.QKANLayer.forward`` handles ``solver == "cutile"`` by calling
    the module-level ``cutile_flash_exact_solver`` after checking
    ``_CUTILE_AVAILABLE``.  Both names are rebound: the solver becomes the
    reference repository's ``cutile_batched_solver``, which on a CPU tensor
    runs the authors' pure-PyTorch scalar recurrence (same gate sequence and
    sign conventions as the cuTile kernel, float32 state).  Nothing else in
    ``qkan`` or in the reference code is modified.
    """
    ref = reference()
    ref.upstream_qkan.cutile_flash_exact_solver = ref.fast_solver.cutile_batched_solver
    ref.upstream_qkan._CUTILE_AVAILABLE = True


# --------------------------------------------------------------------------
# Closed form of the slow QKAN layer (pz ansatz, reps = 1, fast measure)
# --------------------------------------------------------------------------


def _rz(state, angle):
    """Rz(angle) = diag(e^{-i angle/2}, e^{+i angle/2}) on (r0, i0, r1, i1)."""
    r0, i0, r1, i1 = state
    c, s = torch.cos(0.5 * angle), torch.sin(0.5 * angle)
    return (r0 * c + i0 * s, i0 * c - r0 * s, r1 * c - i1 * s, i1 * c + r1 * s)


def _ry(state, angle):
    """Ry(angle) = [[cos, -sin], [sin, cos]](angle/2) on (r0, i0, r1, i1)."""
    r0, i0, r1, i1 = state
    c, s = torch.cos(0.5 * angle), torch.sin(0.5 * angle)
    return (c * r0 - s * r1, c * i0 - s * i1, s * r0 + c * r1, s * i0 + c * i1)


def pz_reps1_coefficients(theta, preacts_weight, preacts_bias):
    """Per-edge coefficient maps of the reps = 1 pz circuit.

    With ``(b0, b1) = Ry(t01) Rz(t00) H|0>`` and ``phi = (w x + b + t10) / 2``
    the final amplitudes are

        alpha = c b0 e^{-i phi} - s b1 e^{+i phi}
        beta  = s b0 e^{-i phi} + c b1 e^{+i phi},   c, s = cos, sin(t11 / 2)

    hence, with ``-c s b0 conj(b1) = rho e^{i gamma}`` and ``psi = phi - gamma/2``,

        |alpha|^2 = (|c b0| - |s b1|)^2 + 4 rho cos^2(psi) = K1 + K2 cos^2(psi)
        |beta|^2  = (|s b0| - |c b1|)^2 + 4 rho sin^2(psi) = K3 + K2 sin^2(psi)

    Both right-hand sides are sums of non-negative terms, so the float32
    evaluation stays accurate even where an amplitude vanishes.

    Args:
        theta: ``(O, I, 2, 2)``; preacts_weight, preacts_bias: ``(O, I, 1)``.

    Returns:
        ``W, C, K1, K2, K3`` of shape ``(O, I)`` in the dtype of ``theta``
        such that ``psi = W * x + C``.  They are computed in float64 (625
        numbers) and stay attached to the autograd graph.
    """
    dtype = theta.dtype
    th = theta.double()
    inv_sqrt2 = 0.7071067811865476
    start = torch.full_like(th[..., 0, 0], inv_sqrt2)
    zero = torch.zeros_like(start)
    state = _ry(_rz((start, zero, start.clone(), zero.clone()), th[..., 0, 0]), th[..., 0, 1])
    r0, i0, r1, i1 = state
    tiny = 1e-300  # keeps d sqrt finite at an exactly vanishing amplitude
    norm_b0 = torch.sqrt(r0 * r0 + i0 * i0 + tiny)
    norm_b1 = torch.sqrt(r1 * r1 + i1 * i1 + tiny)
    c = torch.cos(0.5 * th[..., 1, 1])
    s = torch.sin(0.5 * th[..., 1, 1])
    abs_c, abs_s = c.abs(), s.abs()
    k1 = (abs_c * norm_b0 - abs_s * norm_b1) ** 2
    k2 = 4.0 * abs_c * abs_s * norm_b0 * norm_b1
    k3 = (abs_s * norm_b0 - abs_c * norm_b1) ** 2
    # -c s b0 conj(b1) = u + i v
    kappa = c * s
    u = -kappa * (r0 * r1 + i0 * i1)
    v = -kappa * (i0 * r1 - r0 * i1)
    defined = (u * u + v * v) > 0.0  # gamma is irrelevant (K2 = 0) where this fails
    gamma = torch.atan2(torch.where(defined, v, torch.zeros_like(v)),
                        torch.where(defined, u, torch.ones_like(u)))
    w = 0.5 * preacts_weight[..., 0].double()
    cst = 0.5 * (preacts_bias[..., 0].double() + th[..., 1, 0] - gamma)
    return w.to(dtype), cst.to(dtype), k1.to(dtype), k2.to(dtype), k3.to(dtype)


class _PZReps1Sum(torch.autograd.Function):
    """``y[n, o] = sum_i sqrt(K1 + K2 cos^2(psi)) - sqrt(K3 + K2 sin^2(psi))``

    with ``psi[n, o, i] = W[o, i] * x[n, i] + C[o, i]``; analytic backward.
    """

    @staticmethod
    def forward(ctx, x, w, cst, k1, k2, k3):
        psi = x.unsqueeze(1) * w
        psi += cst
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin_(psi)
        norm_a = torch.addcmul(k1, k2, cos_psi * cos_psi).sqrt_()  # |alpha|
        norm_b = torch.addcmul(k3, k2, sin_psi * sin_psi).sqrt_()  # |beta|
        ctx.save_for_backward(x, w, k2)
        ctx.cache = (cos_psi, sin_psi, norm_a, norm_b)
        return (norm_a - norm_b).sum(dim=2)

    @staticmethod
    def backward(ctx, grad_y):
        x, w, k2 = ctx.saved_tensors
        cos_psi, sin_psi, norm_a, norm_b = ctx.cache
        half_g = (0.5 * grad_y).unsqueeze(2)                       # (N, O, 1)
        zero = norm_a.new_zeros(())
        inv_a = torch.where(norm_a > _NORM_GUARD, norm_a.reciprocal(), zero)
        inv_b = torch.where(norm_b > _NORM_GUARD, norm_b.reciprocal(), zero)
        grad_a2 = half_g * inv_a                                   # dL/d|alpha|^2
        grad_b2 = half_g * inv_b                                   # -dL/d|beta|^2
        grad_k1 = grad_a2.sum(dim=0)
        grad_k3 = -grad_b2.sum(dim=0)
        cos2 = cos_psi * cos_psi
        sin2 = sin_psi * sin_psi
        grad_k2 = (grad_a2 * cos2).sum(dim=0) - (grad_b2 * sin2).sum(dim=0)
        # d|alpha|^2/dpsi = -2 K2 cos sin,  d|beta|^2/dpsi = +2 K2 cos sin
        grad_psi = grad_a2.add_(grad_b2)
        grad_psi *= cos_psi
        grad_psi *= sin_psi
        grad_psi *= -2.0 * k2
        grad_c = grad_psi.sum(dim=0)
        grad_w = (grad_psi * x.unsqueeze(1)).sum(dim=0)
        grad_x = (grad_psi * w).sum(dim=1)
        return grad_x, grad_w, grad_c, grad_k1, grad_k2, grad_k3


def _check_fusable(qkan_module) -> None:
    """Refuse anything but the slow-programmer configuration of the reference."""
    layers = list(qkan_module.layers)
    problems = []
    if len(layers) != 1 or qkan_module.is_map or qkan_module.norm_out or qkan_module.save_act:
        problems.append("expected a single QKANLayer without map/norm_out/save_act")
    if qkan_module.input_id is not None:
        problems.append("input_id is set")
    layer = layers[0]
    if layer.ansatz not in ("pz_encoding", "pz"):
        problems.append(f"ansatz {layer.ansatz!r}")
    if layer.reps != 1:
        problems.append(f"reps {layer.reps}")
    if not layer.fast_measure:
        problems.append("fast_measure is off")
    if not layer.preact_trainable:
        problems.append("preact_trainable is off")
    if layer.is_batchnorm:
        problems.append("is_batchnorm is on")
    if layer.solver != "cutile":
        problems.append(f"solver {layer.solver!r}")
    if tuple(layer.theta.shape) != (layer.out_dim, layer.in_dim, 2, 2):
        problems.append(f"theta shape {tuple(layer.theta.shape)}")
    for name, value in (("postact_weights", 1.0), ("postact_bias", 0.0), ("mask", 1.0)):
        param = getattr(layer, name)
        if param.requires_grad or not bool((param == value).all()):
            problems.append(f"{name} is not the frozen default {value}")
    if problems:
        raise NotImplementedError("slow QKAN layer cannot use the closed form: " + "; ".join(problems))


def fused_slow_qkan(qkan_module, x: torch.Tensor) -> torch.Tensor:
    """Closed-form forward of the slow programmer's QKAN layer.

    Equals ``qkan_module(x)`` for the reference configuration
    (``width=[25, 25], reps=1, ba_trainable=True, preact_trainable=True``):
    ``sum_i (|alpha| - |beta|)[n, o, i] + sum_i base_weight[o, i] * silu(x[n, i])``.
    """
    layer = qkan_module.layers[0]
    w, cst, k1, k2, k3 = pz_reps1_coefficients(layer.theta, layer.preacts_weight, layer.preacts_bias)
    activation = _PZReps1Sum.apply(x, w, cst, k1, k2, k3)
    return activation + F.linear(layer.base_activation(x), layer.base_weight)


# --------------------------------------------------------------------------
# Fast forward for the reference FWP model
# --------------------------------------------------------------------------


def _fast_forward(self, x):
    """Same function as the reference ``FWP.forward`` (see module docstring)."""
    cell = self.fwp_cell
    batch_size, seq_len, _ = x.size()
    n_out, n_in, n_rep, n_par = cell.theta_shape  # (20, 10, 3, 2)

    # 1. Slow programmer, once per distinct input row.
    x_flat = x.reshape(batch_size * seq_len, -1)
    if x_flat.shape[1] == 1:
        rows, inverse = torch.unique(x_flat[:, 0], return_inverse=True)
        rows = rows.unsqueeze(1)
    else:
        rows, inverse = torch.unique(x_flat, dim=0, return_inverse=True)
    encoder = cell.slow_program_encoder
    hidden = fused_slow_qkan(encoder[1], encoder[0](rows))
    res = encoder[2](hidden)
    gates_bl = torch.sigmoid(cell.fast_gate(res)).index_select(0, inverse).view(batch_size, seq_len)
    A_bl = cell.theta_head_A(res).index_select(0, inverse).view(batch_size, seq_len, n_out * n_rep)
    B_bl = cell.theta_head_B(res).index_select(0, inverse).view(batch_size, seq_len, n_in * n_par)

    # 2. Gated fold.  Decay coefficients exactly as in the reference "legacy
    #    cumsum" path; the sum over time is one batched matrix product:
    #    final_theta[b,o,i,r,k] = sum_t coef[b,t] A[b,t,o,r] B[b,t,i,k].
    log_gates = torch.log(gates_bl + 1e-12)
    reverse_gates_cumsum = torch.cumsum(log_gates.flip(dims=[1]), dim=1).flip(dims=[1])
    decay_to_end = torch.exp(reverse_gates_cumsum - log_gates)
    coef = (1 - gates_bl) * decay_to_end
    folded = torch.bmm((coef.unsqueeze(2) * A_bl).transpose(1, 2), B_bl)    # (B, O*R, I*K)
    final_theta = folded.view(batch_size, n_out, n_rep, n_in, n_par).permute(0, 1, 3, 2, 4)

    # 3. Fast QKAN layer on the last input, 4. projection (reference modules).
    last_input = cell.prelinear(x[:, -1, :])
    final_hidden_state = cell.qkan_layer(last_input, final_theta, None)
    regularized = self.dropout(self.batch_norm(final_hidden_state))
    final_predictions = self.output_layer(regularized)
    if self.output_relu:
        final_predictions = F.relu(final_predictions)
    return final_predictions.unsqueeze(0)


_FAST_CLASS = None


def fast_class():
    """``FastCPUFWP``: the reference ``FWP`` with the fast, equivalent forward."""
    global _FAST_CLASS
    if _FAST_CLASS is None:
        base = reference().gqkan_qkanfwp.FWP
        _FAST_CLASS = type("FastCPUFWP", (base,), {"forward": _fast_forward, "__doc__": _fast_forward.__doc__})
    return _FAST_CLASS


def use_fast_forward(model):
    """Switch a reference ``FWP`` instance to the fast forward, in place.

    Only the class (i.e. the ``forward`` method) changes: modules, parameters,
    buffers and state-dict keys remain those created by the reference
    ``make_model``.
    """
    base = reference().gqkan_qkanfwp.FWP
    if type(model) is not base:
        raise TypeError(f"expected the reference FWP model, got {type(model).__name__}")
    _check_fusable(model.fwp_cell.slow_program_encoder[1])
    if model.fwp_cell.qkan_layer.layers[0].solver != "cutile":
        raise ValueError("build the model with fast_solver='cutile' (the CPU-capable fast solver)")
    model.__class__ = fast_class()
    return model


def build_model(args, impl: str = "fast"):
    """Build the authors' model with the reference ``make_model`` for CPU use.

    ``args`` is the reference CLI namespace (``run.sh`` settings) with
    ``device="cpu"`` and ``fast_solver="cutile"``.
    """
    if impl not in ("fast", "reference"):
        raise ValueError(f"unknown impl {impl!r}")
    if args.device != "cpu" or getattr(args, "fast_solver", None) != "cutile":
        raise ValueError("CPU runs need device='cpu' and fast_solver='cutile'")
    install_cpu_solvers()
    model = reference().make_model(args)
    if impl == "fast":
        use_fast_forward(model)
    return model
