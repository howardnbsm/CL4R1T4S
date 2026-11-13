"""Proof that the CPU execution of the authors' model computes the authors' math.

Run:  /home/user/venvs/gqkan/bin/python solar/verify_cpu.py

The reference model needs CUDA only because of two solver strings (see
``cpu_backend.py``).  This script checks, on random inputs and parameters,
that the CPU substitutes equal an INDEPENDENT direct simulation of the same
single-qubit circuits:

    oracle 1  ``statevector_pz``: complex128 state-vector simulation written
              here from the gate definitions in the qkan package source
              (``qkan/torch_qc.py``: H, Rz = diag(e^{-it/2}, e^{it/2}),
              Ry = [[c, -s], [s, c]]; circuit order from
              ``qkan/solver/torch_exact.py::_pz_encoding``; measurement
              |alpha| - |beta| from ``StateVector.measure_z(fast_measure=True)``).
    oracle 2  PennyLane ``default.qubit`` final states for a random subset of
              the same circuits (a different simulator altogether).
    oracle 3  the qkan package's own complex64 ``torch_exact_solver``.

Checked configurations:

    (a) slow QKAN layer: ``qkan.QKAN(width=[25, 25], reps=1, ba_trainable=True,
        preact_trainable=True, solver="cutile")``
          - CPU path R: reference repo's pure-PyTorch recurrence behind the
            patched "cutile" branch (impl="reference");
          - CPU path F: closed form with analytic backward (impl="fast").
    (b) fast QKAN layer: reference ``qkan_fast.QKAN(width=[10, 20], reps=2,
        solver="cutile")`` called with a per-sample theta ``(B, 20, 10, 3, 2)``.

For each: max |CPU float32 - oracle| on outputs (must be < 1e-5) and on
gradients (against float64 autograd through oracle 1).  Finally the two
whole-model implementations are compared with each other and with a float64
end-to-end oracle on real training windows, and the effective trainable
parameter count is checked to be 12,474.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import cpu_backend  # noqa: E402
import protocol  # noqa: E402
from reference_cpu import reference_args, seed_everything  # noqa: E402

TOL = 1e-5
REPORT_PATH = protocol.PROJECT_DIR / "runs" / "solar" / "verify_cpu_report.json"


# --------------------------------------------------------------------------
# Oracle 1: independent complex128 state-vector simulation
# --------------------------------------------------------------------------


def _rz_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Rz(angle) = [[e^{-i angle/2}, 0], [0, e^{+i angle/2}]], shape (..., 2, 2)."""
    phase = torch.exp(-0.5j * angle.to(torch.complex128))
    zero = torch.zeros_like(phase)
    return torch.stack([torch.stack([phase, zero], -1), torch.stack([zero, phase.conj()], -1)], -2)


def _ry_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Ry(angle) = [[cos, -sin], [sin, cos]](angle / 2), shape (..., 2, 2)."""
    c = torch.cos(0.5 * angle).to(torch.complex128)
    s = torch.sin(0.5 * angle).to(torch.complex128)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _apply(gate: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...mn,...n->...m", gate, state)


def statevector_pz(x, theta, preacts_weight=None, preacts_bias=None, fast_measure=True):
    """Direct simulation of the pz data re-uploading circuit of every edge.

    Args:
        x: ``(B, I)`` inputs.
        theta: ``(O, I, R+1, 2)`` shared, or ``(B, O, I, R+1, 2)`` per sample.
        preacts_weight, preacts_bias: ``(O, I, R)`` (data angle ``w x + b``) or
            ``None`` (data angle ``x``).

    Returns:
        ``(B, O, I)`` float64 measurement of each edge circuit.
    """
    x = x.double()
    theta = theta.double()
    if theta.dim() == 4:
        theta = theta.unsqueeze(0)
    reps = theta.shape[-2] - 1
    batch, n_in = x.shape
    n_out = theta.shape[1]
    hadamard = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / np.sqrt(2.0)
    state = torch.zeros(batch, n_out, n_in, 2, dtype=torch.complex128)
    state[..., 0] = 1.0                                      # |0>
    state = _apply(hadamard, state)                          # H
    for layer in range(reps):
        state = _apply(_rz_matrix(theta[..., layer, 0]), state)
        state = _apply(_ry_matrix(theta[..., layer, 1]), state)
        if preacts_weight is None:
            angle = x[:, None, :].expand(batch, n_out, n_in)
        else:
            angle = preacts_weight.double()[..., layer] * x[:, None, :] + preacts_bias.double()[..., layer]
        state = _apply(_rz_matrix(angle), state)             # data re-upload
    state = _apply(_rz_matrix(theta[..., reps, 0]), state)
    state = _apply(_ry_matrix(theta[..., reps, 1]), state)
    amp0, amp1 = state[..., 0].abs(), state[..., 1].abs()
    return amp0 - amp1 if fast_measure else amp0**2 - amp1**2


def layer_oracle(layer, x, theta=None, preacts=True):
    """Full QKANLayer output in float64 built on ``statevector_pz``."""
    th = layer.theta if theta is None else theta
    pw, pb = (layer.preacts_weight, layer.preacts_bias) if preacts else (None, None)
    postacts = statevector_pz(x, th, pw, pb, fast_measure=layer.fast_measure)
    base = torch.einsum("oi,bi->boi", layer.base_weight.double(), torch.nn.functional.silu(x.double()))
    edge = (postacts + layer.postact_bias.double()) * layer.postact_weights.double() + base
    return (edge * layer.mask.double()).sum(dim=2)


# --------------------------------------------------------------------------
# Oracle 2: PennyLane default.qubit on a random subset of edge circuits
# --------------------------------------------------------------------------


def pennylane_pz(theta_edges: np.ndarray, data_angles: np.ndarray) -> np.ndarray:
    """|alpha| - |beta| from PennyLane final states.

    Args:
        theta_edges: ``(n, R+1, 2)`` circuit parameters of n edge circuits.
        data_angles: ``(n, R)`` data re-upload angles of the same circuits.
    """
    import pennylane as qml

    reps = data_angles.shape[1]
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(th, enc):
        qml.Hadamard(wires=0)
        for layer in range(reps):
            qml.RZ(th[:, layer, 0], wires=0)
            qml.RY(th[:, layer, 1], wires=0)
            qml.RZ(enc[:, layer], wires=0)
        qml.RZ(th[:, reps, 0], wires=0)
        qml.RY(th[:, reps, 1], wires=0)
        return qml.state()

    state = np.asarray(circuit(theta_edges, data_angles))
    return np.abs(state[:, 0]) - np.abs(state[:, 1])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _max_abs(a, b) -> float:
    return float((a.detach().double() - b.detach().double()).abs().max())


def _randomise_(layer, rng: torch.Generator) -> None:
    """Move every trainable parameter of a QKANLayer away from its init value."""
    with torch.no_grad():
        layer.theta.copy_(torch.randn(layer.theta.shape, generator=rng) * 1.5)
        if layer.preacts_weight.requires_grad:
            layer.preacts_weight.copy_(1.0 + 0.7 * torch.randn(layer.preacts_weight.shape, generator=rng))
            layer.preacts_bias.copy_(torch.randn(layer.preacts_bias.shape, generator=rng))
        if layer.base_weight.requires_grad:
            layer.base_weight.copy_(0.5 + 0.3 * torch.randn(layer.base_weight.shape, generator=rng))


def _grads(loss, tensors):
    return torch.autograd.grad(loss, tensors, allow_unused=False)


def _grad_report(names, got, want) -> dict:
    report = {}
    for name, g, w in zip(names, got, want):
        scale = float(w.abs().max())
        report[name] = {"max_abs_diff": _max_abs(g, w), "max_abs_grad": scale}
    return report


# --------------------------------------------------------------------------
# (a) slow QKAN layer
# --------------------------------------------------------------------------


def check_slow_layer(seed: int, n_rows: int, x_scale: float) -> dict:
    ref = cpu_backend.reference()
    cpu_backend.install_cpu_solvers()
    from qkan import QKAN  # the class the reference model instantiates

    rng = torch.Generator().manual_seed(1000 + seed)
    module = QKAN(width=[25, 25], reps=1, ba_trainable=True, solver="cutile",
                  preact_trainable=True, c_dtype=torch.float32, seed=seed)
    layer = module.layers[0]
    _randomise_(layer, rng)
    x = (torch.randn(n_rows, 25, generator=rng) * x_scale).requires_grad_(True)
    projection = torch.randn(n_rows, 25, generator=rng)
    params = [layer.theta, layer.preacts_weight, layer.preacts_bias, layer.base_weight]
    names = ["x", "theta", "preacts_weight", "preacts_bias", "base_weight"]

    # oracles
    edge_oracle = statevector_pz(x, layer.theta, layer.preacts_weight, layer.preacts_bias)
    out_oracle = layer_oracle(layer, x)
    grads_oracle = _grads((out_oracle * projection.double()).sum(), [x] + params)

    # CPU path R: patched "cutile" branch of the installed qkan package
    kwargs = dict(device="cpu", ansatz=layer.ansatz, group=layer.group, preacts_trainable=True,
                  fast_measure=True, out_dim=25, dtype=torch.float32)
    edge_r = ref.upstream_qkan.cutile_flash_exact_solver(
        x, layer.theta, layer.preacts_weight, layer.preacts_bias, 1, **kwargs)
    out_r = module(x)
    grads_r = _grads((out_r * projection).sum(), [x] + params)
    assert edge_r.dtype == torch.float32 and out_r.dtype == torch.float32

    # CPU path F: closed form
    cpu_backend._check_fusable(module)
    edge_f = cpu_backend.pz_reps1_edge_activation(layer, x.detach())
    out_f = cpu_backend.fused_slow_qkan(module, x)
    grads_f = _grads((out_f * projection).sum(), [x] + params)
    assert edge_f.dtype == torch.float32 and out_f.dtype == torch.float32

    # oracle 3: the qkan package's complex64 exact solver
    from qkan.solver import torch_exact_solver
    kwargs_exact = dict(kwargs, dtype=torch.complex64)
    edge_exact = torch_exact_solver(x, layer.theta, layer.preacts_weight, layer.preacts_bias, 1, **kwargs_exact)

    # oracle 2: PennyLane on 3000 random edges
    idx = torch.randint(0, n_rows * 625, (3000,), generator=rng)
    b, o, i = idx // 625, (idx % 625) // 25, idx % 25
    theta_edges = layer.theta.detach().double()[o, i].numpy()
    enc = (layer.preacts_weight.detach().double()[o, i, 0] * x.detach().double()[b, i]
           + layer.preacts_bias.detach().double()[o, i, 0]).numpy()[:, None]
    edge_pl = torch.from_numpy(pennylane_pz(theta_edges, enc))

    result = {
        "config": "qkan.QKAN(width=[25,25], reps=1, ba_trainable=True, preact_trainable=True, solver='cutile')",
        "seed": seed, "rows": n_rows, "x_scale": x_scale, "edge_circuits": n_rows * 625,
        # solver boundary: one number per edge circuit, |value| <= 1
        "edge_reference_recurrence_vs_statevector": _max_abs(edge_r, edge_oracle),
        "edge_closed_form_vs_statevector": _max_abs(edge_f, edge_oracle),
        "edge_closed_form_vs_reference_recurrence": _max_abs(edge_f, edge_r),
        "edge_statevector_vs_pennylane": _max_abs(edge_oracle[b, o, i], edge_pl),
        "edge_reference_recurrence_vs_pennylane": _max_abs(edge_r[b, o, i], edge_pl),
        "edge_closed_form_vs_pennylane": _max_abs(edge_f[b, o, i], edge_pl),
        "edge_statevector_vs_qkan_torch_exact_complex64": _max_abs(edge_oracle, edge_exact),
        # layer output: sum over 25 edges plus the SiLU base term
        "max_abs_layer_output": float(out_oracle.detach().abs().max()),
        "layer_reference_recurrence_vs_statevector": _max_abs(out_r, out_oracle),
        "layer_closed_form_vs_statevector": _max_abs(out_f, out_oracle),
        "layer_closed_form_vs_reference_recurrence": _max_abs(out_f, out_r),
        "grad_reference_recurrence_vs_statevector": _grad_report(names, grads_r, grads_oracle),
        "grad_closed_form_vs_statevector": _grad_report(names, grads_f, grads_oracle),
    }
    for key in ("edge_reference_recurrence_vs_statevector", "edge_closed_form_vs_statevector",
                "edge_statevector_vs_pennylane", "edge_reference_recurrence_vs_pennylane",
                "edge_closed_form_vs_pennylane"):
        assert result[key] < TOL, (key, result[key])
    # A layer output adds up 25 edges, so its float32 error scales with its magnitude.
    layer_tol = TOL * max(1.0, result["max_abs_layer_output"])
    for key in ("layer_reference_recurrence_vs_statevector", "layer_closed_form_vs_statevector"):
        assert result[key] < layer_tol, (key, result[key])
    for report in (result["grad_reference_recurrence_vs_statevector"], result["grad_closed_form_vs_statevector"]):
        for name, entry in report.items():
            assert entry["max_abs_diff"] < 1e-4 * max(1.0, entry["max_abs_grad"]), (name, entry)
    return result


# --------------------------------------------------------------------------
# (b) fast QKAN layer with per-sample theta
# --------------------------------------------------------------------------


def check_fast_layer(seed: int, batch: int, theta_scale: float) -> dict:
    ref = cpu_backend.reference()
    rng = torch.Generator().manual_seed(2000 + seed)
    module = ref.qkan_fast.QKAN(width=[10, 20], solver="cutile", c_dtype=torch.float32, reps=2, seed=seed)
    layer = module.layers[0]
    assert tuple(layer.theta.shape) == (20, 10, 3, 2)
    assert not layer.preact_trainable and not layer.base_weight.requires_grad
    theta = (torch.randn(batch, 20, 10, 3, 2, generator=rng) * theta_scale).requires_grad_(True)
    x = (torch.randn(batch, 10, generator=rng) * 1.5).requires_grad_(True)
    projection = torch.randn(batch, 20, generator=rng)
    names = ["x", "theta(per-sample)"]

    edge_oracle = statevector_pz(x, theta)                       # data angle = x (preacts frozen at w=1, b=0)
    out_oracle = layer_oracle(layer, x, theta=theta, preacts=False)
    grads_oracle = _grads((out_oracle * projection.double()).sum(), [x, theta])

    kwargs = dict(device="cpu", ansatz=layer.ansatz, group=layer.group, preacts_trainable=False,
                  fast_measure=True, out_dim=20, dtype=torch.float32)
    edge_cpu = ref.fast_solver.cutile_batched_solver(
        x, theta, layer.preacts_weight, layer.preacts_bias, 2, **kwargs)
    out_cpu = module(x, theta, None)                             # exactly how FWP.forward calls it
    grads_cpu = _grads((out_cpu * projection).sum(), [x, theta])
    assert edge_cpu.dtype == torch.float32 and out_cpu.dtype == torch.float32

    # oracle 3: qkan package exact solver, one sample at a time (it has no batched theta)
    from qkan.solver import torch_exact_solver
    kwargs_exact = dict(kwargs, dtype=torch.complex64)
    edge_exact = torch.cat([
        torch_exact_solver(x[k : k + 1], theta[k], layer.preacts_weight, layer.preacts_bias, 2, **kwargs_exact)
        for k in range(batch)
    ])

    # oracle 2: PennyLane on every edge circuit (batch * 200)
    theta_edges = theta.detach().double().reshape(-1, 3, 2).numpy()
    enc = x.detach().double()[:, None, :].expand(batch, 20, 10).reshape(-1, 1).repeat(1, 2).numpy()
    edge_pl = torch.from_numpy(pennylane_pz(theta_edges, enc)).reshape(batch, 20, 10)

    result = {
        "config": "reference qkan_fast.QKAN(width=[10,20], reps=2, solver='cutile'), theta (B,20,10,3,2)",
        "seed": seed, "batch": batch, "theta_scale": theta_scale, "edge_circuits": batch * 200,
        "edge_cpu_vs_statevector": _max_abs(edge_cpu, edge_oracle),
        "layer_cpu_vs_statevector": _max_abs(out_cpu, out_oracle),
        "edge_statevector_vs_pennylane": _max_abs(edge_oracle, edge_pl),
        "edge_cpu_vs_pennylane": _max_abs(edge_cpu, edge_pl),
        "edge_statevector_vs_qkan_torch_exact_complex64": _max_abs(edge_oracle, edge_exact),
        "max_abs_layer_output": float(out_oracle.detach().abs().max()),
        "grad_cpu_vs_statevector": _grad_report(names, grads_cpu, grads_oracle),
    }
    for key in ("edge_cpu_vs_statevector", "edge_statevector_vs_pennylane", "edge_cpu_vs_pennylane"):
        assert result[key] < TOL, (key, result[key])
    assert result["layer_cpu_vs_statevector"] < TOL * max(1.0, result["max_abs_layer_output"])
    for name, entry in result["grad_cpu_vs_statevector"].items():
        assert entry["max_abs_diff"] < 1e-4 * max(1.0, entry["max_abs_grad"]), (name, entry)
    return result


# --------------------------------------------------------------------------
# whole model: reference forward vs fast forward vs float64 end-to-end oracle
# --------------------------------------------------------------------------


def _float64_oracle_model(model):
    """Deep copy in float64 whose two solvers are the independent simulation.

    The copy runs the UNMODIFIED reference ``FWP.forward`` (legacy cumsum
    path); both QKAN layers get a callable solver, the documented extension
    point of ``QKANLayer.forward``, that calls ``statevector_pz``.
    """
    ref = cpu_backend.reference()
    oracle = copy.deepcopy(model)
    oracle.__class__ = ref.gqkan_qkanfwp.FWP
    oracle = oracle.double()

    def slow_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs):
        return statevector_pz(x, theta, preacts_weight, preacts_bias)

    def fast_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs):
        return statevector_pz(x, theta)

    oracle.fwp_cell.slow_program_encoder[1].layers[0].solver = slow_solver
    oracle.fwp_cell.qkan_layer.layers[0].solver = fast_solver
    return oracle


def _capture_hidden(model) -> dict:
    """Record the tensor entering BatchNorm (the fast QKAN layer output) on every call."""
    store = {}
    model.batch_norm.register_forward_pre_hook(
        lambda module, inputs: store.__setitem__("hidden", inputs[0].detach().clone()))
    return store


def _grad_diff(got: dict, want: dict) -> dict:
    """Worst per-tensor relative gradient difference, max|diff| / max|grad|."""
    worst = {"max_rel_diff": 0.0, "tensor": "", "max_abs_diff": 0.0}
    for name, g_want in want.items():
        diff = _max_abs(got[name], g_want)
        rel = diff / max(float(g_want.abs().max()), 1e-30)
        worst["max_abs_diff"] = max(worst["max_abs_diff"], diff)
        if rel > worst["max_rel_diff"]:
            worst["max_rel_diff"], worst["tensor"] = rel, name
    return worst


def check_whole_model(seed: int, n_windows: int, warm_steps: int, use_oracle: bool) -> dict:
    """Reference forward vs fast forward (vs float64 oracle) on real training windows.

    Outputs in training mode pass through BatchNorm with batch statistics,
    which divides by the batch standard deviation of each hidden feature and
    therefore amplifies float32 rounding noise; the comparison is asserted on
    the BatchNorm input ("hidden") and on eval-mode outputs, and the
    training-mode numbers are reported next to the float32-vs-float64 noise
    floor of the reference implementation itself.
    """
    ref = cpu_backend.reference()
    data = protocol.load()  # sealed: train + val only
    args = reference_args(seed=seed, epochs=1)
    loss_fn = ref.train_loading.peak_aware_loss

    seed_everything(seed)
    model_ref = cpu_backend.build_model(args, impl="reference")
    seed_everything(seed)
    model_fast = cpu_backend.build_model(args, impl="fast")
    assert type(model_ref).__name__ == "FWP" and type(model_fast).__name__ == "FastCPUFWP"
    assert list(model_ref.state_dict()) == list(model_fast.state_dict())
    assert all(torch.equal(a, b) for a, b in
               zip(model_ref.state_dict().values(), model_fast.state_dict().values()))

    rng = np.random.default_rng(seed)
    if warm_steps:
        # A few optimiser steps so that the comparison is not made at the
        # (special) initial point; then both models get the same weights.
        optimiser = torch.optim.Adam(model_fast.parameters(), lr=2.5e-3)
        model_fast.train()
        for _ in range(warm_steps):
            sub = rng.choice(len(data.train), size=32, replace=False)
            optimiser.zero_grad()
            out = model_fast(torch.from_numpy(data.X[sub]).unsqueeze(-1))
            loss_fn(out[-1].squeeze(), torch.from_numpy(data.Y[sub]), alpha=1.0).backward()
            optimiser.step()
        model_ref.load_state_dict(model_fast.state_dict())

    pick = np.sort(rng.choice(len(data.train), size=n_windows, replace=False))
    x = torch.from_numpy(data.X[pick]).unsqueeze(-1)
    y = torch.from_numpy(data.Y[pick])

    models = {"reference": model_ref, "fast": model_fast}
    if use_oracle:
        models["oracle"] = _float64_oracle_model(model_ref)
    hidden = {name: _capture_hidden(model) for name, model in models.items()}
    dropout_p = model_ref.dropout.p

    def run(train: bool, dropout: bool) -> dict:
        outputs = {}
        for name, model in models.items():
            if name == "oracle" and dropout:
                continue  # float64 dropout masks differ from float32 ones
            model.train(train)
            model.dropout.p = dropout_p if dropout else 0.0
            torch.manual_seed(1234)  # same dropout stream for both implementations
            model.zero_grad()
            double = name == "oracle"
            out = model(x.double() if double else x)
            loss = loss_fn(out[-1].squeeze(), y.double() if double else y, alpha=1.0)
            grads = {}
            if train:
                loss.backward()
                grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            outputs[name] = {"out": out.detach(), "hidden": hidden[name]["hidden"], "loss": loss.item(),
                             "grads": grads}
            model.dropout.p = dropout_p
        return outputs

    result = {"seed": seed, "windows": n_windows, "warm_steps": warm_steps,
              "distinct_rows": int(torch.unique(x).numel()), "total_rows": int(x.shape[0] * x.shape[1])}

    def compare(prefix: str, runs: dict, with_grads: bool) -> None:
        pairs = [("fast", "reference")]
        if "oracle" in runs:
            pairs += [("reference", "oracle"), ("fast", "oracle")]
        for a, b in pairs:
            result[f"{prefix}_hidden_{a}_vs_{b}"] = _max_abs(runs[a]["hidden"], runs[b]["hidden"])
            result[f"{prefix}_output_{a}_vs_{b}"] = _max_abs(runs[a]["out"], runs[b]["out"])
            result[f"{prefix}_loss_{a}_vs_{b}"] = abs(runs[a]["loss"] - runs[b]["loss"])
            if with_grads:
                worst = _grad_diff(runs[a]["grads"], runs[b]["grads"])
                result[f"{prefix}_grad_{a}_vs_{b}_max_rel_diff"] = worst["max_rel_diff"]
                result[f"{prefix}_grad_{a}_vs_{b}_worst_tensor"] = worst["tensor"]
        result[f"{prefix}_loss_reference"] = runs["reference"]["loss"]
        result[f"{prefix}_max_abs_output"] = float(runs["reference"]["out"].abs().max())

    # eval mode: BatchNorm running statistics, no dropout
    eval_runs = run(train=False, dropout=False)
    assert eval_runs["reference"]["out"].shape == (1, n_windows, 132)
    compare("eval", eval_runs, with_grads=False)
    # training mode without dropout (the float64 oracle can take part)
    train_runs = run(train=True, dropout=False)
    compare("train_nodropout", train_runs, with_grads=True)
    # training mode with dropout, identical RNG stream
    drop_runs = run(train=True, dropout=True)
    compare("train_dropout", drop_runs, with_grads=True)

    batch_std = train_runs["reference"]["hidden"].std(dim=0)
    result["min_batch_std_of_hidden_feature"] = float(batch_std.min())
    amplification = max(1.0, 1.0 / float(batch_std.min()))
    result["batchnorm_noise_amplification_bound"] = amplification

    for name in ("reference", "fast"):
        total, effective = ref.train_loading.count_effective_trainable(models[name])
        result[f"params_total_{name}"] = total
        result[f"params_effective_trainable_{name}"] = effective
        assert effective == 12474, effective

    for key, value in result.items():
        if "_vs_" not in key or key.endswith("_worst_tensor"):
            continue
        if "_hidden_" in key or key.startswith("eval_output_"):
            assert value < TOL, (key, value)                      # not BatchNorm-amplified
        elif "_output_" in key:
            assert value < TOL * amplification, (key, value)      # training-mode BatchNorm
        elif key.endswith("_max_rel_diff"):
            assert value < 2e-3, (key, value)
    return result


def _print(title: str, result: dict) -> None:
    print(f"--- {title}")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"    {key}:")
            for sub, entry in value.items():
                print(f"        {sub:18s} max|diff| {entry['max_abs_diff']:.3e}   (max|grad| {entry['max_abs_grad']:.3e})")
        elif isinstance(value, float):
            print(f"    {key:55s} {value:.3e}")
        else:
            print(f"    {key:55s} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smaller problem sizes")
    cli = parser.parse_args()
    t0 = time.time()
    report = {"tolerance": TOL, "slow_layer": [], "fast_layer": [], "whole_model": []}

    slow_cases = [(0, 512, 2.0)] if cli.quick else [(0, 2048, 2.0), (1, 2048, 0.5), (2, 1024, 6.0)]
    for seed, rows, scale in slow_cases:
        result = check_slow_layer(seed, rows, scale)
        report["slow_layer"].append(result)
        _print(f"(a) slow QKAN layer, seed {seed}, {rows} rows, x ~ N(0, {scale}^2)", result)

    fast_cases = [(0, 32, 1.5)] if cli.quick else [(0, 32, 1.5), (1, 64, 0.3), (2, 32, 4.0)]
    for seed, batch, scale in fast_cases:
        result = check_fast_layer(seed, batch, scale)
        report["fast_layer"].append(result)
        _print(f"(b) fast QKAN layer, seed {seed}, batch {batch}, theta ~ N(0, {scale}^2)", result)

    model_cases = ([(0, 4, 0, True)] if cli.quick else
                   [(0, 8, 0, True), (0, 8, 30, True), (3, 8, 30, True), (0, 32, 0, False), (1, 32, 30, False)])
    for seed, windows, warm, use_oracle in model_cases:
        result = check_whole_model(seed, windows, warm, use_oracle)
        report["whole_model"].append(result)
        _print(f"whole model, seed {seed}, {windows} real training windows, {warm} warm-up steps, "
               f"float64 oracle: {use_oracle}", result)

    report["seconds"] = time.time() - t0
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"ALL CPU-EQUIVALENCE CHECKS PASSED (tolerance {TOL:g} on outputs) in {report['seconds']:.1f}s")
    print(f"report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
