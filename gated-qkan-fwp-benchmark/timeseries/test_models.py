"""Equivalence proof and speed benchmark for the vectorised GQKAN-QKANFWP.

Run with:  python test_models.py            (asserts + speed table)
           python test_models.py --no-speed (asserts only)
"""

from __future__ import annotations

import math
import sys
import time

import torch

import models
import protocol
from reference_repro import count_parameters, make_reference_model

torch.set_num_threads(1)


def statevector_edge_activations(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Literal complex state-vector simulation of the QKAN edge circuit (test oracle)."""
    alpha0, beta0 = theta[..., 0, 0], theta[..., 0, 1]
    alpha1, beta1 = theta[..., 1, 0], theta[..., 1, 1]
    amp0 = torch.full_like(alpha0, math.sqrt(0.5)).to(torch.complex128)  # H |0>
    amp1 = amp0.clone()

    def rz(a0, a1, angle):
        return torch.exp(-0.5j * angle) * a0, torch.exp(0.5j * angle) * a1

    def ry(a0, a1, angle):
        c, s = torch.cos(angle / 2), torch.sin(angle / 2)
        return c * a0 - s * a1, s * a0 + c * a1

    amp0, amp1 = ry(*rz(amp0, amp1, alpha0), beta0)
    amp0, amp1 = rz(amp0, amp1, x.unsqueeze(-2))
    amp0, amp1 = ry(*rz(amp0, amp1, alpha1), beta1)
    return amp0.abs() - amp1.abs()


def reference_and_copy(seed: int, hidden_size: int = 8, perturb: float = 0.0):
    """A reference GQKAN-QKANFWP and our model carrying a copy of its weights."""
    torch.manual_seed(seed)
    reference = make_reference_model("gqkan_qkanfwp", hidden_size, 2)
    if perturb:
        with torch.no_grad():  # move away from the initialisation, as training would
            for p in reference.parameters():
                if p.requires_grad:
                    p.add_(perturb * torch.randn_like(p))
    ours = models.build_model("gqkan_qkanfwp", window=16, hidden_size=hidden_size)
    ours.load_reference_weights(reference)
    return reference, ours


def test_edge_activation_matches_statevector_simulation():
    torch.manual_seed(0)
    worst = 0.0
    for scale in (0.3, 1.0, 3.0, 10.0):
        theta = (scale * torch.randn(2000, 3, 3, 2, 2)).double()
        x = (scale * torch.randn(2000, 3)).double()
        diff = (models.qkan_edge_activations(x, theta) - statevector_edge_activations(x, theta)).abs().max().item()
        worst = max(worst, diff)
    assert worst < 1e-12, worst  # float64: agreement to rounding
    return worst


def test_gated_memory_weights_match_recursion():
    torch.manual_seed(0)
    gate = torch.rand(5, 64, dtype=torch.float64)
    delta = torch.randn(5, 64, 7, dtype=torch.float64)
    theta = torch.zeros(5, 7, dtype=torch.float64)
    for t in range(64):  # the reference recursion
        theta = (1 - gate[:, t, None]) * delta[:, t] + gate[:, t, None] * theta
    unrolled = torch.einsum("bt,btk->bk", models.gated_memory_weights(gate), delta)
    worst = (theta - unrolled).abs().max().item()
    assert worst < 1e-12, worst
    return worst


def test_same_seed_gives_reference_initial_weights():
    """torch.manual_seed(s) + our constructor == the authors' constructor, bit for bit."""
    for seed in (0, 1, 7):
        torch.manual_seed(seed)
        reference = make_reference_model("gqkan_qkanfwp", 8, 2)
        after_reference = torch.rand(1).item()
        torch.manual_seed(seed)
        ours = models.build_model("gqkan_qkanfwp", window=16)
        after_ours = torch.rand(1).item()
        copy = models.build_model("gqkan_qkanfwp", window=16)
        copy.load_reference_weights(reference)
        for (name, a), b in zip(ours.named_parameters(), copy.parameters()):
            assert torch.equal(a, b), name
        assert after_reference == after_ours  # the RNG stream is left in the same state too


def test_parameter_count_matches_table2():
    reference, ours = reference_and_copy(0)
    _, _, effective = count_parameters(reference)
    count = sum(p.numel() for p in ours.parameters() if p.requires_grad)
    assert count == effective == protocol.paper_param_counts()["GQKAN-QKANFWP"] == 159


def test_outputs_match_reference():
    """max |ours - reference| < 1e-5 on random windows, N in {8, 16, 32, 64}."""
    worst = 0.0
    for seed, perturb in ((0, 0.0), (1, 0.0), (2, 0.3), (5, 0.6)):
        reference, ours = reference_and_copy(seed, perturb=perturb)
        for window in (8, 16, 32, 64):
            for batch, low, high in ((64, -1.0, 1.0), (1, -1.0, 1.0), (4, 0.02, 0.18), (16, -3.0, 3.0)):
                x = low + (high - low) * torch.rand(batch, window, 1)
                with torch.no_grad():
                    expected = reference(x)[-1].squeeze(-1)
                    got = ours(x)
                assert got.shape == (batch,)
                diff = (expected - got).abs().max().item()
                worst = max(worst, diff)
                assert diff < 1e-5, (seed, perturb, window, batch, diff)
    return worst


def test_extreme_weights_differ_only_by_float32_rounding():
    """With huge weights the fast QKAN angles reach ~1e3 rad, where float32 resolves an angle
    to ~1e-4 only.  There the two float32 implementations differ by ~1e-4, which is the same
    size as the float32-vs-float64 difference of ONE implementation: rounding, not a mismatch."""
    import copy

    for seed in (3, 4):
        reference, ours = reference_and_copy(seed, perturb=1.0)
        ours64 = copy.deepcopy(ours).double()
        x = 2 * torch.rand(64, 16, 1) - 1
        with torch.no_grad():
            assert ours.final_theta(x).abs().max() > 100
            to_reference = (reference(x)[-1].squeeze(-1) - ours(x)).abs().max().item()
            own_rounding = (ours(x).double() - ours64(x.double())).abs().max().item()
        assert to_reference < 1e-2 and to_reference < 10 * own_rounding, (to_reference, own_rounding)


def test_outputs_match_reference_on_benchmark_data():
    worst = 0.0
    reference, ours = reference_and_copy(0, perturb=0.3)
    for dataset in ("bessel_j2", "narma_10"):
        for window in (8, 64):
            _, _, xte, _ = protocol.load_torch(dataset, window)
            with torch.no_grad():
                diff = (reference(xte)[-1].squeeze(-1) - ours(xte)).abs().max().item()
            worst = max(worst, diff)
            assert diff < 1e-5, (dataset, window, diff)
    return worst


def test_gradients_match_reference():
    """The loss gradients w.r.t. all 159 parameters agree as well."""
    worst = 0.0
    for seed, perturb in ((0, 0.0), (2, 0.3)):
        reference, ours = reference_and_copy(seed, perturb=perturb)
        copy = models.build_model("gqkan_qkanfwp", window=16)
        for window in (8, 64):
            x = 2 * torch.rand(4, window, 1) - 1
            y = torch.rand(4)
            reference.zero_grad()
            ours.zero_grad()
            torch.nn.functional.mse_loss(reference(x)[-1].squeeze(), y).backward()
            torch.nn.functional.mse_loss(ours(x), y).backward()
            # route the reference gradients through the same name mapping as the weights
            for p in reference.parameters():
                if p.requires_grad and p.grad is not None:
                    p.data.copy_(p.grad)
            copy.load_reference_weights(reference)
            scale = max(p.grad.abs().max().item() for p in ours.parameters())
            for (name, mine), theirs in zip(ours.named_parameters(), copy.parameters()):
                diff = (mine.grad - theirs).abs().max().item() / scale
                worst = max(worst, diff)
                assert diff < 1e-4, (name, window, diff)
            reference, ours = reference_and_copy(seed, perturb=perturb)  # weights were overwritten
    return worst


def steps_per_second(model, window, output, seconds=1.5):
    """Adam training steps per second at batch size 4 (the paper's batch size)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = 2 * torch.rand(4, window, 1) - 1
    y = torch.rand(4)

    def step():
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(output(model(x)), y).backward()
        optimizer.step()

    step()
    count, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        step()
        count += 1
    return count / (time.perf_counter() - start)


def speed_table():
    print("\nTraining steps per second (batch 4, Adam, 1 CPU thread):")
    print(f"{'window':>6s} {'reference loop':>15s} {'vectorised':>11s} {'speed-up':>9s}")
    for window in (8, 16, 32, 64):
        reference, ours = reference_and_copy(0)
        slow = steps_per_second(reference, window, lambda out: out[-1].squeeze())
        fast = steps_per_second(ours, window, lambda out: out)
        print(f"{window:6d} {slow:15.1f} {fast:11.1f} {fast / slow:8.1f}x")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            result = test()
            extra = f" (max difference {result:.2e})" if isinstance(result, float) else ""
            print(f"PASS {name}{extra}")
    print("all model tests passed")
    if "--no-speed" not in sys.argv:
        speed_table()
