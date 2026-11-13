"""How well can ANY model predict the NARMA targets from the input window alone?

In the reference data the NARMA input u_t is a deterministic, smooth function of time and
the target y_{i+N} is a deterministic function of u_0 .. u_{i+N-1}:

    y_{t+1} = 0.3 y_t + 0.05 y_t * sum_{j<n0} y_{t-j} + 1.5 u_{t-n0+1} u_t + 0.1

A window x_i = (u_i .. u_{i+N-1}) therefore determines its target exactly whenever no two
windows coincide (checked below), i.e. the Bayes-optimal error given only the window is 0
in principle.  What limits a model that has learned the true recursion but sees only the
window is (a) the unknown output state at the window start, which is forgotten
geometrically, and (b) the delayed inputs u_{t-n0+1} that lie before the window start.
This script measures both floors on the TEST windows with the exact generator equation:

    oracle_state   true recursion, true earlier inputs, unknown initial state (set to the
                   training-target mean)                      -> fading-memory floor
    oracle_window  true recursion, window only; missing earlier inputs are replaced by
                   the first window value                     -> window-only floor

Usage:  python narma_analysis.py
"""

from __future__ import annotations

import json

import numpy as np

import protocol

ALPHA, BETA, GAMMA, DELTA = 0.3, 0.05, 1.5, 0.1  # as in the reference NARMA_n_Generator


def narma_step(y_history: np.ndarray, u_now: float, u_delayed: float) -> float:
    """One NARMA step.  ``y_history`` holds y_t, y_{t-1}, ..., y_{t-n0+1} (most recent first)."""
    return ALPHA * y_history[0] + BETA * y_history[0] * y_history.sum() + GAMMA * u_delayed * u_now + DELTA


def reference_series(order: int):
    """The reference input u and output y (length 300 each), in float64."""
    protocol.add_reference_to_path()
    from src.datasets.narma_generator import NARMA_n_Generator  # the authors' code

    u, y = NARMA_n_Generator(n_0=order, T=300)
    return np.asarray(u, dtype=np.float64), np.asarray(y, dtype=np.float64)


def simulate_window(u: np.ndarray, start: int, window: int, order: int, initial_state: np.ndarray,
                    earlier_inputs: str) -> float:
    """Run the true recursion over u[start : start + window] and return the prediction of y[start + window].

    ``initial_state`` is the assumed (y_start, y_{start-1}, ..., y_{start-order+1}).
    ``earlier_inputs``: "true" uses the real u before the window (0 before the series start,
    as the generator does); "hold" replaces every u before the window by u[start].
    """
    history = np.array(initial_state, dtype=np.float64)
    for t in range(start, start + window):
        delayed_index = t - order + 1
        if delayed_index >= start:
            u_delayed = u[delayed_index]
        elif earlier_inputs == "true":
            u_delayed = u[delayed_index] if delayed_index >= 0 else 0.0
        else:
            u_delayed = u[start]
        y_next = narma_step(history, u[t], u_delayed)
        history = np.concatenate([[y_next], history[:-1]])
    return history[0]


def memory_decay_rate(order: int) -> float:
    """Per-step factor by which a perturbation of the output state decays (measured)."""
    u, y = reference_series(order)
    start, steps = 100, 12
    exact = np.array([y[start - j] for j in range(order)])
    predictions = []
    for bump in (0.0, 1e-3):
        state = exact.copy()
        state[0] += bump
        predictions.append(simulate_window(u, start, steps, order, state, "true"))
    return float((abs(predictions[1] - predictions[0]) / 1e-3) ** (1.0 / steps))


def analyse(dataset: str, window: int) -> dict:
    order = 5 if dataset == "narma_5" else 10
    u, y = reference_series(order)
    x_train, y_train, x_test, y_test = protocol.load(dataset, window)
    n_train, n_test = len(y_train), len(y_test)

    # the windows are distinct, so window -> target is a function (zero Bayes error in principle)
    windows = np.concatenate([x_train, x_test])[:, :, 0].astype(np.float64)
    distances = np.sqrt(((windows[:, None, :] - windows[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(distances, np.inf)
    smallest_gap = float(distances.min())

    # sanity: with the exact state and exact earlier inputs the recursion IS the generator
    exact = [simulate_window(u, i, window, order, [y[i - j] if i - j >= 0 else 0.0 for j in range(order)], "true")
             for i in range(n_train, n_train + n_test)]
    assert np.max(np.abs(np.array(exact) - y_test)) < 1e-6, "recursion does not reproduce the reference targets"

    guess = np.full(order, float(y_train.mean()))  # unknown state -> training-target mean
    result = {"dataset": dataset, "window": window, "order": order, "smallest_window_distance": smallest_gap,
              "test_target_variance": float(np.var(y_test.astype(np.float64)))}
    for name, earlier in (("oracle_state", "true"), ("oracle_window", "hold")):
        prediction = [simulate_window(u, i, window, order, guess, earlier) for i in range(n_train, n_train + n_test)]
        result[name] = protocol.test_mse(np.array(prediction), y_test)
    result["paper_best"] = protocol.paper_best(dataset, window)
    return result


def main():
    baseline_file = protocol.RUNS_ROOT / "classical_baselines" / "results.json"
    closed_form = {}
    if baseline_file.exists():
        closed_form = {(r["dataset"], r["window"]): r for r in json.loads(baseline_file.read_text())}
    for order in (5, 10):
        print(f"NARMA-{order}: a perturbation of the output state shrinks by a factor "
              f"{memory_decay_rate(order):.3f} per step")
    print()
    print(f"{'dataset':9s} {'N':>3s} {'var(y_test)':>11s} {'paper best':>10s} {'ridge[window,1]':>15s} "
          f"{'closed-form sel.':>16s} {'oracle_state':>12s} {'oracle_window':>13s} {'min window gap':>14s}")
    rows = []
    for dataset in protocol.NARMA_DATASETS:
        for window in protocol.WINDOWS:
            row = analyse(dataset, window)
            fit = closed_form.get((dataset, window))
            ridge = f"{fit['families']['ridge_window']['test_mse']:.2e}" if fit else "n/a"
            selected = f"{fit['test_mse']:.2e}" if fit else "n/a"
            print(f"{dataset:9s} {window:3d} {row['test_target_variance']:11.2e} {row['paper_best']:10.1e} "
                  f"{ridge:>15s} {selected:>16s} {row['oracle_state']:12.2e} {row['oracle_window']:13.2e} "
                  f"{row['smallest_window_distance']:14.2e}")
            rows.append(row)
    out = protocol.RUNS_ROOT / "narma_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
