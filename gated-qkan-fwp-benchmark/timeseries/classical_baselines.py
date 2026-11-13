"""Closed-form baselines for the 24 paper cells, with NO test-set selection.

Every hyperparameter (ridge strength, number of lags, polynomial degree, kernel
width, and the choice between model families) is chosen on a chronological
hold-out carved from the END of the training split (its last 20%, mirroring the
benchmark's own 80/20 split).  The chosen configuration is then refitted on the
full training split and evaluated ONCE on the test split with the paper's metric
(``protocol.test_mse``).  The models are deterministic, so the "5-seed std" is 0.

Families
    ridge_window   ridge regression on [window, 1]            (N + 1 parameters)
    poly_lags      ridge on all monomials of degree <= d of the last K window
                   entries; d in {1, 2, 3} and K are selected on the hold-out
    kernel_ridge   RBF kernel ridge on the last K entries     (NARMA cells only)
    selected       the family with the lowest hold-out MSE    (headline number)

Usage
    python classical_baselines.py                 # all 24 cells -> JSON + markdown
    python classical_baselines.py --datasets narma_10 --windows 16

Outputs: runs/timeseries/classical_baselines/results.json and results.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

import protocol

OUT_DIR = protocol.RUNS_ROOT / "classical_baselines"
HOLDOUT_FRACTION = 0.2
RIDGE_GRID = tuple(10.0 ** e for e in range(-14, 1))  # relative to the number of samples
LAG_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64)
MAX_LAGS_BY_DEGREE = {1: 64, 2: 32, 3: 12}  # keeps the number of monomials moderate
RBF_WIDTH_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)  # length scale in units of sqrt(K) std
KERNEL_LAG_GRID = (4, 6, 8, 10, 12, 16)


# ------------------------------------------------------------------ building blocks
def holdout_split(n_train: int, fraction: float = HOLDOUT_FRACTION) -> int:
    """Number of leading training samples used for fitting during model selection."""
    return n_train - max(1, int(round(fraction * n_train)))


def monomial_features(x: np.ndarray, lags: int, degree: int) -> np.ndarray:
    """All monomials of degree 1..``degree`` of the last ``lags`` window entries.

    x: (n, N) window matrix  ->  (n, C(lags + degree, degree) - 1) feature matrix.
    """
    recent = x[:, -lags:]
    columns = [recent]
    for d in range(2, degree + 1):
        products = [np.prod(recent[:, list(combo)], axis=1)
                    for combo in itertools.combinations_with_replacement(range(lags), d)]
        columns.append(np.stack(products, axis=1))
    return np.concatenate(columns, axis=1)


class RidgePath:
    """Ridge regression on standardised features with an unpenalised intercept.

    One SVD of the feature matrix gives the solution for every ridge strength
    ``lam`` (the penalty is ``lam * n_samples * ||w||^2``), in a numerically stable
    way even when the features are almost collinear (smooth series).
    """

    def __init__(self, features: np.ndarray, y: np.ndarray):
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.y_mean = y.mean()
        z = (features - self.mean) / self.scale
        self.n = len(y)
        u, self.singular, self.vt = np.linalg.svd(z, full_matrices=False)
        self.projected = u.T @ (y - self.y_mean)

    def weights(self, lam: float) -> np.ndarray:
        shrink = self.singular / (self.singular ** 2 + lam * self.n)
        return self.vt.T @ (shrink * self.projected)

    def predict(self, features: np.ndarray, lam: float) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.weights(lam) + self.y_mean


def rbf_kernel(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    squared = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.exp(-np.maximum(squared, 0.0) / (2.0 * length ** 2))


class KernelRidgePath:
    """RBF kernel ridge on standardised inputs; one eigendecomposition serves every ``lam``."""

    def __init__(self, x: np.ndarray, y: np.ndarray, length: float):
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.z = (x - self.mean) / self.scale
        self.length = length * np.sqrt(x.shape[1])
        self.y_mean = y.mean()
        self.n = len(y)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(rbf_kernel(self.z, self.z, self.length))
        self.projected = self.eigenvectors.T @ (y - self.y_mean)

    def predict(self, x: np.ndarray, lam: float) -> np.ndarray:
        dual = self.eigenvectors @ (self.projected / (self.eigenvalues + lam * self.n))
        return rbf_kernel((x - self.mean) / self.scale, self.z, self.length) @ dual + self.y_mean


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


# ---------------------------------------------------------------------- the families
def _select_ridge(features_fit, y_fit, features_val, y_val):
    """Best ridge strength on the hold-out for one feature map: (hold-out MSE, lam)."""
    path = RidgePath(features_fit, y_fit)
    scores = [(mse(path.predict(features_val, lam), y_val), lam) for lam in RIDGE_GRID]
    return min(scores)


def fit_poly_lags(x_train, y_train, x_test, degrees, lag_grid, fixed_lags=None):
    """Select (degree, lags, lam) on the hold-out, refit on all training data, predict test."""
    n_fit = holdout_split(len(y_train))
    window = x_train.shape[1]
    candidates = []
    for degree in degrees:
        lags_options = [fixed_lags] if fixed_lags else [k for k in lag_grid
                                                        if k <= min(window, MAX_LAGS_BY_DEGREE[degree])]
        for lags in lags_options:
            features = monomial_features(x_train, lags, degree)
            score, lam = _select_ridge(features[:n_fit], y_train[:n_fit], features[n_fit:], y_train[n_fit:])
            candidates.append((score, degree, lags, lam))
    score, degree, lags, lam = min(candidates)  # ties -> lower degree, fewer lags
    path = RidgePath(monomial_features(x_train, lags, degree), y_train)
    prediction = path.predict(monomial_features(x_test, lags, degree), lam)
    return {"holdout_mse": score, "prediction": prediction,
            "hyperparameters": {"degree": degree, "lags": lags, "ridge": lam},
            "n_parameters": len(path.mean) + 1}


def fit_kernel_ridge(x_train, y_train, x_test):
    """Select (lags, RBF width, lam) on the hold-out, refit on all training data, predict test."""
    n_fit = holdout_split(len(y_train))
    window = x_train.shape[1]
    candidates = []
    for lags in [k for k in KERNEL_LAG_GRID if k <= window]:
        recent = x_train[:, -lags:]
        for length in RBF_WIDTH_GRID:
            path = KernelRidgePath(recent[:n_fit], y_train[:n_fit], length)
            for lam in RIDGE_GRID:
                candidates.append((mse(path.predict(recent[n_fit:], lam), y_train[n_fit:]), lags, length, lam))
    score, lags, length, lam = min(candidates)
    path = KernelRidgePath(x_train[:, -lags:], y_train, length)
    return {"holdout_mse": score, "prediction": path.predict(x_test[:, -lags:], lam),
            "hyperparameters": {"lags": lags, "rbf_length": length, "ridge": lam},
            "n_parameters": len(y_train) + 1}  # one dual coefficient per training sample + mean


def run_cell(dataset: str, window: int) -> dict:
    """All families for one cell.  The test targets are only touched by ``protocol.test_mse``."""
    x_train, y_train, x_test, y_test = protocol.load(dataset, window)
    x_train64 = x_train[:, :, 0].astype(np.float64)
    x_test64 = x_test[:, :, 0].astype(np.float64)
    y_train64 = y_train.astype(np.float64)

    families = {
        "ridge_window": fit_poly_lags(x_train64, y_train64, x_test64, degrees=(1,), lag_grid=(), fixed_lags=window),
        "poly_lags": fit_poly_lags(x_train64, y_train64, x_test64, degrees=(1, 2, 3), lag_grid=LAG_GRID),
    }
    if dataset in protocol.NARMA_DATASETS:
        families["kernel_ridge"] = fit_kernel_ridge(x_train64, y_train64, x_test64)
    selected = min(families, key=lambda name: families[name]["holdout_mse"])  # hold-out only

    record = {"dataset": dataset, "window": window, "n_train": len(y_train), "n_test": len(y_test),
              "n_fit_for_selection": holdout_split(len(y_train)), "families": {}}
    for name, fit in families.items():
        record["families"][name] = {
            "test_mse": protocol.test_mse(fit["prediction"], y_test),  # the paper's metric
            "holdout_mse": fit["holdout_mse"], "n_parameters": fit["n_parameters"],
            "hyperparameters": fit["hyperparameters"],
        }
    record["selected_family"] = selected
    record["test_mse"] = record["families"][selected]["test_mse"]
    record["n_parameters"] = record["families"][selected]["n_parameters"]
    record["std_over_seeds"] = 0.0  # deterministic closed-form fit
    if protocol.is_paper_cell(dataset, window):
        best = protocol.paper_best(dataset, window)
        record.update(paper_best=best, paper_best_model=protocol.paper_best_model(dataset, window),
                      ratio_to_paper_best=record["test_mse"] / best,
                      ridge_window_ratio=record["families"]["ridge_window"]["test_mse"] / best)
    return record


# --------------------------------------------------------------------------- report
def markdown_table(records: list[dict]) -> str:
    lines = [
        "Closed-form baselines; every hyperparameter chosen on the last 20% of the TRAINING split, "
        "refit on the full training split, test set evaluated once. Deterministic: std over seeds = 0.",
        "",
        "| dataset | N | ridge on [window,1]: test MSE (params) | poly-lags: test MSE (degree, lags, params) "
        "| kernel ridge: test MSE | selected by hold-out | selected test MSE | params | paper best "
        "| ours / paper best | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        fam = r["families"]
        ridge = f"{fam['ridge_window']['test_mse']:.2e} ({fam['ridge_window']['n_parameters']})"
        hp = fam["poly_lags"]["hyperparameters"]
        poly = f"{fam['poly_lags']['test_mse']:.2e} (d={hp['degree']}, K={hp['lags']}, {fam['poly_lags']['n_parameters']})"
        kernel = f"{fam['kernel_ridge']['test_mse']:.2e}" if "kernel_ridge" in fam else "-"
        if "paper_best" in r:
            verdict = "WIN" if round(r["test_mse"], 6) < r["paper_best"] else "lose"
            tail = f"{r['paper_best']:.1e} | {r['ratio_to_paper_best']:.2e} | {verdict}"
        else:
            tail = "non-target | - | -"
        lines.append(f"| {r['dataset']} | {r['window']} | {ridge} | {poly} | {kernel} | {r['selected_family']} "
                     f"| {r['test_mse']:.2e} | {r['n_parameters']} | {tail} |")
    targets = [r for r in records if "paper_best" in r]
    wins = sum(round(r["test_mse"], 6) < r["paper_best"] for r in targets)
    ridge_wins = sum(round(r["families"]["ridge_window"]["test_mse"], 6) < r["paper_best"] for r in targets)
    lines += ["", f"Selected closed-form model beats the paper's best mean on {wins} / {len(targets)} paper cells; "
                  f"plain ridge on [window,1] alone on {ridge_wins} / {len(targets)}."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(protocol.PAPER_DATASETS), choices=list(protocol.ALL_DATASETS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(protocol.WINDOWS))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    records = []
    for dataset in args.datasets:
        for window in args.windows:
            start = time.time()
            record = run_cell(dataset, window)
            record["seconds"] = time.time() - start
            records.append(record)
            print(f"{dataset:24s} w{window:<3d} selected {record['selected_family']:13s} "
                  f"test MSE {record['test_mse']:.3e}  ({record['seconds']:.1f}s)", flush=True)
    table = markdown_table(records)
    print("\n" + table)
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(records, indent=2))
        (OUT_DIR / "results.md").write_text(table + "\n")
        print(f"\nwrote {OUT_DIR / 'results.json'} and {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
