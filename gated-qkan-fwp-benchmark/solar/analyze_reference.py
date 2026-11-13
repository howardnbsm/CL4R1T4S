"""Validation-only reference points for the next phase (the test split stays sealed).

    /home/user/venvs/gqkan/bin/python solar/analyze_reference.py

1. Simple, parameter-free baselines on the VALIDATION windows, scored with
   ``protocol.evaluate_split`` (reference metrics):
     - climatology: the mean training target curve, and the training mean value;
     - last value held, and the mean of the last 12 months held;
     - cycle repeat: the last 132 input months copied forward, and the average
       of the last four 132-month blocks;
     - oracle smooth forecast: the 13-month centred running mean of the TRUE
       future.  No model can produce it; it shows how much peak-amplitude and
       peak-timing error a perfectly accurate *smooth* forecast still has
       against raw monthly targets.
2. The authors' model, re-scored on validation windows from the best-validation
   checkpoints of the CPU reproduction (``runs/solar/reference/seed<k>``), per
   seed and as a seed-ensemble (mean forecast).
3. What the trained slow programmer does: gate values, the effective memory of
   the gated fold (how many months back the fast weights actually look), and
   how many hidden features vary across windows.

Everything here uses train + validation data only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from reference_cpu import DEFAULT_OUT_ROOT, reference_args  # noqa: E402

KEYS = ("scaled_mse", "pae", "pte", "peak_aware_mse")


def _row(name: str, m: dict) -> str:
    return (f"{name:<46s} scaled MSE {m['scaled_mse']:.4f}  PAE {m['pae']:6.2f}  PTE {m['pte']:6.2f}  "
            f"peak-aware loss {m['peak_aware_mse']:.4f}")


def simple_baselines(data) -> dict:
    xtr, ytr = data.split("train")
    xva, yva = data.split("val")
    n = len(xva)
    baselines = {
        "climatology: mean training target curve": np.repeat(ytr.mean(axis=0, keepdims=True), n, axis=0),
        "climatology: training mean value": np.full_like(yva, ytr.mean()),
        "persistence: last value held": np.repeat(xva[:, -1:], 132, axis=1),
        "persistence: mean of last 12 months held": np.repeat(xva[:, -12:].mean(axis=1, keepdims=True), 132, axis=1),
        "cycle repeat: last 132 input months": xva[:, -132:],
        "cycle repeat: mean of last four 132-month blocks": xva.reshape(n, 4, 132).mean(axis=1),
    }
    # Oracle smooth forecast: 13-month centred running mean of the true series
    # (uses only months covered by train + val windows; edges use shorter windows).
    series = data.series
    kernel = np.ones(13)
    smooth = np.convolve(series, kernel, mode="same") / np.convolve(np.ones_like(series), kernel, mode="same")
    starts = np.arange(data.val.start, data.val.stop) + protocol.SEQ_LEN
    baselines["ORACLE (not a forecast): 13-month smoothed truth"] = np.stack(
        [smooth[s : s + protocol.HORIZON] for s in starts])
    results = {}
    for name, pred in baselines.items():
        results[name] = protocol.evaluate_split(pred.astype(np.float32), "val", tag="baseline")
    return results


def load_reference_model(seed: int, root: Path):
    args = reference_args(seed=seed, epochs=100)
    model = cpu_backend.build_model(args, impl="fast")
    state = torch.load(root / f"seed{seed}" / "best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def predict(model, x: np.ndarray) -> np.ndarray:
    out = [model(torch.from_numpy(x[i : i + 64]).unsqueeze(-1))[-1] for i in range(0, len(x), 64)]
    return torch.cat(out).numpy()


@torch.no_grad()
def programmer_statistics(model, x: np.ndarray) -> dict:
    """Gate values and effective memory of the gated fold on the given windows."""
    cell = model.fwp_cell
    xt = torch.from_numpy(x).unsqueeze(-1)
    b, length, _ = xt.shape
    res = cell.slow_program_encoder(xt.reshape(b * length, 1))
    gates = torch.sigmoid(cell.fast_gate(res)).view(b, length).double()
    # weight of time step t in final_theta: (1 - g_t) * prod_{s > t} g_s
    log_g = torch.log(gates + 1e-12)
    decay = torch.exp(torch.cumsum(log_g.flip(1), 1).flip(1) - log_g)
    weight = (1 - gates) * decay
    lag = torch.arange(length - 1, -1, -1, dtype=torch.float64)       # months before the window end
    mass = weight.sum(1)
    mean_lag = (weight * lag).sum(1) / mass
    cumulative = torch.cumsum(weight.flip(1), 1) / mass[:, None]       # from the most recent month back
    lag_90 = (cumulative < 0.9).sum(1).double()                        # months holding 90% of the weight
    hidden = {}
    handle = model.batch_norm.register_forward_pre_hook(lambda m, i: hidden.__setitem__("h", i[0].clone()))
    model(xt)
    handle.remove()
    std = hidden["h"].std(dim=0)
    return {
        "gate_min": float(gates.min()), "gate_mean": float(gates.mean()), "gate_max": float(gates.max()),
        "weight_mass_mean": float(mass.mean()),
        "mean_lag_months": float(mean_lag.mean()),
        "months_holding_90pct_of_weight_mean": float(lag_90.mean()),
        "months_holding_90pct_of_weight_max": float(lag_90.max()),
        "hidden_feature_std_across_windows": [round(float(v), 5) for v in std],
        "hidden_features_with_std_below_1e-3": int((std < 1e-3).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    cli = parser.parse_args()

    data = protocol.load()  # sealed
    report = {"split": "val", "baselines": {}, "authors_model": {}, "programmer": {}}

    print("Simple baselines on the VALIDATION windows (260 windows)")
    for name, m in simple_baselines(data).items():
        report["baselines"][name] = {k: m[k] for k in KEYS}
        print("  " + _row(name, m))

    xva, _ = data.split("val")
    preds = {}
    for seed in cli.seeds:
        if not (cli.root / f"seed{seed}" / "best_model.pth").is_file() or \
                not (cli.root / f"seed{seed}" / "final_metrics.json").is_file():
            print(f"  seed {seed}: reproduction not finished, skipped")
            continue
        model = load_reference_model(seed, cli.root)
        preds[seed] = predict(model, xva)
        report["programmer"][str(seed)] = programmer_statistics(model, xva)
    if preds:
        print("Authors' GQKAN-QKANFWP (CPU reproduction, best-validation checkpoints) on VALIDATION windows")
        per_seed = {}
        for seed, pred in preds.items():
            m = protocol.evaluate_split(pred, "val", tag=f"authors-model seed {seed}")
            per_seed[seed] = m
            report["authors_model"][f"seed{seed}"] = {k: m[k] for k in KEYS}
            print("  " + _row(f"seed {seed}", m))
        for key in KEYS:
            values = np.array([per_seed[s][key] for s in per_seed])
            report["authors_model"][f"mean_{key}"] = float(values.mean())
            report["authors_model"][f"std_population_{key}"] = float(values.std())
        print("  " + _row(f"mean over {len(preds)} seeds", {k: report["authors_model"][f"mean_{k}"] for k in KEYS}))
        ensemble = protocol.evaluate_split(np.mean(list(preds.values()), axis=0), "val", tag="authors-model ensemble")
        report["authors_model"]["seed_ensemble"] = {k: ensemble[k] for k in KEYS}
        print("  " + _row(f"ensemble (mean forecast of {len(preds)} seeds)", ensemble))

        print("Trained slow programmer on VALIDATION windows")
        for seed, stats in report["programmer"].items():
            print(f"  seed {seed}: gate min/mean/max {stats['gate_min']:.3f}/{stats['gate_mean']:.3f}/"
                  f"{stats['gate_max']:.3f}; mean lag {stats['mean_lag_months']:.1f} months; 90% of the fold "
                  f"weight within the last {stats['months_holding_90pct_of_weight_mean']:.0f} months (max "
                  f"{stats['months_holding_90pct_of_weight_max']:.0f}) of 528; hidden features with std < 1e-3: "
                  f"{stats['hidden_features_with_std_below_1e-3']}/20")

    out = cli.root / "analysis_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
