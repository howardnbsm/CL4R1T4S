"""Peak-calibration probe on the closed-form ridge forecaster (cheap test-bed, seconds).

    python solar/v2a/calib_probe.py -> runs/solar/v2a/calib_probe.json

Every calibration parameter is fitted on OUT-OF-BLOCK forecasts of the fold's FIT windows only and then
applied to the fold's evaluation block.  Variants:
  bump s<sigma>      y * (1 + a * Gaussian(tau - argmax y, sigma)); a minimises the out-of-block PAE
  tilt               y * (1 + c * tau / 131) before the bump; c minimises the out-of-block PTE (grid)
  paw<alpha>         peak-aware weighted ridge: per-element weights (1 + alpha * y) as in v1's loss (closed form)
"""
from __future__ import annotations

import json

import numpy as np
import torch

import common as C
from cvvp import RidgeSummaryForecaster, out_of_block_prediction
from final import bump_calibrate, fit_gain, GAIN_GRID

LAM = 0.03


def oob_forecast(fc, x_fit, y_fit):
    z, scale = fc.features(x_fit)
    target = torch.from_numpy(np.asarray(y_fit, dtype=np.float64)) / scale
    pred, _ = out_of_block_prediction(z, target, LAM, 6, "two_sided")
    return np.maximum((pred * scale).numpy(), 0.0)


def tilt(pred, c):
    return pred * (1.0 + c * np.arange(pred.shape[1])[None, :] / (pred.shape[1] - 1))


def fit_tilt(oob, y):
    grid = np.linspace(-0.3, 1.0, 27)
    true_arg = y.argmax(axis=1)
    errs = [np.abs(tilt(oob, c).argmax(axis=1) - true_arg).mean() for c in grid]
    return float(grid[int(np.argmin(errs))]), {f"{c:.2f}": float(e) for c, e in zip(grid, errs)}


def weighted_ridge_predict(fc, x_fit, y_fit, x_ev, alpha):
    """Ridge with per-element weights (1 + alpha * y_scaled) * scale^2 (v1's peak-aware loss), one solve per lead."""
    z, scale = fc.features(x_fit)
    z, scale = z.numpy(), scale.numpy()
    y = np.asarray(y_fit, dtype=np.float64)
    target = y / scale
    w_el = (1.0 + alpha * y) * scale ** 2
    zb = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    pen = LAM * len(z) * np.diag([1.0] * z.shape[1] + [0.0])
    ze, se = fc.features(x_ev)
    zeb = np.concatenate([ze.numpy(), np.ones((len(ze), 1))], axis=1)
    out = np.zeros((len(zeb), y.shape[1]))
    for j in range(y.shape[1]):
        w = w_el[:, j] / w_el[:, j].mean()
        gram = (zb * w[:, None]).T @ zb + pen
        out[:, j] = zeb @ np.linalg.solve(gram, (zb * w[:, None]).T @ target[:, j])
    return np.maximum(out * se.numpy(), 0.0)


def main():
    d = C.data()
    rows = {}
    for fold, (fit, ev) in C.folds().items():
        x_fit, y_fit, x_ev, y_ev = d.X[fit], d.Y[fit], d.X[ev], d.Y[ev]
        fc = RidgeSummaryForecaster(x_fit, y_fit, lam=LAM)
        raw = np.maximum(fc.predict(x_ev).astype(np.float64), 0.0)
        oob = oob_forecast(fc, x_fit, y_fit)
        res = {"raw": C.score(raw.astype(np.float32), y_ev)}
        for sigma in (2.5, 3.75, 6.0):
            gain = fit_gain(oob, y_fit)
            res[f"bump s{sigma:g}"] = dict(C.score(bump_calibrate(raw, gain, sigma).astype(np.float32), y_ev), gain=gain)
        c, curve = fit_tilt(oob, np.asarray(y_fit))
        gain_t = fit_gain(tilt(oob, c), y_fit)
        res["tilt"] = dict(C.score(tilt(raw, c).astype(np.float32), y_ev), c=c)
        res["tilt+bump s3.75"] = dict(C.score(bump_calibrate(tilt(raw, c), gain_t, 3.75).astype(np.float32), y_ev), c=c, gain=gain_t)
        for alpha in (1.0, 3.0):
            res[f"paw{alpha:g}"] = C.score(weighted_ridge_predict(fc, x_fit, y_fit, x_ev, alpha).astype(np.float32), y_ev)
        # diagnostics: where are the maxima
        ta, pa = y_ev.argmax(axis=1), raw.argmax(axis=1)
        res["diag"] = {"true_argmax_lt12": float((ta < 12).mean()), "true_argmax_gt119": float((ta > 119).mean()),
                       "pred_argmax_lt12": float((pa < 12).mean()), "pred_argmax_gt119": float((pa > 119).mean()),
                       "abs_err_gt60": float((np.abs(ta - pa) > 60).mean()),
                       "mean_true_max": float(d.scaler.inverse(y_ev).max(axis=1).mean()) if hasattr(d, "scaler") and hasattr(d.scaler, "inverse") else None,
                       "tilt_curve_min": min(curve.values()), "tilt_curve_at0": curve["0.00"]}
        rows[fold] = res
    names = [k for k in rows["val"] if k != "diag"]
    print(f"{'variant':18s} " + "  ".join(f"{f:^17s}" for f in rows) + "   cv-mean")
    for n in names:
        cv = np.mean([[rows[f][n][k] for k in ("scaled_mse", "pae", "pte")] for f in ("cv1", "cv2", "cv3", "cv4")], axis=0)
        extra = " ".join(f"{k}={[round(rows[f][n][k], 2) for f in rows]}" for k in ("gain", "c") if k in rows["val"][n])
        print(f"{n:18s} " + "  ".join(C.short(rows[f][n]) for f in rows) + f"   {cv[0]:.4f}/{cv[1]:5.1f}/{cv[2]:4.1f}  {extra}")
    for f in rows:
        print(f, {k: (round(v, 3) if v is not None else None) for k, v in rows[f]["diag"].items()})
    (C.RUNS_DIR / "calib_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
