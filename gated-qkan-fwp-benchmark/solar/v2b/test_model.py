"""Tests of the final v2b model (run: python solar/v2b/test_model.py).

1. The TRAINED QKAN layer of the fitted model is numerically identical to ``qkan.QKANLayer``
   (solver "exact", pz ansatz, physical <Z> observable) after copying the parameters: max abs
   difference < 1e-5 in float32 on the real clock angles and on random inputs; the forecasts
   recomputed with the package layer in the loop agree to < 1e-5 as well.
2. ``fit(seed)`` reads the training split only: corrupting every validation window / target
   leaves the fitted model unchanged.
3. ``predict_fn`` is deterministic, row-independent and order-independent; float32 (n, 132).
4. Blocked cross-fitting never shares a target month between fit and held-out windows.
5. Calibration: a Gaussian amplification with FWHM 13 months centred at the forecast's own
   maximum; max(calibrated) = gain * max(forecast); the argmax is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from qkan import QKANLayer

import calib
import common as C
import final
import model as M

warnings.filterwarnings("ignore")
TOL = 1e-5


def test_trained_layer_equals_package():
    d = C.data()
    pf = final.fit(0, "clock+cal")
    mdl = pf.model
    ours = mdl.layer
    ref = QKANLayer(1, mdl.cfg["m"], reps=mdl.cfg["reps"], solver="exact", ansatz="pz_encoding",
                    preact_trainable=True, ba_trainable=False, fast_measure=False)
    assert ref.theta.shape == ours.theta.shape and ref.preacts_weight.shape == ours.preacts_weight.shape
    with torch.no_grad():
        ref.theta.copy_(ours.theta.float())
        ref.preacts_weight.copy_(ours.preacts_weight.float())
        ref.preacts_bias.copy_(ours.preacts_bias.float())
        ref.base_weight.zero_()
    feats = M.featurize(d.X)
    phi = torch.tensor(feats["phi"], dtype=torch.float32)
    rnd = 6.0 * torch.randn(4096, 1, generator=torch.Generator().manual_seed(0))
    ours32 = type(ours)(1, mdl.cfg["m"], mdl.cfg["reps"], True, False, "z")
    ours32.load_state_dict({k: v.float() for k, v in ours.state_dict().items()})
    worst = 0.0
    with torch.no_grad():
        for x in (phi, rnd):
            a, b = ours32(x), ref(x)
            assert a.dtype == torch.float32 and a.shape == b.shape
            worst = max(worst, float((a - b).abs().max()))
    assert worst < TOL, worst
    # forecasts with the package layer in the loop
    with torch.no_grad():
        H = ref(phi).double()
        Z = torch.tensor((feats["ann"] - mdl.mu) / mdl.sd, dtype=torch.float64)
        Fz = torch.cat([Z, (H - mdl.h_stats[0]) / mdl.h_stats[1]], dim=1)
        c = (Fz @ mdl.W + mdl.b).numpy()
    curve_pkg = feats["s"] * (c[:, :mdl.cfg["K"]] @ mdl.B)
    curve = mdl.predict(feats=feats, strength=0.0)
    diff = float(np.abs(curve_pkg - curve).max())
    assert diff < TOL, diff
    print(f"1. trained DARUAN layer == qkan.QKANLayer(exact, <Z>): max abs diff {worst:.2e}; "
          f"forecast with package layer in the loop: max abs diff {diff:.2e}  (< {TOL:g})")
    return pf


def test_train_only(pf):
    d = C.data()
    n_tr = C.N_TRAIN
    keepX, keepY = d.X[n_tr:].copy(), d.Y[n_tr:].copy()
    try:
        d.X[n_tr:] = np.nan
        d.Y[n_tr:] = np.nan
        final._TRAIN_FEATS = None
        pf2 = final.fit(0, "clock+cal")
    finally:
        d.X[n_tr:] = keepX
        d.Y[n_tr:] = keepY
        final._TRAIN_FEATS = None
    dW = float((pf2.model.W - pf.model.W).abs().max())
    assert dW == 0.0 and pf2.model.g_l1 == pf.model.g_l1 and pf2.model.strength == pf.model.strength, dW
    print("2. fit(seed) is unchanged when all validation windows / targets are replaced by NaN (train-only): OK")


def test_predict_fn(pf):
    d = C.data()
    xva, _ = d.split("val")
    p = pf(xva)
    assert p.shape == (C.N_VAL, C.HORIZON) and p.dtype == np.float32 and np.isfinite(p).all()
    rows = np.array([0, 17, 101, 259])
    single = np.concatenate([pf(xva[i:i + 1]) for i in rows])
    assert np.array_equal(single, p[rows])
    perm = np.random.default_rng(0).permutation(len(xva))
    assert np.array_equal(pf(xva[perm]), p[perm])
    p2 = final.fit(0, "clock+cal")(xva)
    assert np.array_equal(p, p2)
    p3 = final.fit(1, "clock+cal")(xva)
    assert not np.array_equal(p, p3)
    print(f"3. predict_fn: float32 {p.shape}, row/order independent, same seed -> identical forecasts, "
          f"other seed differs by max {np.abs(p - p3).max():.2e}: OK")


def test_crossfit_embargo():
    for n in (913, 2084):
        seen = np.zeros(n, dtype=int)
        for tr, held in calib.crossfit_blocks(np.arange(n), 8):
            gap = np.abs(tr[:, None] - held[None, :]).min()
            assert gap >= C.HORIZON, gap  # windows i, j share a target month iff |i - j| < 132
            seen[held] += 1
        assert (seen == 1).all()
    print("4. blocked cross-fitting: every window held out once, >= 132 windows between fit and held-out: OK")


def test_calibration_shape(pf):
    tau = np.arange(C.HORIZON)
    curve = (0.3 + 0.2 * np.sin(2 * np.pi * (tau - 20) / 132.0))[None, :]
    g = 1.3
    out = calib.apply(curve, np.full((1, 1), g))
    t0 = int(curve.argmax())
    assert int(out.argmax()) == t0 and abs(out.max() - g * curve.max()) < 1e-12
    mult = out[0] / curve[0] - 1.0
    assert (mult >= 0).all()
    half = np.interp(6.5, [6, 7], [mult[t0 + 6], mult[t0 + 7]]) / mult[t0]
    assert abs(half - 0.5) < 5e-3, half  # FWHM = 13 months
    mdl = pf.model
    assert 1.0 <= 1 + mdl.strength * (mdl.g_l1 - 1) <= 2.5
    print(f"5. calibration: amplification {g} at the forecast's own maximum, FWHM 13 months, argmax unchanged; "
          f"fitted gain of seed 0 = {1 + mdl.strength * (mdl.g_l1 - 1):.3f} (g_L1 {mdl.g_l1:.2f}, strength {mdl.strength:.2f}): OK")


if __name__ == "__main__":
    assert not C.data().unsealed
    pf = test_trained_layer_equals_package()
    test_train_only(pf)
    test_predict_fn(pf)
    test_crossfit_embargo()
    test_calibration_shape(pf)
    print("PASS")
