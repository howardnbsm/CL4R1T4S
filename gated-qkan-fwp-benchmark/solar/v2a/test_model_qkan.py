"""Unit tests: the QKAN layers INSIDE the v2a candidates equal the qkan package layer.

    python solar/v2a/test_model_qkan.py

For the recommended candidate ("clock") and the in-family variant ("clock_fwp") a model is fitted on a
small slice of the training windows, the parameters of each of its QKAN layers are COPIED into a
``qkan.QKANLayer`` (solver "exact", observable <Z> = ``fast_measure=False``) and both are evaluated on
the model's real layer inputs (float32); the test fails unless max abs diff < 1e-5.  It also checks that
the fitted forecaster's features are exactly [standardised summary, gain * standardised layer output],
i.e. that the QKAN layer output is what reaches the read-out.
"""
from __future__ import annotations

import numpy as np
import torch
from qkan import QKANLayer

import common as C
import cvvp
import final

TOL = 1e-5


def package_layer(in_dim, out_dim, reps, preact_trainable, ba_trainable):
    return QKANLayer(in_dim, out_dim, reps=reps, solver="exact", ansatz="pz_encoding",
                     preact_trainable=preact_trainable, ba_trainable=ba_trainable, fast_measure=False)


def check(candidate):
    d = C.data()
    x, y = d.X[:600], d.Y[:600]
    mcfg, tcfg, _ = final.CANDIDATES[candidate]
    fc = cvvp.fit_cvvp(mcfg, dict(tcfg, steps=min(5, tcfg.get("steps", 0))), x, y, seed=0)
    model, cfg = fc.model, fc.model.cfg
    assert cfg["fast"] == "qkan" and cfg["observable"] == "z"
    feats = model.features(torch.from_numpy(x[:64]))
    query = model.query_vector(feats)                                    # (B, I) real layer inputs
    worst = 0.0
    with torch.no_grad():
        ours = cvvp.hidden(model, feats)                                 # (B, O) what the read-out sees
        if cfg["program"]:
            theta, _ = model.fast_weights(feats["tokens"])               # per-window programmed angles
        else:
            theta = model.theta0.unsqueeze(0).expand(len(query), -1, -1, -1, -1)
        for b in range(len(query) if cfg["program"] else 1):
            ref = package_layer(cfg["I"], cfg["O"], cfg["reps"], False, False)
            assert ref.theta.shape == theta[b].shape
            ref.theta.copy_(theta[b])
            rows = slice(b, b + 1) if cfg["program"] else slice(None)
            worst = max(worst, float((ref(query[rows]) - ours[rows]).abs().max()))
        if cfg["program"]:                                               # slow programmer QKAN layer
            slow = model.slow_qkan
            ref = package_layer(slow.in_dim, slow.out_dim, slow.reps, True, True)
            for name in ("theta", "preacts_weight", "preacts_bias", "base_weight"):
                getattr(ref, name).copy_(getattr(slow, name))
            tok = model.slow_in(feats["tokens"].reshape(-1, feats["tokens"].shape[-1]))
            worst = max(worst, float((ref(tok) - slow(tok)).abs().max()))
        # the read-out features are [standardised summary, gain * standardised QKAN output]
        z, _ = fc.features(x[:64])
        s_mean, s_std, gain = fc.skip
        expect = torch.cat([(feats["summary"].double() - s_mean) / s_std, gain * (ours.double() - fc.z_mean) / fc.z_std], dim=1)
        assert float((z - expect).abs().max()) < 1e-9
    assert worst < TOL, (candidate, worst)
    return worst


if __name__ == "__main__":
    for name in ("clock", "clock_fwp", "clock_fwp_strong"):
        print(f"{name:18s}: QKAN layers vs qkan.QKANLayer(exact, <Z>) max abs diff {check(name):.2e}  (< {TOL:g})")
    print("ALL MODEL-LEVEL QKAN TESTS PASSED")
