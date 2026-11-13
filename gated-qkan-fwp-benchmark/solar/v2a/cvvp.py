"""CVVP training: QKAN-FWP feature extractor + closed-form ridge read-out.

The model is the v2a backbone (``model.GatedQKANFWP2``): patch tokens -> QKAN
slow programmer -> multi-horizon gated fast weights -> programmed QKAN fast
programmer queried with the recent-history summary.  The fast programmer's
output z (O numbers per window) is the ONLY thing the decoder sees:

    forecast = scale * (mean_curve + standardise(z) @ W)            (W: O x 132)

Two-level fit (hybrid closed form / gradient):

  read-out   W is the ridge solution (closed form) on the fit windows.
  programmer the slow programmer, heads, gates, static angles and query are
             trained by gradient descent on the OUT-OF-BLOCK error of that
             ridge read-out ("cross-validated variable projection"): the fit
             windows are cut into contiguous blocks, every block is predicted
             by the ridge solved on the windows at least 132 positions away
             from it (no shared target month), and the mean squared
             out-of-block error is differentiated through the ridge solves
             with respect to the programmer parameters.  The programmer is
             thus trained to produce features whose linear read-out
             GENERALISES across time, not features that fit in-sample.

One optimisation step is one full pass over the fit windows (= one epoch);
at most 100 steps.  ``steps=0`` gives the untrained (random-feature) variant.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

import common as C
from model import GatedQKANFWP2, HORIZON

torch.set_num_threads(1)

CVVP_DEFAULTS = dict(
    steps=60,          # full-batch steps (epochs), <= 100
    lr=3e-3,
    wd=0.0,
    lam=0.1,           # ridge penalty of the read-out (relative to n; features are standardised)
    blocks=6,          # contiguous blocks for the out-of-block objective
    scheme="two_sided",  # "two_sided": a block is predicted from all windows >= 132 positions away;
                         # "forward": rolling origin, a block is predicted from EARLIER windows only
    forward_start=0.35,  # "forward": fraction of the fit windows reserved as the first training segment
    warmup=5,
    clip=1.0,
    alpha=0.0,         # peak-aware weighting (1 + alpha * y) of the out-of-block error
)


def make_cvvp_config(**overrides):
    cfg = dict(CVVP_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown cvvp config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


def hidden(model: GatedQKANFWP2, feats: dict) -> torch.Tensor:
    """Fast-programmer outputs z, (B, O): the only path from the window to the forecast."""
    cfg = model.cfg
    batch = feats["summary"].shape[0]
    if cfg["program"]:
        theta, _ = model.fast_weights(feats["tokens"])
    else:
        theta = model.theta0.unsqueeze(0).expand(batch, -1, -1, -1, -1)
    q0 = model.query(feats["summary"]) if cfg["query"] == "summary" else model.query_bias.expand(batch, -1)
    return model.fast_layer(q0, theta)


def _ridge(z, t, lam):
    """Closed-form ridge (unpenalised intercept) in float64: returns (W, z_mean, t_mean)."""
    z_mean, t_mean = z.mean(dim=0), t.mean(dim=0)
    zc = z - z_mean
    gram = zc.T @ zc + lam * len(z) * torch.eye(z.shape[1], dtype=z.dtype)
    return torch.linalg.solve(gram, zc.T @ (t - t_mean)), z_mean, t_mean


def out_of_block_prediction(z, t, lam, blocks, scheme="two_sided", forward_start=0.35):
    """Out-of-block ridge predictions (differentiable).  Returns ``(pred, first)``: predictions for the
    windows ``first .. n-1``.

    two_sided: every block is predicted by the ridge solved on windows >= 132 positions away (first = 0).
    forward:   rolling origin inside the fit windows; a block is predicted by the ridge solved on the
               windows that END at least 132 positions before it (first = forward_start * n).
    """
    n = len(z)
    first = 0 if scheme == "two_sided" else int(round(forward_start * n))
    edges = [first + round(j * (n - first) / blocks) for j in range(blocks + 1)]
    index = torch.arange(n)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = index < lo - (HORIZON - 1)
        if scheme == "two_sided":
            keep = keep | (index >= hi + (HORIZON - 1))
        w, z_mean, t_mean = _ridge(z[keep], t[keep], lam)
        parts.append((z[lo:hi] - z_mean) @ w + t_mean)
    return torch.cat(parts), first


class CVVPForecaster:
    """A fitted model: programmer (torch module) + closed-form read-out."""

    def __init__(self, model, z_mean, z_std, w, bias, info):
        self.model, self.z_mean, self.z_std, self.w, self.bias, self.info = model, z_mean, z_std, w, bias, info

    @torch.no_grad()
    def features(self, x: np.ndarray):
        feats = self.model.features(torch.from_numpy(np.asarray(x, dtype=np.float32)))
        z = (hidden(self.model, feats).double() - self.z_mean) / self.z_std
        return z, feats["scale"].double()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scaled windows (n, 528) -> scaled forecasts (n, 132)."""
        out = []
        for start in range(0, len(x), 1024):
            z, scale = self.features(x[start:start + 1024])
            out.append(((z @ self.w + self.bias) * scale).float())
        return torch.cat(out).numpy()

    def n_parameters(self) -> dict:
        trainable = int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)) \
            - sum(p.numel() for p in self.model.readout.parameters())
        closed_form = int(self.w.numel() + self.bias.numel())
        return {"programmer_trainable": trainable, "readout_closed_form": closed_form,
                "total": trainable + closed_form}


def fit_cvvp(mcfg, tcfg, x_fit, y_fit, seed=0, x_mon=None, y_mon=None, log=None):
    """Fit one CVVP model on the fit windows.  ``x_mon / y_mon`` are monitored only (never used to fit)."""
    tcfg = make_cvvp_config(**tcfg)
    assert tcfg["steps"] <= 100, "the protocol allows at most 100 epochs"
    torch.manual_seed(seed)
    model = GatedQKANFWP2(dict(mcfg, decoder="basis", baseline="mean"))
    for p in model.readout.parameters():       # the SGD read-out of the basis decoder is not used here
        p.requires_grad_(False)
    xf = torch.from_numpy(np.asarray(x_fit, dtype=np.float32))
    yf = torch.from_numpy(np.asarray(y_fit, dtype=np.float32))
    feats = model.features(xf)
    scale = feats["scale"].double()
    target = yf.double() / scale                                   # normalised targets
    weight = scale ** 2                                            # error in scaled units = scale * normalised error
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["wd"])
    history = []
    t0 = time.time()

    def standardised(z):
        z = z.double()
        return (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)

    for step in range(tcfg["steps"]):
        frac = step / max(1, tcfg["steps"] - 1)
        lr = tcfg["lr"] * (min(1.0, (step + 1) / max(1, tcfg["warmup"])) * 0.5 * (1 + math.cos(math.pi * frac)) + 1e-3)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        z = standardised(hidden(model, feats))
        pred, first = out_of_block_prediction(z, target, tcfg["lam"], tcfg["blocks"], tcfg["scheme"],
                                              tcfg["forward_start"])
        err = (pred - target[first:]) ** 2 * weight[first:]
        if tcfg["alpha"]:
            err = err * (1.0 + tcfg["alpha"] * yf[first:].double())
        loss = err.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg["clip"]:
            torch.nn.utils.clip_grad_norm_(params, tcfg["clip"])
        opt.step()
        history.append(float(loss.detach()))
        if log is not None and (step % 10 == 0 or step == tcfg["steps"] - 1):
            log(f"  step {step + 1:3d} out-of-block mse {history[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        z_raw = hidden(model, feats).double()
        z_mean, z_std = z_raw.mean(dim=0), z_raw.std(dim=0) + 1e-6
        z = (z_raw - z_mean) / z_std
        pred, first = out_of_block_prediction(z, target, tcfg["lam"], tcfg["blocks"], tcfg["scheme"],
                                              tcfg["forward_start"])
        oob = float((((pred - target[first:]) ** 2) * weight[first:]).mean())
        w, zm, tm = _ridge(z, target, tcfg["lam"])
        bias = tm - zm @ w
    info = {"history": history, "oob_mse_final": oob, "seconds": time.time() - t0}
    forecaster = CVVPForecaster(model, z_mean, z_std, w, bias, info)
    info["params"] = forecaster.n_parameters()
    if x_mon is not None:
        info["monitor"] = C.score(forecaster.predict(x_mon), y_mon)
    return forecaster


class RidgeSummaryForecaster:
    """Reference without any QKAN / fast weights: ridge from the standardised recent summary (11 numbers)."""

    def __init__(self, x_fit, y_fit, lam=0.03, mcfg=None):
        from model import Features, make_config
        self.feat = Features(make_config(**(mcfg or {})))
        f = self.feat(torch.from_numpy(np.asarray(x_fit, dtype=np.float32)))
        z = f["summary"].double()
        self.z_mean, self.z_std = z.mean(dim=0), z.std(dim=0) + 1e-6
        target = torch.from_numpy(np.asarray(y_fit, dtype=np.float64)) / f["scale"].double()
        w, zm, tm = _ridge((z - self.z_mean) / self.z_std, target, lam)
        self.w, self.bias = w, tm - zm @ w
        self.info = {}

    @torch.no_grad()
    def features(self, x):
        f = self.feat(torch.from_numpy(np.asarray(x, dtype=np.float32)))
        return (f["summary"].double() - self.z_mean) / self.z_std, f["scale"].double()

    @torch.no_grad()
    def predict(self, x):
        z, scale = self.features(x)
        return ((z @ self.w + self.bias) * scale).float().numpy()

    def n_parameters(self):
        n = int(self.w.numel() + self.bias.numel())
        return {"programmer_trainable": 0, "readout_closed_form": n, "total": n}


if __name__ == "__main__":
    # quick self-check on one fold
    d = C.data()
    fit, ev = C.folds()["cv3"]
    for steps in (0, 40):
        f = fit_cvvp(dict(O=16, I=8, theta0=True), dict(steps=steps), d.X[fit], d.Y[fit], seed=0, log=print)
        print("steps", steps, "eval", C.short(C.score(f.predict(d.X[ev]), d.Y[ev])), "oob", round(f.info["oob_mse_final"], 5),
              f.info["params"], round(f.info["seconds"], 1), "s")
