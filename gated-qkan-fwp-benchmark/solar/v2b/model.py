"""Clock-QKAN forecaster of solar v2 approach B (the consolidated model).

Forecast of one 528-month window ``x`` (scaled units)::

    s     = mean(x) + 1e-3                          44-year activity level (scale)
    z     = 12-month means of the last 132 months / s     (11 linear features)
    phi   = 2 pi * (months since the last minimum of the Gaussian-smoothed window) / 132
    h     = DaruanLayer(1 -> m)(phi)                the QKAN layer (only learned nonlinearity)
    c     = W_lin z~ + W_q h~ + b                   K DCT coefficients (+ optional peak-height head)
    yhat  = s * c @ B                               B: K x 132 orthonormal DCT-II basis

``W_lin, W_q, b`` are a closed-form generalised ridge solution (penalty ``lam_lin``
on the standardised linear features, ``lam_q`` on the standardised QKAN features).
The DARUAN angles / pre-activation frequencies and phases and ``log lam_q`` are
trained by full-batch Adam on a BLOCKED cross-validation objective inside the fit
windows (out-of-block error of the closed-form readout, 132-window embargo), i.e.
the QKAN layer is optimised to produce features that generalise across solar
cycles.  One optimisation step = one pass over the fit windows (``steps <= 100``).

Peak calibration (optional, parameters fitted on out-of-block forecasts of the fit
windows only): a Gaussian amplification (FWHM 13 months) centred at the forecast's
own maximum, see ``calib.py``:

    mode "gain": constant gain  g = 1 + strength * (g_L1 - 1), g_L1 = L1-optimal ratio of
                 the monthly maximum of the truth to the maximum of the forecast.
    mode "head": per-window gain from a peak-height head (an extra closed-form readout
                 column that regresses max(y) / s on the same features).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import calib
import common as C
import features as F
from daruan import DaruanLayer

torch.set_num_threads(1)

DEFAULT = dict(
    K=12, lam_lin=1e-2, lam_q=0.3, m=4, reps=1, observable="z", preact=True,
    steps=60, lr=2e-2, n_blocks=8, train_lam=True,
    arch="daruan",            # "daruan" | "linear" | "tanh" | "none" (ablations)
    head_w=0.0,               # weight of the peak-height head column in the blocked-CV objective
    alpha=0.0,                # peak-aware weight (1 + alpha * y) of the closed-form readout (0 = plain)
    calib="none",             # "none" | "gain" | "head"
    strength=1.0,             # calibration strength in [0, 1]; "auto" = joint MSE/PAE criterion on OOF forecasts
    sigma=calib.SIGMA_DEFAULT, gain_max=2.5,
)


def featurize(x):
    """Window features (float64): scale ``s`` (n,1), linear features ``ann`` (n,11), clock angle ``phi`` (n,1)."""
    x = np.asarray(x, dtype=np.float64)
    s = x.mean(axis=1, keepdims=True) + 1e-3
    cyc = F.cycle_descriptors(x)
    return dict(s=s, ann=F.patch_means(x, 12, 132) / s, phi=2 * np.pi * cyc[:, :1] / 132.0)


def subset(feats, idx):
    return {k: v[idx] for k, v in feats.items()}


class _LinearFeat(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.lin = nn.Linear(d, m)

    def forward(self, u):
        return self.lin(u)


class _TanhFeat(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.lin = nn.Linear(d, m)

    def forward(self, u):
        return torch.tanh(self.lin(u))


def _solve(Fz, T, pen):
    """Generalised ridge with unpenalised intercept: returns (W, b)."""
    n = Fz.shape[0]
    mu_f, b = Fz.mean(0), T.mean(0)
    Fc = Fz - mu_f
    W = torch.linalg.solve(Fc.T @ Fc + n * torch.diag(pen), Fc.T @ (T - b))
    return W, b - mu_f @ W


def _solve_weighted(Fz, Yn, omega, B, pen):
    """Closed-form readout under element weights ``omega`` (n, 132) in the normalised space:
    minimise sum_{n,tau} omega * ((Fz W + b) @ B - Yn)^2 + n * sum_j pen_j |W_j|^2."""
    n, p = Fz.shape
    K = B.shape[0]
    Fa = torch.cat([Fz, torch.ones(n, 1, dtype=Fz.dtype)], dim=1)
    Cn = torch.einsum("kt,nt,lt->nkl", B, omega, B)
    A = torch.einsum("ni,nj,nkl->ikjl", Fa, Fa, Cn).reshape((p + 1) * K, (p + 1) * K)
    rhs = torch.einsum("ni,nk->ik", Fa, (omega * Yn) @ B.T).reshape(-1)
    pen_full = torch.cat([pen, torch.zeros(1, dtype=pen.dtype)]).repeat_interleave(K) * n
    sol = torch.linalg.solve(A + torch.diag(pen_full) + 1e-10 * torch.eye(A.shape[0], dtype=A.dtype), rhs)
    Wb = sol.reshape(p + 1, K)
    return Wb[:p], Wb[p]


class ClockQKAN:
    """Fit on chronologically ordered stride-1 windows; predict any windows."""

    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULT)
        assert not unknown, unknown
        self.cfg = {**DEFAULT, **cfg}
        self.B = C.dct_basis(C.HORIZON, self.cfg["K"])

    # ------------------------------------------------------------------ fitting
    def fit(self, x, y, seed=0, feats=None):
        cfg = self.cfg
        t64 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
        feats = featurize(x) if feats is None else feats
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        s = feats["s"]
        self.mu, self.sd = feats["ann"].mean(0), feats["ann"].std(0) + 1e-9
        Zlin = t64((feats["ann"] - self.mu) / self.sd)
        U = t64(feats["phi"])
        Bt = t64(self.B)
        Yn = t64(y / s)
        T = Yn @ Bt.T
        mx = t64(y.max(axis=1, keepdims=True) / s)
        use_head = cfg["calib"] == "head"
        Tall = torch.cat([T, mx], dim=1) if use_head else T
        colw = torch.ones(Tall.shape[1], dtype=torch.float64)
        if use_head:
            colw[-1] = cfg["head_w"]
        omega = (1.0 + cfg["alpha"] * t64(y)) if cfg["alpha"] > 0 else None
        d_lin, m = Zlin.shape[1], (0 if cfg["arch"] == "none" else cfg["m"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cfg["arch"] == "daruan":
            layer = DaruanLayer(1, m, cfg["reps"], cfg["preact"], False, cfg["observable"]).double()
        elif cfg["arch"] == "linear":
            layer = _LinearFeat(1, m).double()
        elif cfg["arch"] == "tanh":
            layer = _TanhFeat(1, m).double()
        else:
            layer = None
        self.layer = layer
        loglam = nn.Parameter(torch.tensor(float(np.log(cfg["lam_q"])), dtype=torch.float64),
                              requires_grad=bool(cfg["train_lam"]) and layer is not None)
        pen_lin = torch.full((d_lin,), cfg["lam_lin"], dtype=torch.float64)
        idx = np.arange(n)
        splits = [(torch.tensor(tr), torch.tensor(held)) for tr, held in calib.crossfit_blocks(idx, cfg["n_blocks"])]

        def pen():
            return torch.cat([pen_lin, torch.exp(loglam).expand(m)]) if m else pen_lin

        def feats_of(H, tr):
            if layer is None:
                return Zlin, None
            h_mu, h_sd = H[tr].mean(0), H[tr].std(0) + 1e-6
            return torch.cat([Zlin, (H - h_mu) / h_sd], dim=1), (h_mu, h_sd)

        def readout(Fz, tr):
            if omega is None:
                return _solve(Fz[tr], Tall[tr], pen())
            W, b = _solve_weighted(Fz[tr], Yn[tr], omega[tr], Bt, pen())
            if use_head:
                Wh, bh = _solve(Fz[tr], mx[tr], pen())
                W, b = torch.cat([W, Wh], dim=1), torch.cat([b, bh])
            return W, b

        def oof(H):
            out = torch.zeros_like(Tall)
            for tr, held in splits:
                Fz, _ = feats_of(H, tr)
                W, b = readout(Fz, tr)
                out[held] = Fz[held] @ W + b
            return out

        self.history = []
        if layer is not None and cfg["steps"] > 0:
            params = [p for p in layer.parameters() if p.requires_grad] + ([loglam] if loglam.requires_grad else [])
            opt = torch.optim.Adam(params, lr=cfg["lr"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.05)
            for _ in range(cfg["steps"]):
                err = (oof(layer(U)) - Tall) ** 2 * colw
                loss = err.sum() / (len(Tall) * float(colw.sum()))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                self.history.append(float(loss.detach()))
        with torch.no_grad():
            H = layer(U) if layer is not None else None
            all_idx = torch.arange(n)
            Fz, st = feats_of(H, all_idx)
            self.h_stats = st
            self.W, self.b = readout(Fz, all_idx)
            self.pen = pen().clone()
            coef_oof = oof(H).numpy()
        K = cfg["K"]
        self.n_params = (sum(p.numel() for p in layer.parameters() if p.requires_grad) if layer is not None else 0) \
            + self.W.numel() + self.b.numel()
        # out-of-block forecasts of the fit windows (for calibration and diagnostics)
        self.oof_curve = s * (coef_oof[:, :K] @ self.B)
        self.oof_head = s[:, 0] * coef_oof[:, K] if use_head else None
        self.fit_y = y
        self._fit_calibration()
        if cfg["calib"] == "gain":
            self.n_params += 1  # the calibration gain
        return self

    # -------------------------------------------------------------- calibration
    def _gains(self, curve, head):
        cfg = self.cfg
        if cfg["calib"] == "gain":
            return np.full(len(curve), self.g_l1)
        g = head / np.maximum(curve.max(axis=1), 1e-6)
        return np.clip(g, 1.0, cfg["gain_max"])

    def _apply(self, curve, head, strength):
        if self.cfg["calib"] == "none" or strength == 0:
            return curve
        g = 1.0 + strength * (self._gains(curve, head) - 1.0)
        return calib.apply(curve, g[:, None], self.cfg["sigma"])

    def _fit_calibration(self):
        cfg = self.cfg
        self.g_l1, self.strength, self.calib_table = 1.0, 0.0, None
        if cfg["calib"] == "none":
            return
        yo, y = self.oof_curve, self.fit_y
        if cfg["calib"] == "gain":
            # L1-optimal constant ratio between the monthly maximum of the truth and the forecast maximum
            grid = np.arange(1.0, 2.5001, 0.01)
            pm, tm = yo.max(axis=1), y.max(axis=1)
            self.g_l1 = float(grid[np.argmin([np.abs(g * pm - tm).mean() for g in grid])])
        if cfg["strength"] == "auto":
            base_mse = ((yo - y) ** 2).mean()
            base_pae = np.abs(yo.max(1) - y.max(1)).mean()
            rows = []
            for lam in np.arange(0.0, 1.0001, 0.05):
                yc = self._apply(yo, self.oof_head, lam)
                rows.append((lam, ((yc - y) ** 2).mean() / base_mse, np.abs(yc.max(1) - y.max(1)).mean() / base_pae))
            self.calib_table = np.array(rows)
            J = self.calib_table[:, 1] + self.calib_table[:, 2]
            self.strength = float(self.calib_table[np.argmin(J), 0])
        else:
            self.strength = float(cfg["strength"])

    # --------------------------------------------------------------- prediction
    def coef(self, feats):
        with torch.no_grad():
            Z = torch.tensor((feats["ann"] - self.mu) / self.sd, dtype=torch.float64)
            if self.layer is not None:
                H = self.layer(torch.tensor(feats["phi"], dtype=torch.float64))
                Z = torch.cat([Z, (H - self.h_stats[0]) / self.h_stats[1]], dim=1)
            return (Z @ self.W + self.b).numpy()

    def predict(self, x=None, feats=None, strength=None, return_parts=False):
        feats = featurize(x) if feats is None else feats
        K = self.cfg["K"]
        c = self.coef(feats)
        curve = feats["s"] * (c[:, :K] @ self.B)
        head = feats["s"][:, 0] * c[:, K] if self.cfg["calib"] == "head" else None
        out = self._apply(curve, head, self.strength if strength is None else strength)
        if return_parts:
            return out, curve, head
        return out
