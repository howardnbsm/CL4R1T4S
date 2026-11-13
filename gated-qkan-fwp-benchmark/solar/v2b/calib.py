"""Peak calibration of a smooth (MSE-optimal) forecast.

An MSE-optimal forecast is a conditional mean: it averages over the unknown
timing of the next maximum (a blur that grows with lead time) and it contains
no monthly noise, so its maximum under-predicts the maximum of the noisy
monthly truth.  The calibration is a smooth amplification concentrated around
the forecast's own maximum:

    A(tau)     = yhat(tau) * G(tau)          G >= 1: expected-peak gain profile
    tau*       = argmax_tau A(tau)           (the forecast's own, gain-weighted maximum)
    yhat_c     = yhat * (1 + (G(tau*) - 1) * exp(-(tau - tau*)^2 / (2 sigma^2)))

so that max(yhat_c) = A(tau*) at tau*.  ``G`` is fitted on out-of-fold forecasts
of the TRAINING windows only (blocked cross-fitting, see ``crossfit_blocks``);
``sigma`` is fixed a priori (5.5 months: FWHM = 13 months, the standard
smoothing window of the sunspot number), never a single-month spike.
"""

from __future__ import annotations

import numpy as np

HORIZON = 132
SIGMA_DEFAULT = 13.0 / 2.3548


def crossfit_blocks(fit, n_blocks=8, embargo=132):
    """Blocked cross-fitting inside the fit windows.

    Yields ``(train_idx, held_idx)``: ``held_idx`` is a contiguous block of fit
    windows and ``train_idx`` are the fit windows at least ``embargo`` windows away
    from it on either side, so that no target month of a held-out window is a
    training target (windows i and j share target months iff |i - j| < 132).
    """
    fit = np.asarray(fit)
    edges = np.linspace(0, len(fit), n_blocks + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        held = fit[a:b]
        lo, hi = held[0], held[-1]
        train = fit[(fit <= lo - embargo) | (fit >= hi + embargo)]
        yield train, held


def gain_profile(params, tau=None):
    """G(tau) = 1 + g0 + g1 * tau / 131 (clipped at >= 1)."""
    tau = np.arange(HORIZON) if tau is None else tau
    g0, g1 = params[0], params[1]
    return np.maximum(1.0, 1.0 + g0 + g1 * tau / (HORIZON - 1.0))


def apply(yhat, G, sigma=SIGMA_DEFAULT):
    """Apply the calibration with gain profile ``G`` (shape (132,) or (n, 132))."""
    yhat = np.asarray(yhat, dtype=np.float64)
    G = np.broadcast_to(G, yhat.shape)
    A = yhat * G
    tstar = A.argmax(axis=1)
    rows = np.arange(len(yhat))
    gstar = G[rows, tstar]
    tau = np.arange(yhat.shape[1])[None, :]
    bump = np.exp(-0.5 * ((tau - tstar[:, None]) / sigma) ** 2)
    return yhat * (1.0 + (gstar[:, None] - 1.0) * bump)


def peak_stats(yhat, y):
    """(height errors in scaled units, timing errors in months) per window."""
    return np.abs(yhat.max(1) - y.max(1)), np.abs(yhat.argmax(1) - y.argmax(1))


def fit_gain(yhat_oof, y, grid0=None, grid1=None, sigma=SIGMA_DEFAULT, w_pte=1.0):
    """Grid-fit (g0, g1) on out-of-fold forecasts.

    Criterion: PAE / PAE_uncalibrated + w_pte * PTE / PTE_uncalibrated (both relative
    to the uncalibrated forecast, so the two peak metrics are weighted equally for
    w_pte = 1).  Returns (params, table) where table rows are (g0, g1, pae, pte, mse).
    """
    grid0 = np.arange(0.0, 1.01, 0.05) if grid0 is None else grid0
    grid1 = np.arange(0.0, 1.51, 0.1) if grid1 is None else grid1
    h0, t0 = peak_stats(yhat_oof, y)
    pae0, pte0 = h0.mean(), t0.mean()
    best, table = None, []
    for g0 in grid0:
        for g1 in grid1:
            yc = apply(yhat_oof, gain_profile((g0, g1)), sigma)
            h, t = peak_stats(yc, y)
            J = h.mean() / pae0 + w_pte * t.mean() / pte0
            table.append((g0, g1, h.mean(), t.mean(), ((yc - y) ** 2).mean(), J))
            if best is None or J < best[0] - 1e-12:
                best = (J, g0, g1)
    return (best[1], best[2]), np.array(table)
