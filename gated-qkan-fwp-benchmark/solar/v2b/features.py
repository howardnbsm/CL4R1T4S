"""Deterministic featurisations of the 528-month input window (scaled units).

Every function maps ``x`` of shape ``(n, 528)`` (float64) to ``(n, d)`` and uses
nothing but the window itself (no fitted statistics, no future months).
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 528


def patch_means(x, patch=12, months=132):
    """Means of the last ``months`` months in patches of ``patch`` months, oldest first."""
    seg = x[:, SEQ_LEN - months:]
    return seg.reshape(x.shape[0], months // patch, patch).mean(axis=2)


def trailing_means(x, lengths=(3, 6, 12, 24, 36, 48, 72, 96, 132, 198, 264, 396, 528)):
    """Means over the trailing ``L`` months for each ``L`` (multi-resolution level)."""
    return np.stack([x[:, SEQ_LEN - L:].mean(axis=1) for L in lengths], axis=1)


def smooth_edge(x, half=6):
    """Centred running mean of width ``2*half+1`` with half-weight ends (the classical
    13-month smoothing for half=6); near the window edges the kernel is truncated and
    renormalised, so the last value is a one-sided trailing mean."""
    k = np.ones(2 * half + 1)
    k[0] = k[-1] = 0.5
    num = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, x)
    den = np.convolve(np.ones(x.shape[1]), k, mode="same")
    return num / den


def gauss_smooth(x, sigma=8.0):
    """Gaussian smoothing with edge renormalisation (truncated at 3 sigma)."""
    h = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma) ** 2)
    num = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, x)
    den = np.convolve(np.ones(x.shape[1]), k, mode="same")
    return num / den


def demod(x, period=132.0, span=264, harmonics=(1, 2)):
    """Complex demodulation of the last ``span`` months at ``period`` and its harmonics
    (Hann taper, phase referenced to the window end): columns (cos, sin) per harmonic."""
    seg = x[:, SEQ_LEN - span:]
    seg = seg - seg.mean(axis=1, keepdims=True)
    t = np.arange(span) - (span - 1)  # 0 at the window end, negative into the past
    w = np.hanning(span + 2)[1:-1]
    w = w / w.sum()
    cols = []
    for h in harmonics:
        ang = 2 * np.pi * h * t / period
        cols.append(2 * (seg * (w * np.cos(ang))).sum(axis=1))
        cols.append(2 * (seg * (w * np.sin(ang))).sum(axis=1))
    return np.stack(cols, axis=1)


def cycle_descriptors(x, sigma=8.0, min_sep=40, cap=220.0):
    """Event-based descriptors of the solar cycle read off the Gaussian-smoothed window.

    Columns:
      0 months since the last local minimum of the smoothed window (capped)
      1 months since the last local maximum (capped)
      2 smoothed value at that minimum
      3 smoothed value at that maximum
      4 length of the last completed minimum-to-minimum interval (months, capped)
      5 smoothed value at the previous maximum (the one before column 3)
      6 smoothed value now (edge-renormalised)
      7 smoothed slope now (per 12 months)
    Local extrema are extrema over a +-``min_sep`` month neighbourhood that lies
    inside the window; if none is found the cap value / window statistic is used.
    """
    s = gauss_smooth(x, sigma)
    n, T = s.shape
    out = np.zeros((n, 8))
    for r in range(n):
        row = s[r]
        mins, maxs = [], []
        for t in range(min_sep, T):
            lo, hi = t - min_sep, min(T, t + min_sep + 1)
            seg = row[lo:hi]
            # an extremum must be strictly interior to its neighbourhood on the right
            # unless the neighbourhood is cut by the window end (then require >= 6 months margin)
            if hi - 1 - t < 6:
                continue
            if row[t] == seg.min():
                mins.append(t)
            elif row[t] == seg.max():
                maxs.append(t)
        tm = mins[-1] if mins else 0
        tx = maxs[-1] if maxs else 0
        out[r, 0] = min(T - 1 - tm, cap)
        out[r, 1] = min(T - 1 - tx, cap)
        out[r, 2] = row[tm]
        out[r, 3] = row[tx]
        out[r, 4] = min(mins[-1] - mins[-2], cap) if len(mins) >= 2 else 132.0
        out[r, 5] = row[maxs[-2]] if len(maxs) >= 2 else row[tx]
        out[r, 6] = row[-1]
        out[r, 7] = row[-1] - row[-13]
    return out
