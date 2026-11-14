"""Shared helpers of solar v2 approach A: data access, rolling-origin folds, scoring.

Everything goes through ``solar/protocol.py`` (sealed test split).  Nothing in
this package ever passes ``unseal=True``.

Rolling-origin cross-validation INSIDE the training split
---------------------------------------------------------
Window ``i`` has inputs at months ``i .. i+527`` and targets at months
``i+528 .. i+659``.  A fold validates on a block of 260 consecutive training
windows ``[v0, v0+260)`` and fits on windows ``i <= v0 - 132`` only, so the last
training target month (``v0 - 132 + 659 = v0 + 527``) precedes the first
validation target month (``v0 + 528``): no validation target month is ever a
training target (embargo of 131 windows).

    fold  val windows   fit windows  validation targets (cycles)
    cv1   1044..1303    0..912       1880-01..1912-07  (12, 13, 14: weak)
    cv2   1304..1563    0..1172      1901-09..1934-03  (14, 15, 16: weak/medium)
    cv3   1564..1823    0..1432      1923-05..1955-11  (16, 17, 18)
    cv4   1824..2083    0..1692      1945-01..1977-07  (18, 19, 20: strongest)
    val   2084..2343    0..2083      1966-09..1999-04  (official; NO embargo, as v1)

"Far" folds far1..far4 (test-like distance)
-------------------------------------------
The official test windows are 2344..2605, i.e. 261..522 windows after the last training
window (the validation block lies in between and may not be trained on), whereas the
official validation windows are 1..260 windows after it and share target months with the
training targets.  ``farK`` uses the evaluation block of ``cvK`` but fits only on windows
``i <= v0 - 261``, reproducing the train -> test distance (the block of 260 windows in
front of the evaluation block is left out, exactly like the validation block in front of
the test block).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

_SOLAR_DIR = Path(__file__).resolve().parents[1]
if str(_SOLAR_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLAR_DIR))

import protocol  # noqa: E402

PROJECT_DIR = protocol.PROJECT_DIR
RUNS_DIR = PROJECT_DIR / "runs" / "solar" / "v2a"
SEQ_LEN, HORIZON = protocol.SEQ_LEN, protocol.HORIZON
N_TRAIN, N_VAL = protocol.N_TRAIN, protocol.N_VAL
FOLD_VAL_SIZE = 260
EMBARGO = HORIZON  # fit windows i <= v0 - 132

_DATA = None


def data():
    """The sealed protocol data object (train + validation windows only)."""
    global _DATA
    if _DATA is None:
        _DATA = protocol.load()
        assert not _DATA.unsealed
    return _DATA


def folds(names=("cv1", "cv2", "cv3", "cv4", "val")):
    """Return ``{name: (fit_index, eval_index)}`` (numpy index arrays into the sealed windows)."""
    out = {}
    starts = {"cv1": 1044, "cv2": 1304, "cv3": 1564, "cv4": 1824}
    for name in names:
        if name == "val":
            out[name] = (np.arange(0, N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_VAL))
        elif name.startswith("far"):
            v0 = starts["cv" + name[3:]]
            out[name] = (np.arange(0, v0 - FOLD_VAL_SIZE), np.arange(v0, v0 + FOLD_VAL_SIZE))
        else:
            v0 = starts[name]
            assert v0 + FOLD_VAL_SIZE <= N_TRAIN
            out[name] = (np.arange(0, v0 - EMBARGO + 1), np.arange(v0, v0 + FOLD_VAL_SIZE))
            # embargo check: last fit target month < first eval target month
            assert (v0 - EMBARGO) + SEQ_LEN + HORIZON - 1 < v0 + SEQ_LEN
    return out


def score(pred, y):
    """Reference metrics + peak-aware MSE for scaled predictions / targets."""
    m = protocol.metrics(pred, y)
    m["peak_aware_mse"] = protocol.peak_aware_mse(pred, y)
    return m


def short(m):
    return f"{m['scaled_mse']:.4f}/{m['pae']:5.1f}/{m['pte']:4.1f}"


def block_mean_baseline(x, n_blocks=4):
    """Mean of the last ``n_blocks`` 132-month blocks of the input window, (n, 132)."""
    x = np.asarray(x)
    blocks = x[:, SEQ_LEN - n_blocks * HORIZON:].reshape(x.shape[0], n_blocks, HORIZON)
    return blocks.mean(axis=1)
