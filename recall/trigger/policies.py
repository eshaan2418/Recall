"""Memory-trigger policies μ: which 2 fps frames get heavy perception.

Every policy returns a boolean ``processed`` mask over the timeline. Cost is
``processed.sum()`` heavy invocations (plus the always-on cost of the gate
itself, accounted separately in the energy model).

* ``uniform``   — every k-th frame
* ``motion``    — frame-difference energy above a threshold (+ burst)
* ``hand``      — cheap hand-presence detector score above a threshold (+ burst)
* ``learned``   — tiny gate network score above a threshold (+ burst)
* ``oracle``    — ground-truth interaction windows (upper bound, B4)
* ``all``       — always-heavy
"""

from __future__ import annotations

import numpy as np

RATE = 2.0


def uniform(T: int, k: int, offset: int = 0) -> np.ndarray:
    m = np.zeros(T, bool)
    m[offset::k] = True
    return m


def burst(fire: np.ndarray, length: int, refractory: int = 0) -> np.ndarray:
    """Turn sparse firing decisions into processed bursts of ``length`` frames.

    While a burst is active the gate keeps running; a new firing inside the
    burst extends it (so long interactions are covered continuously).
    """
    T = len(fire)
    m = np.zeros(T, bool)
    until = -1
    cool = -1
    for i in range(T):
        if fire[i] and i > cool:
            until = max(until, i + length - 1)
        if i <= until:
            m[i] = True
            if i == until:
                cool = i + refractory
    return m


def threshold_policy(score: np.ndarray, thr: float, length: int = 4, refractory: int = 0) -> np.ndarray:
    return burst(score >= thr, length, refractory)


def motion_score(thumbs: np.ndarray) -> np.ndarray:
    g = thumbs.astype(np.float32).mean(-1)
    d = np.zeros(len(g), np.float32)
    d[1:] = np.abs(g[1:] - g[:-1]).mean((1, 2))
    return d


def oracle(t: np.ndarray, windows: np.ndarray, pad_s: float = 1.0) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in windows:
        m |= (t >= a - pad_s) & (t <= b + pad_s)
    return m


def threshold_for_budget(score: np.ndarray, frac: float, length: int = 4) -> float:
    """Binary-search the threshold so that the processed fraction ≈ ``frac`` (on a calibration set)."""
    lo, hi = float(score.min()) - 1e-6, float(score.max()) + 1e-6
    for _ in range(40):
        mid = (lo + hi) / 2
        f = threshold_policy(score, mid, length).mean()
        if f > frac:
            lo = mid
        else:
            hi = mid
    return hi
