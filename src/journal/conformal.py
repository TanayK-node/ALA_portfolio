"""Conformal-style prediction intervals for realised variance from matured past log-ratios (leak-free by construction)."""
from __future__ import annotations

import numpy as np

WARM = 20


def conformal_bounds(rho: np.ndarray, a: float, groups: np.ndarray | None = None, min_group: int = 19, warm: int = WARM):
    """Return (lo, hi, defined) log-ratio bounds for every origin k using ONLY rho[:k] (and groups[:k+1] for the current group).

    rho[k] is the score log(v_k/h_k) of origin k, known only after the test block ends, i.e. by origin k+1. With groups, the calibration
    set is the past scores in the current origin's group when it has at least ``min_group`` of them, else all past scores.
    """
    n_tot = len(rho)
    lo, hi = np.full(n_tot, np.nan), np.full(n_tot, np.nan)
    for k in range(warm, n_tot):
        past = rho[:k]
        if groups is not None:
            same = past[groups[:k] == groups[k]]
            if len(same) >= min_group:
                past = same
        n = len(past)
        rl, ru = int(np.floor(round((n + 1) * a / 2, 9))), int(np.ceil(round((n + 1) * (1 - a / 2), 9)))   # round: 1 - 0.1 is 0.8999..
        if rl < 1 or ru > n:
            continue
        s = np.sort(past)
        lo[k], hi[k] = s[rl - 1], s[ru - 1]
    return lo, hi, np.isfinite(lo)


def interval_score(y: np.ndarray, l: np.ndarray, u: np.ndarray, a: float) -> np.ndarray:
    return (u - l) + (2 / a) * np.maximum(l - y, 0) + (2 / a) * np.maximum(y - u, 0)
