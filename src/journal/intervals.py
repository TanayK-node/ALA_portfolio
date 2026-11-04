"""Gaussian-theory prediction intervals for the realised variance of an H-day block given the corrected forecast.

Under i.i.d. Gaussian returns, v / h_c = R * G / (c * Q) with Q ~ chi2_{T-N}/(T-1), R = 1 + chi2_{N-1}/chi2_{T-N+1},
G ~ chi2_{H-1}/(H-1), all independent (Proposition 1; G is the sample variance of H test returns over the true
variance of the estimated portfolio). The central (1-a) interval for v is h_c * [q_{a/2}, q_{1-a/2}] of that ratio.
"""
from __future__ import annotations

import numpy as np

from .forecast import c_gauss, c_qlike


def multiplier_value(N: int, T: int, multiplier: str = "c") -> float:
    """Centring multiplier of the forecast h = kappa * w'S w: 'c' (level-unbiased) or 'cqlike' (QLIKE-optimal, needs T >= N + 3)."""
    return {"c": c_gauss, "cqlike": c_qlike}[multiplier](N, T)


def ratio_quantiles(N: int, T: int, H: int, probs, reps: int = 400_000, seed: int = 0, multiplier: str = "c") -> np.ndarray:
    """Quantiles of v / h = R G / (kappa Q) for the forecast h = kappa w'S w.

    The interval for v in variance units is h * quantile, i.e. (w'S w) * quantile of R G / Q, which does not depend on kappa:
    kappa only re-centres the ratio, so the 'c' and 'cqlike' pivots give the same interval for v (same seed, same draws).
    """
    rng = np.random.default_rng(seed)
    Q = rng.chisquare(T - N, reps) / (T - 1)
    R = 1.0 + rng.chisquare(N - 1, reps) / rng.chisquare(T - N + 1, reps)
    G = rng.chisquare(H - 1, reps) / (H - 1)
    return np.quantile(R * G / (multiplier_value(N, T, multiplier) * Q), probs)


def coverage(v: np.ndarray, h: np.ndarray, qlo: float, qhi: float) -> float:
    return float(np.mean((v >= h * qlo) & (v <= h * qhi)))
