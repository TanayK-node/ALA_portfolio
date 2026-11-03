"""Gaussian-theory prediction intervals for the realised variance of an H-day block given the corrected forecast.

Under i.i.d. Gaussian returns, v / h_c = R * G / (c * Q) with Q ~ chi2_{T-N}/(T-1), R = 1 + chi2_{N-1}/chi2_{T-N+1},
G ~ chi2_{H-1}/(H-1), all independent (Proposition 1; G is the sample variance of H test returns over the true
variance of the estimated portfolio). The central (1-a) interval for v is h_c * [q_{a/2}, q_{1-a/2}] of that ratio.
"""
from __future__ import annotations

import numpy as np

from .forecast import c_gauss


def ratio_quantiles(N: int, T: int, H: int, probs, reps: int = 400_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.chisquare(T - N, reps) / (T - 1)
    R = 1.0 + rng.chisquare(N - 1, reps) / rng.chisquare(T - N + 1, reps)
    G = rng.chisquare(H - 1, reps) / (H - 1)
    return np.quantile(R * G / (c_gauss(N, T) * Q), probs)


def coverage(v: np.ndarray, h: np.ndarray, qlo: float, qhi: float) -> float:
    return float(np.mean((v >= h * qlo) & (v <= h * qhi)))
