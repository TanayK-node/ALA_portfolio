"""Kurtosis-aware multiplier for the Gaussian calibration constants (see configs/journal_kurtosis.yaml)."""
from __future__ import annotations

import numpy as np

P, Q = 1.8035, 2.6733          # frozen: fitted on the Student-t simulations only (see config)
CLIP = (1.0, 4.0)


def gaussian_mean_d4(N: int, T: int) -> float:
    """E[mean_i d_i^4] for in-sample Mahalanobis distances (S with divisor T-1) of i.i.d. Gaussian data."""
    a, b = N / 2.0, (T - N - 1) / 2.0
    ex2_beta = a * (a + 1) / ((a + b) * (a + b + 1))
    return ((T - 1) ** 2 / T) ** 2 * ex2_beta


def mardia_ratio(X: np.ndarray) -> float:
    """beta_hat for rows = observations: Mardia kurtosis over its exact Gaussian expectation (about 1 under normality)."""
    T, N = X.shape
    Z = X - X.mean(axis=0)
    S = Z.T @ Z / (T - 1)
    d2 = np.einsum("ti,ti->t", Z @ np.linalg.inv(S), Z)
    return float(np.mean(d2 ** 2) / gaussian_mean_d4(N, T))


def mardia_ratio_batch(Xb: np.ndarray) -> np.ndarray:
    b, T, N = Xb.shape
    Z = Xb - Xb.mean(axis=1, keepdims=True)
    S = np.einsum("bti,btj->bij", Z, Z) / (T - 1)
    Y = np.linalg.solve(S, Z.transpose(0, 2, 1)).transpose(0, 2, 1)
    d2 = np.einsum("bti,bti->bt", Y, Z)
    return (d2 ** 2).mean(axis=1) / gaussian_mean_d4(N, T)


def phi(x: float) -> float:
    return P * x / (1.0 + Q * x)


def multiplier(beta_hat, N: int, T: int):
    return np.clip(beta_hat, *CLIP) ** phi(N / T)


def ew_kurtosis_ratio(X: np.ndarray) -> float:
    """v2 estimator: kurtosis of the equal-weight portfolio return of the window over its Gaussian expectation 3(T-1)/(T+1)."""
    y = np.asarray(X, dtype=float).mean(axis=1)
    T = len(y)
    z = y - y.mean()
    return float(np.mean(z ** 4) / np.mean(z ** 2) ** 2 / (3.0 * (T - 1) / (T + 1)))


def ew_kurtosis_ratio_batch(Xb: np.ndarray) -> np.ndarray:
    y = Xb.mean(axis=2)
    T = y.shape[1]
    z = y - y.mean(axis=1, keepdims=True)
    return (z ** 4).mean(axis=1) / (z ** 2).mean(axis=1) ** 2 / (3.0 * (T - 1) / (T + 1))
