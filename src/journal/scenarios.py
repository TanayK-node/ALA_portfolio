"""Controlled return generators with a FIXED population covariance."""
from __future__ import annotations

import numpy as np

STRUCTURES = ("equicorr", "factor", "hetero")


def make_sigma(structure: str, N: int, seed: int = 7) -> np.ndarray:
    """Pre-specified population covariances (daily units, volatilities of order 1%)."""
    rng = np.random.default_rng(seed)
    if structure == "equicorr":
        rho, s = 0.3, 0.01
        return s * s * ((1 - rho) * np.eye(N) + rho * np.ones((N, N)))
    if structure == "factor":                       # one dominant market factor plus idiosyncratic noise
        beta = rng.uniform(0.6, 1.4, N)
        return (0.008 ** 2) * np.outer(beta, beta) + np.diag(rng.uniform(0.006, 0.012, N) ** 2)
    if structure == "hetero":                       # heterogeneous variances and correlations
        vol = rng.uniform(0.008, 0.02, N)
        A = rng.standard_normal((N, N))
        C = A @ A.T / N + 0.5 * np.eye(N)
        d = np.sqrt(np.diag(C))
        return np.outer(vol, vol) * (C / np.outer(d, d))
    raise ValueError(f"unknown structure {structure!r}")


def draw(scenario: str, b: int, rows: int, Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(b, rows, N) i.i.d. draws with covariance exactly Sigma. ``scenario``: 'gaussian' or 't<df>' (df > 2).

    Student-t: x = L z * sqrt((df-2)/chi2_df) so Cov(x) = Sigma (a t SCALE matrix is NOT its covariance).
    The variance of Q/R estimates needs a finite 4th moment (df > 4); smaller df are refused.
    """
    L = np.linalg.cholesky(Sigma)
    N = Sigma.shape[0]
    if scenario.startswith("skew"):
        return _skewed(scenario, b, rows, N, L, rng)
    z = rng.standard_normal((b, rows, N)) @ L.T
    if scenario == "gaussian":
        return z
    if scenario.startswith("t"):
        df = float(scenario[1:])
        if df <= 4:
            raise ValueError("Student-t df must exceed 4 so the 4th moment exists")
        return z * np.sqrt((df - 2.0) / rng.chisquare(df, size=(b, rows, 1)))
    raise ValueError(f"unknown scenario {scenario!r}")


SKEW_SPECS = {"skew4": (4.0, None), "skew1": (1.0, None), "skewt8": (4.0, 8.0)}     # (gamma shape k, t df or None)


def _skewed(scenario: str, b: int, rows: int, N: int, L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Skewed returns with covariance exactly Sigma: independent centred, unit-variance Gamma(k) innovations
    (skewness 2/sqrt(k), excess kurtosis 6/k) mixed by the Cholesky factor, so every asset and the equal-weight
    portfolio are right-skewed. 'skewt8' additionally applies the Student-t(8) variance mixing used above
    (skewed AND heavy-tailed). The sign is flipped for the negative-skew variant by a leading 'neg' prefix."""
    neg = scenario.startswith("skewneg")
    key = scenario.replace("skewneg", "skew", 1) if neg else scenario
    if key not in SKEW_SPECS:
        raise ValueError(f"unknown scenario {scenario!r}")
    k, df = SKEW_SPECS[key]
    z = (rng.gamma(k, 1.0, size=(b, rows, N)) - k) / np.sqrt(k)
    if neg:
        z = -z
    x = z @ L.T
    if df is not None:
        x = x * np.sqrt((df - 2.0) / rng.chisquare(df, size=(b, rows, 1)))
    return x
