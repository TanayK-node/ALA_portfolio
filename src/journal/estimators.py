"""Re-implementations of two comparators used in the conference paper but absent from the repository.

* ``nl_cov``: analytical nonlinear shrinkage of the sample covariance (Ledoit & Wolf, Annals of Statistics 48(5), 2020).
  Written from the published estimator (Epanechnikov-kernel smoothing of the sample spectral density and its Hilbert
  transform, bandwidth h = n^(-1/3)); only the p <= n branch is implemented. It has NOT been compared with the
  authors' reference code, only checked by properties (see tests/test_journal_estimators.py).
* ``bps_weights``: shrinkage of the unconstrained GMV weights toward equal weights, w = a*w_GMV + (1-a)*(1/N), with the
  intensity that minimises the Gaussian finite-sample expected out-of-sample variance (bona fide: clipped to [0, 1]).
  It uses the exact moments of Proposition 1 and E[w_GMV] = w*, so it is a finite-sample analogue of the
  intensity of Bodnar, Parolya & Schmid (EJOR 266, 2018); it is NOT their formula verbatim.
"""
from __future__ import annotations

import numpy as np

from ..covariance import sample_cov
from ..linalg_tools import symmetrize
from ..optimize import min_variance_closed_form


def nl_cov(X: np.ndarray) -> np.ndarray:
    """Analytical nonlinear shrinkage covariance for rows = observations (demeaned, divisor T-1)."""
    X = np.asarray(X, dtype=float)
    T, p = X.shape
    n = T - 1                                           # effective sample size after demeaning
    if p > n:
        raise ValueError("nl_cov implements the p <= n branch only")
    S = sample_cov(X)
    lam, u = np.linalg.eigh(S)                          # ascending
    lam = np.maximum(lam, np.finfo(float).tiny)
    L = np.tile(lam[:, None], (1, p))                   # L[i, j] = lam_i
    h = n ** (-1.0 / 3.0)
    H = h * L.T                                         # H[i, j] = h * lam_j
    x = (L - L.T) / H                                   # (lam_i - lam_j) / (h lam_j)
    s5 = np.sqrt(5.0)
    ftilde = (3.0 / 4.0 / s5) * np.mean(np.maximum(1 - x ** 2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hf = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / s5 / np.pi) * (1 - x ** 2 / 5.0) * np.log(np.abs((s5 - x) / (s5 + x)))
    edge = np.isclose(np.abs(x), s5)
    Hf[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    Hftilde = np.mean(Hf / H, axis=1)
    c = p / n
    dtilde = lam / ((np.pi * c * lam * ftilde) ** 2 + (1 - c - np.pi * c * lam * Hftilde) ** 2)
    return symmetrize((u * dtilde) @ u.T)


def bps_weights(X: np.ndarray) -> np.ndarray:
    """GMV weights shrunk toward 1/N with a clipped, Gaussian-finite-sample intensity (see module docstring)."""
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    if T - N - 1 <= 0:
        raise ValueError("need T >= N + 2")
    S = sample_cov(X)
    w = min_variance_closed_form(S)
    b = np.full(N, 1.0 / N)
    inv_sum = float(np.ones(N) @ np.linalg.solve(S, np.ones(N)))
    V_hat = (T - 1) / (T - N) / inv_sum                 # unbiased for the true GMV variance V (E[Q] = (T-N)/(T-1))
    Rb = float(b @ S @ b)                               # unbiased for b'Sigma b
    excess = V_hat * (N - 1) / (T - N - 1)              # E[w_GMV' Sigma w_GMV] - V
    num = Rb - V_hat
    a = 0.0 if num <= 0 else num / (num + excess)
    a = float(min(max(a, 0.0), 1.0))
    return a * w + (1 - a) * b


def bps_weights_verbatim(X: np.ndarray) -> np.ndarray:
    """Bodnar, Parolya & Schmid (EJOR 266, 2018) bona fide shrinkage of the GMV weights toward b = 1/N, c = p/n < 1.

    R_b = (1-c) b'S b * 1'S^-1 1 - 1;  alpha = (1-c) R_b / (c + (1-c) R_b), clipped to [0, 1] (bona fide).
    n is the number of observations T; S is the unbiased sample covariance. Formula read from the arXiv version
    (1406.0437, ar5iv HTML); not checked against the authors' code.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    c = N / T
    if c >= 1:
        raise ValueError("verbatim formula implemented for c < 1 only")
    S = sample_cov(X)
    w = min_variance_closed_form(S)
    b = np.full(N, 1.0 / N)
    inv_sum = float(np.ones(N) @ np.linalg.solve(S, np.ones(N)))
    Rb = (1 - c) * float(b @ S @ b) * inv_sum - 1.0
    a = (1 - c) * Rb / (c + (1 - c) * Rb)
    a = float(min(max(a, 0.0), 1.0))
    return a * w + (1 - a) * b
