"""Gaussian finite-sample theory for the estimated global-minimum-variance (GMV) portfolio.

Setting (all results below): T i.i.d. observations x_t ~ N(mu, Sigma) in R^N; S is the
sample covariance with the *demeaned, divide-by-(T-1)* convention, so
W = (T-1) S ~ Wishart_N(n, Sigma) with n = T-1.  The estimated GMV weights are
w^ = S^{-1} 1 / (1' S^{-1} 1).  V = 1/(1' Sigma^{-1} 1) is the true GMV variance.
Two quantities are studied:

    Q = (predicted variance) / V = [1/(1' S^{-1} 1)] / V        (in-sample, "what the optimiser thinks")
    R = (true variance of w^)  / V = w^' Sigma w^ / V            (what the weights really deliver)

These apply ONLY to the unconstrained GMV portfolio built from the raw sample
covariance of i.i.d. Gaussian data. They say nothing about long-only, clipped,
MP-filtered or Ledoit-Wolf portfolios, nor about non-Gaussian / dependent data.

TIER 1 -- exact, standard Wishart results (derivation stated; also checked by simulation)
---------------------------------------------------------------------------------------
Fact (Muirhead 1982, Thm 3.2.12; Dickinson 1974 for the portfolio statement): if
W ~ Wishart_N(n, Sigma) and a != 0 is fixed, then a' Sigma^{-1} a / a' W^{-1} a ~ chi^2_{n-N+1}.
With a = 1 and S = W/n:  1' S^{-1} 1 = n 1' W^{-1} 1, hence

    Q = (1'Sigma^{-1}1)/(1'S^{-1}1) ~ chi^2_{T-N} / (T-1),
    E[Q] = (T-N)/(T-1)        (in-sample variance is biased DOWN by the factor (T-N)/(T-1)).

TIER 2 -- out-of-sample inflation: DERIVED HERE, NOT VERIFIED AGAINST THE LITERATURE
------------------------------------------------------------------------------------
Status: the formulas below come from my own derivation (sketched next). I tried to cross-check
them against Kan & Smith (2008), "The Distribution of the Sample Minimum-Variance Frontier",
Management Science 54(7), 1364-1380 (which gives exact out-of-sample distributions of sample
minimum-variance portfolios) and Okhrin & Schmid (2006), J. Econometrics 134, 235-256, but the
sources could not be retrieved from the sandbox (network egress blocked), so they are NOT checked
against the papers. Convention differences (dof n=T-1 vs T, demeaning) are the usual source of
off-by-one differences with published expressions; here the convention is exactly the one above.
A recalled form V(1 + (N-1)/(T-N-2)) differs from what is derived here by the offset (see R/Q below).

Derivation. Let A = Sigma^{-1/2} S Sigma^{-1/2}, e = Sigma^{-1/2} 1, B = A^{-1}. Then
 1'S^{-1}1 = e'Be, 1'S^{-1} Sigma S^{-1} 1 = e'B^2 e, so R = (e'e)(e'B^2 e)/(e'Be)^2 and
 Q = e'e/(e'Be). A is rotation invariant, so take e along the first axis: Q = 1/B_11 (the Schur
 complement) and R = (B^2)_11 / B_11^2. Block inversion gives R = 1 + a_21' A_22^{-2} a_21.
 Writing W = X'X with i.i.d. N(0,1) entries, W_21 | X_2 ~ N(0, W_22), so W_21 = W_22^{1/2} z with
 z ~ N(0, I_{N-1}) independent of W_22, and a_21' A_22^{-2} a_21 = z' W_22^{-1} z. By the Tier-1 fact in
 dimension N-1, z'W_22^{-1}z = chi^2_{N-1} / chi^2_{n-N+2} (independent chi-squares). Therefore

    R - 1 = chi^2_{N-1} / chi^2_{T-N+1}   =   (N-1)/(T-N+1) * F_{N-1, T-N+1},
    E[R]  = 1 + (N-1)/(T-N-1) = (T-2)/(T-N-1)          (needs T-N-1 > 0),
    R is independent of Q (Schur complement is independent of (W_12, W_22)).

Consequences used for like-with-like comparisons (Jensen: do NOT take square roots of expectations):
    ratio of expectations   E[R]/E[Q]  = (T-1)(T-2) / ((T-N)(T-N-1))
    expectation of ratios   E[R/Q]     = E[R] E[1/Q] = (T-2)/(T-N-1) * (T-1)/(T-N-2)   (needs T-N-2 > 0)
    realised/predicted STD  sqrt(R G / Q), G = chi^2_{H-1}/(H-1) from the H-day out-of-sample sample
    variance; its mean has no simple closed form and is computed from the exact chi-square
    representation by sampling (``expected_std_ratio``), times nothing else (G is included).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from . import config


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def predicted_variance_ratio_mean(N: int, T: int) -> float:
    """TIER 1 (exact). E[predicted variance]/V = (T-N)/(T-1); requires T > N."""
    _require(T > N, "need T > N")
    return (T - N) / (T - 1)


def predicted_variance_dof(N: int, T: int) -> int:
    """TIER 1 (exact). (T-1) * Q ~ chi^2 with T-N degrees of freedom."""
    _require(T > N, "need T > N")
    return T - N


def true_variance_ratio_mean(N: int, T: int) -> float:
    """TIER 2 (derived here, not literature-verified). E[true variance]/V = (T-2)/(T-N-1); needs T-N-1 > 0."""
    _require(T - N - 1 > 0, "need T - N - 1 > 0 for E[R] to exist")
    return (T - 2) / (T - N - 1)


def true_variance_ratio_mean_recalled(N: int, T: int) -> float:
    """DIAGNOSTIC ONLY, not a result: the recalled form 1 + (N-1)/(T-N-2) (offset -2).

    Kept so the Monte Carlo can say whether the offset is -1 (derived above) or -2; it is never used
    for anything else and is not tuned. Needs T-N-2 > 0.
    """
    _require(T - N - 2 > 0, "need T - N - 2 > 0")
    return 1.0 + (N - 1) / (T - N - 2)


def variance_ratio_of_expectations(N: int, T: int) -> float:
    """TIER 2. E[true variance] / E[predicted variance] = (T-1)(T-2)/((T-N)(T-N-1))."""
    return true_variance_ratio_mean(N, T) / predicted_variance_ratio_mean(N, T)


def variance_ratio_mean_per_run(N: int, T: int) -> float:
    """TIER 2. E[true/predicted variance] = E[R] E[1/Q] = (T-2)/(T-N-1) * (T-1)/(T-N-2); needs T-N-2 > 0.

    Uses independence of R and Q and E[1/chi^2_k] = 1/(k-2) with k = T-N.
    """
    _require(T - N - 2 > 0, "need T - N - 2 > 0 for E[R/Q] to exist")
    return true_variance_ratio_mean(N, T) * (T - 1) / (T - N - 2)


def c4(m: int) -> float:
    """E[s]/sigma for the unbiased sample std of m i.i.d. normals: sqrt(2/(m-1)) Gamma(m/2)/Gamma((m-1)/2)."""
    _require(m >= 2, "need m >= 2")
    return float(np.sqrt(2.0 / (m - 1)) * np.exp(gammaln(m / 2) - gammaln((m - 1) / 2)))


def sample_exact_QR(N: int, T: int, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw (Q, R) from their exact distributions (independent): Q = chi2_{T-N}/(T-1),
    R = 1 + chi2_{N-1}/chi2_{T-N+1} (R = 1 when N = 1)."""
    _require(T > N, "need T > N")
    Q = rng.chisquare(T - N, size) / (T - 1)
    R = np.ones(size) if N == 1 else 1.0 + rng.chisquare(N - 1, size) / rng.chisquare(T - N + 1, size)
    return Q, R


def expected_std_ratio(N: int, T: int, H: int = config.HORIZON, n_draws: int = config.THEORY_DRAWS,
                       seed: int = config.SEED) -> tuple[float, float]:
    """E[realised std / predicted std] and its Monte Carlo SE, from the exact representation.

    ratio = sqrt(R * G / Q), G = chi^2_{H-1}/(H-1) (H-day out-of-sample sample variance, ddof = 1).
    The SE reflects only the sampling of this representation, not any model error.
    """
    rng = np.random.default_rng(seed)
    Q, R = sample_exact_QR(N, T, n_draws, rng)
    G = rng.chisquare(H - 1, n_draws) / (H - 1)
    x = np.sqrt(R * G / Q)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n_draws))


def theory_table(N: int, windows: list[int] = config.WINDOWS, H: int = config.HORIZON) -> pd.DataFrame:
    """Closed forms and exact-representation means for each window length T = W."""
    rows = []
    for T in windows:
        m, se = expected_std_ratio(N, T, H)
        rows.append(dict(window=T, N=N, th_pred_var_over_V=predicted_variance_ratio_mean(N, T),
                         th_true_var_over_V=true_variance_ratio_mean(N, T),
                         th_var_ratio_of_expectations=variance_ratio_of_expectations(N, T),
                         th_var_ratio_per_run=variance_ratio_mean_per_run(N, T),
                         th_std_ratio=m, th_std_ratio_se=se,
                         th_sqrt_ratio_of_expectations=float(np.sqrt(variance_ratio_of_expectations(N, T)))))
    return pd.DataFrame(rows)
