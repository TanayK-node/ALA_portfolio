"""5D: sensitivity of the eigenvalue-clipping floor and the Marchenko-Pastur noise variance.

Both sweeps run on the common out-of-sample design (same test periods for every window) and report
the mean risk gap and realised risk with block-bootstrap CIs, plus how many eigenvalues each rule
adjusts on average. Parameters are NOT selected: choosing the value with the best out-of-sample
number would be tuning on the test set. The a-priori settings of the main analysis (clip fraction
0.10, MP sigma^2 = tr(S)/N) are one row among the swept values.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config, evaluate, uncertainty
from .covariance import sample_cov
from .linalg_tools import clip_by_mp, clip_eigenvalues, eig_decompose

CovFn = Callable[[pd.DataFrame], dict[str, tuple[np.ndarray, float]]]


def clip_label(frac: float) -> str:
    """Method label for a clip fraction, e.g. 'clip_0.10'."""
    return f"clip_{frac:.2f}"


def clip_cov_fn(fracs: list[float]) -> CovFn:
    """cov_fn returning raw plus one eigenvalue-clipped Sigma per fraction.

    Floor eps = frac * tr(S)/N (a fraction of the mean eigenvalue); S' = Q diag(max(l, eps)) Q'.
    """
    def fn(train: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
        S = sample_cov(train)
        N = S.shape[0]
        out: dict[str, tuple[np.ndarray, float]] = {"raw": (S, np.nan)}
        for f in fracs:
            Sc, n = clip_eigenvalues(S, f * np.trace(S) / N)
            out[clip_label(f)] = (Sc, float(n))
        return out
    return fn


def mp_sigma2(S: np.ndarray, rule: str) -> float:
    """Noise variance for the MP edge: 'trace' -> tr(S)/N; 'median' -> median eigenvalue of S."""
    if rule == "trace":
        return float(np.trace(S) / S.shape[0])
    if rule == "median":
        return float(np.median(eig_decompose(S)[0]))
    raise ValueError(f"unknown sigma2 rule {rule!r}")


def mp_label(rule: str) -> str:
    """Method label for an MP variance rule, e.g. 'mp_trace'."""
    return f"mp_{rule}"


def mp_cov_fn(rules: tuple[str, ...]) -> CovFn:
    """cov_fn returning raw plus one MP-filtered Sigma per sigma^2 rule (edge l+ = s2 (1+sqrt(N/T))^2)."""
    def fn(train: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
        T, N = train.shape
        S = sample_cov(train)
        out: dict[str, tuple[np.ndarray, float]] = {"raw": (S, np.nan)}
        for r in rules:
            Sm, n_noise, _ = clip_by_mp(S, N, T, sigma2=mp_sigma2(S, r))
            out[mp_label(r)] = (Sm, float(n_noise))
        return out
    return fn


def _summ(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean gap / realised risk with CIs, plus mean n_adjusted and median condition number."""
    keys = ["constraint", "method", "window"]
    ci = uncertainty.summary_with_ci(runs, metrics=("risk_gap", "realized_risk"))
    extra = runs.groupby(keys).agg(n_adjusted_mean=("n_adjusted", "mean"),
                                   cond_median=("cond_number", "median")).reset_index()
    return ci.merge(extra, on=keys)


def run_clip_sweep(returns: pd.DataFrame, fracs: list[float] = config.CLIP_SWEEP,
                   windows: list[int] = config.WINDOWS) -> pd.DataFrame:
    """Clipping-floor sweep on the common-OOS design; includes the raw baseline (eps_fraction = 0)."""
    labels = tuple(clip_label(f) for f in fracs)
    runs = evaluate.run_backtest(returns, windows, origins=evaluate.common_origins(len(returns), windows),
                                 methods=("raw", *labels), cov_fn=clip_cov_fn(fracs))
    s = _summ(runs)
    s["eps_fraction"] = s.method.map({"raw": 0.0, **{clip_label(f): f for f in fracs}})
    return s.drop(columns="method").sort_values(["constraint", "eps_fraction", "window"]).reset_index(drop=True)


def run_mp_sweep(returns: pd.DataFrame, rules: tuple[str, ...] = config.MP_SIGMA2_RULES,
                 windows: list[int] = config.WINDOWS) -> pd.DataFrame:
    """MP noise-variance sweep on the common-OOS design; includes the raw baseline (rule = 'raw')."""
    labels = tuple(mp_label(r) for r in rules)
    runs = evaluate.run_backtest(returns, windows, origins=evaluate.common_origins(len(returns), windows),
                                 methods=("raw", *labels), cov_fn=mp_cov_fn(rules))
    s = _summ(runs)
    s["sigma2_rule"] = s.method.map({"raw": "raw", **{mp_label(r): r for r in rules}})
    return s.drop(columns="method").sort_values(["constraint", "sigma2_rule", "window"]).reset_index(drop=True)
