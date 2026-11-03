"""(N,T) calibration surface on real returns: raw unconstrained GMV, realised/predicted risk vs its exact Gaussian expectation.

For each sample, universe size N and window W: ``n_subsets`` seeded random subsets of the available stocks (the full universe
is used when N equals its size), common origins start..end step H, and for every (subset, origin) the realised/predicted
STANDARD-DEVIATION ratio sqrt(v/h) with v the H-day sample variance of the portfolio return. The benchmark is
``theory.expected_std_ratio(N, W, H)`` (exact chi-square representation including the noise of an H-day variance).
Subsets of one universe and cells sharing test periods are dependent, so intervals resample ORIGINS in blocks and keep the
subset dimension intact (one resampled origin index applies to every subset and every cell).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, theory
from ..evaluate import common_origins
from . import inference


def _ratio(train: np.ndarray, test: np.ndarray) -> float:
    S = np.cov(train, rowvar=False)
    y = np.linalg.solve(S, np.ones(S.shape[0]))
    w = y / y.sum()
    return float(np.sqrt(np.var(test @ w, ddof=1) / (1.0 / y.sum())))


def surface_samples(R: pd.DataFrame, Ns, windows, H: int = 60, n_subsets: int = 20, seed: int = config.SEED,
                    start: int | None = None) -> dict:
    """{(N, W): array (n_subsets_used, n_origins)} of std ratios; also returns the origins list."""
    X = R.to_numpy()
    origins = common_origins(len(R), windows, H, H) if start is None else list(range(start, len(R) - H + 1, H))
    out, subsets = {}, {}
    for N in Ns:
        rng = np.random.default_rng([seed, N])
        full = N >= X.shape[1]
        subs = [np.arange(X.shape[1])] if full else [np.sort(rng.choice(X.shape[1], N, replace=False)) for _ in range(n_subsets)]
        subsets[N] = subs
        for W in windows:
            if W < N + 3:
                continue
            out[(N, W)] = np.array([[_ratio(X[o - W:o][:, s], X[o:o + H][:, s]) for o in origins] for s in subs])
    return dict(samples=out, origins=origins, subsets=subsets)


def cell_table(res: dict, H: int = 60, B: int = 1000, block: int = 3, seed: int = config.SEED) -> pd.DataFrame:
    n_o = len(res["origins"])
    idx = inference.moving_block_indices(n_o, block, B, np.random.default_rng(seed))
    rows = []
    for (N, W), A in res["samples"].items():
        th, th_se = theory.expected_std_ratio(N, W, H)
        meas = float(A.mean())
        boots = A[:, idx].mean(axis=(0, 2)) / th                      # (B,)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append(dict(N=N, W=W, N_over_T=N / W, n_subsets=A.shape[0], n_origins=n_o, measured=meas, theory=th, theory_se=th_se,
                         ratio=meas / th, ratio_lo=float(lo), ratio_hi=float(hi)))
    return pd.DataFrame(rows)


def fit_vs_ratio(tab: pd.DataFrame, res: dict, H: int = 60, B: int = 500, block: int = 3, seed: int = config.SEED) -> dict:
    """OLS ratio = a + b N/T over cells, Pearson r, and an origin-block bootstrap interval for the slope."""
    x, y = tab.N_over_T.to_numpy(), tab.ratio.to_numpy()
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    n_o = len(res["origins"])
    idx = inference.moving_block_indices(n_o, block, B, np.random.default_rng(seed + 1))
    th = {(int(r_.N), int(r_.W)): r_.theory for r_ in tab.itertuples()}
    slopes = []
    for k in range(B):
        yy = np.array([res["samples"][(int(r_.N), int(r_.W))][:, idx[k]].mean() / th[(int(r_.N), int(r_.W))] for r_ in tab.itertuples()])
        slopes.append(np.polyfit(x, yy, 1)[0])
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return dict(intercept=float(a), slope=float(b), pearson_r=r, slope_lo=float(lo), slope_hi=float(hi), cells=len(tab),
                min_ratio=float(y.min()), max_ratio=float(y.max()), n_above_one=int((y > 1).sum()))
