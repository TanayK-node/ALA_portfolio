"""Rolling out-of-sample backtest of min-variance portfolios and diagnostics.

For each window length W and each origin t (training rows [t-W, t), test rows
[t, t+H)): estimate Sigma-hat with each method, build min-variance weights
(unconstrained and long-only), and compare the *predicted* risk sqrt(w'S^w) with
the *realised* out-of-sample risk. Everything is annualised with sqrt(252).

Conventions / caveats (also in the README):
  * Portfolio return on day t is w . r_t with r_t *log* returns (a standard
    approximation to simple returns at daily frequency).
  * Weights are held fixed over the H test days (no drift, no rebalancing).
  * Realised risk is the std of only H=60 daily returns, so it is itself noisy.
  * Training windows of consecutive origins overlap whenever W > STEP, so runs
    are not independent; p-values below are therefore optimistic.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .covariance import ledoit_wolf_cov, sample_cov
from .linalg_tools import (clip_by_mp, clip_eigenvalues, condition_number,
                           inertia, noise_exposure_score)
from .optimize import min_variance_closed_form, min_variance_long_only

log = logging.getLogger(__name__)
ANN = float(np.sqrt(config.TRADING_DAYS))


def estimate_covariances(train: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
    """Return {method: (Sigma-hat, n_adjusted)} on one training window.

    n_adjusted = eigenvalues clipped (``clipped``) or treated as MP noise
    (``mp``); NaN for methods without that notion.
    """
    T, N = train.shape
    S = sample_cov(train)
    eps = config.CLIP_REL * np.trace(S) / N  # floor at a fraction of mean eigenvalue
    S_clip, n_clip = clip_eigenvalues(S, eps)
    S_mp, n_noise, _ = clip_by_mp(S, N, T)
    return {"raw": (S, np.nan), "clipped": (S_clip, float(n_clip)),
            "mp": (S_mp, float(n_noise)), "ledoit_wolf": (ledoit_wolf_cov(train), np.nan)}


def min_var_weights(S: np.ndarray, constraint: str) -> tuple[np.ndarray, bool]:
    """Min-variance weights under 'unconstrained' (solve) or 'long_only' (SLSQP)."""
    if constraint == "unconstrained":
        return min_variance_closed_form(S), True
    if constraint == "long_only":
        return min_variance_long_only(S)
    raise ValueError(f"unknown constraint {constraint!r}")


def oos_stats(w: np.ndarray, test: pd.DataFrame) -> tuple[float, float, float]:
    """(annualised realised risk, annualised mean return, annualised Sharpe)."""
    pr = test.to_numpy() @ w
    sd = float(np.std(pr, ddof=1))
    mean = float(np.mean(pr))
    return sd * ANN, mean * config.TRADING_DAYS, (mean / sd * ANN if sd > 0 else np.nan)


def run_backtest(returns: pd.DataFrame, windows: list[int] = config.WINDOWS,
                 horizon: int = config.HORIZON, step: int = config.STEP,
                 methods: tuple[str, ...] = config.METHODS,
                 constraints: tuple[str, ...] = config.CONSTRAINTS) -> pd.DataFrame:
    """Run the full rolling backtest; one row per (window, origin, method, constraint)."""
    R = returns
    T, N = R.shape
    prev_w: dict[tuple, np.ndarray] = {}
    rows = []
    for W in windows:
        origins = range(W, T - horizon + 1, step)
        if len(origins) == 0:
            log.warning("window %d: not enough data for any run", W)
        for o in origins:
            train, test = R.iloc[o - W:o], R.iloc[o:o + horizon]
            covs = estimate_covariances(train)
            S_raw = covs["raw"][0]
            for method, constraint in itertools.product(methods, constraints):
                S_hat, n_adj = covs[method]
                try:
                    w, ok = min_var_weights(S_hat, constraint)
                except np.linalg.LinAlgError:
                    log.warning("W=%d o=%d %s/%s: singular solve, skipped", W, o, method, constraint)
                    continue
                pred = float(np.sqrt(max(w @ S_hat @ w, 0.0))) * ANN
                real, ann_ret, sharpe = oos_stats(w, test)
                key = (W, method, constraint)
                turnover = (float(np.abs(w - prev_w[key]).sum())
                            if key in prev_w else np.nan)
                prev_w[key] = w
                n_pos, n_zero, n_neg = inertia(S_hat)
                row = dict(window=W, origin=o, method=method, constraint=constraint,
                           train_start=train.index[0], train_end=train.index[-1],
                           test_start=test.index[0], test_end=test.index[-1],
                           predicted_risk=pred, realized_risk=real, risk_gap=real - pred,
                           insample_risk_sample_cov=float(np.sqrt(max(w @ S_raw @ w, 0.0))) * ANN,
                           max_abs_weight=float(np.abs(w).max()), turnover=turnover,
                           cond_number=condition_number(S_hat),
                           n_pos=n_pos, n_zero=n_zero, n_neg=n_neg, n_adjusted=n_adj,
                           oos_return=ann_ret, sharpe=sharpe, converged=ok)
                # Diagnostic always measured in the *raw* window's eigenbasis, so it
                # describes where the weights live relative to the sample noise.
                for k in config.BOTTOM_K:
                    row[f"exposure_k{k}"] = noise_exposure_score(S_raw, w, k)
                rows.append(row)
    return pd.DataFrame(rows)


_AGG_COLS = ["predicted_risk", "realized_risk", "risk_gap", "max_abs_weight", "turnover",
             "cond_number", "sharpe", "oos_return"] + [f"exposure_k{k}" for k in config.BOTTOM_K]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and median of each metric by method x constraint x window (+ run count)."""
    g = runs.groupby(["constraint", "method", "window"])
    out = g[_AGG_COLS].agg(["mean", "median"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_runs", g.size())
    return out.reset_index()


def _within_window_corr(sub: pd.DataFrame, col: str) -> tuple[int, float, float]:
    """Pooled within-window rank correlation (removes window-length effects).

    Within each window, replace both variables by their percentile ranks
    (rank / n, average ties) so every window contributes on the same [0, 1]
    scale and any window-level shift in either variable is removed; then take
    the Pearson correlation of the pooled percentile ranks (= a Spearman
    correlation stratified by window). The p-value treats runs as independent,
    which they are not (overlapping training windows), so it is optimistic.
    """
    d = sub[["window", col, "risk_gap"]].dropna()
    g = d.groupby("window")
    x, y = g[col].rank(pct=True), g["risk_gap"].rank(pct=True)
    if len(d) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return len(d), np.nan, np.nan
    rho, p = pearsonr(x, y)
    return len(d), float(rho), float(p)


def exposure_vs_gap(runs: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of noise-exposure score vs risk gap, raw-Sigma runs.

    Computed for each constraint and each k, with window label:
      * 'all'    -- pooled over windows (confounded: window length shifts both
                    the exposure score and the gap);
      * 'within' -- pooled within-window ranks (added after observing that
                    confound in the 'all' rows; see ``_within_window_corr``);
      * <int>    -- a single window length.
    Reported as measured, with no filtering or multiple-testing correction.
    """
    raw = runs[runs["method"] == "raw"]
    rows = []
    for constraint, k in itertools.product(config.CONSTRAINTS, config.BOTTOM_K):
        col = f"exposure_k{k}"
        sub_c = raw[raw["constraint"] == constraint]
        for label, sub in [("all", sub_c)] + [(int(w), d) for w, d in sub_c.groupby("window")]:
            d = sub[[col, "risk_gap"]].dropna()
            if len(d) < 3 or d[col].nunique() < 2:
                rho = p = np.nan
            else:
                rho, p = spearmanr(d[col], d["risk_gap"])
            rows.append(dict(constraint=constraint, k=k, window=label, n=len(d),
                             spearman_rho=rho, p_value=p))
        n, rho, p = _within_window_corr(sub_c, col)
        rows.append(dict(constraint=constraint, k=k, window="within", n=n,
                         spearman_rho=rho, p_value=p))
    return pd.DataFrame(rows)


def save_results(runs: pd.DataFrame, out_dir=config.RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Write backtest_runs.csv, summary_table.csv, exposure_vs_gap.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"backtest_runs": runs, "summary_table": summarize(runs),
              "exposure_vs_gap": exposure_vs_gap(runs)}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables
