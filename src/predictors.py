"""5C: which training-window quantities predict the out-of-sample risk gap? (pre-registered; see README)

Outcome: risk_gap = realised - predicted annualised risk of the raw-Sigma min-variance portfolio.
Predictors (exactly seven): noise_exposure_score k=1,3,5; log condition number; effective rank;
lambda_min / mean(lambda); max|w|. Statistic: Spearman rho within each window length, pooled by
Fisher-z averaging. Two designs: non-overlapping training windows (stride W; Fisher-z inference) and
stride-60 windows (overlapping; moving-block bootstrap inference).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import bootstrap, config, evaluate, multitest
from .covariance import sample_cov
from .linalg_tools import condition_number, effective_rank, min_eig_ratio

PREDICTORS = ("exposure_k1", "exposure_k3", "exposure_k5", "log_cond", "eff_rank", "lam_ratio", "max_abs_weight")
EXPECTED_SIGN = {"exposure_k1": "+", "exposure_k3": "+", "exposure_k5": "+", "log_cond": "+",
                 "eff_rank": "-", "lam_ratio": "-", "max_abs_weight": "+"}  # recorded, never used for inference
DESIGNS = ("nonoverlap", "stride60_bootstrap")


def nonoverlap_origins(n_obs: int, W: int, horizon: int = config.HORIZON) -> list[int]:
    """Origins W, 2W, 3W, ... with o + horizon <= n_obs (training windows [o-W, o) do not overlap)."""
    return list(range(W, n_obs - horizon + 1, W))


def boot_block_len(W: int, n: int, step: int = config.STEP, base: int = config.BOOT_BLOCK) -> int:
    """Pre-registered block length min(max(base, ceil(W/step)), n // 2), at least 1.

    ceil(W/step) is the number of consecutive stride-`step` origins sharing training data; the cap n//2
    keeps resamples non-degenerate but understates dependence when n is small (long windows).
    """
    return max(1, min(max(base, math.ceil(W / step)), n // 2))


def build_runs(returns: pd.DataFrame, design: str) -> pd.DataFrame:
    """Raw-Sigma backtest runs (both optimizers) under a design, one window at a time."""
    parts = []
    for W in config.WINDOWS:
        origins = (nonoverlap_origins(len(returns), W) if design == "nonoverlap" else None)
        if design not in DESIGNS:
            raise ValueError(design)
        if origins is not None and len(origins) == 0:
            continue
        parts.append(evaluate.run_backtest(returns, windows=[W], methods=("raw",), origins=origins))
    return pd.concat(parts, ignore_index=True)


def add_predictors(runs: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Add log_cond, eff_rank, lam_ratio computed from each run's raw training covariance."""
    rows = []
    for W, o in runs[["window", "origin"]].drop_duplicates().itertuples(index=False):
        S = sample_cov(returns.iloc[o - W:o])
        rows.append(dict(window=W, origin=o, log_cond=float(np.log(condition_number(S))),
                         eff_rank=effective_rank(S), lam_ratio=min_eig_ratio(S)))
    return runs.merge(pd.DataFrame(rows), on=["window", "origin"], how="left")


def _spearman_boot(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman rho of each bootstrap resample (rows of idx); NaN where a resample is constant."""
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    rx, ry = rx - rx.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx**2).sum(1) * (ry**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / den


def _sign_p(stat: np.ndarray) -> float:
    """Bootstrap two-sided sign p: min(1, 2 min(P*(s<=0), P*(s>=0))) with +1 smoothing."""
    s = stat[~np.isnan(stat)]
    B = len(s)
    if B == 0:
        return float("nan")
    return float(min(1.0, 2 * min((np.sum(s <= 0) + 1) / (B + 1), (np.sum(s >= 0) + 1) / (B + 1))))


def _group_rho(x: np.ndarray, y: np.ndarray) -> float:
    return float("nan") if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y)[0])


def test_cell(groups: dict[int, tuple[np.ndarray, np.ndarray]], design: str,
              n_boot: int = config.BOOT_N, seed: int = config.SEED) -> list[dict]:
    """Per-window and pooled tests for one (constraint, predictor) cell. groups: {W: (x, y)} in time order."""
    alpha = config.BOOT_ALPHA
    rows, rho_w, n_w = [], {}, {}
    boots = {}
    for W, (x, y) in groups.items():
        n = len(x)
        rho = _group_rho(x, y)
        rho_w[W], n_w[W] = rho, n
        row = dict(window=W, n=n, rho=rho, p_raw=np.nan, ci_lo=np.nan, ci_hi=np.nan)
        if not np.isnan(rho):
            if design == "nonoverlap":
                row["p_raw"] = float(spearmanr(x, y)[1])
                if n > 3:
                    z, se = np.arctanh(np.clip(rho, -0.999999, 0.999999)), np.sqrt(1.06 / (n - 3))
                    row["ci_lo"], row["ci_hi"] = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))
            else:
                idx = bootstrap.block_bootstrap_indices(n, boot_block_len(W, n), n_boot, np.random.default_rng(seed + W))
                b = _spearman_boot(x, y, idx)
                boots[W] = b
                row["p_raw"] = _sign_p(b)
                if (~np.isnan(b)).any():
                    row["ci_lo"], row["ci_hi"] = (float(v) for v in np.nanquantile(b, [alpha / 2, 1 - alpha / 2]))
        rows.append(row)
    ws, rs, ns = list(groups), np.array([rho_w[W] for W in groups]), np.array([n_w[W] for W in groups])
    pool = multitest.fisher_pool(rs, ns)
    prow = dict(window="pooled", n=pool["n_used"], rho=pool["rho"], p_raw=pool["p"], ci_lo=np.nan, ci_hi=np.nan,
                n_windows_used=pool["k_used"])
    if design == "nonoverlap" and not np.isnan(pool["z"]):
        prow["ci_lo"], prow["ci_hi"] = float(np.tanh(pool["z"] - 1.96 * pool["se"])), float(np.tanh(pool["z"] + 1.96 * pool["se"]))
    if design != "nonoverlap" and boots:
        w = np.array([max(n_w[W] - 3, 0) if n_w[W] > 3 else 0 for W in boots], float)
        B = np.vstack([boots[W] for W in boots])                       # (k, n_boot)
        z = np.arctanh(np.clip(np.nan_to_num(B), -0.999999, 0.999999))
        valid = (~np.isnan(B)) * w[:, None]
        tot = valid.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            zbar = np.where(tot > 0, (z * valid).sum(0) / tot, np.nan)
        rbar = np.tanh(zbar)
        prow["p_raw"] = _sign_p(rbar)
        if (~np.isnan(rbar)).any():
            prow["ci_lo"], prow["ci_hi"] = (float(v) for v in np.nanquantile(rbar, [alpha / 2, 1 - alpha / 2]))
    rows.append(prow)
    return rows


def run_tests(runs: pd.DataFrame, design: str) -> pd.DataFrame:
    """All tests for a design: constraint x predictor x (each window + pooled), with Holm/BH families."""
    out = []
    for cons in config.CONSTRAINTS:
        d = runs[(runs.constraint == cons)].sort_values(["window", "origin"])
        for pred in PREDICTORS:
            groups = {int(W): (g[pred].to_numpy(float), g.risk_gap.to_numpy(float)) for W, g in d.groupby("window")}
            for r in test_cell(groups, design):
                out.append(dict(design=design, constraint=cons, predictor=pred, expected_sign=EXPECTED_SIGN[pred], **r))
    df = pd.DataFrame(out)
    df["family"] = np.where(df.window.astype(str) == "pooled", "primary", "per-window (exploratory)")
    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    for (cons, fam), g in df.groupby(["constraint", "family"]):
        df.loc[g.index, "p_holm"] = multitest.holm_adjust(g.p_raw.to_numpy())
        df.loc[g.index, "p_bh"] = multitest.bh_adjust(g.p_raw.to_numpy())
    return df


def verdicts(tests: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Pre-registered rule: supported iff pooled Holm p < alpha in BOTH designs and same sign of rho."""
    pooled = tests[tests.window.astype(str) == "pooled"]
    a = pooled[pooled.design == "nonoverlap"].set_index(["constraint", "predictor"])
    b = pooled[pooled.design == "stride60_bootstrap"].set_index(["constraint", "predictor"])
    v = pd.DataFrame({"rho_nonoverlap": a.rho, "holm_nonoverlap": a.p_holm, "rho_stride60": b.rho,
                      "holm_stride60": b.p_holm, "expected_sign": a.expected_sign})
    v["same_sign"] = np.sign(v.rho_nonoverlap) == np.sign(v.rho_stride60)
    v["supported"] = (v.holm_nonoverlap < alpha) & (v.holm_stride60 < alpha) & v.same_sign
    return v.reset_index()
