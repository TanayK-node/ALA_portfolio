"""5F: block missingness from late listings -- can pairwise-complete covariance go indefinite?

Scenario: m randomly chosen stocks have their first fraction f of observations missing (all list on the
same date). The pairwise-complete covariance then mixes two samples: pairs among never-missing stocks use
all T rows, every pair that involves a late stock uses only the last (1-f)T rows. Such a patchwork need
not be PSD. We measure the inertia / lambda_min / non-PSD rate per (m, f) over seeds, repair with our
eigenvalue clipping, and compare with the complete-case covariance (drop every row with a NaN, i.e. use
only the last (1-f)T rows for all stocks).

Reference "truth" for judging weights: the full-sample covariance S_full of the UNMASKED data. For any
weight vector w (sum 1) the variance under S_full relative to the best possible,
    ratio(w) = w' S_full w / V_full,   V_full = 1 / (1' S_full^{-1} 1) >= ... ratio >= 1,
equals 1 only for the full-information GMV weights. This is a diagnostic against a fixed reference, not
an out-of-sample test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .covariance import pairwise_cov, sample_cov
from .linalg_tools import (clip_eigenvalues, eig_decompose, frobenius_distance, inertia,
                           is_positive_definite)
from .optimize import min_variance_closed_form


def late_listing_mask(returns: pd.DataFrame, m: int, f: float, seed: int
                      ) -> tuple[pd.DataFrame, np.ndarray]:
    """NaN-out the first round(f*T) rows of m randomly chosen columns; returns (masked, chosen column indices)."""
    N = returns.shape[1]
    if not 0 < m <= N:
        raise ValueError("m must be in 1..N")
    rng = np.random.default_rng([config.SEED, m, int(round(100 * f)), seed])
    cols = np.sort(rng.choice(N, size=m, replace=False))
    k = int(round(f * len(returns)))
    out = returns.copy()
    out.iloc[:k, cols] = np.nan
    return out, cols


def _ratio(w: np.ndarray, S_full: np.ndarray, V_full: float) -> float:
    return float(w @ S_full @ w / V_full)


def run_cell(returns: pd.DataFrame, S_full: np.ndarray, V_full: float, m: int, f: float, seed: int) -> dict:
    """One (m, f, seed) replication: inertia of the pairwise Sigma, clipping repair, complete-case comparison."""
    masked, cols = late_listing_mask(returns, m, f, seed)
    N = returns.shape[1]
    S = pairwise_cov(masked)
    p, z, n = inertia(S)
    vals, _ = eig_decompose(S)
    row = dict(m=m, f=f, seed=seed, late_stocks=",".join(map(str, cols)), n_pos=p, n_zero=z, n_neg=n,
               lambda_min=float(vals[0]), non_psd=n > 0,
               cond=float("inf") if vals[0] <= 0 else float(vals[-1] / vals[0]))
    # weights from the pairwise matrix as it is (indefinite S: 'solve' still returns something)
    try:
        w0 = min_variance_closed_form(S)
        row.update(pw_pred_var=float(w0 @ S @ w0), pw_max_abs_w=float(np.abs(w0).max()),
                   pw_var_ratio=_ratio(w0, S_full, V_full))
    except np.linalg.LinAlgError:
        w0 = None
        row.update(pw_pred_var=np.nan, pw_max_abs_w=np.nan, pw_var_ratio=np.nan)
    # complete-case alternative: only rows where every stock is observed
    cc = masked.dropna()
    Scc = sample_cov(cc)
    wcc = min_variance_closed_form(Scc)
    row.update(cc_rows=len(cc), cc_pd=is_positive_definite(Scc), cc_var_ratio=_ratio(wcc, S_full, V_full),
               cc_max_abs_w=float(np.abs(wcc).max()))
    # repairs: a-priori relative floor (as in the main analysis) and a minimal absolute floor
    floors = {"rel": config.CLIP_REL * np.trace(S) / N, "min": config.CLIP_EPS}
    for tag, eps in floors.items():
        Sc, n_clip = clip_eigenvalues(S, eps)
        pc, zc, nc = inertia(Sc)
        vc, _ = eig_decompose(Sc)
        d_f = frobenius_distance(Sc, S)
        spec_d = float(np.sqrt(np.sum((np.maximum(vals, eps) - vals) ** 2)))  # exact identity for spectral repair
        wc = min_variance_closed_form(Sc)
        row.update({f"clip_{tag}_eps": float(eps), f"clip_{tag}_n_clipped": n_clip,
                    f"clip_{tag}_inertia_pos": pc, f"clip_{tag}_inertia_zero": zc, f"clip_{tag}_inertia_neg": nc,
                    f"clip_{tag}_min_eig_ok": bool(vc[0] >= eps - 1e-12),
                    f"clip_{tag}_frob": d_f, f"clip_{tag}_frob_rel": d_f / frobenius_distance(S, np.zeros_like(S)),
                    f"clip_{tag}_frob_matches_spectrum": bool(abs(d_f - spec_d) <= 1e-9 * max(1.0, d_f)),
                    f"clip_{tag}_max_abs_w": float(np.abs(wc).max()),
                    f"clip_{tag}_var_ratio": _ratio(wc, S_full, V_full)})
        if w0 is not None:
            row[f"clip_{tag}_dw_l1"] = float(np.abs(wc - w0).sum())
            row[f"clip_{tag}_dw_max"] = float(np.abs(wc - w0).max())
        else:
            row[f"clip_{tag}_dw_l1"] = row[f"clip_{tag}_dw_max"] = np.nan
    return row


def run_experiment(returns: pd.DataFrame, ms: list[int] = config.BLOCK_M, fs: list[float] = config.BLOCK_F,
                   n_seeds: int = config.BLOCK_SEEDS) -> pd.DataFrame:
    """All (m, f, seed) replications on the complete-data ``returns`` (no NaNs)."""
    S_full = sample_cov(returns)
    ones = np.ones(S_full.shape[0])
    V_full = 1.0 / float(ones @ np.linalg.solve(S_full, ones))
    return pd.DataFrame([run_cell(returns, S_full, V_full, m, f, s) for m in ms for f in fs for s in range(n_seeds)])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (m, f): non-PSD rate with binomial SE, lambda_min summary, repair cost, weight changes, comparisons."""
    g = df.groupby(["m", "f"])
    out = g.agg(runs=("seed", "size"), non_psd=("non_psd", "sum"), non_psd_rate=("non_psd", "mean"),
                lambda_min_min=("lambda_min", "min"), lambda_min_median=("lambda_min", "median"),
                max_n_neg=("n_neg", "max"), cc_rows=("cc_rows", "first"),
                pw_var_ratio_median=("pw_var_ratio", "median"), cc_var_ratio_median=("cc_var_ratio", "median"),
                clip_rel_var_ratio_median=("clip_rel_var_ratio", "median"),
                clip_rel_frob_rel_median=("clip_rel_frob_rel", "median"),
                clip_min_frob_rel_median=("clip_min_frob_rel", "median")).reset_index()
    out["non_psd_rate_se"] = np.sqrt(out.non_psd_rate * (1 - out.non_psd_rate) / out.runs)
    return out
