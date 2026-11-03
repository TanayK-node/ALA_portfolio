"""Prequential second-stage calibration of the Gaussian-corrected GMV risk forecast. See configs/journal_second_stage.yaml.

At origin k only (i) features built from rows before the origin and (ii) labels y_j = log(v_j / h_G,j) of origins j < k are used.
With stride = H the block of origin k-1 ends at the row before origin k, so its label has matured exactly when it is first used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import inference, metrics

WARM = 10
CLIP = (0.25, 4.0)
LAMBDA = 5.0
FEATURES = ["log_rv20", "pc1_share", "avg_corr", "kurt", "lag_err"]
MODELS = ["gaussian", "constant", "tercile", "ridge"]


def origin_features(R: pd.DataFrame, origin: int, W: int) -> dict:
    """Features from the training window rows [origin-W, origin) only (no label information)."""
    tr = R.iloc[origin - W:origin].to_numpy()
    ew = tr.mean(axis=1)
    C = np.corrcoef(tr, rowvar=False)
    N = C.shape[0]
    ev = np.linalg.eigvalsh(np.cov(tr, rowvar=False))
    return dict(log_rv20=float(np.log(ew[-20:].var(ddof=1) / ew.var(ddof=1))), pc1_share=float(ev[-1] / ev.sum()),
                avg_corr=float((C.sum() - N) / (N * (N - 1))), kurt=float(stats.kurtosis(ew, fisher=True)))


def _clip(g: float) -> float:
    return float(min(max(g, CLIP[0]), CLIP[1]))


def ridge_multiplier(X_past: np.ndarray, y_past: np.ndarray, x_new: np.ndarray, lam: float = LAMBDA) -> float:
    """exp(yhat) * mean(exp(residuals)) from a ridge fit of y on standardised X (intercept unpenalised)."""
    mu, sd = X_past.mean(0), X_past.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X_past - mu) / sd
    ybar = y_past.mean()
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y_past - ybar))
    resid = y_past - ybar - Z @ beta
    yhat = ybar + ((x_new - mu) / sd) @ beta
    return float(np.exp(yhat) * np.mean(np.exp(resid)))


def run_cell(v: np.ndarray, h: np.ndarray, feat: pd.DataFrame, regime: np.ndarray, lam: float = LAMBDA) -> pd.DataFrame:
    """Chronological multipliers for one (sample, window, horizon) cell. Returns one row per origin with g per model."""
    n = len(v)
    y = np.log(v / h)
    Xf = feat[FEATURES[:-1]].to_numpy()
    lag = np.r_[np.nan, y[:-1]]                       # lag_err_j = label of origin j-1 (matured when origin j is formed)
    X = np.column_stack([Xf, lag])
    rows = []
    for k in range(n):
        g = dict(gaussian=1.0, constant=1.0, tercile=1.0, ridge=1.0)
        if k >= WARM:
            past = np.arange(k)
            r = np.exp(y[past])
            g["constant"] = _clip(r.mean())
            same = past[regime[past] == regime[k]]
            g["tercile"] = _clip(np.exp(y[same]).mean()) if len(same) >= 3 else g["constant"]
            idx = past[1:]                                    # rows with a defined lag feature
            g["ridge"] = _clip(ridge_multiplier(X[idx], y[idx], X[k], lam))
        rows.append(dict(k=k, evaluated=k >= WARM, **g))
    return pd.DataFrame(rows)


def evaluate_cell(v, h, g: pd.DataFrame, high: np.ndarray, models=MODELS) -> list[dict]:
    """Per-model statistics on the evaluated origins (k >= WARM), paired against the Gaussian correction and the constant."""
    ev = g.evaluated.to_numpy()
    vv, hh, hi = v[ev], h[ev], high[ev]
    loss = {m: metrics.qlike(vv, g.loc[ev, m].to_numpy() * hh) for m in models}
    out = []
    for m in models:
        gm = g.loc[ev, m].to_numpy()
        ratio = vv / (gm * hh)
        row = dict(model=m, n_eval=int(ev.sum()), qlike=float(loss[m].mean()), mace=float(np.abs(np.log(ratio)).mean()),
                   vr=float(vv.mean() / (gm * hh).mean()), median_ratio=float(np.median(ratio)),
                   frac_clipped=float(np.mean((gm <= CLIP[0] + 1e-12) | (gm >= CLIP[1] - 1e-12))))
        if m != "gaussian":
            for ref in ("gaussian", "constant"):
                if ref == "constant" and m in ("constant",):
                    continue
                r = inference.paired_loss_test(loss[m], loss[ref])
                row.update({f"d_{ref}": r.mean_diff, f"d_{ref}_lo": r.lo, f"d_{ref}_hi": r.hi, f"win_{ref}": r.win_rate})
            d = loss[m] - loss["gaussian"]
            k = int(np.ceil(0.05 * len(d)))
            row["d_gaussian_trim"] = float(np.sort(d)[k:].mean())          # drop the 5% origins with the largest improvement
            row["d_gaussian_high"] = float(d[hi].mean()) if hi.sum() >= 3 else np.nan
            row["n_high"] = int(hi.sum())
        out.append(row)
    return out


def verdict(summary: pd.DataFrame, primary=("ext_A", "ext_B", "ext_AB")) -> pd.DataFrame:
    """Evaluate the pre-specified criteria C1-C7 (configs/journal_second_stage.yaml) over the primary cells."""
    s = summary[summary["sample"].isin(primary)]
    g = s[s.model == "gaussian"].set_index(["sample", "H", "W"])
    c = s[s.model == "constant"].set_index(["sample", "H", "W"])
    rows = []
    for m in ("constant", "tercile", "ridge"):
        d = s[s.model == m].set_index(["sample", "H", "W"])
        n = len(d)
        lower = d.d_gaussian < 0
        crit = {"C1_lower_qlike_share": float(lower.mean()) >= 0.75,
                "C2_ci_share": float((d.d_gaussian_hi < 0).mean()) >= 0.25,
                "C3_lower_mace_share": float((d.mace < g.mace.reindex(d.index)).mean()) >= 0.75,
                "C4_breadth": all(float(lower[grp].mean()) >= 0.5 for key in ("sample", "H") for grp in [lower.index.get_level_values(key) == v
                                                                                                         for v in lower.index.get_level_values(key).unique()]),
                "C5_not_few_periods": (float((d.d_gaussian_trim[lower] < 0).mean()) >= 0.75) if lower.any() else False,
                "C6_high_vol": float(np.nanmedian(d.d_gaussian_high)) <= 0.02,
                "C7_beats_constant": True if m == "constant" else float((d.qlike < c.qlike.reindex(d.index)).mean()) >= 0.5}
        rows.append(dict(model=m, cells=n, share_lower_qlike=float(lower.mean()), share_ci_favours=float((d.d_gaussian_hi < 0).mean()),
                         share_lower_mace=float((d.mace < g.mace.reindex(d.index)).mean()), median_d_gaussian=float(d.d_gaussian.median()),
                         median_d_high_vol=float(np.nanmedian(d.d_gaussian_high)), **crit, supported=all(crit.values())))
    return pd.DataFrame(rows)
