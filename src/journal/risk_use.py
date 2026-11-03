"""Practitioner uses of the calibrated forecasts: daily VaR backtests and volatility targeting (configs/journal_risk_use.yaml)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..evaluate import common_origins, estimate_covariances
from ..optimize import min_variance_closed_form
from . import estimators, forecast, kurtosis

ALPHAS = (0.05, 0.01)
TARGET = 0.01
CAP = 3.0
MIN_DAYS = 1000


def kupiec_lr(x: int, n: int, a: float) -> float:
    p = x / n
    if x == 0 or x == n:
        ll1 = 0.0 if x == 0 else 0.0
    else:
        ll1 = x * np.log(p) + (n - x) * np.log(1 - p)
    ll0 = x * np.log(a) + (n - x) * np.log(1 - a)
    return float(1 - stats.chi2.cdf(max(2 * (ll1 - ll0), 0.0), 1))


def christoffersen_ind(hits: np.ndarray) -> float:
    """Independence LR on the hit sequence (first-order Markov); pooled sequence, block seams ignored."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01, n10, n11 = [int(np.sum((a == i) & (b == j))) for i in (0, 1) for j in (0, 1)]
    if n01 + n11 == 0 or n01 + n00 == 0 or n10 + n11 == 0:
        return np.nan
    pi01, pi11 = n01 / (n00 + n01), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p, n0, n1):
        return (n0 * np.log(1 - p) if n0 else 0.0) + (n1 * np.log(p) if n1 else 0.0)
    lr = 2 * (ll(pi01, n00, n01) + ll(pi11, n10, n11) - ll(pi, n00 + n10, n01 + n11))
    return float(1 - stats.chi2.cdf(max(lr, 0.0), 1))


def origin_records(X: np.ndarray, o: int, W: int, H: int) -> dict[str, tuple[np.ndarray, float]]:
    """{forecast variant: (daily portfolio returns over the block, daily variance forecast)} at origin o. Only rows < o are used for estimation."""
    train, test = X[o - W:o], X[o:o + H]
    N = X.shape[1]
    df = pd.DataFrame(train)
    covs = estimate_covariances(df)
    covs["nl"] = (estimators.nl_cov(train), np.nan)
    out = {}
    for name, (S, _) in covs.items():
        w = min_variance_closed_form(S)
        h = float(w @ S @ w)
        r = test @ w
        if name == "raw":
            beta = kurtosis.ew_kurtosis_ratio(train)
            m = float(kurtosis.multiplier(beta, N, W))
            c, cq = forecast.c_gauss(N, W), forecast.c_qlike(N, W)
            out.update({"plugin_raw": (r, h), "c": (r, c * h), "c_qlike": (r, cq * h), "kappa_cq": (r, cq * m * h)})
            S_raw = S
        else:
            out[name] = (r, h)
    wb = estimators.bps_weights_verbatim(train)
    out["bps_verbatim"] = (test @ wb, float(wb @ S_raw @ wb))
    we = np.full(N, 1.0 / N)
    out["equal_weight"] = (test @ we, float(we @ S_raw @ we))
    return out


def cell_metrics(recs: list[dict[str, tuple[np.ndarray, float]]]) -> list[dict]:
    rows = []
    for v in recs[0]:
        ret = np.concatenate([r[v][0] for r in recs])
        sig = np.concatenate([np.full(len(r[v][0]), np.sqrt(r[v][1])) for r in recs])
        row = dict(variant=v, n_days=len(ret))
        for a in ALPHAS:
            hits = ret < -stats.norm.ppf(1 - a) * sig
            x, n = int(hits.sum()), len(hits)
            row[f"viol_{int(a * 100)}"] = x / n
            row[f"kupiec_p_{int(a * 100)}"] = kupiec_lr(x, n, a)
            row[f"christ_p_{int(a * 100)}"] = christoffersen_ind(hits)
        ratios, expo = [], []
        for r in recs:
            ret_b, h = r[v]
            s = min(CAP, TARGET / np.sqrt(h))
            ratios.append(np.std(s * ret_b, ddof=1) / TARGET)
            expo.append(s)
        ratios = np.array(ratios)
        row.update(vt_mean_ratio=float(ratios.mean()), vt_mean_abs_log=float(np.mean(np.abs(np.log(ratios)))),
                   vt_breach=float(np.mean(ratios > 1.25)), vt_mean_exposure=float(np.mean(expo)), n_blocks=len(ratios))
        rows.append(row)
    return rows
