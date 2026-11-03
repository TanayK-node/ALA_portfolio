"""Gaussian finite-sample correction factors and a leakage-safe rolling-origin forecast engine.

c(N,T)        = (T-1)(T-2) / [(T-N)(T-N-1)]  = E[R]/E[Q]   (unbiased in LEVEL on average; Prop. 1)
c_qlike(N,T)  = (T-1)(T-2) / [(T-N-1)(T-N-2)] = E[R/Q]     (QLIKE-optimal constant multiplier under the
                                                             Gaussian model, because R and Q are independent)
Both apply to the raw unbiased sample covariance with unconstrained GMV weights ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..covariance import sample_cov
from ..evaluate import common_origins, estimate_covariances
from ..optimize import min_variance_closed_form
from . import estimators, metrics


def c_gauss(N: int, T: int) -> float:
    if T - N - 1 <= 0:
        raise ValueError("need T >= N + 2")
    return (T - 1) * (T - 2) / ((T - N) * (T - N - 1))


def c_qlike(N: int, T: int) -> float:
    if T - N - 2 <= 0:
        raise ValueError("need T >= N + 3")
    return (T - 1) * (T - 2) / ((T - N - 1) * (T - N - 2))


def forecast_origin(returns: pd.DataFrame, origin: int, W: int, H: int, proxy: str = "sample",
                    extended: bool = False) -> list[dict]:
    """Forecasts made at ``origin`` using ONLY rows [origin-W, origin); target uses rows [origin, origin+H).

    Returns one dict per (estimator, forecast) with the daily-variance forecast h and the realised proxy v.
    Forecast kinds: 'plugin' for every estimator; 'corrected' (c) and 'corrected_qlike' (c_qlike) for raw only.
    """
    train, test = returns.iloc[origin - W:origin], returns.iloc[origin:origin + H]
    N = train.shape[1]
    covs = estimate_covariances(train)
    if extended:
        covs["nl"] = (estimators.nl_cov(train.to_numpy()), np.nan)
    rows = []
    for method, (S_hat, _) in covs.items():
        try:
            w = min_variance_closed_form(S_hat)
        except np.linalg.LinAlgError:
            rows.append(dict(origin=origin, window=W, horizon=H, method=method, kind="plugin", failed=True))
            continue
        h = float(w @ S_hat @ w)
        v = metrics.realised_variance(test.to_numpy(), w, proxy)
        base = dict(origin=origin, window=W, horizon=H, method=method, v=v, failed=False,
                    test_start=test.index[0], test_end=test.index[-1], train_end=train.index[-1])
        rows.append(dict(base, kind="plugin", h=h))
        if method == "raw":
            rows.append(dict(base, kind="corrected", h=c_gauss(N, W) * h))
            rows.append(dict(base, kind="corrected_qlike", h=c_qlike(N, W) * h))
            if extended:        # weight-based comparators: forecast is w' S_raw w (they have no covariance of their own)
                S_raw = S_hat
                for name, wv in (("bps", estimators.bps_weights(train.to_numpy())), ("bps_verbatim", estimators.bps_weights_verbatim(train.to_numpy())), ("equal_weight", np.full(N, 1.0 / N))):
                    rows.append(dict(base, method=name, kind="plugin", h=float(wv @ S_raw @ wv),
                                     v=metrics.realised_variance(test.to_numpy(), wv, proxy)))
    return rows


def rolling_forecasts(returns: pd.DataFrame, windows=config.WINDOWS, H: int = 20, stride: int | None = None,
                      proxy: str = "sample", extended: bool = False) -> pd.DataFrame:
    """Rolling-origin forecasts on a calendar common to all windows (origins start at max(windows), step = stride = H)."""
    stride = H if stride is None else stride
    origins = common_origins(len(returns), windows, H, stride)
    out = []
    for W in windows:
        for o in origins:
            out += forecast_origin(returns, o, W, H, proxy, extended)
    df = pd.DataFrame(out)
    df["forecast"] = np.where(df.kind == "plugin", df.method, df.method + "_" + df.kind)
    return df


def loss_table(fc: pd.DataFrame) -> pd.DataFrame:
    """Add per-origin QLIKE (drops failed rows, which are counted by the caller)."""
    ok = fc[~fc.failed].copy()
    ok["qlike"] = metrics.qlike(ok.v.to_numpy(), ok.h.to_numpy())
    return ok
