"""Forecast-loss and calibration metrics. All variances are in the same (daily, un-annualised) units."""
from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(v, h) -> np.ndarray:
    """QLIKE L = v/h - log(v/h) - 1 for a realised proxy v > 0 and a forecast h > 0 (elementwise).

    Raises on non-positive or non-finite inputs rather than adding an epsilon silently.
    """
    v, h = np.asarray(v, dtype=float), np.asarray(h, dtype=float)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(h))):
        raise ValueError("QLIKE needs finite inputs")
    if np.any(v <= 0) or np.any(h <= 0):
        raise ValueError("QLIKE needs strictly positive realised proxy and forecast")
    r = v / h
    return r - np.log(r) - 1.0


def realised_variance(test_returns: np.ndarray, w: np.ndarray, proxy: str = "sample") -> float:
    """Realised variance of a fixed-weight portfolio over a test block (rows = days).

    ``sample``: sample variance (ddof=1) of the daily portfolio returns (primary; matches ``evaluate.oos_stats``);
    ``mean_square``: uncentred mean of squared daily portfolio returns (robustness proxy).
    The two are different estimators of the same quantity; they are not interchangeable.
    """
    pr = np.asarray(test_returns, dtype=float) @ np.asarray(w, dtype=float)
    if proxy == "sample":
        return float(np.var(pr, ddof=1))
    if proxy == "mean_square":
        return float(np.mean(pr ** 2))
    raise ValueError(f"unknown proxy {proxy!r}")


def calibration_summary(v: np.ndarray, h: np.ndarray) -> dict[str, float]:
    """Calibration of forecasts h against realised proxies v over forecast origins.

    ratio_of_means = mean(v)/mean(h) (the 'variance ratio' of the conference paper);
    mean_ratio = mean(v/h); median_ratio = median(v/h); bias = mean(log(v/h)) (log-calibration error);
    mace = mean |log(v/h)|; qlike = mean QLIKE. Under perfect calibration ratio_of_means ~ 1, median_ratio ~ 1, bias ~ 0.
    """
    v, h = np.asarray(v, float), np.asarray(h, float)
    lr = np.log(v / h)
    return dict(n=int(len(v)), ratio_of_means=float(v.mean() / h.mean()), mean_ratio=float((v / h).mean()),
                median_ratio=float(np.median(v / h)), log_bias=float(lr.mean()), mace=float(np.abs(lr).mean()),
                qlike=float(qlike(v, h).mean()), ratio_sd=float((v / h).std(ddof=1)) if len(v) > 1 else np.nan)


def calibration_slope(v: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """OLS intercept and slope of log v on log h (Mincer-Zarnowitz in logs); needs >= 3 points, else (nan, nan)."""
    x, y = np.log(np.asarray(h, float)), np.log(np.asarray(v, float))
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)
