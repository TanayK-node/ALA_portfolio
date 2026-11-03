"""Leakage-safe volatility regimes and regime-conditional calibration tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import inference

TRAIN_ROWS = 750          # thresholds use only rows before the first forecast origin (= max window)
VOL_WIN = 60
COVID = ("2020-02-24", "2020-06-30")       # calendar stress window fixed a priori (market fall and rebound), used for flags only


def trailing_vol(R: pd.DataFrame, origin: int, win: int = VOL_WIN) -> float:
    """Annualised std of the equal-weight portfolio over rows [origin-win, origin) -- information available at the origin."""
    return float(R.iloc[origin - win:origin].mean(axis=1).std(ddof=1) * np.sqrt(252))


def vol_thresholds(R: pd.DataFrame, n_train: int = TRAIN_ROWS, win: int = VOL_WIN) -> tuple[float, float]:
    """Terciles of the trailing vol over origins win..n_train, using only rows < n_train."""
    v = [trailing_vol(R, o, win) for o in range(win, n_train + 1)]
    lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    return float(lo), float(hi)


def label_origins(R: pd.DataFrame, origins, H: int) -> pd.DataFrame:
    lo, hi = vol_thresholds(R)
    rows = []
    c0, c1 = pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1])
    for o in origins:
        v = trailing_vol(R, o)
        t0, t1 = R.index[o], R.index[o + H - 1]
        blk = float(R.iloc[o:o + H].mean(axis=1).sum())
        rows.append(dict(origin=o, trailing_vol=v, vol_regime="low" if v <= lo else ("high" if v > hi else "mid"),
                         covid_overlap=bool(t0 <= c1 and t1 >= c0), expost_block_return=blk, expost_large_fall=bool(blk <= -0.05)))
    return pd.DataFrame(rows)


def regime_table(fc: pd.DataFrame, labels: pd.DataFrame, by: str, min_n: int = 12, block: int | None = None) -> pd.DataFrame:
    """Calibration and paired loss difference (corrected - NL) by regime, one row per (window, regime value)."""
    d = fc.merge(labels, on="origin")
    out = []
    for (W, key), g in d.groupby(["window", by]):
        pv = g.pivot(index="origin", columns="forecast", values="qlike")
        vv = g.pivot(index="origin", columns="forecast", values="v")
        hh = g.pivot(index="origin", columns="forecast", values="h")
        n = len(pv)
        row = dict(window=W, regime=str(key), n=n, vr_corr=float(vv["raw_corrected"].mean() / hh["raw_corrected"].mean()),
                   median_ratio_corr=float(np.median(vv["raw_corrected"] / hh["raw_corrected"])),
                   qlike_raw=float(pv["raw"].mean()), qlike_corr=float(pv["raw_corrected"].mean()),
                   qlike_nl=float(pv["nl"].mean()), qlike_lw=float(pv["ledoit_wolf"].mean()), adequate=n >= min_n)
        if n >= min_n:
            r = inference.paired_loss_test(pv["raw_corrected"].to_numpy(), pv["nl"].to_numpy(), block=block)
            row.update(delta_vs_nl=r.mean_diff, delta_lo=r.lo, delta_hi=r.hi, win=r.win_rate)
        out.append(row)
    return pd.DataFrame(out)
