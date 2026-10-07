"""Price download, caching, cleaning and log returns."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance.

    Returns a DataFrame (index = date, columns = tickers). Imports yfinance
    lazily so cached runs work without network access.
    """
    import yfinance as yf  # lazy: only needed on a cache miss

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (network or symbol issue).")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.reindex(columns=tickers).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices


def load_or_download(path: Path = config.PRICES_CSV,
                     tickers: list[str] = config.TICKERS) -> pd.DataFrame:
    """Load prices from the CSV cache if present, else download and cache.

    The cache is never refreshed automatically; delete data/prices.csv to
    force a re-download.
    """
    if path.exists():
        log.info("Loading cached prices from %s", path)
        return pd.read_csv(path, index_col=0, parse_dates=True)
    log.info("Cache miss: downloading %d tickers", len(tickers))
    prices = download_prices(tickers, config.START_DATE.isoformat(),
                             config.END_DATE.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


def clean_prices(prices: pd.DataFrame,
                 max_missing_frac: float = config.MAX_MISSING_FRAC,
                 ffill_limit: int = config.FFILL_LIMIT
                 ) -> tuple[pd.DataFrame, list[str]]:
    """Drop sparse tickers, forward-fill small gaps only.

    Steps: (1) drop all-NaN rows (non-trading days); (2) drop tickers whose
    missing fraction exceeds ``max_missing_frac`` and report them;
    (3) forward-fill gaps of at most ``ffill_limit`` days (longer gaps stay NaN).
    Returns (cleaned prices, list of dropped tickers).
    """
    prices = prices.dropna(how="all")
    miss = prices.isna().mean()
    dropped = [str(t) for t in miss[miss > max_missing_frac].index]
    if dropped:
        log.warning("Dropping tickers with >%.0f%% missing: %s",
                    100 * max_missing_frac, dropped)
    prices = prices.drop(columns=dropped).ffill(limit=ffill_limit)
    return prices, dropped


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = log(P_t / P_{t-1}); NaNs are preserved (first row dropped)."""
    return np.log(prices).diff().iloc[1:]


def get_returns(path: Path = config.PRICES_CSV
                ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Full data pipeline.

    Returns (returns_complete, returns_with_nan, dropped_tickers):
      * returns_with_nan: log returns keeping NaNs (missing-data experiment);
      * returns_complete: rows with any NaN removed (used by the backtest).
    """
    prices, dropped = clean_prices(load_or_download(path))
    r_nan = log_returns(prices)
    # Guard against inf from zero/negative prices in bad vendor data.
    r_nan = r_nan.replace([np.inf, -np.inf], np.nan)
    return r_nan.dropna(how="any"), r_nan, dropped
