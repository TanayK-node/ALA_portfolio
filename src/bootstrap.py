"""Moving-block bootstrap over the sequence of out-of-sample periods.

Backtest periods are ordered in time and can be serially dependent (volatility
regimes), so we resample *blocks* of consecutive periods rather than single ones.

Moving-block bootstrap (Kunsch 1989; Liu & Singh 1992): for a series x_1..x_n and
block length L, draw ceil(n/L) block starts uniformly from {0, ..., n-L}, concatenate
the blocks x_s..x_{s+L-1}, truncate to n, and recompute the statistic (here the
mean). Intervals are plain percentile intervals. With very few periods (the common
design has only a handful) the bootstrap distribution is coarse and the intervals
are rough; they should be read as indicative, not exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


def block_bootstrap_indices(n: int, block_len: int, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Index matrix (n_boot, n) of moving-block bootstrap resamples of range(n).

    ``block_len`` is capped at n. The same index matrix can be applied to several
    aligned series (e.g. two methods on the same periods) to make *paired*
    resamples.
    """
    L = max(1, min(block_len, n))
    n_blocks = -(-n // L)  # ceil(n / L)
    starts = rng.integers(0, n - L + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


@dataclass(frozen=True)
class CI:
    """Point estimate (sample mean) with percentile bootstrap interval."""
    mean: float
    lo: float
    hi: float

    @property
    def excludes_zero(self) -> bool:
        """True iff the interval does not contain 0."""
        return bool(self.lo > 0 or self.hi < 0)


def mean_ci(values: np.ndarray, block_len: int = config.BOOT_BLOCK,
            n_boot: int = config.BOOT_N, alpha: float = config.BOOT_ALPHA,
            seed: int = config.SEED, idx: np.ndarray | None = None) -> CI:
    """Block-bootstrap CI for the mean of ``values`` (ordered in time).

    Pass a precomputed ``idx`` (from ``block_bootstrap_indices``) to reuse the
    same resamples across series. With fewer than 2 values the interval collapses
    to the point estimate.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)] if idx is None else x
    if len(x) < 2:
        m = float(x.mean()) if len(x) else float("nan")
        return CI(m, m, m)
    if idx is None:
        idx = block_bootstrap_indices(len(x), block_len, n_boot, np.random.default_rng(seed))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(float(x.mean()), float(lo), float(hi))


def paired_mean_diff_ci(a: np.ndarray, b: np.ndarray, block_len: int = config.BOOT_BLOCK,
                        n_boot: int = config.BOOT_N, alpha: float = config.BOOT_ALPHA,
                        seed: int = config.SEED) -> CI:
    """Block-bootstrap CI for mean(a - b) on *aligned* periods (same resamples for both).

    Pairing removes the period effect shared by the two series.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired series must be aligned (same length)")
    return mean_ci(a - b, block_len, n_boot, alpha, seed)
