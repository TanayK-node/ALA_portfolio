"""Bootstrap confidence intervals for backtest summaries (moving blocks over OOS periods)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bootstrap, config

CI_METRICS = ("risk_gap", "realized_risk", "sharpe")


def _per_origin(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide table: rows = origin (time order), columns = (constraint, method, window)."""
    return runs.pivot_table(index="origin", columns=["constraint", "method", "window"],
                            values=metric, aggfunc="first").sort_index()


def summary_with_ci(runs: pd.DataFrame, metrics: tuple[str, ...] = CI_METRICS,
                    block_len: int = config.BOOT_BLOCK, n_boot: int = config.BOOT_N,
                    seed: int = config.SEED) -> pd.DataFrame:
    """Mean of each metric per constraint x method x window with block-bootstrap 95% CI.

    The bootstrap resamples blocks of consecutive OOS periods (``block_len``); the
    same index matrix is reused for all series with the same number of periods, so
    resamples are consistent across methods. Columns: ``<metric>_mean``, ``_lo``,
    ``_hi``, plus ``n_periods``. Intervals with few periods are crude (see
    ``bootstrap``).
    """
    cache: dict[int, np.ndarray] = {}
    rows = {}
    for metric in metrics:
        wide = _per_origin(runs, metric)
        for key in wide.columns:
            x = wide[key].dropna().to_numpy()
            if len(x) >= 2 and len(x) not in cache:
                cache[len(x)] = bootstrap.block_bootstrap_indices(
                    len(x), block_len, n_boot, np.random.default_rng(seed))
            ci = bootstrap.mean_ci(x, idx=cache.get(len(x)))
            d = rows.setdefault(key, {"n_periods": len(x)})
            d.update({f"{metric}_mean": ci.mean, f"{metric}_lo": ci.lo, f"{metric}_hi": ci.hi})
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["constraint", "method", "window"])
    return out.reset_index()


def compare_designs(old_runs: pd.DataFrame, new_runs: pd.DataFrame) -> pd.DataFrame:
    """Per-window-OOS vs common-OOS mean risk gap / realised risk, by constraint x method x window."""
    keys = ["constraint", "method", "window"]
    agg = lambda r: r.groupby(keys).agg(n=("origin", "size"), gap=("risk_gap", "mean"),
                                        realized=("realized_risk", "mean"))
    old, new = agg(old_runs), agg(new_runs)
    out = old.join(new, lsuffix="_old", rsuffix="_new")
    out["gap_diff"] = out["gap_new"] - out["gap_old"]
    out["realized_diff"] = out["realized_new"] - out["realized_old"]
    return out.reset_index()
