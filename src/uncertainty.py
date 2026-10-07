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


PAIRS = (("clipped", "raw"), ("mp", "raw"), ("ledoit_wolf", "raw"), ("ledoit_wolf", "mp"))
PAIR_METRICS = ("risk_gap", "realized_risk")


def paired_differences(runs: pd.DataFrame, pairs=PAIRS, metrics: tuple[str, ...] = PAIR_METRICS,
                       block_len: int = config.BOOT_BLOCK, n_boot: int = config.BOOT_N,
                       seed: int = config.SEED) -> pd.DataFrame:
    """Block-bootstrap CIs for paired differences of per-period metrics: mean(A - B) over aligned periods.

    A and B are estimators run on the SAME periods (the common-OOS design), so pairing removes the
    period effect they share. diff = A - B: for risk_gap a negative value means A shrinks the gap
    relative to B; for realized_risk a negative value means A realised LOWER risk than B (better).
    ``excludes_zero`` is the percentile interval not containing 0; intervals are unadjusted for the
    many comparisons made and, with few periods, are too narrow (see ``coverage_check``).
    """
    rows = []
    for metric in metrics:
        wide = _per_origin(runs, metric)
        for cons in config.CONSTRAINTS:
            for W in sorted(runs.window.unique()):
                for a, b in pairs:
                    ka, kb = (cons, a, W), (cons, b, W)
                    if ka not in wide.columns or kb not in wide.columns:
                        continue
                    d = (wide[ka] - wide[kb]).dropna().to_numpy()
                    ci = bootstrap.mean_ci(d, block_len, n_boot, seed=seed)
                    rows.append(dict(constraint=cons, window=int(W), comparison=f"{a} - {b}", metric=metric,
                                     n_periods=len(d), block_len=min(block_len, len(d)), diff_mean=ci.mean,
                                     diff_lo=ci.lo, diff_hi=ci.hi, excludes_zero=ci.excludes_zero))
    return pd.DataFrame(rows)


def add_paired_to_summary(summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Append `<metric>_diff_vs_raw`, `_lo`, `_hi`, `_excl0` columns (method - raw) to a summary table."""
    vs = paired[paired.comparison.str.endswith(" - raw")].copy()
    vs["method"] = vs.comparison.str.replace(" - raw", "", regex=False)
    parts = []
    for metric in vs.metric.unique():
        m = vs[vs.metric == metric].rename(columns={"diff_mean": f"{metric}_diff_vs_raw", "diff_lo": f"{metric}_diff_vs_raw_lo",
                                                    "diff_hi": f"{metric}_diff_vs_raw_hi", "excludes_zero": f"{metric}_diff_vs_raw_excl0"})
        parts.append(m[["constraint", "method", "window", f"{metric}_diff_vs_raw", f"{metric}_diff_vs_raw_lo",
                        f"{metric}_diff_vs_raw_hi", f"{metric}_diff_vs_raw_excl0"]])
    out = summary
    for p in parts:
        out = out.merge(p, on=["constraint", "method", "window"], how="left")
    return out


def block_sensitivity(runs: pd.DataFrame, block_lens: tuple[int, ...] = config.BOOT_BLOCK_SENSITIVITY,
                      n_boot: int = config.BOOT_N) -> pd.DataFrame:
    """How many paired differences exclude zero under each block length (L = 1 is the plain i.i.d. bootstrap)."""
    rows = []
    for L in block_lens:
        p = paired_differences(runs, block_len=L, n_boot=n_boot)
        rows.append(dict(block_len=L, n_comparisons=len(p), n_exclude_zero=int(p.excludes_zero.sum()),
                         n_exclude_zero_vs_raw=int(p[p.comparison.str.endswith(" - raw")].excludes_zero.sum())))
    return pd.DataFrame(rows)


def coverage_check(n: int, block_len: int, rho: float = 0.0, reps: int = config.BOOT_COVER_REPS,
                   n_boot: int = config.BOOT_COVER_B, seed: int = config.SEED) -> dict:
    """Simulated coverage of the nominal 95% block-bootstrap percentile CI for a mean.

    Data: AR(1) with coefficient ``rho`` (0 = i.i.d.) and true mean 0, length ``n``. Returns the fraction of
    replications whose interval contains 0 and its binomial standard error. Coverage below 0.95 means the
    intervals are too narrow, so 'excludes zero' flags are anti-conservative at that n.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        e = rng.standard_normal(n)
        x = e.copy()
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]
        ci = bootstrap.mean_ci(x, block_len, n_boot, seed=int(rng.integers(1 << 31)))
        hits += int(ci.lo <= 0 <= ci.hi)
    cov = hits / reps
    return dict(n=n, block_len=min(block_len, n), rho=rho, reps=reps, coverage=cov,
                se=float(np.sqrt(cov * (1 - cov) / reps)))
