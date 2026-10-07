"""Universe / sub-period resolution for run_all.py and the cross-universe comparison.

A *run spec* says where prices come from, which sub-period of the return history to use, and where
results go. With NO flags the legacy layout is kept (nifty30, full sample, results written to
``results/``), so existing outputs are never touched. With ``--universe`` and/or ``--period`` results go
to ``results/<universe>/`` (``results/<universe>/<period>/`` for a sub-period).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class RunSpec:
    """Resolved run configuration."""
    universe: str
    period: str
    period_frac: tuple[float, float]
    tickers: list[str]
    prices_csv: Path
    results_dir: Path
    legacy: bool


def parse_period(s: str) -> tuple[str, tuple[float, float]]:
    """'full' | 'first_half' | 'second_half' | '<a>:<b>' (fractions in [0, 1], a < b) -> (label, (a, b))."""
    if s in config.PERIOD_PRESETS:
        return s, config.PERIOD_PRESETS[s]
    m = re.fullmatch(r"\s*([0-9.]+)\s*:\s*([0-9.]+)\s*", s)
    if not m:
        raise ValueError(f"bad --period {s!r}: use one of {sorted(config.PERIOD_PRESETS)} or '<start>:<end>' fractions")
    a, b = float(m.group(1)), float(m.group(2))
    if not 0.0 <= a < b <= 1.0:
        raise ValueError("period fractions must satisfy 0 <= start < end <= 1")
    return f"{a:g}-{b:g}", (a, b)


def resolve_run(universe: str | None = None, period: str | None = None) -> RunSpec:
    """Resolve CLI flags into a RunSpec (see module docstring for the output-directory rule)."""
    legacy = universe is None and period is None
    uni = universe or config.DEFAULT_UNIVERSE
    if uni not in config.UNIVERSES:
        raise ValueError(f"unknown universe {uni!r}; configured: {sorted(config.UNIVERSES)}")
    label, frac = parse_period(period or "full")
    u = config.UNIVERSES[uni]
    if legacy:
        out = config.RESULTS_DIR
    else:
        out = config.RESULTS_DIR / uni / ("" if label == "full" else label)
    return RunSpec(uni, label, frac, list(u["tickers"]), Path(u["prices_csv"]), out, legacy)


def missing_data_message(spec: RunSpec) -> str:
    """Instructions printed (instead of downloading) when a universe's price cache is absent."""
    return (f"No cached prices for universe '{spec.universe}': expected {spec.prices_csv}.\n"
            f"The sandbox cannot download them. Create the cache on a machine with internet access:\n"
            f"    python -m src.data {spec.universe}\n"
            f"(or put a CSV with a date index and one adjusted-close column per ticker at that path;\n"
            f" tickers are listed in config.UNIVERSES['{spec.universe}']['tickers']), then re-run.")


def slice_period(returns: pd.DataFrame, frac: tuple[float, float]) -> pd.DataFrame:
    """Rows [round(a*T), round(b*T)) of the return history."""
    T = len(returns)
    return returns.iloc[int(round(frac[0] * T)):int(round(frac[1] * T))]


def feasible_windows(n_obs: int, windows: list[int] | None = None, horizon: int = config.HORIZON,
                     step: int = config.STEP, min_periods: int = config.MIN_WINDOW_PERIODS) -> list[int]:
    """Window lengths that yield at least ``min_periods`` per-window backtest origins on n_obs rows."""
    ws = list(config.WINDOWS if windows is None else windows)
    return [W for W in ws if len(range(W, n_obs - horizon + 1, step)) >= min_periods]


def set_windows(windows: list[int]) -> None:
    """Replace config.WINDOWS IN PLACE (same list object).

    Many functions bind ``config.WINDOWS`` as a default argument at import time; mutating the list in
    place makes those defaults follow the change. Intended for run_all.py (one universe per process).
    """
    if not windows:
        raise ValueError("no feasible window lengths for this sample")
    config.WINDOWS[:] = list(windows)


def _read(results_dir: Path, name: str) -> pd.DataFrame | None:
    f = Path(results_dir) / name
    return pd.read_csv(f) if f.exists() else None


def headline_metrics(results_dir: Path) -> dict[str, float | str]:
    """Flat dict of headline numbers read from a results directory (whatever files exist)."""
    d: dict[str, float | str] = {}
    full = _read(results_dir, "full_sample_diagnostics.csv")
    if full is not None:
        v = dict(zip(full.iloc[:, 0], full["value"]))
        for k in ("N", "T", "condition_number", "lambda_min", "top1_variance_share", "n_above_mp_edge"):
            if k in v:
                d[k] = float(v[k])
        d["positive_definite"] = str(v.get("positive_definite"))
    miss = _read(results_dir, "scenario_missing_data.csv")
    if miss is not None:
        d["random_mask_non_psd"] = f"{int(miss.non_psd.sum())}/{len(miss)}"
    s = _read(results_dir, "summary_table.csv")
    if s is not None:
        for cons, tag in (("unconstrained", "unc"), ("long_only", "lo")):
            for _, r in s[(s.constraint == cons) & (s.method == "raw")].iterrows():
                d[f"raw_{tag}_gap_pts_W{int(r.window)}"] = 100 * r.risk_gap_mean
        for m in ("mp", "ledoit_wolf"):
            for _, r in s[(s.constraint == "unconstrained") & (s.method == m)].iterrows():
                d[f"{m}_unc_gap_pts_W{int(r.window)}"] = 100 * r.risk_gap_mean
    e = _read(results_dir, "exposure_vs_gap.csv")
    if e is not None:
        for cons, tag in (("unconstrained", "unc"), ("long_only", "lo")):
            r = e[(e.constraint == cons) & (e.k == config.PLOT_K) & (e.window.astype(str) == "within")]
            if len(r):
                d[f"exposure_k{config.PLOT_K}_within_rho_{tag}"] = float(r.spearman_rho.iloc[0])
                d[f"exposure_k{config.PLOT_K}_within_p_{tag}"] = float(r.p_value.iloc[0])
    p5 = Path(results_dir) / "phase5"
    c = _read(p5, "summary_common_oos.csv")
    if c is not None:
        for _, r in c[(c.constraint == "unconstrained") & (c.method == "raw")].iterrows():
            d[f"common_oos_raw_unc_gap_pts_W{int(r.window)}"] = 100 * r.risk_gap_mean
    t = _read(p5, "theory_vs_empirical.csv")
    if t is not None:
        r = t.iloc[0]
        d[f"std_ratio_theory_W{int(r.window)}"], d[f"std_ratio_measured_W{int(r.window)}"] = float(r.th_std_ratio), float(r.emp_std_ratio)
    b = _read(p5, "block_missing_summary.csv")
    if b is not None:
        d["block_missing_non_psd"] = f"{int(b.non_psd.sum())}/{int(b.runs.sum())}"
    v = _read(p5, "predictor_verdicts.csv")
    if v is not None:
        d["predictors_supported"] = f"{int(v.supported.sum())}/{len(v)}"
    return d


def compare(results_dirs: dict[str, Path]) -> pd.DataFrame:
    """Metric x run table from several results directories (keys become column names)."""
    cols = {name: headline_metrics(path) for name, path in results_dirs.items()}
    df = pd.DataFrame(cols)
    return df.reindex(sorted(df.index, key=lambda k: (not k[0].isupper(), k))) if len(df) else df
