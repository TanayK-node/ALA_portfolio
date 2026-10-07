"""Phase 5 orchestration: robustness and theory extensions.

Each ``run_5x`` function computes its tables, writes them to ``config.PHASE5_DIR``
and returns what the findings report needs. Existing outputs in ``results/`` are
never touched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, evaluate, plots, uncertainty


def _out(out_dir: Path | None) -> Path:
    d = Path(out_dir) if out_dir else config.PHASE5_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_5a(returns: pd.DataFrame, old_runs: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """5A: common-OOS backtest, summary with bootstrap CIs, old-vs-common comparison, figure."""
    out = _out(out_dir)
    runs = evaluate.run_backtest_common_oos(returns)
    runs.to_csv(out / "backtest_runs_common_oos.csv", index=False)
    keys = ["constraint", "method", "window"]
    ci = uncertainty.summary_with_ci(runs)
    ci_cols = keys + ["n_periods"] + [c for c in ci.columns if c.endswith(("_lo", "_hi"))]
    summary = evaluate.summarize(runs).merge(ci[ci_cols], on=keys)
    summary.to_csv(out / "summary_common_oos.csv", index=False)
    cmp_ = uncertainty.compare_designs(old_runs, runs)
    cmp_.to_csv(out / "oos_design_comparison.csv", index=False)
    fig = plots.fig_common_oos_realized(ci, out)
    return dict(runs=runs, summary=summary, ci=ci, comparison=cmp_, figure=fig,
                origins=evaluate.common_origins(len(returns)))


def print_design_comparison(cmp_: pd.DataFrame) -> None:
    """Side-by-side old (per-window OOS) vs common-OOS mean risk gap, in percentage points."""
    for cons in config.CONSTRAINTS:
        c = cmp_[cmp_.constraint == cons]
        cols = {}
        for m in config.METHODS:
            d = c[c.method == m].set_index("window")
            cols[f"{m} old"] = d.gap_old * 100
            cols[f"{m} new"] = d.gap_new * 100
        t = pd.DataFrame(cols)
        t.insert(0, "n_old", c[c.method == "raw"].set_index("window").n_old)
        t.insert(1, "n_new", c[c.method == "raw"].set_index("window").n_new)
        print(f"\nMean risk gap (pts), {cons}: per-window OOS (old) vs common OOS (new)")
        print(t.round(2).to_string())
