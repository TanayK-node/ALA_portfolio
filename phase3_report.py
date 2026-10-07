"""Phase 3: rolling backtest on real cached data; writes results/*.csv."""
from __future__ import annotations

import logging

import pandas as pd

from src import config, data, evaluate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    returns, _, _ = data.get_returns()
    runs = evaluate.run_backtest(returns)
    t = evaluate.save_results(runs)
    print(f"{len(runs)} runs; runs per window:\n{runs.groupby('window')['origin'].nunique().to_string()}")
    s = t["summary_table"]
    cols = ["constraint", "method", "window", "n_runs", "predicted_risk_mean", "realized_risk_mean",
            "risk_gap_mean", "risk_gap_median", "max_abs_weight_mean", "turnover_mean",
            "cond_number_median", "sharpe_mean", "exposure_k3_mean"]
    print("\n== Summary (annualised risk) ==")
    print(s[cols].round(4).to_string(index=False))
    print("\n== Spearman: exposure vs risk gap (raw Sigma) ==")
    print(t["exposure_vs_gap"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
