"""Run the full pipeline end to end and write everything to results/.

    python run_all.py

Steps: data (cached) -> full-sample diagnostics -> broken-Sigma scenarios ->
rolling backtest + tables -> figures -> results/key_findings.md.
"""
from __future__ import annotations

import logging
import random
import time

import numpy as np
import pandas as pd

from src import config, covariance as cv, data, diagnostics, evaluate, plots


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(config.SEED)
    np.random.seed(config.SEED)  # all stochastic steps also take explicit seeds
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    returns, _, dropped = data.get_returns()
    print(f"[1/5] data: {returns.shape[0]} days x {returns.shape[1]} stocks "
          f"({returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}); dropped: {dropped or 'none'}")

    full = diagnostics.full_sample_report(returns)
    pd.Series(full).astype(str).to_csv(out / "full_sample_diagnostics.csv", header=["value"])
    print(f"[2/5] full-sample: inertia={full['inertia']}, cond={full['condition_number']:.1f}, "
          f"PD={full['positive_definite']}")

    short = diagnostics.short_window_report(returns)
    short.to_csv(out / "scenario_short_window.csv", index=False)
    missing = cv.missing_data_scenarios(returns)
    missing.to_csv(out / "scenario_missing_data.csv", index=False)
    print(f"[3/5] scenarios: {len(short)} window sizes; non-PSD in {int(missing.non_psd.sum())}/{len(missing)} masked runs")

    runs = evaluate.run_backtest(returns)
    tables = evaluate.save_results(runs)
    print(f"[4/5] backtest: {len(runs)} runs")

    paths = plots.make_all(returns, runs, tables["exposure_vs_gap"], tables["summary_table"])
    print(f"[5/5] figures: {len(paths)} written")

    diagnostics.write_key_findings(out / "key_findings.md", full, dropped, short, missing,
                                   tables["summary_table"], tables["exposure_vs_gap"], len(runs))
    print(f"done in {time.time() - t0:.1f}s -> {out}/ (start with key_findings.md)")


if __name__ == "__main__":
    main()
