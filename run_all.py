"""Run the full pipeline end to end and write everything to results/.

    python run_all.py                                   # legacy layout: nifty30, full sample -> results/
    python run_all.py --phase5                          # also run the Phase 5 robustness/theory extensions
    python run_all.py --universe nifty_next             # -> results/nifty_next/   (needs data/prices_nifty_next.csv)
    python run_all.py --universe nifty30 --period second_half     # -> results/nifty30/second_half/
    python run_all.py --period 0.25:0.75                # custom sub-period (fractions of the history)

Steps: data (cached) -> full-sample diagnostics -> broken-Sigma scenarios -> rolling backtest + tables
-> figures -> results/key_findings.md [-> Phase 5 -> results/phase5/key_findings_phase5.md].
Window lengths with too few out-of-sample periods in the chosen sample are dropped automatically.
With no --universe/--period flag the output layout is the original one, so existing results are untouched.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time

import numpy as np
import pandas as pd

from src import config, covariance as cv, data, diagnostics, evaluate, plots, universes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=sorted(config.UNIVERSES), default=None)
    ap.add_argument("--period", default=None, help="full | first_half | second_half | <start>:<end> (fractions)")
    ap.add_argument("--phase5", action="store_true", help="also run Phase 5 (adds roughly a minute)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        spec = universes.resolve_run(a.universe, a.period)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not spec.legacy and not spec.prices_csv.exists():     # never attempt a download for non-default runs
        print(universes.missing_data_message(spec), file=sys.stderr)
        return 2

    random.seed(config.SEED)
    np.random.seed(config.SEED)  # all stochastic steps also take explicit seeds
    out = spec.results_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    returns, _, dropped = data.get_returns(spec.prices_csv, spec.tickers)
    returns = universes.slice_period(returns, spec.period_frac)
    windows = universes.feasible_windows(len(returns))
    skipped = [w for w in config.WINDOWS if w not in windows]
    try:
        universes.set_windows(windows)
    except ValueError:
        print(f"error: the chosen sample ({len(returns)} days) is too short for any window length "
              f"{config.WINDOWS} with at least {config.MIN_WINDOW_PERIODS} out-of-sample periods.", file=sys.stderr)
        return 2
    tag = "" if spec.legacy else f" [{spec.universe}, period {spec.period}]"
    print(f"[1/5] data{tag}: {returns.shape[0]} days x {returns.shape[1]} stocks "
          f"({returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}); dropped: {dropped or 'none'}; "
          f"windows: {windows}" + (f" (skipped, too few periods: {skipped})" if skipped else ""))

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
    tables = evaluate.save_results(runs, out)
    print(f"[4/5] backtest: {len(runs)} runs")

    paths = plots.make_all(returns, runs, tables["exposure_vs_gap"], tables["summary_table"], out)
    print(f"[5/5] figures: {len(paths)} written")

    diagnostics.write_key_findings(out / "key_findings.md", full, dropped, short, missing,
                                   tables["summary_table"], tables["exposure_vs_gap"], len(runs))

    if a.phase5:
        from src import phase5

        print("Phase 5 (robustness and theory extensions):")
        phase5.run_phase5(returns, runs, out / "phase5")
    print(f"done in {time.time() - t0:.1f}s -> {out}/ (start with key_findings.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
