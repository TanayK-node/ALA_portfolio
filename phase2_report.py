"""Phase 2: broken-Sigma scenarios with inertia reports (real cached data)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config, covariance as cv, data, linalg_tools as lt, optimize as op


def short_window_report(returns: pd.DataFrame) -> pd.DataFrame:
    """Scenario (a): sample cov from T rows, T from below N to above N."""
    N = returns.shape[1]
    rows = []
    for T in sorted({N // 2, N - 1, N, N + 1, 2 * N, *config.WINDOWS}):
        if T > len(returns):
            continue
        S, (p, z, n) = cv.short_window_cov(returns, T)
        try:
            w = op.min_variance_closed_form(S)
            solve_ok, max_w = True, float(np.abs(w).max())
        except np.linalg.LinAlgError:
            solve_ok, max_w = False, float("nan")
        w_s = lt.min_variance_weights_spectral(S)
        rows.append(dict(T=T, N=N, n_pos=p, n_zero=z, n_neg=n,
                         pd_cholesky=lt.is_positive_definite(S),
                         cond=lt.condition_number(S), solve_ok=solve_ok,
                         max_abs_w_solve=max_w, max_abs_w_spectral=float(np.abs(w_s).max())))
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    _, r_nan, _ = data.get_returns()
    r_full = r_nan.dropna()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    print("== Scenario (a): short windows (first T rows of the sample) ==")
    a = short_window_report(r_full)
    print(a.to_string(index=False))
    a.to_csv(config.RESULTS_DIR / "scenario_short_window.csv", index=False)

    print("\n== Scenario (b): random masking + pairwise-complete cov ==")
    b = cv.missing_data_scenarios(r_full)
    b.to_csv(config.RESULTS_DIR / "scenario_missing_data.csv", index=False)
    agg = b.groupby("mask_frac").agg(
        runs=("seed", "size"), non_psd=("non_psd", "sum"),
        frac_non_psd=("non_psd", "mean"), lambda_min_min=("lambda_min", "min"),
        lambda_min_median=("lambda_min", "median"), max_n_neg=("n_neg", "max"))
    print(agg)
    print("\nPer-run inertia (n_pos, n_zero, n_neg):")
    for frac, g in b.groupby("mask_frac"):
        print(f" mask {frac:.0%}:",
              [(int(r.n_pos), int(r.n_zero), int(r.n_neg)) for r in g.itertuples()])

    print("\n== Frontier smoke check (full sample, raw Sigma, long-only) ==")
    S = cv.sample_cov(r_full)
    risk, ret, W = op.efficient_frontier(r_full.mean().to_numpy(), S, 20)
    print(f"frontier points: {len(risk)}; risk range "
          f"[{risk.min():.5f}, {risk.max():.5f}] daily")


if __name__ == "__main__":
    main()
