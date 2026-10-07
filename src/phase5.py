"""Phase 5 orchestration: robustness and theory extensions.

Each ``run_5x`` function computes its tables, writes them to ``config.PHASE5_DIR``
and returns what the findings report needs. Existing outputs in ``results/`` are
never touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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


def run_5b(returns: pd.DataFrame, common_runs: pd.DataFrame, out_dir: Path | None = None,
           n_sims: int = config.MC_SIMS) -> dict:
    """5B: theory vs Monte Carlo (Gaussian / Student-t / bootstrap) vs the measured backtest.

    Empirical points use the raw, unconstrained portfolios of the common-OOS backtest, with
    moving-block bootstrap intervals over OOS periods. Writes theory_vs_empirical.csv,
    theory_mc_checks.csv and the figure.
    """
    from . import bootstrap, simulate, theory  # local import keeps module import light

    out = _out(out_dir)
    Sigma = simulate.true_sigma(returns)
    N = Sigma.shape[0]
    th = theory.theory_table(N)
    pool = returns.to_numpy()
    wide, checks = [], []
    emp = common_runs[(common_runs.method == "raw") & (common_runs.constraint == "unconstrained")]
    for i, row in th.iterrows():
        W = int(row.window)
        rec = row.to_dict()
        for k, scen in enumerate(simulate.SCENARIOS):
            sim = simulate.simulate_gmv(Sigma, W, scen, n_sims, seed=config.SEED + 100 * k + W, pool=pool)
            for name, (m, se) in simulate.summarize_sim(sim).items():
                rec[f"mc_{scen}_{name}"], rec[f"mc_{scen}_{name}_se"] = m, se
            if scen == "gaussian":
                checks += simulate.gaussian_checks(sim, N, W, row)
        e = emp[emp.window == W].sort_values("origin")
        ratio = (e.realized_risk / e.predicted_risk).to_numpy()
        ci = bootstrap.mean_ci(ratio)
        rec.update(emp_n_periods=len(e), emp_std_ratio=ci.mean, emp_std_ratio_lo=ci.lo,
                   emp_std_ratio_hi=ci.hi, emp_var_ratio_per_run=float(np.mean(ratio ** 2)))
        wide.append(rec)
    table, chk = pd.DataFrame(wide), pd.DataFrame(checks)
    # Post-hoc DIAGNOSTIC (labelled; excluded from the pass/fail tally): the recalled offset -2 form.
    alt = []
    for _, r in table.iterrows():
        v = theory.true_variance_ratio_mean_recalled(N, int(r.window))
        se = r.mc_gaussian_true_var_over_V_se
        alt.append(dict(window=int(r.window), quantity="true_var_over_V (recalled -2 offset)",
                        tier="recalled-diagnostic", theory=v, mc_mean=r.mc_gaussian_true_var_over_V,
                        mc_se=se, z=(r.mc_gaussian_true_var_over_V - v) / se,
                        ok=bool(abs((r.mc_gaussian_true_var_over_V - v) / se) <= config.MC_Z_TOL), ks_p=np.nan))
    chk = pd.concat([chk, pd.DataFrame(alt)], ignore_index=True)
    table.to_csv(out / "theory_vs_empirical.csv", index=False)
    chk.to_csv(out / "theory_mc_checks.csv", index=False)
    fig = plots.fig_theory_vs_empirical(table, out)
    return dict(table=table, checks=chk, figure=fig, sigma_cond=float(np.linalg.cond(Sigma)))


def run_5c(returns: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """5C: pre-registered predictor comparison (two designs). Writes predictor_tests.csv,
    predictor_verdicts.csv and the forest plot."""
    from . import predictors

    out = _out(out_dir)
    parts, runs = [], {}
    for d in predictors.DESIGNS:
        runs[d] = predictors.add_predictors(predictors.build_runs(returns, d), returns)
        parts.append(predictors.run_tests(runs[d], d))
    tests = pd.concat(parts, ignore_index=True)
    ver = predictors.verdicts(tests)
    tests.to_csv(out / "predictor_tests.csv", index=False)
    ver.to_csv(out / "predictor_verdicts.csv", index=False)
    fig = plots.fig_predictor_forest(tests, out)
    nruns = {d: runs[d][runs[d].constraint == "unconstrained"].groupby("window").size().to_dict() for d in runs}
    return dict(tests=tests, verdicts=ver, figure=fig, runs_per_window=nruns)
