import numpy as np
import pandas as pd
import pytest

from src import config, evaluate as ev, uncertainty as un


@pytest.fixture(scope="module")
def runs():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=600)
    f = rng.standard_normal((600, 1)) * 0.01
    R = pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 8)) + rng.standard_normal((600, 8)) * 0.01, index=idx)
    return ev.run_backtest_common_oos(R, windows=[60, 120], horizon=60, step=60)


def test_paired_diff_matches_manual_mean_and_contains_it(runs):
    p = un.paired_differences(runs, n_boot=300)
    r = p[(p.constraint == "unconstrained") & (p.window == 60) & (p.comparison == "mp - raw") & (p.metric == "risk_gap")].iloc[0]
    a = runs[(runs.constraint == "unconstrained") & (runs.window == 60) & (runs.method == "mp")].sort_values("origin").risk_gap.to_numpy()
    b = runs[(runs.constraint == "unconstrained") & (runs.window == 60) & (runs.method == "raw")].sort_values("origin").risk_gap.to_numpy()
    assert r.diff_mean == pytest.approx(np.mean(a - b)) and r.diff_lo <= r.diff_mean <= r.diff_hi
    assert len(p) == 2 * 2 * len(un.PAIRS) * 2          # constraints x windows x pairs x metrics


def test_antisymmetry_of_pairs(runs):
    ab = un.paired_differences(runs, pairs=(("mp", "raw"),), n_boot=300)
    ba = un.paired_differences(runs, pairs=(("raw", "mp"),), n_boot=300)
    np.testing.assert_allclose(ab.diff_mean.to_numpy(), -ba.diff_mean.to_numpy())
    np.testing.assert_allclose(ab.diff_lo.to_numpy(), -ba.diff_hi.to_numpy())   # same resamples -> mirrored interval


def test_summary_gets_paired_columns(runs):
    summ = un.add_paired_to_summary(ev.summarize(runs), un.paired_differences(runs, n_boot=200))
    assert {"risk_gap_diff_vs_raw", "risk_gap_diff_vs_raw_excl0", "realized_risk_diff_vs_raw_lo"} <= set(summ.columns)
    assert summ[summ.method == "raw"].risk_gap_diff_vs_raw.isna().all()          # no raw-vs-raw row
    assert summ[summ.method == "mp"].risk_gap_diff_vs_raw.notna().all()
    assert len(summ) == len(ev.summarize(runs))


def test_block_sensitivity_counts(runs):
    s = un.block_sensitivity(runs, block_lens=(1, 2), n_boot=200)
    assert list(s.block_len) == [1, 2] and (s.n_exclude_zero <= s.n_comparisons).all()
    assert (s.n_exclude_zero_vs_raw <= s.n_exclude_zero).all()


def test_coverage_check_reasonable_and_reproducible():
    a = un.coverage_check(200, 3, rho=0.0, reps=300, n_boot=200, seed=1)
    assert a == un.coverage_check(200, 3, rho=0.0, reps=300, n_boot=200, seed=1)
    assert 0.88 <= a["coverage"] <= 1.0                  # large n, iid: close to nominal
    b = un.coverage_check(6, 1, rho=0.0, reps=300, n_boot=200, seed=1)
    assert b["coverage"] < a["coverage"] + 0.02          # tiny samples cover no better than large ones
    assert 0 <= b["se"] < 0.05
