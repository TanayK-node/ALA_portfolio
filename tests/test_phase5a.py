import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import config, evaluate as ev, uncertainty as un


@pytest.fixture(scope="module")
def R():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=500)
    f = rng.standard_normal((500, 1)) * 0.01
    return pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 8)) + rng.standard_normal((500, 8)) * 0.01, index=idx)


# ---- bootstrap
def test_block_indices_shape_range_and_contiguity():
    idx = bs.block_bootstrap_indices(10, 3, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10) and idx.min() >= 0 and idx.max() <= 9
    # inside each full block of 3 the indices are consecutive
    d = np.diff(idx[:, :9].reshape(50, 3, 3), axis=2)
    assert (d == 1).all()


def test_block_len_capped_and_short_series():
    idx = bs.block_bootstrap_indices(2, 5, 7, np.random.default_rng(1))
    assert idx.shape == (7, 2)


def test_mean_ci_contains_mean_and_is_deterministic():
    x = np.random.default_rng(2).standard_normal(12)
    a, b = bs.mean_ci(x, n_boot=500), bs.mean_ci(x, n_boot=500)
    assert a == b and a.lo <= a.mean <= a.hi and a.mean == pytest.approx(x.mean())


def test_constant_series_has_zero_width_and_excludes_zero():
    ci = bs.mean_ci(np.full(8, 0.5), n_boot=200)
    assert ci.lo == ci.hi == 0.5 and ci.excludes_zero


def test_paired_diff_identical_series_is_zero():
    x = np.random.default_rng(3).standard_normal(9)
    ci = bs.paired_mean_diff_ci(x, x, n_boot=200)
    assert ci.mean == 0 and ci.lo == 0 and ci.hi == 0 and not ci.excludes_zero
    with pytest.raises(ValueError):
        bs.paired_mean_diff_ci(x, x[:-1])


def test_single_value_collapses():
    ci = bs.mean_ci(np.array([1.5]))
    assert ci.mean == ci.lo == ci.hi == 1.5


# ---- common OOS
def test_common_origins():
    o = ev.common_origins(500, windows=[60, 120, 250], horizon=60, step=60)
    assert o[0] == 250 and all(b - a == 60 for a, b in zip(o, o[1:])) and o[-1] <= 440


def test_common_oos_same_periods_and_train_lengths(R):
    runs = ev.run_backtest_common_oos(R, windows=[60, 120, 250], horizon=60, step=60)
    per_w = runs.groupby("window").origin.apply(lambda s: tuple(sorted(set(s))))
    assert per_w.nunique() == 1  # identical origin sequence for every window
    assert (runs.test_start > runs.train_end).all()
    lens = ((runs.origin - runs.window))
    assert (lens >= 0).all()
    # same test dates for every window at a given origin
    td = runs.groupby("origin").test_start.nunique()
    assert (td == 1).all()


def test_common_oos_largest_window_matches_per_window_design(R):
    old = ev.run_backtest(R, windows=[60, 250], horizon=60, step=60)
    new = ev.run_backtest_common_oos(R, windows=[60, 250], horizon=60, step=60)
    a = old[old.window == 250].reset_index(drop=True)
    b = new[new.window == 250].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # largest window has identical origins in both designs


def test_default_backtest_unchanged_and_bad_origins_rejected(R):
    with pytest.raises(ValueError):
        ev.run_backtest(R, windows=[250], origins=[100])
    with pytest.raises(ValueError):
        ev.run_backtest(R, windows=[60], origins=[490])


def test_summary_with_ci_and_compare(R):
    new = ev.run_backtest_common_oos(R, windows=[60, 120], horizon=60, step=60)
    old = ev.run_backtest(R, windows=[60, 120], horizon=60, step=60)
    ci = un.summary_with_ci(new, n_boot=200)
    assert (ci.risk_gap_lo <= ci.risk_gap_mean + 1e-12).all() and (ci.risk_gap_mean <= ci.risk_gap_hi + 1e-12).all()
    assert len(ci) == 2 * len(config.METHODS) * len(config.CONSTRAINTS)
    cmp_ = un.compare_designs(old, new)
    np.testing.assert_allclose(cmp_.gap_diff, cmp_.gap_new - cmp_.gap_old)
    # window 120 is the largest here, so both designs share its origins -> identical results
    assert np.allclose(cmp_[cmp_.window == 120].gap_diff, 0)
