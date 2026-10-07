import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src import config


@pytest.fixture(scope="module")
def runs_and_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    f = rng.standard_normal((400, 1)) * 0.01
    R = pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 8)) + rng.standard_normal((400, 8)) * 0.01, index=idx)
    return ev.run_backtest(R, windows=[60, 120], horizon=40, step=40), R


def test_shape_and_no_lookahead(runs_and_returns):
    runs, _ = runs_and_returns
    assert set(runs.method) == set(config.METHODS) and set(runs.constraint) == set(config.CONSTRAINTS)
    assert (runs.test_start > runs.train_end).all()
    # 2 methods x ... one row per (window, origin, method, constraint)
    assert not runs.duplicated(["window", "origin", "method", "constraint"]).any()


def test_gap_definition_and_ranges(runs_and_returns):
    runs, _ = runs_and_returns
    np.testing.assert_allclose(runs.risk_gap, runs.realized_risk - runs.predicted_risk)
    assert runs.predicted_risk.gt(0).all() and runs.realized_risk.gt(0).all()
    for k in config.BOTTOM_K:
        assert runs[f"exposure_k{k}"].between(0, 1).all()
    lo = runs[runs.constraint == "long_only"]
    assert (lo.max_abs_weight <= 1 + 1e-9).all()


def test_raw_unconstrained_predicted_risk_matches_analytic(runs_and_returns):
    # min-variance risk^2 = 1 / (1' S^{-1} 1)
    runs, R = runs_and_returns
    r = runs[(runs.method == "raw") & (runs.constraint == "unconstrained")].iloc[0]
    S = np.cov(R.iloc[r.origin - r.window:r.origin].to_numpy(), rowvar=False)
    expected = np.sqrt(1 / (np.ones(8) @ np.linalg.solve(S, np.ones(8)))) * ev.ANN
    assert r.predicted_risk == pytest.approx(expected, rel=1e-9)


def test_deterministic_and_turnover_first_nan(runs_and_returns):
    runs, R = runs_and_returns
    again = ev.run_backtest(R, windows=[60, 120], horizon=40, step=40)
    pd.testing.assert_frame_equal(runs, again)
    first = runs.groupby(["window", "method", "constraint"]).head(1)
    assert first.turnover.isna().all()


def test_spearman_table_known_relation():
    df = pd.DataFrame(dict(method="raw", constraint="unconstrained", window=60,
                           risk_gap=np.arange(10.0)))
    for k in config.BOTTOM_K:
        df[f"exposure_k{k}"] = np.arange(10.0) ** 2  # monotone -> rho = 1
    out = ev.exposure_vs_gap(df)
    row = out[(out.constraint == "unconstrained") & (out.window == "all") & (out.k == 1)].iloc[0]
    assert row.spearman_rho == pytest.approx(1.0)
