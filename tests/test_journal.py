"""Tests for the journal-extension modules (src/journal)."""
import numpy as np
import pandas as pd
import pytest

from src.journal import forecast, inference, metrics, scenarios


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(3)
    n, N = 400, 8
    A = rng.standard_normal((N, N))
    X = rng.standard_normal((n, N)) @ A.T * 0.01
    return pd.DataFrame(X, index=pd.bdate_range("2020-01-01", periods=n), columns=[f"s{i}" for i in range(N)])


def test_qlike_zero_when_perfect_and_positive_otherwise():
    assert metrics.qlike(2.0, 2.0) == pytest.approx(0.0)
    assert np.all(metrics.qlike(np.array([1.0, 3.0]), np.array([2.0, 2.0])) > 0)


def test_qlike_hand_value():
    assert metrics.qlike(2.0, 1.0) == pytest.approx(2 - np.log(2) - 1)


def test_qlike_rejects_nonpositive_or_nonfinite():
    for v, h in [(0.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf)]:
        with pytest.raises(ValueError):
            metrics.qlike(v, h)


def test_correction_factors_formulas_and_ordering():
    assert forecast.c_gauss(30, 60) == pytest.approx(59 * 58 / (30 * 29))
    assert forecast.c_qlike(30, 60) == pytest.approx(59 * 58 / (29 * 28))
    assert forecast.c_qlike(30, 60) > forecast.c_gauss(30, 60) > 1
    assert forecast.c_qlike(30, 60) / forecast.c_gauss(30, 60) == pytest.approx((60 - 30) / (60 - 30 - 2))
    with pytest.raises(ValueError):
        forecast.c_gauss(30, 31)
    with pytest.raises(ValueError):
        forecast.c_qlike(30, 32)


def test_gaussian_correction_is_unbiased_in_level_by_simulation():
    rng = np.random.default_rng(11)
    N, T, b = 6, 40, 4000
    Sigma = scenarios.make_sigma("hetero", N)
    X = scenarios.draw("gaussian", b, T, Sigma, rng)
    Xc = X - X.mean(1, keepdims=True)
    S = Xc.transpose(0, 2, 1) @ Xc / (T - 1)
    y = np.linalg.solve(S, np.ones((b, N, 1)))[..., 0]
    s = y.sum(1)
    w = y / s[:, None]
    pred = 1 / s
    true = np.einsum("bi,ij,bj->b", w, Sigma, w)
    ratio = (forecast.c_gauss(N, T) * pred).mean() / true.mean()
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_student_t_covariance_is_sigma():
    rng = np.random.default_rng(5)
    Sigma = scenarios.make_sigma("equicorr", 4)
    X = scenarios.draw("t8", 400, 400, Sigma, rng).reshape(-1, 4)
    assert np.allclose(np.cov(X, rowvar=False) / Sigma, 1.0, atol=0.08)
    with pytest.raises(ValueError):
        scenarios.draw("t4", 2, 5, Sigma, rng)


def test_forecast_uses_only_past_data(returns):
    """Changing ANY observation at or after origin+H must change nothing; changing the test block changes v but not h."""
    o, W, H = 200, 60, 20
    base = pd.DataFrame(forecast.forecast_origin(returns, o, W, H))
    after = returns.copy()
    after.iloc[o + H:] *= 7.0                       # future beyond the target block
    t = pd.DataFrame(forecast.forecast_origin(after, o, W, H))
    pd.testing.assert_frame_equal(base, t)
    block = returns.copy()
    block.iloc[o:o + H] *= 3.0                       # the target block itself
    b = pd.DataFrame(forecast.forecast_origin(block, o, W, H))
    assert np.allclose(b.h.to_numpy(), base.h.to_numpy())      # forecasts depend on training rows only
    assert not np.allclose(b.v.to_numpy(), base.v.to_numpy())  # realised proxy does depend on the block
    train_changed = returns.copy()
    train_changed.iloc[o - 5] *= 2.0
    c = pd.DataFrame(forecast.forecast_origin(train_changed, o, W, H))
    assert not np.allclose(c.h.to_numpy(), base.h.to_numpy())


def test_rolling_forecasts_shared_origins_and_kinds(returns):
    fc = forecast.rolling_forecasts(returns, windows=[60, 90], H=20)
    assert fc.groupby("window").origin.apply(lambda s: tuple(sorted(set(s)))).nunique() == 1
    assert set(fc.forecast) >= {"raw", "raw_corrected", "raw_corrected_qlike", "ledoit_wolf", "mp", "clipped"}
    corr = fc[fc.forecast == "raw_corrected"].set_index(["window", "origin"]).h
    raw = fc[fc.forecast == "raw"].set_index(["window", "origin"]).h
    N = returns.shape[1]
    ratios = (corr / raw).groupby(level="window").first()
    assert ratios[60] == pytest.approx(forecast.c_gauss(N, 60))


def test_realised_variance_proxies(returns):
    w = np.full(8, 1 / 8)
    x = returns.iloc[:20].to_numpy()
    pr = x @ w
    assert metrics.realised_variance(x, w) == pytest.approx(np.var(pr, ddof=1))
    assert metrics.realised_variance(x, w, "mean_square") == pytest.approx(np.mean(pr ** 2))


def test_paired_test_reproducible_and_detects_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 120)
    b = a + 0.5 + rng.normal(0, 0.2, 120)            # B clearly worse, so a - b < 0
    r1 = inference.paired_loss_test(a, b, seed=9, B=500)
    r2 = inference.paired_loss_test(a, b, seed=9, B=500)
    assert r1 == r2 and r1.hi < 0 and r1.win_rate > 0.9 and r1.robust_win
    r3 = inference.paired_loss_test(a, b, seed=10, B=500)
    assert r3.lo != r1.lo                              # a different seed changes the stochastic output
    null = inference.paired_loss_test(a, a + rng.normal(0, 0.2, 120), seed=3, B=500)
    assert null.lo < 0 < null.hi


def test_bootstrap_indices_in_range_and_stationary_runs():
    rng = np.random.default_rng(0)
    idx = inference.moving_block_indices(30, 4, 50, rng)
    assert idx.shape == (50, 30) and idx.min() >= 0 and idx.max() < 30
    s = inference.stationary_indices(30, 5, 50, rng)
    assert s.shape == (50, 30) and s.min() >= 0 and s.max() < 30
    assert inference.block_length_rule(8) == 2 and inference.block_length_rule(151) == 5


def test_extended_forecasts_add_nl_bps_equal_weight(returns):
    fc = forecast.rolling_forecasts(returns, windows=[60], H=20, extended=True)
    assert {"nl", "bps", "equal_weight"} <= set(fc.forecast)
    ok = fc[~fc.failed]
    assert (ok.h > 0).all() and (ok.v > 0).all()
    ew = ok[ok.forecast == "equal_weight"].iloc[0]
    S = returns.iloc[ew.origin - 60:ew.origin].cov().to_numpy()
    assert ew.h == pytest.approx(np.full(8, 1 / 8) @ S @ np.full(8, 1 / 8))


def test_skewed_draws_have_exact_covariance_and_positive_skew_and_neg_flips_sign():
    import numpy as np
    from src.journal import scenarios
    rng = np.random.default_rng(5)
    Sig = scenarios.make_sigma("factor", 6)
    X = scenarios.draw("skew4", 1, 400_000, Sig, rng)[0]
    assert np.allclose(np.cov(X.T), Sig, rtol=0.02, atol=1e-6 * 1 + 2e-3 * np.abs(Sig).max())
    sk = ((X - X.mean(0)) ** 3).mean(0) / X.var(0) ** 1.5
    assert (sk > 0.3).all() and (sk < 2.5).all()
    Xn = scenarios.draw("skewneg4", 1, 400_000, Sig, rng)[0]
    skn = ((Xn - Xn.mean(0)) ** 3).mean(0) / Xn.var(0) ** 1.5
    assert (skn < -0.3).all()
    Xt = scenarios.draw("skewt8", 1, 400_000, Sig, rng)[0]
    assert np.allclose(np.cov(Xt.T), Sig, rtol=0.05, atol=5e-3 * np.abs(Sig).max())
