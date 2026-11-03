import numpy as np

from src.journal import kurtosis, scenarios


def test_gaussian_expectation_matches_simulation_and_beta_hat_is_about_one():
    rng = np.random.default_rng(0)
    N, T = 10, 60
    X = rng.standard_normal((4000, T, N))
    b = kurtosis.mardia_ratio_batch(X)
    assert abs(b.mean() - 1.0) < 0.01


def test_beta_hat_detects_elliptical_heavy_tails_but_not_gamma_skew_in_many_assets():
    rng = np.random.default_rng(1)
    Sig = scenarios.make_sigma("equicorr", 10)
    t8 = scenarios.draw("t8", 400, 250, Sig, rng)
    g4 = scenarios.draw("skew4", 400, 250, Sig, rng)
    assert kurtosis.mardia_ratio_batch(t8).mean() > 1.25
    assert abs(kurtosis.mardia_ratio_batch(g4).mean() - 1.0) < 0.15


def test_batch_and_single_agree_and_multiplier_properties():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 8))
    assert abs(kurtosis.mardia_ratio(X) - kurtosis.mardia_ratio_batch(X[None])[0]) < 1e-10
    assert kurtosis.multiplier(0.7, 8, 80) == 1.0                    # never deflates
    assert kurtosis.multiplier(10.0, 8, 80) == kurtosis.multiplier(4.0, 8, 80)       # capped
    assert kurtosis.multiplier(2.0, 30, 60) > kurtosis.multiplier(2.0, 5, 250) > 1.0


def test_ew_kurtosis_ratio_is_about_one_under_normality_and_detects_t_tails():
    rng = np.random.default_rng(3)
    Sig = scenarios.make_sigma("equicorr", 10)
    g = scenarios.draw("gaussian", 3000, 120, Sig, rng)
    assert abs(kurtosis.ew_kurtosis_ratio_batch(g).mean() - 1.0) < 0.03
    t8 = scenarios.draw("t8", 3000, 250, Sig, rng)
    assert kurtosis.ew_kurtosis_ratio_batch(t8).mean() > 1.2
    assert abs(kurtosis.ew_kurtosis_ratio(g[0]) - kurtosis.ew_kurtosis_ratio_batch(g[:1])[0]) < 1e-10
