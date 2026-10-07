import numpy as np
import pandas as pd
import pytest

from src import covariance as cv, optimize as op, simulate as sm, theory as th


@pytest.fixture(scope="module")
def Sigma():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 5))
    return A.T @ A / 200 + 0.2 * np.eye(5)


def test_gmv_stats_matches_unbatched_reference(Sigma):
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.standard_normal((4, 30, 5)), rng.standard_normal((4, 20, 5))
    pred, true, real = sm.gmv_stats(Xtr, Xte, Sigma)
    for i in range(4):
        S = cv.sample_cov(Xtr[i])
        w = op.min_variance_closed_form(S)
        assert pred[i] == pytest.approx(w @ S @ w)
        assert true[i] == pytest.approx(w @ Sigma @ w)
        assert real[i] == pytest.approx(np.var(Xte[i] @ w, ddof=1))


def test_true_variance_at_least_V(Sigma):
    sim = sm.simulate_gmv(Sigma, W=30, scenario="gaussian", n_sims=400, H=20, seed=3)
    assert (sim.true_var >= sim.V - 1e-15).all()  # nothing beats the true GMV variance


def test_simulation_is_reproducible_and_scenarios_differ(Sigma):
    a = sm.simulate_gmv(Sigma, 40, "gaussian", 200, 20, seed=5)
    b = sm.simulate_gmv(Sigma, 40, "gaussian", 200, 20, seed=5)
    pd.testing.assert_frame_equal(a, b)
    t = sm.simulate_gmv(Sigma, 40, "student_t", 200, 20, seed=5)
    assert not np.allclose(a.pred_var, t.pred_var)
    with pytest.raises(ValueError):
        sm.simulate_gmv(Sigma, 40, "nope", 10, 20)
    with pytest.raises(ValueError):
        sm.simulate_gmv(Sigma, 40, "bootstrap", 10, 20)  # needs a pool


def test_student_t_has_target_covariance(Sigma):
    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(Sigma)
    X = sm._draw("student_t", 4000, 50, L, rng, None).reshape(-1, 5)
    # df=5 -> heavy tails but finite covariance; tolerance is loose
    assert np.allclose(np.cov(X, rowvar=False), Sigma, rtol=0.15, atol=0.03)


def test_bootstrap_scenario_runs(Sigma):
    pool = np.random.default_rng(2).standard_normal((300, 5))
    sim = sm.simulate_gmv(Sigma, 40, "bootstrap", 100, 20, seed=1, pool=pool)
    assert len(sim) == 100 and (sim.pred_var > 0).all()


def test_wishart_theory_holds_in_small_gaussian_simulation(Sigma):
    """Monte Carlo check of tier 1 and tier 2 with the pre-stated 3-SE tolerance (small N, moderate T)."""
    N, W = 5, 25
    sim = sm.simulate_gmv(Sigma, W, "gaussian", 20000, 25, seed=11)
    row = th.theory_table(N, windows=[W], H=25).iloc[0]
    checks = pd.DataFrame(sm.gaussian_checks(sim, N, W, row))
    assert checks.ok.all(), checks
