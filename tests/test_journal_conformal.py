import numpy as np

from src.journal import conformal


def test_conformal_uses_only_past_scores():
    rng = np.random.default_rng(0)
    rho = rng.normal(size=60)
    lo0, hi0, d0 = conformal.conformal_bounds(rho, 0.2)
    rho2 = rho.copy()
    rho2[40:] += 100.0                            # change the future of origin 39 (rho[39] itself is not used at k=39)
    lo1, hi1, d1 = conformal.conformal_bounds(rho2, 0.2)
    assert np.array_equal(lo0[:40], lo1[:40], equal_nan=True) and np.array_equal(hi0[:40], hi1[:40], equal_nan=True)
    assert not d0[:conformal.WARM].any()


def test_conformal_coverage_is_nominal_for_exchangeable_scores():
    rng = np.random.default_rng(1)
    cov = []
    for _ in range(300):
        rho = rng.standard_t(6, size=120)
        lo, hi, d = conformal.conformal_bounds(rho, 0.1)
        cov.append(((rho >= lo) & (rho <= hi))[d].mean())
    assert 0.90 <= np.mean(cov) <= 0.93      # conservative by construction (rank rounding), never below nominal


def test_group_conditioning_falls_back_to_pooled_when_group_small():
    rng = np.random.default_rng(2)
    rho = rng.normal(size=80)
    g = np.zeros(80, dtype=int)
    g[::2] = 1
    lo_g, hi_g, _ = conformal.conformal_bounds(rho, 0.2, groups=g, min_group=500)   # never enough -> pooled
    lo_p, hi_p, _ = conformal.conformal_bounds(rho, 0.2)
    assert np.array_equal(lo_g, lo_p, equal_nan=True)


def test_interval_score_penalises_misses():
    y, l, u = np.array([0.0, 5.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    s = conformal.interval_score(y, l, u, 0.1)
    assert s[0] == 2.0 and s[1] > s[0]
