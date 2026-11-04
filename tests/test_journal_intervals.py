import numpy as np

from src.journal import intervals
from src.journal.scenarios import make_sigma, draw
from src.journal.forecast import c_gauss


def test_interval_coverage_is_nominal_under_gaussian_returns():
    N, T, H, reps = 10, 80, 20, 3000
    rng = np.random.default_rng(1)
    Sig = make_sigma("equicorr", N)
    q = intervals.ratio_quantiles(N, T, H, [0.05, 0.95], reps=200_000, seed=2)
    hit = 0
    for X in draw("gaussian", reps, T + H, Sig, rng):
        S = np.cov(X[:T].T)
        w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
        h = c_gauss(N, T) * float(w @ S @ w)
        v = np.var(X[T:] @ w, ddof=1)
        hit += (h * q[0] <= v <= h * q[1])
    assert abs(hit / reps - 0.90) < 0.03


def test_interval_is_invariant_to_the_centring_multiplier():
    """h*q is the same in variance units whether the pivot uses c or c_QLIKE (kappa only re-centres v/h)."""
    from src.journal.forecast import c_qlike
    N, T, H = 30, 60, 20
    sxx = 0.7                                                      # a plug-in variance w'S w
    pr = [0.05, 0.95]
    qc = intervals.ratio_quantiles(N, T, H, pr, reps=100_000, seed=3, multiplier="c")
    qq = intervals.ratio_quantiles(N, T, H, pr, reps=100_000, seed=3, multiplier="cqlike")
    assert np.allclose(c_gauss(N, T) * sxx * qc, c_qlike(N, T) * sxx * qq, rtol=1e-12)
    assert np.all(qq < qc)                                         # the QLIKE-centred ratio is smaller by c/c_qlike


def test_interval_nominal_coverage_with_cqlike_centre():
    from src.journal.forecast import c_qlike
    N, T, H, reps = 10, 80, 20, 3000
    rng = np.random.default_rng(5)
    Sig = make_sigma("equicorr", N)
    q = intervals.ratio_quantiles(N, T, H, [0.05, 0.95], reps=200_000, seed=2, multiplier="cqlike")
    hit = 0
    for X in draw("gaussian", reps, T + H, Sig, rng):
        S = np.cov(X[:T].T)
        w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
        h = c_qlike(N, T) * float(w @ S @ w)
        v = np.var(X[T:] @ w, ddof=1)
        hit += (h * q[0] <= v <= h * q[1])
    assert abs(hit / reps - 0.90) < 0.03
