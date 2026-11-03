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
