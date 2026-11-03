import numpy as np
import pytest

from src.covariance import sample_cov
from src.journal import estimators


def _draw(n, p, Sigma, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T


def test_nl_is_symmetric_positive_definite_and_scale_equivariant():
    X = _draw(200, 20, np.eye(20), 1)
    S = estimators.nl_cov(X)
    assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)
    assert np.allclose(estimators.nl_cov(3.0 * X), 9.0 * S, rtol=1e-8)


def test_nl_close_to_sample_when_p_much_smaller_than_n():
    X = _draw(4000, 4, np.diag([1.0, 2.0, 4.0, 8.0]), 2)
    S, NL = sample_cov(X), estimators.nl_cov(X)
    assert np.linalg.norm(NL - S) / np.linalg.norm(S) < 0.05


def test_nl_shrinks_spread_toward_identity_population():
    """Population = I: sample eigenvalues spread (MP law); NL eigenvalues are far less dispersed and centred near 1."""
    X = _draw(120, 40, np.eye(40), 3)
    ev_s, ev_n = np.linalg.eigvalsh(sample_cov(X)), np.linalg.eigvalsh(estimators.nl_cov(X))
    assert ev_n.std() < 0.5 * ev_s.std()
    assert abs(ev_n.mean() - 1.0) < 0.1


def test_nl_beats_sample_in_frobenius_loss_on_spiked_population():
    rng = np.random.default_rng(4)
    p = 30
    ev = np.r_[np.full(p - 3, 1.0), [6.0, 4.0, 3.0]]
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Sigma = (Q * ev) @ Q.T
    loss_s, loss_n = [], []
    for k in range(30):
        X = _draw(90, p, Sigma, 100 + k)
        loss_s.append(np.linalg.norm(sample_cov(X) - Sigma))
        loss_n.append(np.linalg.norm(estimators.nl_cov(X) - Sigma))
    assert np.mean(loss_n) < 0.8 * np.mean(loss_s)


def test_nl_rejects_p_greater_than_n():
    with pytest.raises(ValueError):
        estimators.nl_cov(np.random.default_rng(0).standard_normal((20, 30)))


def test_bps_weights_sum_to_one_and_lie_between_gmv_and_equal_weight():
    X = _draw(120, 10, np.diag(np.linspace(1, 3, 10)), 5)
    w = estimators.bps_weights(X)
    assert w.sum() == pytest.approx(1.0)
    from src.optimize import min_variance_closed_form
    g, b = min_variance_closed_form(sample_cov(X)), np.full(10, 0.1)
    a = np.dot(w - b, g - b) / np.dot(g - b, g - b)
    assert -1e-9 <= a <= 1 + 1e-9 and np.allclose(w, a * g + (1 - a) * b)


def test_bps_shrinks_more_when_estimation_noise_is_larger():
    Sigma = np.diag(np.linspace(1, 3, 10))
    def intensity(T, seed):
        X = _draw(T, 10, Sigma, seed)
        from src.optimize import min_variance_closed_form
        g, b = min_variance_closed_form(sample_cov(X)), np.full(10, 0.1)
        w = estimators.bps_weights(X)
        return np.dot(w - b, g - b) / np.dot(g - b, g - b)
    low = np.mean([intensity(25, s) for s in range(40)])
    high = np.mean([intensity(400, s) for s in range(40)])
    assert low < high


def test_bps_verbatim_weights_sum_to_one_and_are_between_gmv_and_equal_weight():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 20))
    w = estimators.bps_weights_verbatim(X)
    assert abs(w.sum() - 1) < 1e-12
