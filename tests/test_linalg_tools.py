"""Tests for src/linalg_tools.py."""
import numpy as np
import pytest

from src import linalg_tools as lt


def rand_orth(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q


def mat_with_spectrum(vals, seed=0):
    Q = rand_orth(len(vals), np.random.default_rng(seed))
    return Q @ np.diag(vals) @ Q.T


@pytest.fixture
def spd():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((40, 6))
    return A.T @ A / 40 + 0.5 * np.eye(6)


@pytest.mark.parametrize("vals,expected", [
    ([3, 2, 1], (3, 0, 0)),
    ([3, 2, 0], (2, 1, 0)),
    ([3, -1, -2], (1, 0, 2)),
    ([1, 0, 0, -1, -5], (1, 2, 2)),
    ([-1, -2], (0, 0, 2)),
])
def test_inertia_known_signature(vals, expected):
    assert lt.inertia(mat_with_spectrum(np.array(vals, float))) == expected


def test_inertia_scale_invariant():
    S = mat_with_spectrum(np.array([2.0, 1.0, 0.0, -1.0]))
    assert lt.inertia(1e-8 * S) == lt.inertia(S) == (2, 1, 1)


def test_definiteness_checks():
    assert lt.is_positive_definite(mat_with_spectrum(np.array([1.0, 2.0])))
    assert not lt.is_positive_definite(mat_with_spectrum(np.array([1.0, -2.0])))
    assert lt.is_psd(mat_with_spectrum(np.array([1.0, 0.0])))
    assert not lt.is_psd(mat_with_spectrum(np.array([1.0, -0.1])))


def test_condition_number():
    assert lt.condition_number(mat_with_spectrum(np.array([1.0, 4.0, 100.0]))) == pytest.approx(100.0)
    assert lt.condition_number(mat_with_spectrum(np.array([0.0, 1.0]))) == float("inf")
    assert lt.condition_number(mat_with_spectrum(np.array([-1.0, 1.0]))) == float("inf")


def test_eig_decompose_orthonormal_ascending(spd):
    vals, Q = lt.eig_decompose(spd)
    assert np.all(np.diff(vals) >= 0)
    np.testing.assert_allclose(Q.T @ Q, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(Q @ np.diag(vals) @ Q.T, spd, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_normal_form_sums_to_quadratic_form(seed):
    rng = np.random.default_rng(seed)
    S = mat_with_spectrum(rng.standard_normal(8), seed)  # indefinite is fine too
    w = rng.standard_normal(8)
    y, c = lt.normal_form_transform(S, w)
    assert c.sum() == pytest.approx(w @ S @ w, rel=1e-10, abs=1e-12)
    assert np.sum(y**2) == pytest.approx(w @ w)


def test_spectral_min_variance_matches_solve(spd):
    ones = np.ones(6)
    direct = np.linalg.solve(spd, ones)
    direct /= direct.sum()
    np.testing.assert_allclose(lt.min_variance_weights_spectral(spd), direct, atol=1e-10)


def test_spectral_min_variance_singular_uses_pinv():
    # Singular S whose null space is orthogonal to 1: pinv solution still sums to 1.
    S = np.diag([1.0, 2.0, 0.0])
    w = lt.min_variance_weights_spectral(S)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(w))


def test_clip_output_psd_symmetric():
    S = mat_with_spectrum(np.array([-2.0, -0.1, 0.0, 1e-9, 0.5, 3.0]))
    eps = 1e-3
    out, n = lt.clip_eigenvalues(S, eps)
    assert n == 4
    np.testing.assert_allclose(out, out.T, atol=0)
    assert np.linalg.eigvalsh(out).min() >= eps - 1e-12
    assert lt.is_positive_definite(out)


def test_clip_noop_when_all_above_eps(spd):
    out, n = lt.clip_eigenvalues(spd, 1e-6)
    assert n == 0
    np.testing.assert_allclose(out, spd, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_sylvester_law(seed):
    rng = np.random.default_rng(seed)
    S = mat_with_spectrum(np.array([4, 2, 1, 0, 0, -1, -3.0]), seed)
    P = lt.random_invertible(7, rng)
    holds, i0, i1 = lt.verify_sylvester(S, P)
    assert holds and i0 == i1 == (3, 2, 2)


def test_sylvester_fails_for_singular_P():
    # Non-invertible P is outside the theorem: inertia can change.
    S = np.diag([1.0, 1.0])
    P = np.array([[1.0, 0.0], [0.0, 0.0]])
    holds, _, _ = lt.verify_sylvester(S, P)
    assert not holds


def test_noise_exposure_bounds_and_extremes(spd):
    vals, Q = lt.eig_decompose(spd)
    rng = np.random.default_rng(3)
    for k in (1, 3, 5):
        assert 0.0 <= lt.noise_exposure_score(spd, rng.standard_normal(6), k) <= 1.0
    # w in span of bottom-2 eigenvectors -> score(k=2) == 1
    w = 0.7 * Q[:, 0] - 1.3 * Q[:, 1]
    assert lt.noise_exposure_score(spd, w, 2) == pytest.approx(1.0)
    assert lt.noise_exposure_score(spd, w, 1) < 1.0
    # w = top eigenvector -> score(k=5) == 0
    assert lt.noise_exposure_score(spd, Q[:, -1], 5) == pytest.approx(0.0, abs=1e-12)
    assert lt.noise_exposure_score(spd, Q[:, 0], 6) == pytest.approx(1.0)


def test_mp_edge_formula():
    assert lt.mp_noise_edge(100, 400, 2.0) == pytest.approx(2.0 * 2.25)
    assert lt.mp_noise_edge(100, 100) == pytest.approx(4.0)


def test_clip_by_mp_preserves_trace_and_flattens_noise():
    rng = np.random.default_rng(0)
    N, T = 50, 400
    X = rng.standard_normal((T, N))
    X[:, 0] += 3 * rng.standard_normal(T)  # one strong common factor-like direction
    S = np.cov(X, rowvar=False)
    out, n_noise, edge = lt.clip_by_mp(S, N, T, sigma2=1.0)
    assert 0 < n_noise < N
    assert np.trace(out) == pytest.approx(np.trace(S))
    vals = np.linalg.eigvalsh(out)
    assert np.sum(vals <= edge) == n_noise
    assert lt.is_positive_definite(out)
