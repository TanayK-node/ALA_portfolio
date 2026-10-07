import logging

import numpy as np
import pytest

from src import optimize as op


@pytest.fixture
def spd():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((200, 6))
    return A.T @ A / 200 + 0.1 * np.eye(6)


def test_closed_form_sums_to_one_and_is_stationary(spd):
    w = op.min_variance_closed_form(spd)
    assert w.sum() == pytest.approx(1.0)
    g = spd @ w  # stationarity: S w = lambda * 1
    np.testing.assert_allclose(g, g.mean() * np.ones(6), atol=1e-10)


def test_closed_form_beats_equal_weight(spd):
    w = op.min_variance_closed_form(spd)
    ew = np.full(6, 1 / 6)
    assert w @ spd @ w <= ew @ spd @ ew + 1e-12


def test_long_only_feasible_and_not_better_than_unconstrained(spd):
    w, ok = op.min_variance_long_only(spd)
    assert ok and w.min() >= 0 and w.sum() == pytest.approx(1.0)
    wu = op.min_variance_closed_form(spd)
    assert w @ spd @ w >= wu @ spd @ wu - 1e-10


def test_long_only_matches_closed_form_when_closed_form_is_positive():
    S = np.diag([1.0, 2.0, 4.0])  # S^{-1}1 > 0 elementwise
    w, _ = op.min_variance_long_only(S)
    np.testing.assert_allclose(w, op.min_variance_closed_form(S), atol=1e-5)


def test_long_only_handles_indefinite_without_raising(caplog):
    S = np.diag([1.0, -1.0, 2.0])
    with caplog.at_level(logging.WARNING):
        w, _ = op.min_variance_long_only(S)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0) and w.min() >= 0


def test_frontier_monotone_return_and_feasible(spd):
    mu = np.linspace(0.0002, 0.001, 6)
    risk, ret, W = op.efficient_frontier(mu, spd, n_points=10)
    assert len(risk) > 3
    assert np.all(np.diff(ret) > -1e-12)
    assert np.all(W >= 0) and np.allclose(W.sum(axis=1), 1.0)
    # Variance is non-decreasing along the frontier from the min-variance point.
    assert np.all(np.diff(risk) > -1e-8)
