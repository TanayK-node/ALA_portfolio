import numpy as np
import pandas as pd
import pytest

from src.journal import second_stage as ss


def _cell(n=60, seed=0):
    rng = np.random.default_rng(seed)
    h = np.exp(rng.normal(-9, 0.3, n))
    v = h * np.exp(rng.normal(0.5, 0.2, n))                    # forecasts biased low by about exp(0.5)
    feat = pd.DataFrame(rng.standard_normal((n, 4)), columns=ss.FEATURES[:-1])
    regime = rng.choice(["low", "mid", "high"], n)
    return v, h, feat, regime


def test_warmup_uses_gaussian_and_constant_is_qlike_optimal():
    v, h, feat, reg = _cell()
    g = ss.run_cell(v, h, feat, reg)
    assert (g.loc[:ss.WARM - 1, ["constant", "tercile", "ridge"]] == 1.0).all().all()
    k = 30
    assert g.constant[k] == pytest.approx(np.mean(v[:k] / h[:k]), rel=1e-9)
    assert 1.4 < g.constant[40] < 2.0                              # recovers the bias exp(0.5)+


def test_multipliers_use_only_past_labels():
    v, h, feat, reg = _cell()
    g0 = ss.run_cell(v, h, feat, reg)
    k = 35
    v2, h2 = v.copy(), h.copy()
    v2[k:] *= 50.0                                                  # change this origin's and all later labels
    g1 = ss.run_cell(v2, h2, feat, reg)
    for m in ("constant", "tercile", "ridge"):
        assert np.allclose(g0[m][:k + 1], g1[m][:k + 1]), m          # multiplier at origin k cannot see label k
    assert not np.allclose(g0.constant[k + 1:], g1.constant[k + 1:])


def test_ridge_recovers_a_linear_signal_and_clip_bounds_hold():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.standard_normal((n, 3))
    y = 0.3 + 0.5 * x[:, 0] + rng.normal(0, 0.05, n)
    g = ss.ridge_multiplier(x[:-1], y[:-1], x[-1], lam=1.0)
    assert g == pytest.approx(np.exp(0.3 + 0.5 * x[-1, 0]), rel=0.08)
    assert ss._clip(100.0) == ss.CLIP[1] and ss._clip(0.0) == ss.CLIP[0]


def test_origin_features_depend_only_on_training_rows():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01)
    f0 = ss.origin_features(R, 200, 100)
    R2 = R.copy()
    R2.iloc[200:] *= 9
    assert ss.origin_features(R2, 200, 100) == pytest.approx(f0, rel=1e-12)       # equal up to BLAS rounding
    R3 = R.copy()
    R3.iloc[150] *= 9
    assert ss.origin_features(R3, 200, 100) != pytest.approx(f0, rel=1e-6)


def test_evaluate_cell_reports_positive_qlike_and_pairing():
    v, h, feat, reg = _cell()
    g = ss.run_cell(v, h, feat, reg)
    rows = ss.evaluate_cell(v, h, g, reg == "high")
    by = {r["model"]: r for r in rows}
    assert by["constant"]["qlike"] < by["gaussian"]["qlike"]       # the data are biased, a constant multiplier must help
    assert by["constant"]["d_gaussian_hi"] < 0
