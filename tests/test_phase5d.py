import numpy as np
import pandas as pd
import pytest

from src import config, evaluate as ev, linalg_tools as lt, sweeps as sw
from src.covariance import sample_cov


@pytest.fixture(scope="module")
def R():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=420)
    f = rng.standard_normal((420, 1)) * 0.01
    return pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 8)) + rng.standard_normal((420, 8)) * 0.01, index=idx)


def test_cov_fn_hook_default_is_unchanged(R):
    a = ev.run_backtest(R, windows=[60, 120], horizon=60, step=60)
    b = ev.run_backtest(R, windows=[60, 120], horizon=60, step=60, cov_fn=ev.estimate_covariances)
    pd.testing.assert_frame_equal(a, b)


def test_clip_labels_and_floor_properties(R):
    train = R.iloc[:60]
    fracs = [0.01, 0.1, 0.5]
    out = sw.clip_cov_fn(fracs)(train)
    S = out["raw"][0]
    np.testing.assert_allclose(S, sample_cov(train))
    ns = [out[sw.clip_label(f)][1] for f in fracs]
    assert ns == sorted(ns)                                  # larger floor clips at least as many
    for f in fracs:
        Sc = out[sw.clip_label(f)][0]
        eps = f * np.trace(S) / S.shape[0]
        assert np.linalg.eigvalsh(Sc).min() >= eps - 1e-12   # floor respected
        assert lt.condition_number(Sc) <= np.linalg.eigvalsh(S).max() / eps * (1 + 1e-9)
    assert out[sw.clip_label(0.01)][1] == 0 or out[sw.clip_label(0.01)][1] <= out[sw.clip_label(0.5)][1]


def test_tiny_floor_equals_raw(R):
    out = sw.clip_cov_fn([1e-9])(R.iloc[:120])
    np.testing.assert_allclose(out[sw.clip_label(1e-9)][0], out["raw"][0], atol=1e-12)
    assert out[sw.clip_label(1e-9)][1] == 0


def test_mp_sigma2_rules_and_trace_preservation(R):
    train = R.iloc[:60]
    S = sample_cov(train)
    assert sw.mp_sigma2(S, "trace") == pytest.approx(np.trace(S) / 8)
    assert sw.mp_sigma2(S, "median") == pytest.approx(np.median(np.linalg.eigvalsh(S)))
    with pytest.raises(ValueError):
        sw.mp_sigma2(S, "nope")
    out = sw.mp_cov_fn(("trace", "median"))(train)
    for r in ("trace", "median"):
        Sm, n = out[sw.mp_label(r)]
        assert np.trace(Sm) == pytest.approx(np.trace(S))   # noise flattening preserves the trace
        edge = lt.mp_noise_edge(8, 60, sw.mp_sigma2(S, r))
        assert n == int(np.sum(np.linalg.eigvalsh(S) <= edge))


def test_sweeps_run_and_shapes(R):
    clip = sw.run_clip_sweep(R, fracs=[0.05, 0.5], windows=[60, 120])
    assert set(clip.window) == {60, 120}
    assert set(clip.eps_fraction) == {0.0, 0.05, 0.5}
    assert {"risk_gap_mean", "risk_gap_lo", "risk_gap_hi", "n_adjusted_mean"} <= set(clip.columns)
    mp = sw.run_mp_sweep(R, windows=[60, 120])
    assert set(mp.sigma2_rule) == {"raw", "trace", "median"}
    # raw baseline rows are identical regardless of the sweep method (same data, same weights)
    base_a = clip[clip.eps_fraction == 0].risk_gap_mean.to_numpy()
    base_b = mp[mp.sigma2_rule == "raw"].risk_gap_mean.to_numpy()
    np.testing.assert_allclose(base_a, base_b)
