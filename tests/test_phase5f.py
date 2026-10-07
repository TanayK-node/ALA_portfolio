import numpy as np
import pandas as pd
import pytest

from src import block_missing as bm, config, linalg_tools as lt
from src.covariance import pairwise_cov


@pytest.fixture(scope="module")
def regime():
    """3 stocks: identical early (210 rows), late (90 rows) x2 = -x1, x3 = x1.
    One late-listed stock => pairwise correlations (+-1, +-1, 0.4): provably indefinite."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal(300) * 0.01
    e = lambda: rng.standard_normal(300) * 1e-5
    x1 = g + e()
    x2 = np.where(np.arange(300) < 210, g, -g) + e()
    x3 = g + e()
    return pd.DataFrame({"a": x1, "b": x2, "c": x3}, index=pd.bdate_range("2020-01-01", periods=300))


def test_frobenius_distance_basic_and_spectral_identity():
    A, B = np.array([[1.0, 2.0], [3.0, 4.0]]), np.zeros((2, 2))
    assert lt.frobenius_distance(A, B) == pytest.approx(np.sqrt(30))
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 6))
    S = (X + X.T) / 2
    Sc, _ = lt.clip_eigenvalues(S, 0.3)
    vals = np.linalg.eigvalsh(S)
    assert lt.frobenius_distance(Sc, S) == pytest.approx(np.sqrt(np.sum((np.maximum(vals, 0.3) - vals) ** 2)))


def test_mask_shape_columns_and_reproducibility():
    rng = np.random.default_rng(2)
    R = pd.DataFrame(rng.standard_normal((200, 10)))
    a, cols = bm.late_listing_mask(R, 4, 0.3, seed=7)
    assert len(cols) == 4 and a.isna().sum().sum() == 4 * 60
    assert a.iloc[:60, cols].isna().all().all() and a.iloc[60:].notna().all().all()
    others = [c for c in range(10) if c not in cols]
    assert a.iloc[:, others].notna().all().all()
    b, cols2 = bm.late_listing_mask(R, 4, 0.3, seed=7)
    assert (cols == cols2).all()
    seen = {tuple(bm.late_listing_mask(R, 4, 0.3, seed=s)[1]) for s in range(10)}
    assert len(seen) > 1                                   # different seeds pick different stocks
    with pytest.raises(ValueError):
        bm.late_listing_mask(R, 0, 0.3, 0)


def test_constructed_late_listing_is_indefinite_and_repair_verified(regime):
    masked, _ = bm.late_listing_mask(regime, 1, 0.7, seed=0)
    S = pairwise_cov(masked)
    assert lt.inertia(S)[2] >= 1 and not lt.is_positive_definite(S)   # the premise: pairwise Sigma not PSD
    df = bm.run_experiment(regime, ms=[1], fs=[0.7], n_seeds=6)
    assert df.non_psd.all() and (df.n_neg >= 1).all()
    for tag in ("rel", "min"):
        assert (df[f"clip_{tag}_inertia_pos"] == 3).all() and (df[f"clip_{tag}_inertia_neg"] == 0).all()
        assert df[f"clip_{tag}_min_eig_ok"].all() and df[f"clip_{tag}_frob_matches_spectrum"].all()
        assert (df[f"clip_{tag}_frob"] > 0).all() and (df[f"clip_{tag}_n_clipped"] >= 1).all()
    assert (df.clip_rel_frob >= df.clip_min_frob - 1e-15).all()       # larger floor costs at least as much
    assert (df.cc_pd).all() and (df.cc_rows == 90).all()               # complete case = last 90 rows, PD
    assert (df.clip_rel_dw_l1 > 0).all() and np.isfinite(df.pw_var_ratio).all()


def test_all_variance_ratios_at_least_one_when_psd_inputs():
    rng = np.random.default_rng(3)
    f = rng.standard_normal((500, 1)) * 0.01
    R = pd.DataFrame(f @ rng.uniform(.6, 1.4, (1, 8)) + rng.standard_normal((500, 8)) * 0.01)
    df = bm.run_experiment(R, ms=[2], fs=[0.3], n_seeds=5)
    assert (df.cc_var_ratio >= 1 - 1e-9).all()             # nothing beats the full-information GMV under S_full
    assert (df.clip_rel_var_ratio >= 1 - 1e-9).all()


def test_experiment_deterministic_and_summary(regime):
    a = bm.run_experiment(regime, ms=[1], fs=[0.5, 0.7], n_seeds=4)
    b = bm.run_experiment(regime, ms=[1], fs=[0.5, 0.7], n_seeds=4)
    pd.testing.assert_frame_equal(a, b)
    s = bm.summarize(a)
    assert list(s.columns[:4]) == ["m", "f", "runs", "non_psd"] and (s.runs == 4).all()
    assert (s.non_psd_rate_se >= 0).all() and (s.non_psd <= s.runs).all()
