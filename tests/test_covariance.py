import numpy as np
import pandas as pd
import pytest

from src import covariance as cv
from src import linalg_tools as lt


@pytest.fixture
def rets():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((300, 12)) * 0.01)


def test_sample_cov_matches_pandas(rets):
    np.testing.assert_allclose(cv.sample_cov(rets), rets.cov().to_numpy(), atol=1e-14)


def test_pairwise_equals_sample_without_nans(rets):
    np.testing.assert_allclose(cv.pairwise_cov(rets), cv.sample_cov(rets), atol=1e-14)


def test_short_window_is_singular(rets):
    S, (p, z, n) = cv.short_window_cov(rets, T=8)
    assert p + z + n == 12 and z >= 12 - 8 + 1 and n == 0


def test_ledoit_wolf_is_pd(rets):
    assert lt.is_positive_definite(cv.ledoit_wolf_cov(rets.iloc[:8]))


def test_mask_is_reproducible_and_scenarios_shape(rets):
    a = cv.masked_returns(rets, 0.1, seed=5)
    b = cv.masked_returns(rets, 0.1, seed=5)
    pd.testing.assert_frame_equal(a, b)
    df = cv.missing_data_scenarios(rets, fracs=[0.1], n_seeds=3)
    assert len(df) == 3 and (df[["n_pos", "n_zero", "n_neg"]].sum(axis=1) == 12).all()


def test_short_window_report_inertia_theory():
    from src import diagnostics as dg
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.standard_normal((900, 30)) * 0.01)
    rep = dg.short_window_report(r).set_index("T")
    for T in (15, 29, 30):  # rank = T-1  =>  N-T+1 zero eigenvalues
        assert rep.loc[T, "n_zero"] == 30 - T + 1
    assert rep.loc[60, "n_zero"] == 0 and rep.loc[60, "pd_cholesky"]


def test_md_table_renders():
    from src import diagnostics as dg
    t = dg.md_table(pd.DataFrame({"a": [1, 2], "b": [0.12345, 2.0]}), "{:.2f}")
    assert t.splitlines()[0] == "| a | b |" and "| 1 | 0.12 |" in t
