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
