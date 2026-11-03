import numpy as np
import pandas as pd
import pytest

from src.journal import regimes, scenarios, sim_dynamic, surface


@pytest.mark.parametrize("kind", ["garch", "regime"])
def test_unconditional_covariance_equals_sigma(kind):
    Sigma = scenarios.make_sigma("hetero", 4)
    rng = np.random.default_rng(1)
    X, aux = sim_dynamic.simulate(kind, Sigma, reps=200, length=1500, rng=rng, burn=200)
    S = np.einsum("rti,rtj->ij", X, X) / (X.shape[0] * X.shape[1])
    assert np.allclose(S / Sigma, 1.0, atol=0.08)
    assert aux.mean() == pytest.approx(1.0 if kind == "garch" else sim_dynamic.regime_scales(0.99, 0.95, 2.5)[2], abs=0.03)


def test_regime_scales_and_invalid_parameters():
    m_c, m_s, pi_s = sim_dynamic.regime_scales(0.99, 0.95, 2.5)
    assert pi_s == pytest.approx(1 / 6) and (1 - pi_s) * m_c + pi_s * m_s == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sim_dynamic.regime_scales(0.99, 0.95, 8.0)
    with pytest.raises(ValueError):
        sim_dynamic.simulate("garch", np.eye(2), 2, 5, np.random.default_rng(0), params=dict(a=0.3, b=0.7))


def test_garch_has_volatility_clustering_and_iid_does_not():
    Sigma = np.eye(3) * 1e-4
    X, _ = sim_dynamic.simulate("garch", Sigma, 300, 800, np.random.default_rng(2))
    sq = (X.mean(axis=2) ** 2)
    ac = np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in sq])
    Xi = scenarios.draw("gaussian", 300, 800, Sigma, np.random.default_rng(2))
    sqi = Xi.mean(axis=2) ** 2
    aci = np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in sqi])
    assert ac > 0.1 > abs(aci)


def _returns(n=900, N=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, N)) * 0.01, index=pd.bdate_range("2015-01-01", periods=n))


def test_regime_labels_use_only_past_data():
    R = _returns()
    base = regimes.label_origins(R, [760, 820], 20)
    R2 = R.copy()
    R2.iloc[760:] *= 5.0                              # change every observation from row 760 on
    chg = regimes.label_origins(R2, [760, 820], 20)
    assert regimes.vol_thresholds(R) == regimes.vol_thresholds(R2)        # thresholds use rows < 750 only
    assert base.trailing_vol.iloc[0] == chg.trailing_vol.iloc[0]           # origin 760 looks at rows 700..759 only
    assert base.trailing_vol.iloc[1] != chg.trailing_vol.iloc[1]           # origin 820 sees rows 760.. so it does change
