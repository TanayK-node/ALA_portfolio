import numpy as np
import pandas as pd
import pytest

from src import config, evaluate as ev, linalg_tools as lt, multitest as mt, predictors as pr


# ---- linalg additions
def test_effective_rank_and_min_eig_ratio():
    assert lt.effective_rank(np.eye(7) * 3.0) == pytest.approx(7.0)
    v = np.ones((5, 1))
    assert lt.effective_rank(v @ v.T) == pytest.approx(1.0)
    S = np.diag([1.0, 2.0, 3.0])
    assert lt.effective_rank(S) == pytest.approx(36 / 14)          # (1+2+3)^2 / (1+4+9)
    assert lt.min_eig_ratio(S) == pytest.approx(1.0 / 2.0)         # 1 / mean(2)
    assert lt.min_eig_ratio(np.eye(4)) == pytest.approx(1.0)


# ---- multiple testing
def test_holm_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(mt.holm_adjust(p), [0.03, 0.06, 0.06, 0.02])


def test_bh_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(mt.bh_adjust(p), [0.02, 0.04, 0.04, 0.02])


def test_adjustments_ordering_nan_and_bounds():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20)
    p[3] = np.nan
    h, b = mt.holm_adjust(p), mt.bh_adjust(p)
    assert np.isnan(h[3]) and np.isnan(b[3])
    ok = ~np.isnan(p)
    assert (h[ok] >= p[ok] - 1e-12).all() and (b[ok] <= h[ok] + 1e-12).all() and (h[ok] <= 1).all()
    assert np.isnan(mt.holm_adjust(np.array([np.nan]))).all()


def test_fisher_pool_properties():
    r = mt.fisher_pool(np.array([0.3, 0.3, 0.3]), np.array([13, 13, 13]))
    assert r["rho"] == pytest.approx(0.3) and r["k_used"] == 3 and r["total_weight"] == pytest.approx(30)
    assert r["se"] == pytest.approx(np.sqrt(1.06 / 30))
    # windows with n <= 3 or NaN rho get zero weight
    r2 = mt.fisher_pool(np.array([0.5, 0.9, np.nan]), np.array([23, 3, 40]))
    assert r2["rho"] == pytest.approx(0.5) and r2["k_used"] == 1
    assert np.isnan(mt.fisher_pool(np.array([0.5]), np.array([3]))["rho"])
    # strongly correlated pooled evidence is significant, null is not
    assert mt.fisher_pool(np.array([0.8, 0.7]), np.array([30, 30]))["p"] < 1e-6
    assert mt.fisher_pool(np.array([0.0, 0.0]), np.array([30, 30]))["p"] == pytest.approx(1.0)


# ---- designs
def test_nonoverlap_origins_disjoint_and_counts():
    assert pr.nonoverlap_origins(1238, 60) == list(range(60, 1179, 60))
    assert pr.nonoverlap_origins(1238, 750) == [750]
    assert pr.nonoverlap_origins(1238, 500) == [500, 1000]
    for W in (60, 250):
        o = pr.nonoverlap_origins(1238, W)
        assert all(b - a == W for a, b in zip(o, o[1:]))  # training windows [o-W,o) are disjoint


def test_boot_block_len_rule():
    assert pr.boot_block_len(60, 19) == 3
    assert pr.boot_block_len(250, 16) == 5
    assert pr.boot_block_len(750, 8) == 4      # capped at n//2
    assert pr.boot_block_len(60, 2) == 1


@pytest.fixture(scope="module")
def R():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=900)
    f = rng.standard_normal((900, 1)) * 0.01
    return pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 6)) + rng.standard_normal((900, 6)) * 0.01, index=idx)


def test_build_runs_and_predictor_columns(R, monkeypatch):
    monkeypatch.setattr(config, "WINDOWS", [60, 120])
    monkeypatch.setattr(pr.config, "WINDOWS", [60, 120])
    a = pr.add_predictors(pr.build_runs(R, "nonoverlap"), R)
    assert set(a.method) == {"raw"} and set(a.constraint) == set(config.CONSTRAINTS)
    # non-overlapping: training windows of consecutive runs of one window length are disjoint
    g = a[(a.window == 60) & (a.constraint == "unconstrained")].sort_values("origin")
    assert (g.train_start.iloc[1:].to_numpy() > g.train_end.iloc[:-1].to_numpy()).all()
    assert (a.test_start > a.train_end).all()
    row = a.iloc[0]
    S = np.cov(R.iloc[row.origin - row.window:row.origin].to_numpy(), rowvar=False)
    assert row.eff_rank == pytest.approx(lt.effective_rank(S))
    assert row.lam_ratio == pytest.approx(lt.min_eig_ratio(S))
    assert row.log_cond == pytest.approx(np.log(lt.condition_number(S)))
    with pytest.raises(ValueError):
        pr.build_runs(R, "nope")


def test_spearman_boot_matches_direct_and_sign_p():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(15)
    y = x + 0.5 * rng.standard_normal(15)
    idx = np.vstack([np.arange(15), rng.integers(0, 15, 15)])
    b = pr._spearman_boot(x, y, idx)
    from scipy.stats import spearmanr
    assert b[0] == pytest.approx(spearmanr(x, y)[0])
    assert b[1] == pytest.approx(spearmanr(x[idx[1]], y[idx[1]])[0])
    assert pr._sign_p(np.full(100, 0.5)) < 0.05 and pr._sign_p(np.concatenate([np.ones(50), -np.ones(50)])) == 1.0


def _cell(rho_strength, seed=0):
    rng = np.random.default_rng(seed)
    g = {}
    for W, n in ((60, 19), (90, 13), (120, 9)):
        x = rng.standard_normal(n)
        g[W] = (x, rho_strength * x + rng.standard_normal(n))
    return g


@pytest.mark.parametrize("design", pr.DESIGNS)
def test_cell_detects_strong_signal_and_not_noise(design):
    strong = {r["window"]: r for r in pr.test_cell(_cell(6.0), design, n_boot=500)}
    assert strong["pooled"]["rho"] > 0.7 and strong["pooled"]["p_raw"] < 0.01
    null = {r["window"]: r for r in pr.test_cell(_cell(0.0, seed=3), design, n_boot=500)}
    assert null["pooled"]["p_raw"] > 0.05 or abs(null["pooled"]["rho"]) < 0.4  # null shouldn't look strong
    assert strong[60]["n"] == 19


def test_cell_small_windows_are_nan_not_hidden():
    g = {250: (np.arange(4.0), np.arange(4.0)), 500: (np.arange(2.0), np.arange(2.0)), 750: (np.array([1.0]), np.array([1.0]))}
    rows = {r["window"]: r for r in pr.test_cell(g, "nonoverlap", n_boot=100)}
    assert np.isnan(rows[500]["rho"]) and np.isnan(rows[750]["rho"]) and rows[750]["n"] == 1
    assert rows["pooled"]["n_windows_used"] == 1   # only W=250 (n=4 -> weight 1)


def test_run_tests_families_and_verdict_rule(R, monkeypatch):
    monkeypatch.setattr(config, "WINDOWS", [60, 120])
    monkeypatch.setattr(pr.config, "WINDOWS", [60, 120])
    allt = []
    for d in pr.DESIGNS:
        runs = pr.add_predictors(pr.build_runs(R, d), R)
        t = pr.run_tests(runs, d)
        allt.append(t)
        prim = t[t.family == "primary"]
        assert len(prim) == 2 * len(pr.PREDICTORS)                 # 7 pooled tests per constraint
        assert (prim.p_holm >= prim.p_raw - 1e-12).all() and (prim.p_bh <= prim.p_holm + 1e-12).all()
    v = pr.verdicts(pd.concat(allt))
    assert len(v) == 2 * len(pr.PREDICTORS) and v.supported.dtype == bool
    fake = v.copy()
    fake.loc[0, ["holm_nonoverlap", "holm_stride60", "rho_nonoverlap", "rho_stride60"]] = [0.01, 0.01, 0.3, 0.3]
    fake.loc[1, ["holm_nonoverlap", "holm_stride60", "rho_nonoverlap", "rho_stride60"]] = [0.01, 0.01, 0.3, -0.3]
    sup = ((fake.holm_nonoverlap < .05) & (fake.holm_stride60 < .05) & (np.sign(fake.rho_nonoverlap) == np.sign(fake.rho_stride60)))
    assert sup[0] and not sup[1]
