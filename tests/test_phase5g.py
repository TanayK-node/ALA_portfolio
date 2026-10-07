import numpy as np
import pandas as pd
import pytest

import run_all
from src import config, data, phase5, universes


@pytest.fixture
def restore_windows():
    saved = list(config.WINDOWS)
    yield
    config.WINDOWS[:] = saved


def test_parse_period():
    assert universes.parse_period("full") == ("full", (0.0, 1.0))
    assert universes.parse_period("second_half") == ("second_half", (0.5, 1.0))
    assert universes.parse_period("0.25:0.75") == ("0.25-0.75", (0.25, 0.75))
    for bad in ("nope", "0.5:0.2", "0:1.5", "a:b"):
        with pytest.raises(ValueError):
            universes.parse_period(bad)


def test_resolve_run_layouts():
    leg = universes.resolve_run()
    assert leg.legacy and leg.results_dir == config.RESULTS_DIR and leg.universe == config.DEFAULT_UNIVERSE
    u = universes.resolve_run("nifty30")
    assert not u.legacy and u.results_dir == config.RESULTS_DIR / "nifty30"
    s = universes.resolve_run("nifty_next", "second_half")
    assert s.results_dir == config.RESULTS_DIR / "nifty_next" / "second_half" and s.period_frac == (0.5, 1.0)
    assert universes.resolve_run(period="first_half").results_dir == config.RESULTS_DIR / "nifty30" / "first_half"
    with pytest.raises(ValueError):
        universes.resolve_run("nasdaq")


def test_configured_universes_are_sane():
    assert set(config.UNIVERSES) >= {"nifty30", "nifty_next"}
    nn = config.NIFTY_NEXT_TICKERS
    assert len(nn) == 30 and len(set(nn)) == 30 and all(t.endswith(".NS") for t in nn)
    assert not set(nn) & set(config.TICKERS)                  # disjoint from the main universe
    assert config.UNIVERSES["nifty_next"]["prices_csv"].name == "prices_nifty_next.csv"


def test_feasible_windows_and_slice():
    assert universes.feasible_windows(1238) == config.WINDOWS
    assert universes.feasible_windows(619) == [60, 90, 120, 250]
    assert universes.feasible_windows(100) == []
    R = pd.DataFrame(np.zeros((100, 2)))
    assert len(universes.slice_period(R, (0.0, 0.5))) == 50 and len(universes.slice_period(R, (0.5, 1.0))) == 50
    assert len(universes.slice_period(R, (0.25, 0.75))) == 50


def test_set_windows_is_in_place_and_rejects_empty(restore_windows):
    ref = config.WINDOWS
    universes.set_windows([60, 90])
    assert config.WINDOWS is ref and ref == [60, 90]
    with pytest.raises(ValueError):
        universes.set_windows([])


def test_missing_cache_prints_instructions_and_never_downloads(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(config.UNIVERSES, "nifty_next", dict(tickers=["A.NS"], prices_csv=tmp_path / "prices_nifty_next.csv"))
    monkeypatch.setattr(data, "download_prices", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download attempted")))
    rc = run_all.main(["--universe", "nifty_next"])
    err = capsys.readouterr().err
    assert rc == 2 and "python -m src.data nifty_next" in err and "prices_nifty_next.csv" in err
    assert run_all.main(["--universe", "nifty_next", "--period", "bogus"]) == 2


@pytest.fixture
def synthetic_universe(monkeypatch, tmp_path, restore_windows):
    """A tiny universe whose price cache lives in tmp_path; results also redirected to tmp_path."""
    rng = np.random.default_rng(0)
    T, N = 1000, 8
    f = rng.standard_normal((T, 1)) * 0.01
    r = f @ rng.uniform(.5, 1.5, (1, N)) + rng.standard_normal((T, N)) * 0.01
    idx = pd.bdate_range("2020-01-01", periods=T)
    tickers = [f"S{i}.NS" for i in range(N)]
    pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=tickers).to_csv(tmp_path / "prices_syn.csv")
    monkeypatch.setitem(config.UNIVERSES, "syn", dict(tickers=tickers, prices_csv=tmp_path / "prices_syn.csv"))
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


def test_run_all_end_to_end_on_synthetic_universe(synthetic_universe):
    assert run_all.main(["--universe", "syn"]) == 0
    out = synthetic_universe / "results" / "syn"
    for f in ("backtest_runs.csv", "summary_table.csv", "exposure_vs_gap.csv", "key_findings.md",
              "full_sample_diagnostics.csv", "fig1_eigenvalue_spectrum.png"):
        assert (out / f).exists(), f
    runs = pd.read_csv(out / "backtest_runs.csv")
    assert set(runs.window) == {60, 90, 120, 250, 500, 750}       # 1000 days: every window feasible
    assert not (synthetic_universe / "results" / "backtest_runs.csv").exists()   # legacy dir untouched
    m = universes.headline_metrics(out)
    assert m["N"] == 8.0 and "raw_unc_gap_pts_W60" in m and "random_mask_non_psd" in m
    assert f"exposure_k{config.PLOT_K}_within_rho_unc" in m
    cmp_ = universes.compare({"syn": out, "syn_again": out})
    assert list(cmp_.columns) == ["syn", "syn_again"] and cmp_.loc["N"].eq(8.0).all()


def test_run_all_sub_period_drops_windows_and_reports_infeasible(synthetic_universe, capsys):
    assert run_all.main(["--universe", "syn", "--period", "first_half"]) == 0
    out = synthetic_universe / "results" / "syn" / "first_half"
    assert set(pd.read_csv(out / "backtest_runs.csv").window) == {60, 90, 120, 250}   # 500, 750 need more history
    assert run_all.main(["--universe", "syn", "--period", "0.0:0.1"]) == 2             # 100 days: nothing feasible
    assert "too short" in capsys.readouterr().err


def test_phase5_smoke(synthetic_universe):
    # direct Phase 5 smoke test with tiny Monte Carlo / bootstrap settings
    R, _, _ = data.get_returns(synthetic_universe / "prices_syn.csv", config.UNIVERSES["syn"]["tickers"])
    universes.set_windows(universes.feasible_windows(len(R)))
    from src import evaluate
    old = evaluate.run_backtest(R)
    res = phase5.run_phase5(R, old, synthetic_universe / "p5", n_sims=300, cover_reps=20, verbose=False)
    assert set(res) == set("abcdef")
    for f in ("summary_common_oos.csv", "theory_vs_empirical.csv", "predictor_tests.csv", "sweep_clip.csv",
              "sweep_mp.csv", "paired_differences_common_oos.csv", "bootstrap_coverage_check.csv", "block_missing.csv"):
        assert (synthetic_universe / "p5" / f).exists(), f
    assert max(res["f"]["runs"].m) < R.shape[1]                    # m is capped below N (N = 8 here)
