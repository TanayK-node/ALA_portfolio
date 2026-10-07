import numpy as np
import pandas as pd
import pytest

from src import config, data, evaluate, findings5, phase5, universes


@pytest.fixture(scope="module")
def p5dir(tmp_path_factory):
    """A tiny but complete Phase 5 output directory (N = 8 synthetic stocks, small Monte Carlo)."""
    tmp = tmp_path_factory.mktemp("p5")
    rng = np.random.default_rng(0)
    T, N = 1000, 8
    f = rng.standard_normal((T, 1)) * 0.01
    r = f @ rng.uniform(.5, 1.5, (1, N)) + rng.standard_normal((T, N)) * 0.01
    idx = pd.bdate_range("2020-01-01", periods=T)
    R = pd.DataFrame(r, index=idx, columns=[f"S{i}" for i in range(N)])
    saved = list(config.WINDOWS)
    try:
        universes.set_windows(universes.feasible_windows(len(R)))
        phase5.run_phase5(R, evaluate.run_backtest(R), tmp, n_sims=300, cover_reps=15, verbose=False)
    finally:
        config.WINDOWS[:] = saved
    return tmp


def test_render_has_all_sections_and_is_clean(p5dir):
    text = findings5.render(p5dir)
    for h in ("## 5A.", "## 5B.", "## 5C.", "## 5D.", "## 5E.", "## 5F."):
        assert h in text
    assert "np.int64" not in text and "np.float64" not in text and "nan" not in text.lower().replace("financ", "")
    assert "not verified against the literature" in text           # tier-2 status is always stated
    assert "pre-registered rule" in text.replace("pre-registered predictor", "pre-registered rule") or "Predictors supported" in text


def test_numbers_come_from_the_tables(p5dir):
    text = findings5.render(p5dir)
    chk = pd.read_csv(p5dir / "theory_mc_checks.csv")
    main = chk[chk.tier != "recalled-diagnostic"]
    assert f"Checks within tolerance: {int(main.ok.sum())} of {len(main)}" in text
    ver = pd.read_csv(p5dir / "predictor_verdicts.csv")
    assert f"{int(ver.supported.sum())} of {len(ver)} cells" in text
    bm = pd.read_csv(p5dir / "block_missing.csv")
    assert f"Non-PSD runs: {int(bm.non_psd.sum())} of {len(bm)}" in text


def test_heading_level_shift(p5dir):
    assert findings5.render(p5dir, level=3).count("\n### 5A.") == 1
    assert "\n## 5A." in findings5.render(p5dir) and "#####" not in findings5.render(p5dir, level=3)


def test_write_and_update_readme_idempotent(p5dir, tmp_path):
    path = findings5.write(p5dir)
    assert path.name == "key_findings_phase5.md" and path.read_text().startswith("# Phase 5 key findings")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{findings5.START}\nOLD\n{findings5.END}\noutro\n")
    findings5.update_readme(readme, p5dir)
    once = readme.read_text()
    assert once.startswith("intro\n") and once.endswith("outro\n") and "OLD" not in once and "\n### 5A." in once and "#####" not in once
    findings5.update_readme(readme, p5dir)
    assert readme.read_text() == once                                # idempotent
    bad = tmp_path / "bad.md"
    bad.write_text("no markers here")
    with pytest.raises(ValueError):
        findings5.update_readme(bad, p5dir)
