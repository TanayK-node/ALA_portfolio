"""Auto-generated Phase 5 findings: every number is computed from the CSVs in results/phase5/.

    python -m src.findings5                    # (re)write results/phase5/key_findings_phase5.md
    python -m src.findings5 --update-readme    # also refresh the marked results block in README.md

Nothing here is typed by hand except neutral descriptive text; statements about the data (counts,
maxima, whether an interval contains a value) are evaluated on the tables at render time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .diagnostics import md_table

START, END = "<!-- PHASE5-RESULTS:START -->", "<!-- PHASE5-RESULTS:END -->"
CONS = config.CONSTRAINTS
METHODS = config.METHODS


def _r(d: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(d) / name)


def _pts(x: float) -> str:
    return f"{100 * x:.2f}"


def _pivot(df: pd.DataFrame, index: str, columns: str, values: str, scale: float = 100.0, fmt: str = "{:.2f}",
           col_order: list | None = None) -> pd.DataFrame:
    p = df.pivot(index=index, columns=columns, values=values) * scale
    if col_order is not None:
        p = p[[c for c in col_order if c in p.columns]]
    return p.map(lambda v: "" if pd.isna(v) else fmt.format(v)).reset_index()


def _sec5a(d: Path) -> list[str]:
    s, cmp_, runs = _r(d, "summary_common_oos.csv"), _r(d, "oos_design_comparison.csv"), _r(d, "backtest_runs_common_oos.csv")
    n = int(s.n_periods.max())
    L = ["## 5A. Common out-of-sample periods", "",
         f"{n} test blocks of {config.HORIZON} days, {str(runs.test_start.min())[:10]} to {str(runs.test_end.max())[:10]}; "
         f"every window length W is trained on the W days before each of the same origins "
         f"(first origin = the largest window, {max(runs.window)}).", ""]
    for cons in CONS:
        c = s[s.constraint == cons]
        L += [f"**{cons}: mean risk gap (realised − predicted), annualised, percentage points**", "",
              md_table(_pivot(c, "window", "method", "risk_gap_mean", col_order=list(METHODS))), ""]
        raw = c[c.method == "raw"].sort_values("window")
        L += ["Raw 95% moving-block-bootstrap interval for the mean gap: " +
              "; ".join(f"W={int(r.window)}: [{_pts(r.risk_gap_lo)}, {_pts(r.risk_gap_hi)}]" for r in raw.itertuples()), "",
              f"**{cons}: mean realised risk, annualised, %**", "",
              md_table(_pivot(c, "window", "method", "realized_risk_mean", col_order=list(METHODS))), ""]
    cmp_["abs_diff"] = cmp_.gap_diff.abs()
    top = cmp_.loc[cmp_.abs_diff.idxmax()]
    L += ["**Per-window OOS design (old) vs common OOS (new): mean risk gap, pts**", ""]
    for cons in CONS:
        c = cmp_[cmp_.constraint == cons]
        t = c.assign(v=lambda x: x.gap_old.map(_pts) + " → " + x.gap_new.map(_pts)).pivot(index="window", columns="method", values="v")
        L += [f"{cons} (old → new)", "", md_table(t[list(METHODS)].reset_index()), ""]
    L += [f"Largest absolute change in mean gap between the two designs: {_pts(top.abs_diff)} pts "
          f"({top.constraint}, {top.method}, W={int(top.window)}).", ""]
    return L


def _sec5b(d: Path) -> list[str]:
    t, chk = _r(d, "theory_vs_empirical.csv"), _r(d, "theory_mc_checks.csv")
    main, alt = chk[chk.tier != "recalled-diagnostic"], chk[chk.tier == "recalled-diagnostic"]
    N = int(t.N.iloc[0])
    L = ["## 5B. Theory overlay (unconstrained raw GMV, i.i.d. Gaussian)", "",
         "Tier 1 (exact, Wishart): in-sample variance ratio `(T-1)Q ~ chi2_{T-N}`. Tier 2 (derived in `src/theory.py`, "
         "**not verified against the literature**): out-of-sample variance ratio `R-1 ~ chi2_{N-1}/chi2_{T-N+1}`. "
         f"Monte Carlo: N = {N}, {config.MC_SIMS} simulations per window, true Σ = shrunk empirical Σ "
         f"(shrinkage {config.MC_SHRINK}). Pre-stated tolerance: |MC − theory| ≤ {config.MC_Z_TOL:g} standard errors; "
         f"KS mismatch flagged at p < {config.MC_KS_ALPHA}.", "",
         f"Checks within tolerance: {int(main.ok.sum())} of {len(main)}; max |z| = {main.z.abs().max():.2f}; "
         f"min KS p-value = {main.ks_p.min():.3f}.", ""]
    rows = []
    for r in t.itertuples():
        rows.append({"T": int(r.window),
                     "E[pred]/V theory": f"{r.th_pred_var_over_V:.4f}", "MC": f"{r.mc_gaussian_pred_var_over_V:.4f}±{r.mc_gaussian_pred_var_over_V_se:.4f}",
                     "E[true]/V theory": f"{r.th_true_var_over_V:.4f}", "MC ": f"{r.mc_gaussian_true_var_over_V:.4f}±{r.mc_gaussian_true_var_over_V_se:.4f}",
                     "E[true/pred] theory": f"{r.th_var_ratio_per_run:.3f}", "MC  ": f"{r.mc_gaussian_var_ratio_per_run:.3f}±{r.mc_gaussian_var_ratio_per_run_se:.3f}"})
    L += [md_table(pd.DataFrame(rows)), "", "**Realised / predicted risk (std ratio): theory, simulations, measured**", ""]
    rows = []
    for r in t.itertuples():
        inside = r.emp_std_ratio_lo <= r.th_std_ratio <= r.emp_std_ratio_hi
        rows.append({"T": int(r.window), "theory": f"{r.th_std_ratio:.3f}", "MC Gaussian": f"{r.mc_gaussian_std_ratio:.3f}",
                     f"MC Student-t (df={config.MC_T_DF})": f"{r.mc_student_t_std_ratio:.3f}", "MC bootstrap of real returns": f"{r.mc_bootstrap_std_ratio:.3f}",
                     "measured [95% CI]": f"{r.emp_std_ratio:.3f} [{r.emp_std_ratio_lo:.3f}, {r.emp_std_ratio_hi:.3f}]",
                     "CI contains Gaussian theory": str(bool(inside))})
    L += [md_table(pd.DataFrame(rows)), "",
          f"Measured ratio above the Gaussian theory at {int((t.emp_std_ratio > t.th_std_ratio).sum())} of {len(t)} windows; "
          f"measured CI contains the theory value at {int(((t.emp_std_ratio_lo <= t.th_std_ratio) & (t.th_std_ratio <= t.emp_std_ratio_hi)).sum())} of {len(t)} "
          f"(n = {int(t.emp_n_periods.max())} periods, so the data cannot discriminate well). "
          "Student-t and bootstrap simulations show how far non-Gaussian data departs from the theory.", "",
          "**Diagnostic only: the recalled offset-2 form `1 + (N-1)/(T-N-2)` against the same simulation**", "",
          md_table(alt[["window", "theory", "mc_mean", "mc_se", "z"]].round(4)), ""]
    return L


def _sec5c(d: Path) -> list[str]:
    tests, ver = _r(d, "predictor_tests.csv"), _r(d, "predictor_verdicts.csv")
    pooled = tests[tests.window.astype(str) == "pooled"]
    per = tests[(tests.window.astype(str) != "pooled") & (tests.constraint == "unconstrained") & (tests.predictor == "exposure_k1")].copy()
    per["window"] = per.window.astype(int)           # the column mixes 'pooled' and numbers; sort windows numerically
    L = ["## 5C. Pre-registered predictor comparison", "",
         "Analysis plan: see the pre-registration section above (committed before any 5C analysis). Pooled statistic = Fisher-z "
         "average of within-window Spearman ρ; Holm and BH within the 7-predictor family of each (design, portfolio) cell.", "",
         f"**Predictors supported by the pre-registered rule (Holm p < 0.05 in both designs, same sign): "
         f"{int(ver.supported.sum())} of {len(ver)} cells.**", "", "Runs per window (unconstrained):", "",
         md_table(per.pivot(index="window", columns="design", values="n").reset_index()), "",
         "**Verdicts**", "", md_table(ver.round(3)), "", "**All pooled (primary-family) tests**", "",
         md_table(pooled[["design", "constraint", "predictor", "n", "n_windows_used", "rho", "ci_lo", "ci_hi", "p_raw", "p_holm", "p_bh"]].round(3)), ""]
    sig = pooled[pooled.p_holm < 0.05]
    L += ["Pooled tests with Holm p < 0.05 in at least one design: " +
          ("; ".join(f"{r.design}/{r.constraint}/{r.predictor} (ρ = {r.rho:.2f}, Holm p = {r.p_holm:.3f})" for r in sig.itertuples()) or "none") + ".", ""]
    return L


def _sec5d(d: Path) -> list[str]:
    clip, mp = _r(d, "sweep_clip.csv"), _r(d, "sweep_mp.csv")
    L = ["## 5D. Sensitivity sweeps (common-OOS design)", "",
         f"Clipping floor = fraction × tr(S)/N, fractions {config.CLIP_SWEEP} (0 = raw). A-priori setting in the main analysis: "
         f"{config.CLIP_REL:g}. Parameters are not selected.", ""]
    for cons in CONS:
        c = clip[clip.constraint == cons]
        L += [f"**{cons}: mean risk gap (pts) by floor fraction**", "", md_table(_pivot(c, "window", "eps_fraction", "risk_gap_mean")), "",
              f"**{cons}: mean realised risk (%)**", "", md_table(_pivot(c, "window", "eps_fraction", "realized_risk_mean")), ""]
    c = clip[clip.constraint == CONS[0]]
    L += ["**Mean number of eigenvalues clipped**", "", md_table(_pivot(c, "window", "eps_fraction", "n_adjusted_mean", scale=1.0)), "",
          f"MP noise variance: σ² from {list(config.MP_SIGMA2_RULES)} (a-priori rule: trace).", ""]
    for cons in CONS:
        m = mp[mp.constraint == cons]
        L += [f"**{cons}: MP sweep, mean risk gap (pts) / realised risk (%)**", "",
              md_table(_pivot(m, "window", "sigma2_rule", "risk_gap_mean")), "", md_table(_pivot(m, "window", "sigma2_rule", "realized_risk_mean")), ""]
    m = mp[mp.constraint == CONS[0]]
    L += ["**Mean number of eigenvalues treated as noise**", "", md_table(_pivot(m[m.sigma2_rule != "raw"], "window", "sigma2_rule", "n_adjusted_mean", scale=1.0)), ""]
    return L


def _sec5e(d: Path) -> list[str]:
    p, sens, cov = _r(d, "paired_differences_common_oos.csv"), _r(d, "paired_block_sensitivity.csv"), _r(d, "bootstrap_coverage_check.csv")
    n = int(p.n_periods.max())
    trivial = int(((p.excludes_zero) & (p.diff_mean.abs() < 0.001)).sum())
    c0 = cov[(cov.n == n) & (cov.block_len == min(config.BOOT_BLOCK, n)) & (cov.rho == 0.0)]
    c1 = cov[(cov.n == n) & (cov.block_len == min(config.BOOT_BLOCK, n)) & (cov.rho == 0.5)]
    L = ["## 5E. Uncertainty (moving-block bootstrap over OOS periods)", "",
         f"{n} periods per window, block length {config.BOOT_BLOCK}, {config.BOOT_N} resamples, 95% percentile intervals; "
         "differences are method − comparator on identical periods (negative gap difference = smaller gap; negative realised-risk "
         "difference = lower risk). Unadjusted for multiple comparisons.", "",
         f"Paired comparisons: {len(p)}; intervals excluding zero: {int(p.excludes_zero.sum())}; flagged differences smaller than "
         f"0.1 percentage point in absolute size: {trivial}.", ""]
    if len(c0) and len(c1):
        L += [f"**Simulated coverage of the nominal 95% interval at n = {n}, block length {config.BOOT_BLOCK}:** "
              f"{c0.coverage.iloc[0]:.3f} (i.i.d. data) and {c1.coverage.iloc[0]:.3f} (AR(1), coefficient 0.5). "
              "Coverage below 0.95 means the intervals are too narrow and the 'excludes zero' flags are anti-conservative.", ""]
    L += ["**Coverage check (all settings)**", "", md_table(cov.round(3)), "", "**Block-length sensitivity (number of comparisons excluding zero)**", "",
          md_table(sens), ""]
    for metric, lab in (("risk_gap", "mean risk gap difference (pts)"), ("realized_risk", "mean realised risk difference (pts)")):
        for cons in CONS:
            s = p[(p.metric == metric) & (p.constraint == cons)].copy()
            s["v"] = (100 * s.diff_mean).map("{:.2f}".format) + np.where(s.excludes_zero, "*", "")
            L += [f"**{cons}: {lab}** (* = interval excludes 0)", "", md_table(s.pivot(index="window", columns="comparison", values="v").reset_index()), ""]
    return L


def _sec5f(d: Path) -> list[str]:
    df, s = _r(d, "block_missing.csv"), _r(d, "block_missing_summary.csv")
    bad = df[df.non_psd]
    L = ["## 5F. Block missingness: late-listed stocks", "",
         f"m ∈ {[int(x) for x in sorted(df.m.unique())]} stocks have the first fraction f ∈ {[float(x) for x in sorted(df.f.unique())]} of the sample missing "
         f"(all list on the same date), {int(s.runs.iloc[0])} seeds per cell; pairwise-complete covariance. Reference for judging weights: the "
         "full-sample covariance of the unmasked data (variance ratio 1 = full-information GMV).", "",
         md_table(s[["m", "f", "runs", "non_psd", "non_psd_rate", "non_psd_rate_se", "lambda_min_min", "lambda_min_median",
                     "pw_var_ratio_median", "clip_rel_var_ratio_median", "cc_var_ratio_median"]], "{:.3g}"), "",
         f"Non-PSD runs: {len(bad)} of {len(df)}.", ""]
    if len(bad):
        nneg = f"{int(bad.n_neg.min())}" if bad.n_neg.min() == bad.n_neg.max() else f"{int(bad.n_neg.min())}–{int(bad.n_neg.max())}"
        L += [f"In the {len(bad)} non-PSD runs: number of negative eigenvalues per matrix {nneg}; "
              f"λ_min from {bad.lambda_min.min():.2e} to {bad.lambda_min.max():.2e}; negative predicted variance in "
              f"{int((bad.pw_pred_var < 0).sum())}; max|w| of the unrepaired weights {bad.pw_max_abs_w.min():.2f}–{bad.pw_max_abs_w.max():.2f}.", "",
              "**Variance ratio against the full-sample covariance (median over non-PSD runs)**", "",
              md_table(pd.DataFrame([{"pairwise, unrepaired": bad.pw_var_ratio.median(), "clipped, relative floor": bad.clip_rel_var_ratio.median(),
                                      "clipped, minimal floor": bad.clip_min_var_ratio.median(), "complete-case": bad.cc_var_ratio.median()}]), "{:.3f}"), "",
              f"Repair cost ‖S_clipped − S‖_F / ‖S‖_F: relative floor {bad.clip_rel_frob_rel.min():.4f}–{bad.clip_rel_frob_rel.max():.4f}, "
              f"minimal floor {bad.clip_min_frob_rel.min():.4f}–{bad.clip_min_frob_rel.max():.4f}. "
              f"Median L1 change in weights (unrepaired → relative floor): {bad.clip_rel_dw_l1.median():.2f}.", "",
              f"Complete-case variance ratio is lower (better) than the relative-floor repair in "
              f"{int((bad.cc_var_ratio < bad.clip_rel_var_ratio).sum())} of {len(bad)} non-PSD runs.", ""]
    L += [f"Verification over all {len(df)} runs: repaired matrix positive definite (relative floor) in "
          f"{int((df.clip_rel_inertia_neg == 0).sum())}; minimum eigenvalue ≥ floor in {int(df.clip_rel_min_eig_ok.sum())}; "
          f"Frobenius distance equals the spectral distance in {int((df.clip_rel_frob_matches_spectrum & df.clip_min_frob_matches_spectrum).sum())}; "
          f"complete-case covariance positive definite in {int(df.cc_pd.sum())}.", ""]
    return L


def render(phase5_dir: Path, level: int = 2) -> str:
    """Markdown for all sections; ``level`` is the heading level of the section titles (2 = '## 5A. ...')."""
    d = Path(phase5_dir)
    body = ["Every number below was computed from the tables in `results/phase5/` (regenerate with "
            "`python run_all.py --phase5`). Risks are annualised (`std·sqrt(252)`, log returns); gaps are in percentage points.", ""]
    for f in (_sec5a, _sec5b, _sec5c, _sec5d, _sec5e, _sec5f):
        body += f(d)
    text = "\n".join(body)
    if level > 2:
        text = "\n".join(("#" * (level - 2) + ln) if ln.startswith("#") else ln for ln in text.splitlines())
    return text


def write(phase5_dir: Path | None = None) -> Path:
    """Write results/phase5/key_findings_phase5.md."""
    d = Path(phase5_dir or config.PHASE5_DIR)
    path = d / "key_findings_phase5.md"
    path.write_text("# Phase 5 key findings (auto-generated)\n\n" + render(d) + "\n")
    return path


def update_readme(readme: Path, phase5_dir: Path | None = None) -> None:
    """Replace the text between the PHASE5-RESULTS markers in README.md with freshly rendered results."""
    text = Path(readme).read_text()
    if START not in text or END not in text:
        raise ValueError("README.md has no PHASE5-RESULTS markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    block = render(Path(phase5_dir or config.PHASE5_DIR), level=3)   # sections become '### '
    Path(readme).write_text(head + START + "\n" + block + "\n" + END + tail)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    p = write(a.dir)
    print(f"wrote {p}")
    if a.update_readme:
        update_readme(config.ROOT / "README.md", a.dir)
        print("README.md results block refreshed")


if __name__ == "__main__":
    main()
