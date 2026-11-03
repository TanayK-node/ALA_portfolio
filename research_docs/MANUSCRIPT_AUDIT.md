# Manuscript audit: `ALA_IEEE_paper_v6.pdf` (6 pages)

Label key: **[V]** verified, **[S]** suspected, **[P]** proposed, **[U]** unresolved.

## 1. Mathematical audit of Proposition 1 **[V]**

Independent derivation (not copied from the paper) and an independent Monte Carlo
(`research_docs/audit_scripts/audit_theorem.py`, does not import `src/theory.py` or `src/simulate.py`; random non-diagonal Σ,
arbitrary mean, 60 000 replications per cell):

* With A = Σ^{-1/2} Σ̂ Σ^{-1/2}, e = Σ^{-1/2}1, B = A^{-1}: Q = e'e/(e'Be), R = (e'e)(e'B²e)/(e'Be)². Rotating e to the first axis,
  Q = 1/B₁₁ (a Schur complement) and R = 1 + a₂₁'A₂₂⁻²a₂₁. Conditionally on A₂₂, a₂₁ ~ N(0, A₂₂/(T-1)), so the quadratic form is
  z'[(T-1)A₂₂]⁻¹z = χ²_{N-1}/χ²_{T-N+1} with independent numerator/denominator. Hence:
  (T-1)Q ~ χ²_{T-N}; R-1 ~ χ²_{N-1}/χ²_{T-N+1} = (N-1)/(T-N+1)·F_{N-1,T-N+1}; Q ⟂ R.
* E[Q] = (T-N)/(T-1), E[R] = (T-2)/(T-N-1) (needs T ≥ N+2), Var(R) finite iff T ≥ N+4. **All match the paper.**
* The paper's scaling and degrees of freedom are correct; the ratio is a χ² ratio, equal to an F variable times (N-1)/(T-N+1).
* Monte Carlo: |z| for E[Q] and E[R] ≤ 2.22 over ten tests (one at 2.22, N=10, T=60, p≈0.03, within what ten tests produce);
  KS p-values 0.09-0.98 for both exact distributions in all five (N,T) cells; corr(Q,R) between -0.003 and +0.007.
* Scope is stated correctly: raw unbiased Σ̂, unconstrained GMV, i.i.d. Gaussian. The paper does not extend it to other estimators.

### Findings that matter for the journal paper

* **M1 [V] c(N,T) is a ratio-of-expectations correction; per-forecast ratios are still biased.** Because R ⟂ Q,
  E[R/(cQ)] = (T-N)/(T-N-2) > 1: **+7.1% at (N,T)=(30,60)**, +2.2% at (30,120), +0.9% at (30,250) (simulation matches).
  The QLIKE-optimal constant multiplier (minimising E[R·G/(aQ)] + log a) is **E[R/Q] = (T-1)(T-2)/[(T-N-1)(T-N-2)]**, not c:
  4.214 vs 3.933 at (30,60). Numerically the *median* of R/(cQ) is ≈ 1.003, so c is nearly median-calibrated.
  The abstract says "unbiased ... for the true variance of the estimated portfolio", which is correct, but the paper then judges
  the forecast with QLIKE and variance ratios. Part of the "residual miscalibration" (up to 24%) must therefore be compared
  with the right Gaussian benchmark. The repo's C2 comparison (`theory.expected_std_ratio`, 5B) includes the test-block noise G and the Jensen
  effect, which is the like-with-like target for the *raw* forecast. Whether Table II's corrected-forecast variance ratios (1.09-1.49) were compared
  with the matching benchmark (≈ (T-N)/(T-N-2) above 1 under Gaussian data) cannot be checked, because that code is absent. The journal version should (a) state which target is used,
  (b) report c and c_QLIKE side by side, and (c) not describe c as "calibrated" without this qualification. **This is a candidate new contribution.**
* **M2 [U] Novelty wording.** `src/theory.py` says tier 2 was "derived here, not verified against the literature", but the paper says the mean
  of (ii) "equals the expected out-of-sample GMV variance of Kempf and Memmel". These must be reconciled by reading Kempf-Memmel (2006),
  Okhrin-Schmid (2006), Bodnar-Schmid (2008/2009) and Frahm-Memmel (2010). The F-type representation of R may already be in that
  literature; the journal paper should claim only what a reading confirms.
* **M3 [V] Conditional vs unconditional accuracy** is stated in the paper ("for a given Σ̂ the corrected forecast can be too high or too low").
  Fine. Add the explicit variance of cQ-R, which the exact representation gives for T ≥ N+4 **[P]**.
* **M4.** The theorem cannot be applied to long-only, shrinkage, clipped, factor or robust estimators. The paper respects this
  (it evaluates raw-only correction against others' forecasts). Keep this separation in every new table **[P]**.

## 2. Numerical reconciliation of reported counts **[V]**

The counting unit is (sample, horizon, window, alternative) with six alternatives (raw, clipped, MP, LW, NL, BPS) and two horizons:

| Paper count | Reconstruction | Result |
|---|---|---|
| 348 comparisons | 5 samples; windows 6 each except A+B with W > 1.5N = 85.5 giving 5; (6·4+5)·2·6 | **348** |
| 276 (four extended samples) | (6+6+6+5)·2·6 | **276** |
| 132 (W ≤ 120, extended) | (3+3+3+2)·2·6 | **132** |
| 48 (W = 250) | 4·2·6 | **48** |
| 72 (development) | 6·2·6 | **72** |
| Periods | H=20: 24 / 90 / 151; H=60: 8 / 30 / 50 | reproduced from `range(750, T-H+1, H)` |

So the denominators are internally consistent. The *numerators* (347, 219, 126, 11 of 48, 8 of 72, 5 losses) cannot be recomputed here because
the data and the QLIKE/NL/BPS code are absent. **Dependence caveat [S]:** the 348 units are not independent (A 2011-26 = A hist + dev;
A+B contains A and B; all windows share test periods; two horizons overlap). The paper already says so for the surface cells; it should
say it for these counts too, and the journal version should report a single master experiment table from which all counts are computed (spec 3.3).

## 3. Discrepancies and risks

| ID | Paper statement | Evidence | Action |
|---|---|---|---|
| D1 | T = 1238 (dev), T = 3784 (extended), T = 2550 (hist) | cached dev file has 6 all-zero holiday rows; a fresh download gave 1233, and 2550 + 1 + 1233 = 3784 exactly | report dev T = 1233 or add the footnote drafted earlier; rerun dev results if rows are removed |
| D2 | "universe A (the same 30 stocks)" in 2011-26 | RESOLVED 2026-11-02: a new Yahoo download (`data/prices_2011_2026.csv`) gives A = 30, B = 27 (ICICIGI, INDIGO, HAL dropped), A+B = 57 and T = 2550 / 3784 exactly. Yahoo is flaky: an earlier download (2026-10-29) lost TECHM.NS and one bulk call lost VEDL.NS (a single-ticker retry worked) | ship the price CSVs with the code; add a download check that asserts N and T |
| D3 | block length 5 (H=20) or 3 (H=60) | `config.BOOT_BLOCK = 3`; sensitivity set is 1-4 | add 5 and the selection rule to config |
| D4 | "overlapping prices of the two downloads agree within 1e-6" | holds on shared dates (max diff 7.8e-7), cannot see extra dates | say "on common dates" |
| D5 | Table I "This work: 8 estimators" | repo implements raw, clipped, MP, LW only; NL, BPS, 1/N absent | provide the code or reduce the claim |
| D6 | Student-t ratio "2.458 vs 2.018 Gaussian" | `theory_vs_empirical.csv`: t = 2.458, Gaussian MC 2.015, theory 2.018 **[V]** | none |
| D7 | Repository commit/time claims (dc9485e at 15:50 UTC, 139f182 at 15:53) | `git log` shows both commits at those timestamps **[V]** | keep the "self-reported timestamps" caveat |

## 4. Claims that rest on the single development sample and prior exploration

The 2011-2021 period was used after the method was fixed, but it was downloaded and analysed within the same research project and
now also appears inside the 2011-26 samples. The journal version must call it a *historical test on surviving stocks*, never an untouched holdout
(spec 6.2). Statements about "robustness" combine overlapping data and should not be read as independent replications.

## 5. Citation notes

Reference-list metadata were checked against Crossref/arXiv earlier in this conversation; three entries still need the fixes listed there
([3] page range, [36] issue number, [15] article-number style). The journal version needs a fresh literature pass (spec 14.1), in particular
Kempf-Memmel, Frahm-Memmel, Bodnar-Schmid and Okhrin-Schmid for M2, and risk-forecast-calibration papers (Patton; Engle-Sheppard; Diebold-Mariano).
