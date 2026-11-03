# Implementation plan for the FMPM journal extension

Basis: `REPOSITORY_AUDIT.md`, `MANUSCRIPT_AUDIT.md`, `CLAIM_TRACEABILITY.csv`. Principle: preserve `src/`, `results/`, `results/phase5/`
untouched (one minimal bug fix, listed below); add a separate `src/journal/` package and `results/journal_extension/`.
Nothing is committed or pushed.

## 1. Decisions and blockers (need the authors)

| ID | Question | Why it blocks | Default if unanswered |
|---|---|---|---|
| B1 (PARTLY RESOLVED: `data/prices_2011_2026.csv` downloaded from Yahoo on 2026-11-02, reproduces N and T of the paper; the authors' original files were not supplied) | Where are the 2011-2021 prices and the Nifty Next 50 prices used in the paper? (`data/prices_nifty_next.csv`, 2011-26 file for universe A) | Historical test, B, A+B, 60-cell surface, 347/348 counts cannot be reproduced; the spec forbids substituting new data | Work on the development file only and mark everything else "not run: data missing" |
| B2 | Where is the code that produced Table II (QLIKE, corrected forecast, NL, BPS, 1/N, win rate, H=20)? | Without it the baseline cannot be reconciled | Re-implement from the paper's definitions in `src/journal/` and label it a re-implementation, not a reproduction |
| D1 | Dev cleaning policy: keep T=1238, or drop the 6 vendor-holiday rows (T=1232) | Changes T and all dev tables slightly | Report both as a sensitivity (spec 3.6); primary = existing policy so that conference results are unchanged |
| D2 | Calibration target for the journal paper: c (unbiased level), c_QLIKE = E[R/Q], or median-ratio | M1 shows they differ by up to 7% at (30,60) | c is primary (it is the conference method); c_QLIKE and the empirical constant multiplier are secondary baselines |
| D3 | Reconcile Kempf-Memmel / Bodnar-Schmid with Prop 1(ii) | Novelty statement | Claim only the application to forecast calibration until read |

## 2. Stages

**Stage C1 (this session): foundations that need no new data.** `src/journal/` with
`metrics.py` (QLIKE, calibration ratio/bias/MACE, realised-variance proxies), `forecast.py` (c, c_QLIKE, corrected forecast, rolling-origin
engine with a leakage-safe interface, strictly positive checks), `scenarios.py` (Gaussian, Student-t with fixed covariance, optional
stationary mixture), `bootstrap_ext.py` (moving-block and stationary bootstrap, paired differences), `runmeta.py` (experiment IDs, git SHA, data hash,
package versions, seeds), `configs/*.yaml`, tests (QLIKE, c, no-look-ahead, seed reproducibility, scenario covariance), the single bug fix in
`src/plots.py`, a Gaussian + Student-t validation Monte Carlo over the (N, T) grid, and a development-sample baseline
(raw / clipped / MP / LW with plug-in and corrected forecasts, H = 20 and 60) with paired bootstrap comparisons.

**Stage C2: simulation extension.** GARCH-type volatility clustering (constant-correlation), two-state regime switching, skewed generator only
if its covariance and dependence can be validated (otherwise optional/not run), N/T surface with fixed seeds, Monte Carlo SEs.

**Stage C3: empirical extension on the development data (and on all data once B1 is answered).** (N,T) random-subset calibration surface with
clustered inference, causal volatility regimes (thresholds from a training period), calibration vs portfolio-performance table, data-policy sensitivity.

**Stage C4: second-stage calibration, only if C2/C3 show stable residual error.** Prequential, matured-label-only constant multiplier and
regularised log-linear model; must beat the constant multiplier out of sample or be reported as a negative result.

**Stage D: reporting.** `EXPERIMENT_LEDGER.md`, `RESULTS_SUMMARY.md`, `LIMITATIONS_AND_FAILED_EXPERIMENTS.md`, journal manuscript draft,
conference-extension disclosure, FMPM guideline check against the live Springer page, and the list of references needing verification.

## 3. What is reused and what is new

Reused as-is: `data.py`, `covariance.py`, `optimize.py`, `linalg_tools.py`, `evaluate.run_backtest*`, `bootstrap.py`, `theory.py`,
`simulate.py` (Gaussian/t/bootstrap), Phase 5 outputs. New: QLIKE and calibration metrics, the corrected-forecast engine, paired
inference for forecast losses, non-i.i.d. scenarios, surface, regimes, second stage. Not duplicated: the 36 Gaussian MC checks, predictor test,
sweeps, block-missingness.

## 4. Pre-specified primary analyses (frozen before the new runs)

1. Gaussian validation: E[Q], E[R], mean cQ·V ratio to E[R]; accepted if |z| ≤ 3 and KS p ≥ 0.01 per cell (same tolerances as 5B).
2. Primary empirical comparison: mean QLIKE difference, corrected-raw vs each alternative's own forecast, same origins, non-overlapping blocks
   (stride = H), block bootstrap with the length rule L = max(1, round(n^{1/3})) fixed here; Holm over the family of (window, alternative) cells
   within a horizon. Everything else is labelled secondary/exploratory.
3. No tuning on the development period beyond what the conference paper already did; hyperparameters are fixed in `configs/`.

## 5. Risks

Few development periods (24 at H=20, 8 at H=60) give weak inference; coverage of the bootstrap at these lengths is about 79% (paper).
Dev window overlaps with the conference analysis, so it is a transparent re-analysis, not an out-of-sample test.
Long simulations are run with chunking and fixed seeds; failures are logged, never dropped.

## 6. Status (2026-11-03)
C1 done; extended samples reproduced (NL exact, BPS approximate); C2 (GARCH, regime) done; C3 (surface, regimes) done; skewed generator and interval coverage not done;
C4 done (negative result); the Table III predictor test is reproduced; Stage D manuscript, the skewed generator and interval coverage remain.

## 7. Stage D (2026-11-03)
Draft in `manuscript/MANUSCRIPT_DRAFT.md`. FMPM guidelines NOT verified (Springer page blocked by cookie redirect). Fresh literature pass, skewed generator, interval coverage, figure review remain.

## 8. Stage D follow-up (2026-11-03)
Done: BPS verbatim check (Results 11), figure review + fixes + `figures` stage, interval-coverage experiment (Results 12), partial literature pass (Kan-Smith 2008, Dickinson 1974 pre-empt Prop 1 novelty). Not done: skewed generator, full-text reading of Kempf-Memmel/Kan-Smith, FMPM guidelines check.
