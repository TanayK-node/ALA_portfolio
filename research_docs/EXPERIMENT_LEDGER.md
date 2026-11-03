# Experiment ledger (journal extension)

All outputs under `results/journal_extension/`; metadata JSON in `metadata/`. Code state: commit `d8437d3` plus the uncommitted working tree
(nothing committed or pushed). Python 3.13, numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.9.0.

| ID | Experiment | Config | Seed | Output | Status |
|---|---|---|---|---|---|
| J-AUDIT-THM | independent derivation + Monte Carlo of Prop. 1 (5 (N,T) cells, 60k reps, random Σ, arbitrary mean) | `research_docs/audit_scripts/audit_theorem.py` | 20261102 | printed table, summarised in MANUSCRIPT_AUDIT s1 | success |
| J-BASELINE-DEV | dev-sample rolling forecasts, H = 20 and 60, raw/clipped/MP/LW plug-in + raw corrected (c) and corrected (c_QLIKE); paired tests; data policies P1 (existing) and P3 (drop 6 zero rows); proxy `sample` (primary) and `mean_square` (P1 only) | `configs/journal_baseline_dev.yaml` | 42 | `empirical/forecasts_*.csv`, `calibration_summary.csv`, `paired_tests.csv` | success; 0 failed solves |
| J-VALIDATE-MC | Gaussian / Student-t (df 5, 8, 15) Monte Carlo, N ∈ {5..50}, T ∈ {60..500}, 3 covariance structures, 4000 reps/cell, T ≥ N+5 | `configs/journal_validation_mc.yaml` | 20261102 (SeedSequence per cell) | `simulations/gaussian_t_validation.csv` (360 cells), 3 figures | success; failure rate 0 in all cells |
| J-UNIT | `tests/test_journal.py` (12), `tests/test_journal_estimators.py` (7), `tests/test_journal_dynamic.py` (5), `tests/test_journal_second_stage.py` (5) plus the 117 existing tests | - | - | pytest | 146 passed (the earlier plotting failure was fixed in `src/plots.py`) |

| J-EXTENDED-5SAMPLES | five conference samples (dev from cached file; hist A, ext A, ext B, ext A+B from `data/prices_2011_2026.csv`), H = 20 and 60, raw/clipped/MP/LW/NL/BPS/1-N plus corrected forecast; paired block bootstrap with the conference's block lengths (5 for H=20, 3 for H=60), B = 2000 | `run_journal.py extended` | 42 | `extended/forecasts_*.csv`, `table2_all_rows.csv`, `paired_tests_extended.csv`, `research_docs/audit_scripts/paper_table2_reference.csv` (hand-typed from the PDF, for comparison only) | success; 0 failed solves |

| J-DYNAMIC-MC | common-volatility GARCH(1,1) (a=0.10, b=0.85) and two-state volatility-regime process (p_stay 0.99/0.95, stress scale 2.5), unconditional covariance = Sigma; N in {10,30}, T in {60,120,250}, 2 structures, 3000 reps; origin-conditional breakdowns | `run_journal.py dynamic` | 20261103 | `simulations/dynamic_validation.csv` | success |
| J-SURFACE | (N,T) calibration surface, 20 seeded random subsets per N, 6 windows, H=60, dev (N 10..30) and 2011-26 A+B (N 10..50), origin-block bootstrap (block 3, B=1000) | `run_journal.py surface` | 42 | `empirical/calibration_surface.csv`, `calibration_surface_fits.csv` | success |
| J-REGIMES | causal volatility terciles (thresholds from rows < 750), COVID calendar flag, ex-post large-fall flag (diagnostic only); calibration and paired corrected-vs-NL by regime, min n 12 | `run_journal.py regimes` | 42 | `empirical/regime_calibration.csv`, `regime_labels_*.csv` | success; dev cells inadequate (n 6-10) |

| J-SECOND-STAGE | prequential second-stage calibration (Gaussian, constant, tercile, ridge; ridge lambda 1/5/25 as sensitivity) on matured labels only; criteria C1-C7 fixed in `configs/journal_second_stage.yaml` before the first run; controls on simulated iid / GARCH / regime paths with the dev covariance | `run_journal.py second_stage` | 42 (controls 20261104) | `second_stage/second_stage_summary.csv`, `second_stage_verdict.csv`, `second_stage_controls.csv`, `multipliers_*.csv`, `figures/fig_second_stage.png` | success; no model meets the criteria |

| J-PREDICTORS-HIST | conference pre-specified predictor test (7 predictors x 2 optimizers, non-overlapping and stride-60 designs, Holm, pre-registered rule) on 2011-21 universe A via `src/predictors`; volatility control; Gaussian signal-in-principle (2000 pairs per window) and power (1000 reps, window-specific effects) re-implemented | `run_journal.py predictors` | 42 | `predictors/predictor_tests_hist_A.csv`, `predictor_verdicts_hist_A.csv`, `volatility_control_hist_A.csv`, `signal_by_window_hist_A.csv`, `signal_and_power_hist_A.csv` | success |

Not run (planned stages): skewed generator, interval coverage, stationary-bootstrap sensitivity, historical-sample predictor test and the late-listing/MC items that need the authors' original files.
Seed reproducibility is covered by tests for the bootstrap and by SeedSequence-keyed cells; a full second run of the Monte Carlo to confirm bit-identical output was not done.
