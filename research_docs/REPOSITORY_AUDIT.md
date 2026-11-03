# Repository audit (Stage A/B)

Audited 2026-11-02 against the working tree at branch `claude/portfolio-quadratic-forms-dvu76s`
(16 commits, clean tree at start, remote `TanayK-node/ALA_portfolio`). Nothing was committed, pushed or deleted.
Label key: **[V]** verified by running/recomputing, **[S]** suspected, **[U]** unresolved / needs the authors.

## 1. Headline findings

1. **[V] The repository reproduces only the development-sample parts of the paper.** `data/prices.csv` holds
   2021-10-08 to 2026-10-06 for the 30 Nifty 50 stocks. There is no 2011-2021 data and no
   `data/prices_nifty_next.csv`, so the historical test, universe B and A+B cannot be rerun (`run_all.py` refuses
   to download for non-default universes).
2. **[V] The paper's central method has no code in the repository.** A search of `src/`, `tests/`, the root scripts
   and the notebook finds no QLIKE loss, no corrected forecast `c(N,T)·w'Σ̂w` (except `theory.variance_ratio_of_expectations`,
   which computes the factor but is never applied to a forecast), no win-rate statistic, no nonlinear shrinkage (NL),
   no BPS weight shrinkage, no 1/N backtest, and no H = 20 run (`config.HORIZON = 60`). So Table II, Fig. 2b/2c,
   the 60-cell calibration surface, and the counts 347/348, 126/132, 276/219 cannot be traced to repository code.
   They were presumably produced by code that is not in this repo.
3. **[V] Phase 5 (5A-5G) is real, tested and partly reused by the paper**: common-OOS backtest, Gaussian/t/bootstrap Monte Carlo,
   the pre-specified predictor test, clipping/MP sweeps, bootstrap diagnostics, block-missingness, multi-universe support.
   Numbers quoted from it in the paper match the committed CSVs (see `CLAIM_TRACEABILITY.csv`).
4. **[V] Test suite: 116 passed, 1 failed** (`tests/test_phase5g.py::test_phase5_smoke`). The failure is in
   `src/plots.py:302` (`fig_theory_vs_empirical`): the error-bar half-widths `emp_std_ratio - emp_std_ratio_lo` can be
   negative when the sample mean falls outside its percentile bootstrap interval (possible with the 300-sim, few-period smoke
   configuration). It is a plotting-robustness defect, not a scientific one. Left unfixed in the audit; fixed minimally in
   the implementation stage and recorded in the ledger.
5. **[V] The cached development data contain 6 vendor-holiday rows with all-zero returns** (2026-01-15, 05-01, 05-28,
   06-26, 09-14, 10-02), so T = 1238 includes artificial zero-return days. A fresh download had 1233 rows. This is why
   2550 + 1 + 1238 ≠ 3784 (see MANUSCRIPT_AUDIT D1).

> **Update 2026-11-02 (later):** items 1 and 2 above are now addressed outside the original code: `data/prices_2011_2026.csv` (new Yahoo download) and `src/journal/` (QLIKE, corrected forecast, NL, BPS, 1/N, paired tests). See RESULTS_SUMMARY section 5.

## 2. File inventory

| File/path | Purpose | Inputs | Outputs | In paper? | Reusable? | Issues |
|---|---|---|---|---|---|---|
| `src/config.py` | all tunables, universes, seeds | - | constants | yes | yes | `END_DATE=date.today()` (only used on cache miss); `BOOT_BLOCK=3`, paper also uses 5 for H=20 (not in config) |
| `src/data.py` | download, cache, clean, log returns | `data/prices.csv` | returns | yes | yes | forward-fill (limit 3) turns missing prices into zero returns; vendor-holiday rows not removed |
| `src/covariance.py` | sample cov (ddof=1), pairwise cov, LW, masks | returns | matrices | yes | yes | LW via sklearn (benchmark only) |
| `src/linalg_tools.py` | inertia, clipping, MP filter, exposure score, pseudo-inverse | matrices | diagnostics | yes (C4) | yes | not line-audited in this pass **[U]** |
| `src/optimize.py` | closed-form GMV (`solve`), long-only SLSQP | S | weights | yes | yes | uses `solve`, weights normalised to sum 1 **[V]**; SLSQP failure logged but solver message/constraint residual not stored (spec 3.5) |
| `src/evaluate.py` | rolling backtest, common origins, summaries, exposure-vs-gap | returns | runs, tables | yes | yes | realised risk = std of H daily portfolio returns (sample std, ddof=1); predicted = sqrt(w'Σ̂w); annualised sqrt(252); log returns; no QLIKE |
| `src/theory.py` | Wishart/chi-square theory for Q and R | N,T | closed forms, exact-representation draws | yes | yes | docstring says tier 2 is "derived here, not verified against literature" while paper cites Kempf-Memmel **[U]** |
| `src/simulate.py` | Gaussian / Student-t / bootstrap GMV Monte Carlo with known Σ | Σ, N, W | per-sim variances | yes | yes | t-scaling `sqrt((df-2)/chi2)` keeps covariance = Σ **[V]**; only i.i.d. scenarios (no GARCH, regime, skew) |
| `src/bootstrap.py`, `src/uncertainty.py` | moving-block bootstrap, paired CIs, coverage check | loss series | CIs | yes | yes | percentile intervals only; mean can lie outside the CI (plot defect above); `uncertainty.py` not line-audited **[U]** |
| `src/predictors.py`, `src/multitest.py` | pre-specified predictor test, Holm | runs | verdicts | yes (C3) | yes | not line-audited **[U]** |
| `src/sweeps.py`, `src/block_missing.py` | clipping/MP sweeps; late-listing experiment | returns | tables | yes | yes | not line-audited **[U]** |
| `src/universes.py`, `compare_universes.py` | universe/period resolution | CSV paths | spec | partly | yes | depends on missing CSVs |
| `src/phase5.py`, `src/findings5.py`, `src/plots.py` | orchestration, auto-findings, figures | - | `results/phase5/*` | yes | yes | smoke-test failure above |
| `run_all.py`, `phase{1,2,3}_report.py` | pipeline entry points | - | `results/` | - | yes | `run_all.py --phase5` reproduces dev outputs only |
| `notebooks/main.ipynb` | demo (21 cells, 11 code) | - | figures | - | exploratory | no cell mentions QLIKE or a corrected forecast; not executed in this audit **[U]** |
| `tests/` (13 files) | unit/integration tests | - | - | - | yes | 116 pass, 1 fail |
| `results/`, `results/phase5/` | committed outputs | - | - | yes | frozen | preserved untouched |

## 3. Experiment inventory (what exists)

| Experiment | Where | Data | Status |
|---|---|---|---|
| 1. Full-sample inertia/conditioning | `diagnostics`, `results/full_sample_diagnostics.csv` | dev | exists; N=30, T=1238, cond 35.6, 4 eigenvalues above MP edge **[V]** |
| 2. Short-window / missing-data scenarios | `results/scenario_*.csv` | dev | exists |
| 3. Rolling backtest (per-window and common OOS), H=60 only | `evaluate`, `results/phase5/backtest_runs_common_oos.csv` | dev | exists; 8 origins at 750..1170 |
| 4. Gaussian/t/bootstrap MC (N=30 only) | `simulate`, `theory_mc_checks.csv` | synthetic from dev Σ | exists; 36/36 checks pass, max|z|=1.98, min KS p=0.067 **[V]** |
| 5. Predictor test (14 cells) | `predictors`, `predictor_verdicts.csv` | dev | exists; 0/14 supported **[V]** |
| 6. Clip / MP sweeps; bootstrap coverage; block-missingness | `sweeps`, `uncertainty`, `block_missing` | dev | exists; 450 runs, 11 non-PSD **[V]** |
| 7. QLIKE comparison, H=20, corrected forecast, NL/BPS/1/N | - | - | **absent** |
| 8. 60-cell (N,T) calibration surface, random subsets | - | - | **absent** (only N=30, six windows) |
| 9. Historical 2011-21 test, universe B, A+B | - | - | **absent (data missing)** |
| 10. Non-i.i.d. simulations (GARCH, regime, skew), residual calibration, regime analysis | - | - | **absent** |

## 4. Data audit (development file) **[V]**

| Check | Result |
|---|---|
| Shape / range | 1239 rows x 30 tickers, 2021-10-08 to 2026-10-06, strictly increasing, 0 duplicate dates |
| Non-positive prices | 0 |
| Missing cells | 1 (NESTLEIND.NS on 2026-10-02); 1 cell forward-filled |
| Calendar gaps | max 5 calendar days, one gap > 4 days |
| Extreme returns | max |r| = 18.4%, none above 20% |
| All-stock zero-return rows | 6 (listed above): vendor holidays repeating the prior close |
| Policy P1 (existing): ffill limit 3, complete rows | T = 1238 |
| Policy P2 (strict: no ffill, complete-case prices) | T = 1237 |
| Policy P3 (P1 minus all-zero rows) | T = 1232 |
| Effect on Σ̂ | cond(S): 35.61 under both P1 and P3; mean annual vol 23.95% (P1) vs 24.01% (P3) |

The zero rows barely change the covariance but they do change T and slightly lower realised variance in any block
that contains them, so they should be reported or removed (decision recorded in IMPLEMENTATION_PLAN D2).
Survivorship: the universe is today's constituents; no point-in-time data are available, so this stays a stated limitation.

## 5. Numerical-implementation checklist (spec 3.5)

| Item | Status |
|---|---|
| Linear solve rather than explicit inverse | **[V]** `optimize.min_variance_closed_form` |
| Weights sum to one | **[V]** `x / x.sum()` |
| Sample covariance `ddof=1` | **[V]** `np.cov` default, tested in `test_covariance.py` |
| Symmetrisation | **[V]** `symmetrize` before use |
| Pseudo-inverse usage | **[U]** in `linalg_tools` (spectral GMV for singular S); not used in the backtest path |
| SLSQP success flag | recorded as `converged`; message, objective, constraint residual **not** recorded (gap) |
| Weight-sum error, min eigenvalue, condition number, max |w| | condition number and max |w| recorded; others not |
| Raw vs repaired covariance kept separate | **[V]** estimators are separate columns/methods; block-missing experiment reports unrepaired and repaired |
| Annualisation | sqrt(252) on std, consistent in `evaluate` |

## 6. Not yet audited (declared, not hidden)

`linalg_tools.py`, `uncertainty.py`, `predictors.py`, `sweeps.py`, `block_missing.py`, `findings5.py`, most of `plots.py`, and
execution of `notebooks/main.ipynb`. These are reused as black boxes; their committed outputs are treated as the paper's evidence
only where the paper's number matched the CSV.
