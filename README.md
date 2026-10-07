# Portfolio risk as a quadratic form
**Why Markowitz fails, and how Σ's eigenstructure explains it** — Advanced Linear Algebra, Module 2
(quadratic forms, congruent transformations, Sylvester's law of inertia, orthogonal diagonalization).

Portfolio variance is the quadratic form `wᵀΣw`. Writing `Σ = QΛQᵀ` and `y = Qᵀw` gives the normal form
`wᵀΣw = Σᵢ λᵢ yᵢ²`. The minimum-variance weights `w ∝ Σ⁻¹1 = Σᵢ λᵢ⁻¹ qᵢ(qᵢᵀ1)` amplify the directions with the
*smallest* eigenvalues — the ones estimated worst from short or noisy data. We use eigen-decomposition, inertia
and definiteness to **detect** that (condition number, inertia, noise-exposure score) and **fix** it (eigenvalue
clipping, Marchenko-Pastur filtering, benchmarked against Ledoit-Wolf).

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, yfinance, scikit-learn, pytest
python -m pytest -q                      # unit tests
python run_all.py                        # full pipeline (~10 s with cached prices), writes results/
```

* Prices are downloaded once and cached in `data/prices.csv`; later runs reload the cache and never re-download.
  Delete the file to force a refresh (the date range is then recomputed from today's date, so numbers will change).
  `data/prices.csv` is committed so the project reproduces without network access.
* `notebooks/main.ipynb` is a thin walk-through that calls `src/` (needs `pip install -r requirements-dev.txt`
  to re-execute; it is committed with outputs).
* Phase scripts `phase1_report.py`, `phase2_report.py`, `phase3_report.py` print the intermediate diagnostics.

## Layout

```
src/config.py        every tunable: tickers, dates, windows, tolerances, seeds
src/data.py          download, cache, clean (drop sparse tickers, forward-fill ≤3 days), log returns
src/linalg_tools.py  the linear algebra, written by us (only np.linalg.eigh / cholesky are solvers)
src/covariance.py    sample / pairwise-complete / Ledoit-Wolf (sklearn, benchmark only) + broken-Σ scenarios
src/optimize.py      min-variance (closed form via solve, long-only SLSQP), efficient frontier
src/evaluate.py      rolling backtest, summary table, exposure-vs-gap hypothesis test
src/diagnostics.py   full-sample + short-window reports, auto-generated results/key_findings.md
src/plots.py         all figures
tests/               pytest (linear algebra properties, optimizers, backtest, scenarios)
```

`linalg_tools.py` contains: `eig_decompose`, `inertia`, `is_positive_definite` (attempted Cholesky), `is_psd`,
`condition_number`, `normal_form_transform`, `verify_sylvester`, `clip_eigenvalues`, `mp_noise_edge`,
`clip_by_mp`, `min_variance_weights_spectral`, `noise_exposure_score`. No black-box "nearest PSD" routine is used.

## Outputs (`results/`)

| File | Meaning |
|---|---|
| `key_findings.md` | Auto-generated summary; **every number is computed in the run**. Start here. |
| `full_sample_diagnostics.csv` | Eigenvalue summary, inertia, condition number, PD check of the full-sample Σ |
| `scenario_short_window.csv` | Broken-Σ (a): sample covariance from T rows vs N; inertia, Cholesky PD flag, `solve` vs spectral weights |
| `scenario_missing_data.csv` | Broken-Σ (b): random masking + pairwise-complete Σ; inertia and λ_min per (mask, seed). Measured, not assumed to be non-PSD |
| `backtest_runs.csv` | One row per (window, origin, method, constraint): predicted vs realised risk, risk gap, max‖w‖∞, turnover, κ, inertia, exposure scores, Sharpe |
| `summary_table.csv` | Mean/median of the metrics by constraint × method × window |
| `exposure_vs_gap.csv` | Spearman ρ between noise-exposure score and risk gap (raw Σ): pooled (`all`), `within`-window, and per window |
| `fig1_eigenvalue_spectrum.png` | Eigenvalue spectrum (log): raw vs clipped vs MP-filtered, MP edge marked |
| `fig2_efficient_frontier.png` | Long-only frontier for raw vs repaired Σ; dots = same portfolios out-of-sample |
| `fig3_weight_comparison.png` | Min-variance weights, raw vs repaired Σ |
| `fig4_risk_contribution.png` | Risk share λᵢyᵢ² per principal direction of the raw min-variance portfolio |
| `fig5_predicted_vs_realized.png` | Predicted vs realised risk by window and method (mean over runs) |
| `fig6_exposure_vs_gap_*.png` | Exposure score vs risk gap, pooled and within-window ranks, ρ in the titles |
| `fig7_condition_number.png` | Condition number vs window length (log y) |

Figures 1–4 illustrate the mechanism on **one** training window (`DEMO_WINDOW`, the last non-overlapping origin);
the aggregate evidence is figures 5–7 and the tables.

## Design choices and defaults (all in `config.py`)

| Choice | Default | Why |
|---|---|---|
| Inertia tolerance | `|λ| ≤ 1e-10·max|λ|` counts as zero | relative, so it is scale-free (daily covariances are ~1e-4) |
| Pseudo-inverse rank cutoff | `λ > 1e-10·λmax` | same idea; singular Σ gives the min-variance portfolio *within the retained subspace* |
| Clipping floor in the backtest | `eps = 0.10 · tr(Σ)/N` (relative) | an absolute 1e-6 is below every sample eigenvalue once T > N and would make "clipped" ≡ "raw". Chosen a priori, **not tuned** |
| MP noise variance | `σ² = tr(Σ)/N` | standard but slightly conservative; eigenvalues below the edge are replaced by their mean (trace-preserving) |
| Annualisation | `std·√252`, mean·252, log returns | applied identically to predicted and realised risk |
| Backtest roll | train `W` days, test next `H=60`, step `60` | non-overlapping out-of-sample blocks |
| Weights in test | held fixed for `H` days | no drift or rebalancing; turnover is `‖wₜ − wₜ₋₁‖₁` between consecutive origins |
| Broken-Σ (b) | masks 5/10/20 %, 20 seeds each | reported as measured |
| Seed | 42 | global seed plus explicit `default_rng(seed)` everywhere |

## Honest limitations

* **One universe, one period.** 30 large-cap Nifty stocks over ~5 years; nothing here is evidence about other markets or regimes.
* **Different out-of-sample periods per window length.** Origins start at `W`, so W=750 only tests the later part of the
  sample while W=60 covers all of it. Cross-window comparisons — and especially Sharpe ratios — are confounded by period.
  Sharpe is reported but min-variance portfolios are not designed to maximise it.
* **Runs are not independent.** Training windows overlap whenever `W > 60`; the reported p-values are optimistic, and the
  hypothesis table contains many unadjusted tests (a few small p-values are expected by chance alone).
* **Realised risk is noisy.** It is the standard deviation of only 60 daily returns.
* **The exposure-score hypothesis.** The pooled correlation across windows is confounded by window length (short windows
  raise both the score and the gap). We therefore also report a `within`-window version, **added after seeing the pooled
  result**; the original pooled and per-window rows are kept. Read `key_findings.md` for what the data say.
* **Missing-data scenario.** In this universe with this many observations, random masking did not make the
  pairwise-complete covariance non-PSD in our runs (see `scenario_missing_data.csv`). Indefiniteness would need far fewer
  observations or non-random missingness; we did not tune the masks to force it. The Sylvester demo in the notebook
  therefore uses a *constructed* indefinite matrix (Σ − cI), clearly labelled as such.
* **Platform dependence.** For *exactly singular* Σ, whether `np.linalg.solve` raises or returns rounding-noise weights
  depends on the BLAS/LAPACK build (it differed between Windows and Linux in our runs). Inertia and the spectral weights
  are stable; treat the `max_abs_w_solve` column of `scenario_short_window.csv` for singular rows as illustrative. All
  backtest windows (W ≥ 60 > N = 30) are non-singular and unaffected.
* **Eigenvalue clipping** with the chosen floor is a no-op when no eigenvalue is below it (typically long windows), so it
  coincides with raw there. A floor sensitivity sweep is not included.
* **Log-return approximation.** Portfolio returns are `w·r` with log returns r, a standard daily approximation.
* **No transaction costs, no short-sale costs.** The unconstrained portfolios take large short positions.
