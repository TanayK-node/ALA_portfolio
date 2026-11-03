# Limitations, failures and open items

* **Data (blocker B1, partly resolved 2026-11-02):** `data/prices_2011_2026.csv` was re-downloaded from Yahoo and reproduces N = 30 / 27 / 57 and T = 2550 / 3784, but it is not the authors' original file; vendor adjustments can drift, so numbers may differ slightly from the conference paper. Original text:  2011-2021 prices, universe B (27 Nifty Next 50 stocks) and A+B are not in the repository; every claim about them is marked not reproducible.
  A fresh Yahoo download in this project did not contain TECHM.NS, so the paper's "same 30 stocks" for 2011-26 may not be reproducible from vendor data today.
* **Code (blocker B2, mostly resolved):** NL, BPS, 1/N, win rate and the paired tests were re-implemented in `src/journal/`. NL reproduces the paper's numbers to 3 decimals; BPS is a finite-sample analogue of Bodnar-Parolya-Schmid, not their formula, and differs by up to 0.03 QLIKE at W<=90. The 60-cell surface code is still absent.
* **Dev data quality:** 6 vendor-holiday rows with zero returns in a T = 1238 sample; the conference results include them. Sensitivity (P3) is reported; the primary analysis keeps the existing policy.
* **Inference:** 24 and 8 development periods; bootstrap p-values are heuristic (sign test on the centred bootstrap distribution); block length rule n^(1/3) was fixed before the runs but is not the conference's block length 5.
* **Simulation scope:** only i.i.d. Gaussian and Student-t with a fixed covariance and three covariance structures; no volatility clustering, regime switching or skewness yet. The future-block target uses H = 60 only.
* **Novelty:** the F-type representation of R and its relation to Kempf-Memmel / Bodnar-Schmid is unresolved (M2); do not claim it as new until read.
* **Failed/changed:** `test_phase5_smoke` failed at the start (plot error bars); fixed with a one-line clamp in `src/plots.py`. The first draft of `figures.py` used `DataFrame.T` instead of the column "T" and failed; fixed.
* **Not independently re-executed:** notebook, `linalg_tools`, `uncertainty`, `predictors`, `sweeps`, `block_missing` internals (REPOSITORY_AUDIT s6).

* **Dynamic simulations (C2):** the GARCH and regime generators keep a fixed correlation structure and move only a common volatility scale, with illustrative parameters (not fitted to data); no skewed generator, no dynamic correlations, no interval-coverage experiment.
* **Surface (C3):** the paper does not say how its random subsets were drawn, so cell values differ slightly from the paper; development inference rests on 8 origins.
* **Regimes (C3):** vol terciles are causal; the COVID flag and the large-fall flag use the test block (calendar window fixed a priori; large-fall is ex-post and diagnostic only). Development and COVID cells have too few origins for inference.
* **Second stage (C4):** failed its pre-specified criteria (see RESULTS_SUMMARY 9). The numeric thresholds are the authors' choices written before the run, not from the specification; cells are dependent, so the shares are descriptive. Only one family of features and one regularisation (lambda 5) was pre-specified; lambda 25 looked better but was not selected.
* **Predictor test:** the test itself reproduces; the power analysis is my re-implementation (not the paper's script) and differs by up to 0.12 (s1 ideal 0.60 vs 0.48). lambda_min/mean(lambda) for the long-only portfolio meets the rule but is not discussed in the paper.
* **Not started:** the manuscript draft and the Springer guideline check, the skewed generator, interval coverage.
