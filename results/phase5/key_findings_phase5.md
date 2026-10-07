# Phase 5 key findings (auto-generated)

Every number below was computed from the tables in `results/phase5/` (regenerate with `python run_all.py --phase5`). Risks are annualised (`std·sqrt(252)`, log returns); gaps are in percentage points.

## 5A. Common out-of-sample periods

8 test blocks of 60 days, 2024-10-24 to 2026-09-24; every window length W is trained on the W days before each of the same origins (first origin = the largest window, 750).

**unconstrained: mean risk gap (realised − predicted), annualised, percentage points**

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 8.76 | 6.90 | 4.25 | 3.84 |
| 90 | 5.64 | 5.51 | 4.08 | 3.64 |
| 120 | 3.52 | 3.48 | 3.17 | 2.40 |
| 250 | 1.63 | 1.63 | 2.64 | 1.46 |
| 500 | 1.24 | 1.24 | 2.25 | 1.29 |
| 750 | 1.03 | 1.03 | 2.17 | 1.08 |

Raw 95% moving-block-bootstrap interval for the mean gap: W=60: [6.13, 11.66]; W=90: [3.40, 8.90]; W=120: [1.54, 6.19]; W=250: [-0.14, 3.20]; W=500: [-0.26, 2.58]; W=750: [-0.23, 2.50]

**unconstrained: mean realised risk, annualised, %**

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 15.36 | 13.90 | 11.52 | 11.51 |
| 90 | 13.15 | 13.05 | 11.80 | 11.63 |
| 120 | 12.18 | 12.14 | 11.52 | 11.21 |
| 250 | 11.27 | 11.27 | 11.67 | 11.00 |
| 500 | 11.05 | 11.05 | 11.38 | 10.98 |
| 750 | 11.11 | 11.11 | 11.59 | 11.07 |

**long_only: mean risk gap (realised − predicted), annualised, percentage points**

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 3.00 | 2.94 | 3.30 | 2.95 |
| 90 | 2.34 | 2.34 | 2.84 | 2.37 |
| 120 | 1.30 | 1.30 | 2.00 | 1.50 |
| 250 | 0.63 | 0.63 | 1.53 | 0.87 |
| 500 | 0.91 | 0.91 | 1.72 | 1.07 |
| 750 | 0.83 | 0.83 | 1.67 | 0.96 |

Raw 95% moving-block-bootstrap interval for the mean gap: W=60: [1.60, 5.36]; W=90: [0.03, 5.31]; W=120: [-0.86, 3.91]; W=250: [-1.00, 2.17]; W=500: [-0.58, 2.31]; W=750: [-0.45, 2.35]

**long_only: mean realised risk, annualised, %**

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 11.71 | 11.68 | 11.39 | 11.30 |
| 90 | 11.45 | 11.45 | 11.47 | 11.17 |
| 120 | 11.10 | 11.10 | 11.35 | 10.93 |
| 250 | 10.94 | 10.94 | 11.41 | 10.90 |
| 500 | 11.09 | 11.09 | 11.46 | 11.06 |
| 750 | 11.23 | 11.23 | 11.65 | 11.21 |

**Per-window OOS design (old) vs common OOS (new): mean risk gap, pts**

unconstrained (old → new)

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 8.30 → 8.76 | 6.42 → 6.90 | 4.13 → 4.25 | 3.78 → 3.84 |
| 90 | 5.14 → 5.64 | 4.99 → 5.51 | 3.54 → 4.08 | 3.15 → 3.64 |
| 120 | 3.70 → 3.52 | 3.68 → 3.48 | 3.04 → 3.17 | 2.41 → 2.40 |
| 250 | 1.34 → 1.63 | 1.34 → 1.63 | 1.96 → 2.64 | 1.17 → 1.46 |
| 500 | 1.29 → 1.24 | 1.29 → 1.24 | 2.40 → 2.25 | 1.35 → 1.29 |
| 750 | 1.03 → 1.03 | 1.03 → 1.03 | 2.17 → 2.17 | 1.08 → 1.08 |

long_only (old → new)

| window | raw | clipped | mp | ledoit_wolf |
|---|---|---|---|---|
| 60 | 2.88 → 3.00 | 2.83 → 2.94 | 3.14 → 3.30 | 2.90 → 2.95 |
| 90 | 2.25 → 2.34 | 2.25 → 2.34 | 2.55 → 2.84 | 2.31 → 2.37 |
| 120 | 1.55 → 1.30 | 1.54 → 1.30 | 1.94 → 2.00 | 1.67 → 1.50 |
| 250 | 0.48 → 0.63 | 0.48 → 0.63 | 1.05 → 1.53 | 0.68 → 0.87 |
| 500 | 1.05 → 0.91 | 1.05 → 0.91 | 1.93 → 1.72 | 1.22 → 1.07 |
| 750 | 0.83 → 0.83 | 0.83 → 0.83 | 1.67 → 1.67 | 0.96 → 0.96 |

Largest absolute change in mean gap between the two designs: 0.68 pts (unconstrained, mp, W=250).

## 5B. Theory overlay (unconstrained raw GMV, i.i.d. Gaussian)

Tier 1 (exact, Wishart): in-sample variance ratio `(T-1)Q ~ chi2_{T-N}`. Tier 2 (derived in `src/theory.py`, **not verified against the literature**): out-of-sample variance ratio `R-1 ~ chi2_{N-1}/chi2_{T-N+1}`. Monte Carlo: N = 30, 10000 simulations per window, true Σ = shrunk empirical Σ (shrinkage 0.5). Pre-stated tolerance: |MC − theory| ≤ 3 standard errors; KS mismatch flagged at p < 0.01.

Checks within tolerance: 36 of 36; max |z| = 1.98; min KS p-value = 0.067.

| T | E[pred]/V theory | MC | E[true]/V theory | MC  | E[true/pred] theory | MC   |
|---|---|---|---|---|---|---|
| 60 | 0.5085 | 0.5087±0.0013 | 2.0000 | 2.0007±0.0038 | 4.214 | 4.213±0.014 |
| 90 | 0.6742 | 0.6739±0.0013 | 1.4915 | 1.4929±0.0016 | 2.289 | 2.294±0.005 |
| 120 | 0.7563 | 0.7566±0.0011 | 1.3258 | 1.3239±0.0010 | 1.793 | 1.790±0.003 |
| 250 | 0.8835 | 0.8837±0.0009 | 1.1324 | 1.1320±0.0004 | 1.293 | 1.293±0.001 |
| 500 | 0.9419 | 0.9422±0.0006 | 1.0618 | 1.0616±0.0002 | 1.132 | 1.131±0.001 |
| 750 | 0.9613 | 0.9613±0.0005 | 1.0403 | 1.0403±0.0001 | 1.085 | 1.085±0.001 |

**Realised / predicted risk (std ratio): theory, simulations, measured**

| T | theory | MC Gaussian | MC Student-t (df=5) | MC bootstrap of real returns | measured [95% CI] | CI contains Gaussian theory |
|---|---|---|---|---|---|---|
| 60 | 2.018 | 2.015 | 2.458 | 2.274 | 2.462 [1.921, 3.203] | True |
| 90 | 1.498 | 1.499 | 1.760 | 1.625 | 1.864 [1.451, 2.566] | True |
| 120 | 1.329 | 1.327 | 1.526 | 1.410 | 1.516 [1.199, 2.029] | True |
| 250 | 1.131 | 1.131 | 1.230 | 1.161 | 1.177 [0.986, 1.359] | True |
| 500 | 1.059 | 1.060 | 1.118 | 1.075 | 1.132 [0.979, 1.265] | True |
| 750 | 1.037 | 1.037 | 1.075 | 1.048 | 1.105 [0.973, 1.265] | True |

Measured ratio above the Gaussian theory at 6 of 6 windows; measured CI contains the theory value at 6 of 6 (n = 8 periods, so the data cannot discriminate well). Student-t and bootstrap simulations show how far non-Gaussian data departs from the theory.

**Diagnostic only: the recalled offset-2 form `1 + (N-1)/(T-N-2)` against the same simulation**

| window | theory | mc_mean | mc_se | z |
|---|---|---|---|---|
| 60 | 2.036 | 2.001 | 0.0038 | -9.137 |
| 90 | 1.5 | 1.493 | 0.0016 | -4.43 |
| 120 | 1.329 | 1.324 | 0.001 | -5.709 |
| 250 | 1.133 | 1.132 | 0.0004 | -2.771 |
| 500 | 1.062 | 1.062 | 0.0002 | -2.284 |
| 750 | 1.04 | 1.04 | 0.0001 | -0.989 |

## 5C. Pre-registered predictor comparison

Analysis plan: see the pre-registration section above (committed before any 5C analysis). Pooled statistic = Fisher-z average of within-window Spearman ρ; Holm and BH within the 7-predictor family of each (design, portfolio) cell.

**Predictors supported by the pre-registered rule (Holm p < 0.05 in both designs, same sign): 0 of 14 cells.**

Runs per window (unconstrained):

| window | nonoverlap | stride60_bootstrap |
|---|---|---|
| 60 | 19 | 19 |
| 90 | 13 | 19 |
| 120 | 9 | 18 |
| 250 | 4 | 16 |
| 500 | 2 | 12 |
| 750 | 1 | 8 |

**Verdicts**

| constraint | predictor | rho_nonoverlap | holm_nonoverlap | rho_stride60 | holm_stride60 | expected_sign | same_sign | supported |
|---|---|---|---|---|---|---|---|---|
| unconstrained | exposure_k1 | 0.043 | 1 | -0.095 | 0.8 | + | False | False |
| unconstrained | exposure_k3 | -0.045 | 1 | -0.005 | 0.816 | + | True | False |
| unconstrained | exposure_k5 | 0.198 | 1 | 0.139 | 0.8 | + | True | False |
| unconstrained | log_cond | -0.284 | 0.722 | -0.379 | 0.007 | + | True | False |
| unconstrained | eff_rank | 0.261 | 0.722 | 0.374 | 0.007 | - | True | False |
| unconstrained | lam_ratio | 0.282 | 0.722 | 0.234 | 0.23 | - | True | False |
| unconstrained | max_abs_weight | 0.067 | 1 | -0.071 | 0.816 | + | False | False |
| long_only | exposure_k1 | -0.327 | 0.195 | -0.086 | 0.978 | + | True | False |
| long_only | exposure_k3 | -0.339 | 0.195 | -0.103 | 0.978 | + | True | False |
| long_only | exposure_k5 | 0.133 | 0.454 | 0.207 | 0.266 | + | True | False |
| long_only | log_cond | -0.496 | 0.017 | -0.462 | 0.266 | + | True | False |
| long_only | eff_rank | 0.462 | 0.032 | 0.454 | 0.266 | - | True | False |
| long_only | lam_ratio | 0.461 | 0.032 | 0.316 | 0.266 | - | True | False |
| long_only | max_abs_weight | -0.306 | 0.195 | -0.136 | 0.266 | + | True | False |

**All pooled (primary-family) tests**

| design | constraint | predictor | n | n_windows_used | rho | ci_lo | ci_hi | p_raw | p_holm | p_bh |
|---|---|---|---|---|---|---|---|---|---|---|
| nonoverlap | unconstrained | exposure_k1 | 45 | 4 | 0.043 | -0.299 | 0.375 | 0.811 | 1 | 0.811 |
| nonoverlap | unconstrained | exposure_k3 | 45 | 4 | -0.045 | -0.377 | 0.297 | 0.802 | 1 | 0.811 |
| nonoverlap | unconstrained | exposure_k5 | 45 | 4 | 0.198 | -0.15 | 0.502 | 0.264 | 1 | 0.462 |
| nonoverlap | unconstrained | log_cond | 45 | 4 | -0.284 | -0.567 | 0.059 | 0.103 | 0.722 | 0.319 |
| nonoverlap | unconstrained | eff_rank | 45 | 4 | 0.261 | -0.084 | 0.55 | 0.137 | 0.722 | 0.319 |
| nonoverlap | unconstrained | lam_ratio | 45 | 4 | 0.282 | -0.061 | 0.566 | 0.106 | 0.722 | 0.319 |
| nonoverlap | unconstrained | max_abs_weight | 45 | 4 | 0.067 | -0.277 | 0.396 | 0.708 | 1 | 0.811 |
| nonoverlap | long_only | exposure_k1 | 45 | 4 | -0.327 | -0.598 | 0.012 | 0.059 | 0.195 | 0.082 |
| nonoverlap | long_only | exposure_k3 | 45 | 4 | -0.339 | -0.607 | -0.002 | 0.049 | 0.195 | 0.082 |
| nonoverlap | long_only | exposure_k5 | 45 | 4 | 0.133 | -0.214 | 0.451 | 0.454 | 0.454 | 0.454 |
| nonoverlap | long_only | log_cond | 45 | 4 | -0.496 | -0.714 | -0.19 | 0.002 | 0.017 | 0.013 |
| nonoverlap | long_only | eff_rank | 45 | 4 | 0.462 | 0.147 | 0.691 | 0.005 | 0.032 | 0.013 |
| nonoverlap | long_only | lam_ratio | 45 | 4 | 0.461 | 0.147 | 0.691 | 0.005 | 0.032 | 0.013 |
| nonoverlap | long_only | max_abs_weight | 45 | 4 | -0.306 | -0.583 | 0.035 | 0.078 | 0.195 | 0.09 |
| stride60_bootstrap | unconstrained | exposure_k1 | 92 | 6 | -0.095 | -0.382 | 0.087 | 0.219 | 0.8 | 0.306 |
| stride60_bootstrap | unconstrained | exposure_k3 | 92 | 6 | -0.005 | -0.206 | 0.245 | 0.816 | 0.816 | 0.816 |
| stride60_bootstrap | unconstrained | exposure_k5 | 92 | 6 | 0.139 | -0.083 | 0.351 | 0.2 | 0.8 | 0.306 |
| stride60_bootstrap | unconstrained | log_cond | 92 | 6 | -0.379 | -0.566 | -0.187 | 0.001 | 0.007 | 0.003 |
| stride60_bootstrap | unconstrained | eff_rank | 92 | 6 | 0.374 | 0.193 | 0.518 | 0.001 | 0.007 | 0.003 |
| stride60_bootstrap | unconstrained | lam_ratio | 92 | 6 | 0.234 | 0.006 | 0.472 | 0.046 | 0.23 | 0.107 |
| stride60_bootstrap | unconstrained | max_abs_weight | 92 | 6 | -0.071 | -0.334 | 0.127 | 0.408 | 0.816 | 0.476 |
| stride60_bootstrap | long_only | exposure_k1 | 92 | 6 | -0.086 | -0.615 | 0.158 | 0.489 | 0.978 | 0.57 |
| stride60_bootstrap | long_only | exposure_k3 | 92 | 6 | -0.103 | -0.484 | 0.161 | 0.694 | 0.978 | 0.694 |
| stride60_bootstrap | long_only | exposure_k5 | 92 | 6 | 0.207 | -0.006 | 0.603 | 0.059 | 0.266 | 0.103 |
| stride60_bootstrap | long_only | log_cond | 92 | 6 | -0.462 | -0.63 | -0.045 | 0.04 | 0.266 | 0.093 |
| stride60_bootstrap | long_only | eff_rank | 92 | 6 | 0.454 | 0.044 | 0.606 | 0.039 | 0.266 | 0.093 |
| stride60_bootstrap | long_only | lam_ratio | 92 | 6 | 0.316 | -0.088 | 0.538 | 0.075 | 0.266 | 0.105 |
| stride60_bootstrap | long_only | max_abs_weight | 92 | 6 | -0.136 | -0.429 | -0.024 | 0.038 | 0.266 | 0.093 |

Pooled tests with Holm p < 0.05 in at least one design: nonoverlap/long_only/log_cond (ρ = -0.50, Holm p = 0.017); nonoverlap/long_only/eff_rank (ρ = 0.46, Holm p = 0.032); nonoverlap/long_only/lam_ratio (ρ = 0.46, Holm p = 0.032); stride60_bootstrap/unconstrained/log_cond (ρ = -0.38, Holm p = 0.007); stride60_bootstrap/unconstrained/eff_rank (ρ = 0.37, Holm p = 0.007).

## 5D. Sensitivity sweeps (common-OOS design)

Clipping floor = fraction × tr(S)/N, fractions [0.01, 0.05, 0.1, 0.25, 0.5] (0 = raw). A-priori setting in the main analysis: 0.1. Parameters are not selected.

**unconstrained: mean risk gap (pts) by floor fraction**

| window | 0.0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| 60 | 8.76 | 8.76 | 8.24 | 6.90 | 4.73 | 3.03 |
| 90 | 5.64 | 5.64 | 5.64 | 5.51 | 4.22 | 2.68 |
| 120 | 3.52 | 3.52 | 3.52 | 3.48 | 2.62 | 1.58 |
| 250 | 1.63 | 1.63 | 1.63 | 1.63 | 1.46 | 0.90 |
| 500 | 1.24 | 1.24 | 1.24 | 1.24 | 1.20 | 0.95 |
| 750 | 1.03 | 1.03 | 1.03 | 1.03 | 1.02 | 0.86 |

**unconstrained: mean realised risk (%)**

| window | 0.0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| 60 | 15.36 | 15.36 | 14.96 | 13.90 | 12.42 | 11.55 |
| 90 | 13.15 | 13.15 | 13.15 | 13.05 | 12.22 | 11.42 |
| 120 | 12.18 | 12.18 | 12.18 | 12.14 | 11.53 | 11.02 |
| 250 | 11.27 | 11.27 | 11.27 | 11.27 | 11.13 | 10.86 |
| 500 | 11.05 | 11.05 | 11.05 | 11.05 | 11.03 | 10.92 |
| 750 | 11.11 | 11.11 | 11.11 | 11.11 | 11.11 | 11.07 |

**long_only: mean risk gap (pts) by floor fraction**

| window | 0.0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| 60 | 3.00 | 3.00 | 2.98 | 2.94 | 2.68 | 2.12 |
| 90 | 2.34 | 2.34 | 2.34 | 2.34 | 2.17 | 1.72 |
| 120 | 1.30 | 1.30 | 1.30 | 1.30 | 1.16 | 0.88 |
| 250 | 0.63 | 0.63 | 0.63 | 0.63 | 0.60 | 0.42 |
| 500 | 0.91 | 0.91 | 0.91 | 0.91 | 0.89 | 0.76 |
| 750 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.77 |

**long_only: mean realised risk (%)**

| window | 0.0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| 60 | 11.71 | 11.71 | 11.70 | 11.68 | 11.57 | 11.33 |
| 90 | 11.45 | 11.45 | 11.45 | 11.45 | 11.38 | 11.20 |
| 120 | 11.10 | 11.10 | 11.10 | 11.10 | 11.02 | 10.92 |
| 250 | 10.94 | 10.94 | 10.94 | 10.94 | 10.91 | 10.87 |
| 500 | 11.09 | 11.09 | 11.09 | 11.09 | 11.09 | 11.04 |
| 750 | 11.23 | 11.23 | 11.23 | 11.23 | 11.23 | 11.23 |

**Mean number of eigenvalues clipped**

| window | 0.0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| 60 |  | 0.00 | 1.38 | 4.12 | 10.12 | 16.25 |
| 90 |  | 0.00 | 0.00 | 1.50 | 8.25 | 15.50 |
| 120 |  | 0.00 | 0.00 | 0.25 | 7.88 | 15.00 |
| 250 |  | 0.00 | 0.00 | 0.00 | 4.25 | 14.12 |
| 500 |  | 0.00 | 0.00 | 0.00 | 2.12 | 13.62 |
| 750 |  | 0.00 | 0.00 | 0.00 | 1.50 | 13.00 |

MP noise variance: σ² from ['trace', 'median'] (a-priori rule: trace).

**unconstrained: MP sweep, mean risk gap (pts) / realised risk (%)**

| window | median | raw | trace |
|---|---|---|---|
| 60 | 4.10 | 8.76 | 4.25 |
| 90 | 3.58 | 5.64 | 4.08 |
| 120 | 2.56 | 3.52 | 3.17 |
| 250 | 1.46 | 1.63 | 2.64 |
| 500 | 1.16 | 1.24 | 2.25 |
| 750 | 0.96 | 1.03 | 2.17 |

| window | median | raw | trace |
|---|---|---|---|
| 60 | 11.79 | 15.36 | 11.52 |
| 90 | 11.64 | 13.15 | 11.80 |
| 120 | 11.40 | 12.18 | 11.52 |
| 250 | 11.15 | 11.27 | 11.67 |
| 500 | 11.00 | 11.05 | 11.38 |
| 750 | 11.10 | 11.11 | 11.59 |

**long_only: MP sweep, mean risk gap (pts) / realised risk (%)**

| window | median | raw | trace |
|---|---|---|---|
| 60 | 2.89 | 3.00 | 3.30 |
| 90 | 2.05 | 2.34 | 2.84 |
| 120 | 1.26 | 1.30 | 2.00 |
| 250 | 0.58 | 0.63 | 1.53 |
| 500 | 0.85 | 0.91 | 1.72 |
| 750 | 0.80 | 0.83 | 1.67 |

| window | median | raw | trace |
|---|---|---|---|
| 60 | 11.60 | 11.71 | 11.39 |
| 90 | 11.19 | 11.45 | 11.47 |
| 120 | 11.03 | 11.10 | 11.35 |
| 250 | 10.92 | 10.94 | 11.41 |
| 500 | 11.07 | 11.09 | 11.46 |
| 750 | 11.23 | 11.23 | 11.65 |

**Mean number of eigenvalues treated as noise**

| window | median | trace |
|---|---|---|
| 60 | 24.88 | 28.38 |
| 90 | 24.50 | 27.75 |
| 120 | 24.38 | 27.75 |
| 250 | 24.00 | 27.25 |
| 500 | 22.88 | 27.00 |
| 750 | 21.25 | 26.50 |

## 5E. Uncertainty (moving-block bootstrap over OOS periods)

8 periods per window, block length 3, 2000 resamples, 95% percentile intervals; differences are method − comparator on identical periods (negative gap difference = smaller gap; negative realised-risk difference = lower risk). Unadjusted for multiple comparisons.

Paired comparisons: 96; intervals excluding zero: 61; flagged differences smaller than 0.1 percentage point in absolute size: 15.

**Simulated coverage of the nominal 95% interval at n = 8, block length 3:** 0.767 (i.i.d. data) and 0.583 (AR(1), coefficient 0.5). Coverage below 0.95 means the intervals are too narrow and the 'excludes zero' flags are anti-conservative.

**Coverage check (all settings)**

| n | block_len | rho | reps | coverage | se |
|---|---|---|---|---|---|
| 8 | 1 | 0 | 1000 | 0.874 | 0.01 |
| 8 | 1 | 0.5 | 1000 | 0.651 | 0.015 |
| 8 | 3 | 0 | 1000 | 0.767 | 0.013 |
| 8 | 3 | 0.5 | 1000 | 0.583 | 0.016 |
| 12 | 1 | 0 | 1000 | 0.906 | 0.009 |
| 12 | 1 | 0.5 | 1000 | 0.659 | 0.015 |
| 12 | 3 | 0 | 1000 | 0.812 | 0.012 |
| 12 | 3 | 0.5 | 1000 | 0.666 | 0.015 |
| 19 | 1 | 0 | 1000 | 0.911 | 0.009 |
| 19 | 1 | 0.5 | 1000 | 0.703 | 0.014 |
| 19 | 3 | 0 | 1000 | 0.872 | 0.011 |
| 19 | 3 | 0.5 | 1000 | 0.754 | 0.014 |

**Block-length sensitivity (number of comparisons excluding zero)**

| block_len | n_comparisons | n_exclude_zero | n_exclude_zero_vs_raw |
|---|---|---|---|
| 1 | 96 | 55 | 44 |
| 2 | 96 | 56 | 44 |
| 3 | 96 | 61 | 46 |
| 4 | 96 | 73 | 53 |

**unconstrained: mean risk gap difference (pts)** (* = interval excludes 0)

| window | clipped - raw | ledoit_wolf - mp | ledoit_wolf - raw | mp - raw |
|---|---|---|---|---|
| 60 | -1.87* | -0.40 | -4.92* | -4.52* |
| 90 | -0.13* | -0.44 | -2.00* | -1.56* |
| 120 | -0.04* | -0.77* | -1.12* | -0.35 |
| 250 | 0.00* | -1.18* | -0.18* | 1.01* |
| 500 | -0.00 | -0.96* | 0.05 | 1.01* |
| 750 | -0.00 | -1.09* | 0.05* | 1.13* |

**long_only: mean risk gap difference (pts)** (* = interval excludes 0)

| window | clipped - raw | ledoit_wolf - mp | ledoit_wolf - raw | mp - raw |
|---|---|---|---|---|
| 60 | -0.06* | -0.36 | -0.05 | 0.31 |
| 90 | -0.00 | -0.48* | 0.03 | 0.50 |
| 120 | -0.00 | -0.50* | 0.20* | 0.70* |
| 250 | 0.00* | -0.67* | 0.24* | 0.90* |
| 500 | -0.00 | -0.65* | 0.16* | 0.82* |
| 750 | -0.00* | -0.71* | 0.12* | 0.83* |

**unconstrained: mean realised risk difference (pts)** (* = interval excludes 0)

| window | clipped - raw | ledoit_wolf - mp | ledoit_wolf - raw | mp - raw |
|---|---|---|---|---|
| 60 | -1.46* | -0.01 | -3.85* | -3.84* |
| 90 | -0.10* | -0.17 | -1.52* | -1.35* |
| 120 | -0.04* | -0.31 | -0.97* | -0.66* |
| 250 | 0.00* | -0.67* | -0.26* | 0.41 |
| 500 | -0.00* | -0.41 | -0.07 | 0.34 |
| 750 | -0.00 | -0.52 | -0.04* | 0.48 |

**long_only: mean realised risk difference (pts)** (* = interval excludes 0)

| window | clipped - raw | ledoit_wolf - mp | ledoit_wolf - raw | mp - raw |
|---|---|---|---|---|
| 60 | -0.03 | -0.09 | -0.41* | -0.32 |
| 90 | 0.00 | -0.30* | -0.28 | 0.02 |
| 120 | -0.00 | -0.41* | -0.17* | 0.25 |
| 250 | 0.00* | -0.52* | -0.04 | 0.48* |
| 500 | -0.00* | -0.40* | -0.04* | 0.37 |
| 750 | -0.00* | -0.44* | -0.02 | 0.42* |

## 5F. Block missingness: late-listed stocks

m ∈ [5, 10, 15] stocks have the first fraction f ∈ [0.3, 0.5, 0.7] of the sample missing (all list on the same date), 50 seeds per cell; pairwise-complete covariance. Reference for judging weights: the full-sample covariance of the unmasked data (variance ratio 1 = full-information GMV).

| m | f | runs | non_psd | non_psd_rate | non_psd_rate_se | lambda_min_min | lambda_min_median | pw_var_ratio_median | clip_rel_var_ratio_median | cc_var_ratio_median |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.3 | 50 | 0 | 0 | 0 | 4.91e-05 | 5.64e-05 | 1.03 | 1.03 | 1.03 |
| 5 | 0.5 | 50 | 0 | 0 | 0 | 2.78e-05 | 4.86e-05 | 1.02 | 1.02 | 1.05 |
| 5 | 0.7 | 50 | 1 | 0.02 | 0.0198 | -1.63e-06 | 2.7e-05 | 1.23 | 1.21 | 1.28 |
| 10 | 0.3 | 50 | 0 | 0 | 0 | 4.85e-05 | 5.32e-05 | 1.04 | 1.04 | 1.03 |
| 10 | 0.5 | 50 | 0 | 0 | 0 | 2.95e-05 | 3.67e-05 | 1.04 | 1.04 | 1.05 |
| 10 | 0.7 | 50 | 3 | 0.06 | 0.0336 | -5.57e-06 | 1.21e-05 | 1.39 | 1.3 | 1.28 |
| 15 | 0.3 | 50 | 0 | 0 | 0 | 4.87e-05 | 5.51e-05 | 1.04 | 1.04 | 1.03 |
| 15 | 0.5 | 50 | 0 | 0 | 0 | 2.59e-05 | 4.03e-05 | 1.05 | 1.05 | 1.05 |
| 15 | 0.7 | 50 | 7 | 0.14 | 0.0491 | -5.53e-06 | 1.22e-05 | 1.42 | 1.31 | 1.28 |

Non-PSD runs: 11 of 450.

In the 11 non-PSD runs: number of negative eigenvalues per matrix 1; λ_min from -5.57e-06 to -1.53e-07; negative predicted variance in 3; max|w| of the unrepaired weights 0.13–10.16.

**Variance ratio against the full-sample covariance (median over non-PSD runs)**

| pairwise, unrepaired | clipped, relative floor | clipped, minimal floor | complete-case |
|---|---|---|---|
| 56.809 | 1.410 | 15.569 | 1.283 |

Repair cost ‖S_clipped − S‖_F / ‖S‖_F: relative floor 0.0101–0.0127, minimal floor 0.0006–0.0030. Median L1 change in weights (unrepaired → relative floor): 18.78.

Complete-case variance ratio is lower (better) than the relative-floor repair in 10 of 11 non-PSD runs.

Verification over all 450 runs: repaired matrix positive definite (relative floor) in 450; minimum eigenvalue ≥ floor in 450; Frobenius distance equals the spectral distance in 450; complete-case covariance positive definite in 450.

