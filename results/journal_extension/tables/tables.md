### T1_comparison_counts

| W | ext_comparisons | ext_lower_mean_qlike | ext_ci_below_zero | ext_ci_and_win_rate | ext_significant_losses | dev_comparisons | dev_ci_and_win_rate |
|---|---|---|---|---|---|---|---|
| 60 | 36 | 36 | 36 | 36 | 0 | 12 | 5 |
| 90 | 48 | 48 | 48 | 46 | 0 | 12 | 3 |
| 120 | 48 | 48 | 48 | 44 | 0 | 12 | 0 |
| 250 | 48 | 48 | 45 | 11 | 0 | 12 | 0 |
| 500 | 48 | 48 | 29 | 0 | 0 | 12 | 0 |
| 750 | 48 | 48 | 14 | 0 | 5 | 12 | 0 |
| all | 276 | 276 | 220 | 137 | 5 | 72 | 8 |

### T2_ext_A_H20_qlike

| W | periods | vr_raw | vr_corr | q_raw | q_corr | q_lw | q_nl | q_bps | q_ew |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 151 | 5.248 | 1.334 | 3.407 | 0.387 | 1.139 | 1.039 | 0.890 | 0.480 |
| 90 | 151 | 2.898 | 1.310 | 1.437 | 0.435 | 0.956 | 0.875 | 0.812 | 0.516 |
| 120 | 151 | 2.262 | 1.290 | 0.896 | 0.397 | 0.703 | 0.668 | 0.682 | 0.422 |
| 250 | 151 | 1.471 | 1.148 | 0.547 | 0.419 | 0.553 | 0.538 | 0.550 | 0.450 |
| 500 | 151 | 1.253 | 1.111 | 0.443 | 0.402 | 0.457 | 0.444 | 0.458 | 0.476 |
| 750 | 151 | 1.183 | 1.093 | 0.491 | 0.466 | 0.504 | 0.495 | 0.505 | 0.535 |

### T3_bps_variants

| W | bps | bps_verbatim | gap_verbatim_minus_analogue |
|---|---|---|---|
| 60 | 1.006 | 0.981 | -0.025 |
| 90 | 0.969 | 0.956 | -0.013 |
| 120 | 0.845 | 0.836 | -0.008 |
| 250 | 0.571 | 0.571 | -0.001 |
| 500 | 0.436 | 0.437 | 0.000 |
| 750 | 0.463 | 0.464 | 0.000 |

### T4_regimes_ext_A_H20_W60

| regime | n | vr_corr | median_ratio_corr | qlike_corr | qlike_nl | delta_vs_nl | delta_lo | delta_hi |
|---|---|---|---|---|---|---|---|---|
| high | 32 | 1.154 | 0.779 | 0.295 | 0.208 | 0.086 | -0.076 | 0.363 |
| low | 62 | 1.460 | 1.294 | 0.252 | 0.707 | -0.455 | -0.607 | -0.306 |
| mid | 57 | 1.514 | 1.020 | 0.584 | 1.866 | -1.282 | -3.505 | -0.146 |

### T5_interval_coverage

| level | W | cells | mean_coverage | min_coverage | max_coverage |
|---|---|---|---|---|---|
| 0.800 | 60 | 8 | 0.643 | 0.583 | 0.695 |
| 0.800 | 90 | 10 | 0.588 | 0.417 | 0.700 |
| 0.800 | 120 | 10 | 0.563 | 0.375 | 0.682 |
| 0.800 | 250 | 10 | 0.447 | 0.250 | 0.583 |
| 0.800 | 500 | 10 | 0.429 | 0.250 | 0.589 |
| 0.800 | 750 | 10 | 0.369 | 0.200 | 0.500 |
| 0.900 | 60 | 8 | 0.745 | 0.625 | 0.800 |
| 0.900 | 90 | 10 | 0.728 | 0.640 | 0.795 |
| 0.900 | 120 | 10 | 0.683 | 0.375 | 0.801 |
| 0.900 | 250 | 10 | 0.590 | 0.375 | 0.711 |
| 0.900 | 500 | 10 | 0.531 | 0.250 | 0.675 |
| 0.900 | 750 | 10 | 0.524 | 0.400 | 0.633 |

### T6_second_stage_verdict

| model | cells | share_lower_qlike | share_ci_favours | share_lower_mace | median_d_gaussian | median_d_high_vol | supported |
|---|---|---|---|---|---|---|---|
| constant | 34 | 0.706 | 0.029 | 0.000 | -0.037 | 0.046 | False |
| tercile | 34 | 0.676 | 0.029 | 0.000 | -0.032 | 0.013 | False |
| ridge | 34 | 0.794 | 0.059 | 0.294 | -0.058 | 0.045 | False |

### T7_level_bias_by_scenario

| scenario | cells | mean_bias_raw | mean_bias_c | worst_bias_c |
|---|---|---|---|---|
| gaussian | 90 | -0.337 | -0.000 | -0.015 |
| t15 | 90 | -0.359 | -0.048 | -0.122 |
| t5 | 90 | -0.432 | -0.186 | -0.386 |
| t8 | 90 | -0.385 | -0.100 | -0.239 |

### T8_predictors

| constraint | predictor | rho_nonoverlap | holm_nonoverlap | rho_stride60 | holm_stride60 | expected_sign | same_sign | supported |
|---|---|---|---|---|---|---|---|---|
| unconstrained | exposure_k1 | 0.031 | 1.000 | 0.088 | 1.000 | + | True | False |
| unconstrained | exposure_k3 | 0.102 | 1.000 | 0.077 | 1.000 | + | True | False |
| unconstrained | exposure_k5 | 0.113 | 1.000 | 0.067 | 1.000 | + | True | False |
| unconstrained | log_cond | -0.321 | 0.019 | -0.357 | 0.007 | + | True | True |
| unconstrained | eff_rank | 0.297 | 0.035 | 0.316 | 0.007 | - | True | True |
| unconstrained | lam_ratio | 0.163 | 0.697 | 0.243 | 0.030 | - | True | False |
| unconstrained | max_abs_weight | -0.093 | 1.000 | -0.093 | 0.456 | + | True | False |
| long_only | exposure_k1 | 0.014 | 1.000 | 0.014 | 1.000 | + | True | False |
| long_only | exposure_k3 | 0.029 | 1.000 | -0.027 | 1.000 | + | False | False |
| long_only | exposure_k5 | 0.147 | 0.726 | 0.014 | 1.000 | + | True | False |
| long_only | log_cond | -0.472 | 0.000 | -0.502 | 0.007 | + | True | True |
| long_only | eff_rank | 0.426 | 0.000 | 0.461 | 0.007 | - | True | True |
| long_only | lam_ratio | 0.344 | 0.006 | 0.383 | 0.007 | - | True | True |
| long_only | max_abs_weight | -0.119 | 0.845 | -0.131 | 0.028 | + | True | False |

### T10_constants

| N | T | c | c_qlike | ratio |
|---|---|---|---|---|
| 10 | 60 | 1.397 | 1.455 | 1.042 |
| 10 | 120 | 1.171 | 1.193 | 1.019 |
| 10 | 250 | 1.077 | 1.086 | 1.008 |
| 10 | 750 | 1.024 | 1.027 | 1.003 |
| 30 | 60 | 3.933 | 4.214 | 1.071 |
| 30 | 120 | 1.753 | 1.793 | 1.023 |
| 30 | 250 | 1.282 | 1.293 | 1.009 |
| 30 | 750 | 1.082 | 1.085 | 1.003 |
| 50 | 60 | 38.022 | 47.528 | 1.250 |
| 50 | 120 | 2.907 | 2.993 | 1.029 |
| 50 | 250 | 1.552 | 1.567 | 1.010 |
| 50 | 750 | 1.145 | 1.148 | 1.003 |

### T9_cqlike_vs_c

| W | cells | c_qlike_lower | ci_below_zero | mean_qlike_diff | vr_c | vr_c_qlike |
|---|---|---|---|---|---|---|
| 60 | 8 | 8 | 6 | -0.039 | 1.370 | 1.281 |
| 90 | 10 | 10 | 8 | -0.022 | 1.333 | 1.281 |
| 120 | 10 | 10 | 6 | -0.012 | 1.279 | 1.248 |
| 250 | 10 | 10 | 4 | -0.004 | 1.212 | 1.200 |
| 500 | 10 | 10 | 0 | -0.001 | 1.194 | 1.189 |
| 750 | 10 | 10 | 0 | -0.001 | 1.191 | 1.188 |

### T11_conformal_coverage

| level | adequate_cells | mean_cov_gauss | mean_cov_pooled | mean_cov_vol | C1_share_within_0p05 | C2_mean_gap | C3_share_beats_gauss | vol_beats_pooled_share |
|---|---|---|---|---|---|---|---|---|
| 0.800 | 40 | 0.500 | 0.754 | 0.763 | 0.625 | -0.046 | 0.975 | 0.525 |
| 0.900 | 40 | 0.622 | 0.860 | 0.876 | 0.650 | -0.040 | 0.900 | 0.375 |

### T12_kurtosis_simulation

| scenario | beta_mean | bias_c | bias_kc | qlike_cq | qlike_kcq | vr_cq | vr_kcq |
|---|---|---|---|---|---|---|---|
| garch | 1.059 | -0.026 | -0.008 | 0.128 | 0.127 | 0.999 | 0.981 |
| gaussian | 1.000 | 0.000 | 0.012 | 0.035 | 0.036 | 0.972 | 0.962 |
| regime | 1.228 | -0.067 | -0.027 | 0.139 | 0.137 | 1.044 | 1.001 |
| skew1 | 1.332 | -0.069 | -0.020 | 0.050 | 0.050 | 1.045 | 0.993 |
| skew4 | 1.086 | -0.018 | 0.004 | 0.039 | 0.040 | 0.990 | 0.969 |
| skewt8 | 1.477 | -0.114 | -0.053 | 0.066 | 0.060 | 1.102 | 1.028 |
| t15 | 1.151 | -0.046 | -0.017 | 0.043 | 0.043 | 1.020 | 0.989 |
| t5 | 1.899 | -0.186 | -0.101 | 0.104 | 0.086 | 1.205 | 1.084 |
| t8 | 1.372 | -0.097 | -0.045 | 0.059 | 0.055 | 1.080 | 1.019 |

### T13_kurtosis_market

| W | cells | beta_hat | lower_qlike | ci_below_zero | mean_qlike_diff | vr_c_qlike | vr_kappa |
|---|---|---|---|---|---|---|---|
| 60 | 4 | 1.441 | 3 | 0 | -0.006 | 1.273 | 1.088 |
| 90 | 6 | 1.714 | 6 | 0 | -0.059 | 1.273 | 1.042 |
| 120 | 6 | 1.756 | 6 | 0 | -0.061 | 1.275 | 1.061 |
| 250 | 6 | 2.227 | 6 | 0 | -0.037 | 1.233 | 1.056 |
| 500 | 6 | 2.882 | 6 | 0 | -0.010 | 1.178 | 1.057 |
| 750 | 6 | 3.339 | 6 | 0 | -0.007 | 1.144 | 1.052 |

### T14_india_vs_us

| result | India | US |
|---|---|---|
| Corrected forecast has lower mean QLIKE (all comparisons) | 347 of 348 | 348 of 348 |
| Interval below zero, extended samples | 220 of 276 | 248 of 276 |
| Interval below zero, W <= 120 | 132 of 132 | 132 of 132 |
| Interval below zero, W >= 500 | 43 of 96 | 68 of 96 |
| Comparisons where the correction wins fewer than half the origins | 5 | 13 |
| c_QLIKE lower QLIKE than c (cells) | 58 of 58 | 58 of 58 |
| c_QLIKE interval below zero (cells) | 24 of 58 | 34 of 58 |
| Gaussian interval coverage, nominal 80% / 90% | 50.2% / 63.0% | 44.4% / 55.0% |
| Conformal interval coverage, nominal 80% / 90% | 75.4% / 86.0% | 85.1% / 92.9% |
| Kurtosis multiplier: lower QLIKE (cells) | 33 of 34 | 34 of 34 |
| Kurtosis multiplier: interval below zero (cells) | 0 of 34 | 6 of 34 |
| Kurtosis multiplier: realised/forecast before -> after | 1.23 -> 1.06 | 1.37 -> 1.20 |

### T15_var_backtest

| market | windows | variant | viol5 | kup5 | viol1 |
|---|---|---|---|---|---|
| India | all W | plugin_raw | 0.108 | 0.853 | 0.060 |
| India | all W | c | 0.058 | 0.529 | 0.023 |
| India | all W | c_qlike | 0.056 | 0.500 | 0.022 |
| India | all W | kappa_cq | 0.050 | 0.235 | 0.019 |
| India | all W | nl | 0.078 | 0.853 | 0.036 |
| India | all W | ledoit_wolf | 0.081 | 0.824 | 0.038 |
| India | all W | bps_verbatim | 0.080 | 0.824 | 0.037 |
| India | all W | equal_weight | 0.055 | 0.324 | 0.021 |
| India | W<=120 | plugin_raw | 0.157 | 1.000 | 0.096 |
| India | W<=120 | c | 0.065 | 1.000 | 0.026 |
| India | W<=120 | c_qlike | 0.062 | 0.938 | 0.024 |
| India | W<=120 | kappa_cq | 0.054 | 0.125 | 0.020 |
| India | W<=120 | nl | 0.097 | 1.000 | 0.048 |
| India | W<=120 | ledoit_wolf | 0.102 | 1.000 | 0.052 |
| India | W<=120 | bps_verbatim | 0.099 | 1.000 | 0.050 |
| India | W<=120 | equal_weight | 0.059 | 0.688 | 0.024 |
| US | all W | plugin_raw | 0.112 | 0.882 | 0.066 |
| US | all W | c | 0.063 | 0.588 | 0.029 |
| US | all W | c_qlike | 0.061 | 0.588 | 0.028 |
| US | all W | kappa_cq | 0.055 | 0.324 | 0.025 |
| US | all W | nl | 0.080 | 0.824 | 0.041 |
| US | all W | ledoit_wolf | 0.081 | 0.853 | 0.042 |
| US | all W | bps_verbatim | 0.079 | 0.824 | 0.041 |
| US | all W | equal_weight | 0.053 | 0.294 | 0.026 |
| US | W<=120 | plugin_raw | 0.160 | 1.000 | 0.103 |
| US | W<=120 | c | 0.072 | 1.000 | 0.033 |
| US | W<=120 | c_qlike | 0.069 | 1.000 | 0.031 |
| US | W<=120 | kappa_cq | 0.060 | 0.688 | 0.027 |
| US | W<=120 | nl | 0.098 | 1.000 | 0.052 |
| US | W<=120 | ledoit_wolf | 0.099 | 1.000 | 0.054 |
| US | W<=120 | bps_verbatim | 0.095 | 1.000 | 0.052 |
| US | W<=120 | equal_weight | 0.059 | 0.562 | 0.027 |

### T16_vol_targeting

| market | windows | variant | vt_abs_log | breach | exposure |
|---|---|---|---|---|---|
| India | all W | plugin_raw | 0.413 | 0.549 | 1.723 |
| India | all W | c | 0.235 | 0.208 | 1.265 |
| India | all W | c_qlike | 0.234 | 0.198 | 1.251 |
| India | all W | kappa_cq | 0.249 | 0.162 | 1.192 |
| India | all W | nl | 0.281 | 0.389 | 1.611 |
| India | all W | ledoit_wolf | 0.299 | 0.425 | 1.635 |
| India | all W | bps_verbatim | 0.299 | 0.403 | 1.556 |
| India | all W | equal_weight | 0.317 | 0.193 | 1.066 |
| India | W<=120 | plugin_raw | 0.593 | 0.829 | 2.009 |
| India | W<=120 | c | 0.229 | 0.233 | 1.196 |
| India | W<=120 | c_qlike | 0.226 | 0.215 | 1.171 |
| India | W<=120 | kappa_cq | 0.239 | 0.168 | 1.109 |
| India | W<=120 | nl | 0.322 | 0.535 | 1.775 |
| India | W<=120 | ledoit_wolf | 0.353 | 0.591 | 1.818 |
| India | W<=120 | bps_verbatim | 0.348 | 0.546 | 1.670 |
| India | W<=120 | equal_weight | 0.321 | 0.236 | 1.117 |
| US | all W | plugin_raw | 0.445 | 0.533 | 1.808 |
| US | all W | c | 0.288 | 0.253 | 1.348 |
| US | all W | c_qlike | 0.287 | 0.241 | 1.333 |
| US | all W | kappa_cq | 0.299 | 0.206 | 1.268 |
| US | all W | nl | 0.325 | 0.376 | 1.699 |
| US | all W | ledoit_wolf | 0.335 | 0.384 | 1.717 |
| US | all W | bps_verbatim | 0.325 | 0.350 | 1.643 |
| US | all W | equal_weight | 0.381 | 0.189 | 1.184 |
| US | W<=120 | plugin_raw | 0.611 | 0.816 | 2.136 |
| US | W<=120 | c | 0.277 | 0.309 | 1.321 |
| US | W<=120 | c_qlike | 0.273 | 0.285 | 1.294 |
| US | W<=120 | kappa_cq | 0.280 | 0.241 | 1.223 |
| US | W<=120 | nl | 0.358 | 0.521 | 1.919 |
| US | W<=120 | ledoit_wolf | 0.371 | 0.535 | 1.948 |
| US | W<=120 | bps_verbatim | 0.350 | 0.462 | 1.804 |
| US | W<=120 | equal_weight | 0.347 | 0.207 | 1.269 |

### T17_same_target

| market | target | vs | comparisons | lower | ci_below | ci_above | mean_diff |
|---|---|---|---|---|---|---|---|
| India | raw minimum-variance | raw | 34 | 1.000 | 0.794 | 0.000 | -1.216 |
| India | raw minimum-variance | clipped | 34 | 1.000 | 0.794 | 0.000 | -0.796 |
| India | raw minimum-variance | mp | 34 | 0.853 | 0.235 | 0.000 | -0.047 |
| India | raw minimum-variance | ledoit_wolf | 34 | 1.000 | 0.794 | 0.000 | -0.317 |
| India | raw minimum-variance | nl | 34 | 1.000 | 0.824 | 0.000 | -0.230 |
| India | equal-weight (control) | raw | 34 | 0.441 | 0.000 | 0.147 | 0.112 |
| India | equal-weight (control) | clipped | 34 | 0.441 | 0.000 | 0.147 | 0.112 |
| India | equal-weight (control) | mp | 34 | 0.441 | 0.000 | 0.147 | 0.108 |
| India | equal-weight (control) | ledoit_wolf | 34 | 0.706 | 0.000 | 0.088 | 0.042 |
| India | equal-weight (control) | nl | 34 | 0.471 | 0.000 | 0.147 | 0.094 |
| US | raw minimum-variance | raw | 34 | 1.000 | 0.882 | 0.000 | -1.816 |
| US | raw minimum-variance | clipped | 34 | 1.000 | 0.882 | 0.000 | -0.918 |
| US | raw minimum-variance | mp | 34 | 0.735 | 0.324 | 0.118 | 0.001 |
| US | raw minimum-variance | ledoit_wolf | 34 | 1.000 | 0.824 | 0.000 | -0.388 |
| US | raw minimum-variance | nl | 34 | 1.000 | 0.853 | 0.000 | -0.403 |
| US | equal-weight (control) | raw | 34 | 0.853 | 0.000 | 0.000 | -0.050 |
| US | equal-weight (control) | clipped | 34 | 0.853 | 0.000 | 0.000 | -0.050 |
| US | equal-weight (control) | mp | 34 | 0.853 | 0.000 | 0.000 | -0.060 |
| US | equal-weight (control) | ledoit_wolf | 34 | 0.941 | 0.000 | 0.000 | -0.178 |
| US | equal-weight (control) | nl | 34 | 0.853 | 0.000 | 0.000 | -0.081 |

### T18_portfolio_outcomes

| market | windows | portfolio | ann_vol | max_drawdown | ann_return | mean_max_abs_w |
|---|---|---|---|---|---|---|
| India | W<=120 | raw | 0.173 | -0.328 | 0.144 | 0.262 |
| India | W<=120 | clipped | 0.159 | -0.310 | 0.143 | 0.222 |
| India | W<=120 | mp | 0.145 | -0.305 | 0.116 | 0.118 |
| India | W<=120 | ledoit_wolf | 0.144 | -0.310 | 0.136 | 0.144 |
| India | W<=120 | nl | 0.143 | -0.304 | 0.126 | 0.139 |
| India | W<=120 | bps_verbatim | 0.155 | -0.348 | 0.143 | 0.162 |
| India | W<=120 | equal_weight | 0.168 | -0.433 | 0.139 | 0.031 |
| India | W>=500 | raw | 0.138 | -0.316 | 0.135 | 0.165 |
| India | W>=500 | clipped | 0.138 | -0.316 | 0.135 | 0.165 |
| India | W>=500 | mp | 0.141 | -0.330 | 0.125 | 0.121 |
| India | W>=500 | ledoit_wolf | 0.137 | -0.317 | 0.135 | 0.147 |
| India | W>=500 | nl | 0.136 | -0.315 | 0.133 | 0.144 |
| India | W>=500 | bps_verbatim | 0.137 | -0.322 | 0.135 | 0.155 |
| India | W>=500 | equal_weight | 0.167 | -0.433 | 0.139 | 0.029 |
| US | W<=120 | raw | 0.174 | -0.355 | 0.100 | 0.376 |
| US | W<=120 | clipped | 0.156 | -0.309 | 0.104 | 0.262 |
| US | W<=120 | mp | 0.153 | -0.344 | 0.073 | 0.142 |
| US | W<=120 | ledoit_wolf | 0.146 | -0.305 | 0.102 | 0.183 |
| US | W<=120 | nl | 0.147 | -0.301 | 0.102 | 0.196 |
| US | W<=120 | bps_verbatim | 0.154 | -0.309 | 0.109 | 0.220 |
| US | W<=120 | equal_weight | 0.165 | -0.387 | 0.123 | 0.031 |
| US | W>=500 | raw | 0.145 | -0.380 | 0.066 | 0.229 |
| US | W>=500 | clipped | 0.145 | -0.378 | 0.067 | 0.226 |
| US | W>=500 | mp | 0.152 | -0.357 | 0.068 | 0.159 |
| US | W>=500 | ledoit_wolf | 0.144 | -0.361 | 0.070 | 0.203 |
| US | W>=500 | nl | 0.144 | -0.364 | 0.068 | 0.207 |
| US | W>=500 | bps_verbatim | 0.144 | -0.370 | 0.070 | 0.212 |
| US | W>=500 | equal_weight | 0.165 | -0.386 | 0.123 | 0.029 |

### T19_comparator_roles

| estimator | own-portfolio QLIKE (348 comparisons) | same-target QLIKE | VaR and volatility targeting | portfolio outcomes |
|---|---|---|---|---|
| Raw plug-in | yes | yes | yes | yes |
| Eigenvalue clipping | yes | yes | yes | yes |
| Marchenko-Pastur filter | yes | yes | yes | yes |
| Ledoit-Wolf | yes | yes | yes | yes |
| Nonlinear shrinkage | yes | yes | yes | yes |
| Weight shrinkage (BPS) | yes | no | yes | yes |
| Equal weights | no | control only | yes | yes |
