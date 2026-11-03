"""Manuscript tables built ONLY from saved CSVs under results/journal_extension/ (markdown + csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.astype(object).iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in r) + " |")
    return "\n".join(out)


def build(out_root: Path) -> dict[str, pd.DataFrame]:
    R, T = out_root, out_root / "tables"
    T.mkdir(parents=True, exist_ok=True)
    tabs: dict[str, pd.DataFrame] = {}

    tests = pd.read_csv(R / "extended" / "paired_tests_extended.csv")
    ext = tests[tests["sample"] != "dev"]
    rows = []
    for W, g in tests.groupby("W"):
        e, dv = g[g["sample"] != "dev"], g[g["sample"] == "dev"]
        rows.append(dict(W=W, ext_comparisons=len(e), ext_lower_mean_qlike=int((e.delta < 0).sum()), ext_ci_below_zero=int((e.hi < 0).sum()),
                         ext_ci_and_win_rate=int(e.robust.sum()), ext_significant_losses=int(e.loses.sum()),
                         dev_comparisons=len(dv), dev_ci_and_win_rate=int(dv.robust.sum())))
    t = pd.DataFrame(rows)
    tot = dict(W="all", **{c: int(t[c].sum()) for c in t.columns if c != "W"})
    tabs["T1_comparison_counts"] = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)

    t2 = pd.read_csv(R / "extended" / "table2_all_rows.csv")
    t2 = t2[(t2["sample"] == "ext_A") & (t2.H == 20)][["W", "periods", "vr_raw", "vr_corr", "q_raw", "q_corr", "q_lw", "q_nl", "q_bps", "q_ew"]]
    tabs["T2_ext_A_H20_qlike"] = t2

    b = pd.read_csv(R / "bps_check" / "bps_variants.csv")
    p = b.pivot_table(index="W", columns="alt", values="mean_alt").reset_index()
    p["gap_verbatim_minus_analogue"] = p.bps_verbatim - p.bps
    tabs["T3_bps_variants"] = p

    reg = pd.read_csv(R / "empirical" / "regime_calibration.csv")
    r = reg[(reg["sample"] == "ext_A") & (reg.H == 20) & (reg.window == 60) & (reg.kind == "causal")]
    tabs["T4_regimes_ext_A_H20_W60"] = r[["regime", "n", "vr_corr", "median_ratio_corr", "qlike_corr", "qlike_nl", "delta_vs_nl", "delta_lo", "delta_hi"]]

    iv = pd.read_csv(R / "intervals" / "interval_coverage.csv")
    tabs["T5_interval_coverage"] = iv.groupby(["level", "W"], as_index=False).agg(cells=("n", "size"), mean_coverage=("coverage", "mean"),
                                                                                    min_coverage=("coverage", "min"), max_coverage=("coverage", "max"))

    tabs["T6_second_stage_verdict"] = pd.read_csv(R / "second_stage" / "second_stage_verdict.csv")[
        ["model", "cells", "share_lower_qlike", "share_ci_favours", "share_lower_mace", "median_d_gaussian", "median_d_high_vol", "supported"]]

    sim = pd.read_csv(R / "simulations" / "gaussian_t_validation.csv")
    tabs["T7_level_bias_by_scenario"] = sim.groupby("scenario", as_index=False).agg(
        cells=("N", "size"), mean_bias_raw=("level_bias_raw", "mean"), mean_bias_c=("level_bias_c", "mean"), worst_bias_c=("level_bias_c", "min"))

    pv = pd.read_csv(R / "predictors" / "predictor_verdicts_hist_A.csv")
    tabs["T8_predictors"] = pv

    from .forecast import c_gauss, c_qlike
    tabs["T10_constants"] = pd.DataFrame([dict(N=N, T=Tw, c=c_gauss(N, Tw), c_qlike=c_qlike(N, Tw), ratio=c_qlike(N, Tw) / c_gauss(N, Tw))
                                          for N in (10, 30, 50) for Tw in (60, 120, 250, 750) if Tw >= N + 5])
    cq = pd.read_csv(R / "cqlike" / "cqlike_vs_c.csv")
    tabs["T9_cqlike_vs_c"] = cq.groupby("W", as_index=False).agg(
        cells=("delta", "size"), c_qlike_lower=("delta", lambda x: int((x < 0).sum())), ci_below_zero=("hi", lambda x: int((x < 0).sum())),
        mean_qlike_diff=("delta", "mean"), vr_c=("vr_c", "mean"), vr_c_qlike=("vr_cq", "mean"))

    cv = pd.read_csv(R / "conformal" / "conformal_verdict.csv")
    tabs["T11_conformal_coverage"] = cv[["level", "adequate_cells", "mean_cov_gauss", "mean_cov_pooled", "mean_cov_vol", "C1_share_within_0p05",
                                         "C2_mean_gap", "C3_share_beats_gauss", "vol_beats_pooled_share"]]
    kv = pd.read_csv(R / "simulations" / "kurtosis_validation_v2.csv")
    tabs["T12_kurtosis_simulation"] = kv.groupby("scenario", as_index=False)[["beta_mean", "bias_c", "bias_kc", "qlike_cq", "qlike_kcq", "vr_cq", "vr_kcq"]].mean()
    ke = pd.read_csv(R / "kurtosis" / "kurtosis_vs_cqlike.csv")
    tabs["T13_kurtosis_market"] = ke.groupby("W", as_index=False).agg(
        cells=("delta", "size"), beta_hat=("beta_mean", "mean"), lower_qlike=("delta", lambda x: int((x < 0).sum())), ci_below_zero=("hi", lambda x: int((x < 0).sum())),
        mean_qlike_diff=("delta", "mean"), vr_c_qlike=("vr_cq", "mean"), vr_kappa=("vr_kcq", "mean"))

    rp = R.parent / "journal_extension_us" / "replication_india_vs_us.csv"
    if rp.exists():
        r = pd.read_csv(rp).set_index("market")
        f = lambda a, b: [f"{int(r.loc[m][a])} of {int(r.loc[m][b])}" for m in ("india", "us")]
        pc = lambda c: [f"{100 * r.loc[m][c]:.1f}%" for m in ("india", "us")]
        rows = {"Corrected forecast has lower mean QLIKE (all comparisons)": f("lower_mean_qlike", "comparisons"),
                "Interval below zero, extended samples": f("ext_ci_below_zero", "ext_comparisons"),
                "Interval below zero, W <= 120": f("short_ci_below_zero", "short_comparisons"),
                "Interval below zero, W >= 500": f("long_ci_below_zero", "long_comparisons"),
                "Comparisons where the correction wins fewer than half the origins": [str(int(r.loc[m].win_rate_losses)) for m in ("india", "us")],
                "c_QLIKE lower QLIKE than c (cells)": f("cqlike_lower_cells", "cqlike_cells"),
                "c_QLIKE interval below zero (cells)": f("cqlike_ci_below_zero", "cqlike_cells"),
                "Gaussian interval coverage, nominal 80% / 90%": [f"{100 * r.loc[m].gauss_cov_80:.1f}% / {100 * r.loc[m].gauss_cov_90:.1f}%" for m in ("india", "us")],
                "Conformal interval coverage, nominal 80% / 90%": [f"{100 * r.loc[m].conf_cov_80:.1f}% / {100 * r.loc[m].conf_cov_90:.1f}%" for m in ("india", "us")],
                "Kurtosis multiplier: lower QLIKE (cells)": f("kurt_lower_cells", "kurt_cells"),
                "Kurtosis multiplier: interval below zero (cells)": f("kurt_ci_below_zero", "kurt_cells"),
                "Kurtosis multiplier: realised/forecast before -> after": [f"{r.loc[m].kurt_vr_before:.2f} -> {r.loc[m].kurt_vr_after:.2f}" for m in ("india", "us")]}
        tabs["T14_india_vs_us"] = pd.DataFrame({"result": list(rows), "India": [v[0] for v in rows.values()], "US": [v[1] for v in rows.values()]})

    ru_rows = []
    for mk, base in (("India", R), ("US", R.parent / "journal_extension_us")):
        f = base / "risk_use" / "risk_use_cells.csv"
        if not f.exists():
            continue
        ru = pd.read_csv(f)
        ru = ru[ru["sample"].isin(["ext_A", "ext_B", "ext_AB"])]
        for grp, sub in (("W<=120", ru[ru.W <= 120]), ("all W", ru)):
            g = sub.groupby("variant").agg(viol5=("viol_5", "mean"), viol1=("viol_1", "mean"), kup5=("kupiec_p_5", lambda x: float((x < 0.05).mean())),
                                           vt_abs_log=("vt_mean_abs_log", "mean"), breach=("vt_breach", "mean"), exposure=("vt_mean_exposure", "mean")).reset_index()
            g.insert(0, "windows", grp)
            g.insert(0, "market", mk)
            ru_rows.append(g)
    if ru_rows:
        allru = pd.concat(ru_rows, ignore_index=True)
        allru.to_csv(T / "risk_use_summary.csv", index=False)
        order = ["plugin_raw", "c", "c_qlike", "kappa_cq", "nl", "ledoit_wolf", "bps_verbatim", "equal_weight"]
        t15 = allru[allru.variant.isin(order)].copy()
        t15["variant"] = pd.Categorical(t15.variant, order, ordered=True)
        t15 = t15.sort_values(["market", "windows", "variant"], ascending=[True, False, True])
        tabs["T15_var_backtest"] = t15[["market", "windows", "variant", "viol5", "kup5", "viol1"]].reset_index(drop=True)
        tabs["T16_vol_targeting"] = t15[["market", "windows", "variant", "vt_abs_log", "breach", "exposure"]].reset_index(drop=True)

    st_rows, po_rows = [], []
    for mk, base in (("India", R), ("US", R.parent / "journal_extension_us")):
        f = base / "same_target" / "fixed_target_tests.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        t = t[~t.alt.str.startswith("level_") & t["sample"].isin(["ext_A", "ext_B", "ext_AB"])]
        for tg, lab in (("gmv", "raw minimum-variance"), ("ew", "equal-weight (control)")):
            for alt in ("raw", "clipped", "mp", "ledoit_wolf", "nl"):
                g = t[(t.target == tg) & (t.alt == alt)]
                st_rows.append(dict(market=mk, target=lab, vs=alt, comparisons=len(g), lower=float((g.delta < 0).mean()), ci_below=float((g.hi < 0).mean()),
                                    ci_above=float((g.lo > 0).mean()), mean_diff=float(g.delta.mean())))
        o = pd.read_csv(base / "same_target" / "portfolio_outcomes.csv")
        o = o[o["sample"].isin(["ext_A", "ext_B", "ext_AB"])]
        for grp, sub in (("W<=120", o[o.W <= 120]), ("W>=500", o[o.W >= 500])):
            g = sub.groupby("portfolio")[["ann_vol", "max_drawdown", "ann_return", "mean_max_abs_w"]].mean().reset_index()
            g.insert(0, "windows", grp)
            g.insert(0, "market", mk)
            po_rows.append(g)
    if st_rows:
        sdf = pd.DataFrame(st_rows)
        sdf.to_csv(T / "same_target_summary.csv", index=False)
        tabs["T17_same_target"] = sdf
        order = ["raw", "clipped", "mp", "ledoit_wolf", "nl", "bps_verbatim", "equal_weight"]
        pdf = pd.concat(po_rows, ignore_index=True)
        pdf["portfolio"] = pd.Categorical(pdf.portfolio, order, ordered=True)
        tabs["T18_portfolio_outcomes"] = pdf.sort_values(["market", "windows", "portfolio"], ascending=[True, True, True]).reset_index(drop=True)
        tabs["T18_portfolio_outcomes"]["windows"] = tabs["T18_portfolio_outcomes"]["windows"]
        pdf.to_csv(T / "portfolio_outcomes_summary.csv", index=False)
    tabs["T19_comparator_roles"] = pd.DataFrame([
        ("Raw plug-in", "yes", "yes", "yes", "yes"), ("Eigenvalue clipping", "yes", "yes", "yes", "yes"), ("Marchenko-Pastur filter", "yes", "yes", "yes", "yes"),
        ("Ledoit-Wolf", "yes", "yes", "yes", "yes"), ("Nonlinear shrinkage", "yes", "yes", "yes", "yes"), ("Weight shrinkage (BPS)", "yes", "no", "yes", "yes"),
        ("Equal weights", "no", "control only", "yes", "yes")],
        columns=["estimator", "own-portfolio QLIKE (348 comparisons)", "same-target QLIKE", "VaR and volatility targeting", "portfolio outcomes"])

    for name, df in tabs.items():
        df = df.copy()
        for c in ("W", "periods", "cells", "n", "comparisons", "c_qlike_lower", "ci_below_zero", "adequate_cells", "lower_qlike"):
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].round().astype(int)
        tabs[name] = df
        df.to_csv(T / f"{name}.csv", index=False)
    (T / "tables.md").write_text("\n\n".join(f"### {k}\n\n{to_md(v)}" for k, v in tabs.items()) + "\n")
    return tabs
