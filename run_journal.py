"""Journal-extension pipeline (separate from run_all.py; writes only to results/journal_extension/).

    python run_journal.py baseline            # dev-sample rolling forecasts, paired tests (needs data/prices.csv)
    python run_journal.py validate [--reps N] # Gaussian + Student-t Monte Carlo grid (no market data needed)
    python run_journal.py dynamic             # GARCH and regime-switching Monte Carlo (no market data)
    python run_journal.py surface             # (N,T) subset calibration surface on real data
    python run_journal.py regimes             # causal volatility / calendar regime calibration on real data
    python run_journal.py second_stage        # prequential second-stage calibration + controls (needs the extended stage output)
    python run_journal.py predictors          # Table III / Sec. V-C: pre-specified predictor test on 2011-21, vol control, signal-in-principle and power
    python run_journal.py extended            # five paper samples, H=20/60, NL/BPS added (needs data/prices_2011_2026.csv)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, data
from src.journal import forecast, inference, metrics, runmeta, scenarios
from src.multitest import holm_adjust
from src.simulate import gmv_stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "journal_extension"
ALTS = ["raw", "clipped", "mp", "ledoit_wolf"]


def _returns(policy: str) -> pd.DataFrame:
    R, _, _ = data.get_returns(config.PRICES_CSV, config.TICKERS)
    if policy == "P3_drop_zero_rows":
        R = R[~(R == 0).all(axis=1)]
    elif policy != "P1_existing":
        raise ValueError(policy)
    return R


def baseline() -> None:
    d = OUT / "empirical"
    d.mkdir(parents=True, exist_ok=True)
    summaries, tests = [], []
    for policy in ("P1_existing", "P3_drop_zero_rows"):
        R = _returns(policy)
        for H in (20, 60):
            for proxy in (("sample", "mean_square") if policy == "P1_existing" else ("sample",)):
                fc = forecast.rolling_forecasts(R, H=H, proxy=proxy)
                n_fail = int(fc.failed.sum())
                lt = forecast.loss_table(fc)
                lt.assign(policy=policy, proxy=proxy).to_csv(d / f"forecasts_{policy}_H{H}_{proxy}.csv", index=False)
                for (W, name), g in lt.groupby(["window", "forecast"]):
                    s = metrics.calibration_summary(g.v.to_numpy(), g.h.to_numpy())
                    summaries.append(dict(policy=policy, proxy=proxy, H=H, window=W, forecast=name, failed_origins=n_fail, **s))
                if proxy != "sample":
                    continue
                fam = []
                for W in config.WINDOWS:
                    pivot = lt[lt.window == W].pivot(index="origin", columns="forecast", values="qlike")
                    for alt in ALTS + ["raw_corrected_qlike"]:
                        r = inference.paired_loss_test(pivot["raw_corrected"].to_numpy(), pivot[alt].to_numpy())
                        fam.append(dict(policy=policy, H=H, window=W, A="raw_corrected", B=alt, primary=alt in ALTS,
                                        n=r.n, mean_diff=r.mean_diff, ci_lo=r.lo, ci_hi=r.hi, win_rate=r.win_rate,
                                        win_lo=r.win_lo, win_hi=r.win_hi, p_boot=r.p_two_sided, block=r.block,
                                        robust_win=r.robust_win))
                f = pd.DataFrame(fam)
                prim = f.primary
                f.loc[prim, "p_holm"] = holm_adjust(f.loc[prim, "p_boot"].to_numpy())
                tests.append(f)
    pd.DataFrame(summaries).to_csv(d / "calibration_summary.csv", index=False)
    pd.concat(tests).to_csv(d / "paired_tests.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "baseline.json", "J-BASELINE-DEV", ROOT, config="configs/journal_baseline_dev.yaml",
                       data_file="data/prices.csv", data_sha256=runmeta.file_sha256(config.PRICES_CSV), seed=config.SEED,
                       status="success")
    print("baseline done ->", d)


def _cell(structure: str, scen: str, N: int, T: int, reps: int, seed: int, H: int = 60, batch: int = 200) -> dict:
    ss = np.random.SeedSequence([seed, scenarios.STRUCTURES.index(structure), ["gaussian", "t5", "t8", "t15", "skew4", "skew1", "skewneg4", "skewt8"].index(scen), N, T])
    rng = np.random.default_rng(ss)
    Sigma = scenarios.make_sigma(structure, N)
    one = np.ones(N)
    V = 1.0 / float(one @ np.linalg.solve(Sigma, one))
    pred, true, real = [], [], []
    for lo in range(0, reps, batch):
        b = min(batch, reps - lo)
        Xtr, Xte = scenarios.draw(scen, b, T, Sigma, rng), scenarios.draw(scen, b, H, Sigma, rng)
        p, t, r = gmv_stats(Xtr, Xte, Sigma)
        pred.append(p), true.append(t), real.append(r)
    pred, true, real = map(np.concatenate, (pred, true, real))
    ok = np.isfinite(pred) & np.isfinite(true) & np.isfinite(real) & (pred > 0) & (real > 0)
    fail = 1.0 - ok.mean()
    pred, true, real = pred[ok], true[ok], real[ok]
    n = len(pred)
    c, cq = forecast.c_gauss(N, T), forecast.c_qlike(N, T)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    out = dict(structure=structure, scenario=scen, N=N, T=T, reps=reps, n_ok=n, fail_rate=float(fail), V=V,
               EQ_mc=float(np.mean(pred / V)), EQ_se=se(pred / V), EQ_th=(T - N) / (T - 1),
               ER_mc=float(np.mean(true / V)), ER_se=se(true / V), ER_th=(T - 2) / (T - N - 1), c=c, c_qlike=cq)
    # known-Sigma target: level bias of the corrected forecast, per-run ratio distribution
    d = c * pred - true
    out.update(level_bias_c=float(d.mean() / true.mean()), level_bias_c_se=float(d.std(ddof=1) / np.sqrt(n) / true.mean()),
               level_bias_raw=float((pred - true).mean() / true.mean()),
               ratio_known_mean=float(np.mean(true / (c * pred))), ratio_known_median=float(np.median(true / (c * pred))),
               ratio_known_mean_se=se(true / (c * pred)),
               mace_known_c=float(np.mean(np.abs(np.log(true / (c * pred))))),
               mace_known_cq=float(np.mean(np.abs(np.log(true / (cq * pred))))))
    # noisy future-block target (H days): reported separately from the known-Sigma target
    for tag, f in (("raw", 1.0), ("c", c), ("cq", cq)):
        out[f"qlike_block_{tag}"] = float(metrics.qlike(real, f * pred).mean())
        out[f"vr_block_{tag}"] = float(real.mean() / (f * pred).mean())
    out["z_EQ"] = (out["EQ_mc"] - out["EQ_th"]) / out["EQ_se"] if scen == "gaussian" else np.nan
    out["z_ER"] = (out["ER_mc"] - out["ER_th"]) / out["ER_se"] if scen == "gaussian" else np.nan
    out["z_level_bias_c"] = out["level_bias_c"] / out["level_bias_c_se"]
    return out


def validate(reps: int, grid: dict | None = None) -> None:
    d = OUT / "simulations"
    d.mkdir(parents=True, exist_ok=True)
    g = grid or dict(N=[5, 10, 20, 30, 40, 50], T=[60, 90, 120, 250, 500], structures=list(scenarios.STRUCTURES),
                     scenarios=["gaussian", "t5", "t8", "t15"])
    rows, t0 = [], time.time()
    seed = 20261102
    for st in g["structures"]:
        for sc in g["scenarios"]:
            for N in g["N"]:
                for T in g["T"]:
                    if T < N + 5:
                        continue
                    rows.append(_cell(st, sc, N, T, reps, seed))
        print(f"  {st} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    name = "gaussian_t_validation.csv" if grid is None else "gaussian_t_validation_smoke.csv"
    df.to_csv(d / name, index=False)
    gz = df[df.scenario == "gaussian"]
    runmeta.write_meta(OUT / "metadata" / ("validate.json" if grid is None else "validate_smoke.json"), "J-VALIDATE-MC", ROOT,
                       config="configs/journal_validation_mc.yaml", reps=reps, seed=seed, cells=len(df),
                       gaussian_cells=len(gz), gaussian_max_abs_z=float(gz[["z_EQ", "z_ER"]].abs().max().max()),
                       max_fail_rate=float(df.fail_rate.max()), status="success")
    print("validate done ->", d / name, f"({len(df)} cells, {time.time() - t0:.0f}s)")


def validate_skew(reps: int) -> None:
    """Same (N, T, structure) grid as `validate`, with skewed return generators; writes a NEW file."""
    d = OUT / "simulations"
    rows, t0, seed = [], time.time(), 20261103
    for st in scenarios.STRUCTURES:
        for sc in ("skew4", "skew1", "skewneg4", "skewt8"):
            for N in (5, 10, 20, 30, 40, 50):
                for T in (60, 90, 120, 250, 500):
                    if T >= N + 5:
                        rows.append(_cell(st, sc, N, T, reps, seed))
        print(f"  {st} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(d / "skewed_validation.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "validate_skew.json", "J-VALIDATE-SKEW", ROOT, reps=reps, seed=seed, cells=len(df),
                       scenarios=["skew4 (gamma k=4: skew 1.0, ex.kurt 1.5)", "skew1 (k=1: skew 2.0, ex.kurt 6)", "skewneg4", "skewt8 (k=4 plus t8 mixing)"],
                       max_fail_rate=float(df.fail_rate.max()), status="success")
    print("validate_skew done ->", d / "skewed_validation.csv", f"({len(df)} cells, {time.time() - t0:.0f}s)")


def cqlike_check() -> None:
    """Empirical comparison of the two constants: corrected with c (level-unbiased) vs c_QLIKE (loss-optimal under Gaussian i.i.d.)."""
    d = OUT / "cqlike"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("dev", "hist_A", "ext_A", "ext_B", "ext_AB"):
        for H in (20, 60):
            lt = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            L = 5 if H == 20 else 3
            for W in sorted(lt.window.unique()):
                g = lt[lt.window == W]
                pv = g.pivot(index="origin", columns="forecast", values="qlike")
                hh = g.pivot(index="origin", columns="forecast", values="h")
                vv = g.pivot(index="origin", columns="forecast", values="v")
                r = inference.paired_loss_test(pv["raw_corrected_qlike"].to_numpy(), pv["raw_corrected"].to_numpy(), block=L)
                rows.append(dict(sample=name, H=H, W=int(W), n=r.n, q_c=float(pv["raw_corrected"].mean()),
                                 q_cq=float(pv["raw_corrected_qlike"].mean()), delta=r.mean_diff, lo=r.lo, hi=r.hi,
                                 win=r.win_rate, vr_c=float(vv["raw_corrected"].mean() / hh["raw_corrected"].mean()),
                                 vr_cq=float(vv["raw_corrected_qlike"].mean() / hh["raw_corrected_qlike"].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(d / "cqlike_vs_c.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "cqlike.json", "J-CQLIKE-EMPIRICAL", ROOT, data_files=["data/prices_2011_2026.csv"], status="success")
    print("cqlike_check done ->", d)


def conformal_stage() -> None:
    """Conformal vs Gaussian-theory prediction intervals for realised variance (configs/journal_conformal.yaml)."""
    from src.journal import conformal, intervals, regimes
    d = OUT / "conformal"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name in ("hist_A", "ext_A", "ext_B", "ext_AB"):
        R = _sample_returns(name)
        N = R.shape[1]
        for H in (20, 60):
            fc = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            L = 5 if H == 20 else 3
            origins = sorted(fc.origin.unique())
            grp = regimes.label_origins(R, origins, H).set_index("origin").vol_regime.loc[origins].to_numpy()
            for W in sorted(fc.window.unique()):
                g = fc[fc.window == W].set_index("origin")
                cq = g[g.forecast == "raw_corrected_qlike"].loc[origins]
                cc = g[g.forecast == "raw_corrected"].loc[origins]
                v, h = cq.v.to_numpy(), cq.h.to_numpy()
                y = np.log(v)
                rho = np.log(v / h)
                for lvl in (0.8, 0.9):
                    a = 1 - lvl
                    lo_p, hi_p, ok = conformal.conformal_bounds(rho, a)
                    lo_v, hi_v, _ = conformal.conformal_bounds(rho, a, groups=grp)
                    ql, qh = intervals.ratio_quantiles(N, int(W), H, [a / 2, 1 - a / 2], seed=config.SEED)
                    shift = np.log(cc.h.to_numpy() / h)                       # log(h_c / h_cq): re-centre Gaussian quantiles on the c_QLIKE forecast
                    lo_g, hi_g = np.log(h) + np.log(ql) + 0 * shift - np.log(c_ratio := forecast.c_qlike(N, int(W)) / forecast.c_gauss(N, int(W))), \
                        np.log(h) + np.log(qh) - np.log(c_ratio)
                    rho_c = np.log(v / cc.h.to_numpy())
                    lo_c, hi_c, _ = conformal.conformal_bounds(rho_c, a)
                    variants = {"gauss": (lo_g, hi_g), "conformal_pooled": (np.log(h) + lo_p, np.log(h) + hi_p),
                                "conformal_vol": (np.log(h) + lo_v, np.log(h) + hi_v),
                                "conformal_pooled_c_centre": (np.log(cc.h.to_numpy()) + lo_c, np.log(cc.h.to_numpy()) + hi_c)}
                    for vn, (l, u) in variants.items():
                        m = ok
                        hit = ((y >= l) & (y <= u))[m].astype(float)
                        idx = inference.moving_block_indices(len(hit), L, 1000, rng)
                        boot = hit[idx[:, :len(hit)]].mean(axis=1) if len(hit) else np.array([np.nan])
                        rows.append(dict(sample=name, H=H, W=int(W), level=lvl, variant=vn, n_eval=int(m.sum()),
                                         coverage=float(hit.mean()) if len(hit) else np.nan,
                                         lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
                                         width=float((u - l)[m].mean()) if m.any() else np.nan,
                                         score=float(conformal.interval_score(y[m], l[m], u[m], a).mean()) if m.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(d / "conformal_intervals.csv", index=False)
    a = df[df.n_eval >= 20]
    ver = []
    for lvl in (0.8, 0.9):
        pv = a[a.level == lvl].pivot_table(index=["sample", "H", "W"], columns="variant", values=["coverage", "score"])
        cp = pv["coverage"]["conformal_pooled"]
        ver.append(dict(level=lvl, adequate_cells=len(cp), mean_cov_pooled=cp.mean(), mean_cov_vol=pv["coverage"]["conformal_vol"].mean(),
                        mean_cov_gauss=pv["coverage"]["gauss"].mean(),
                        C1_share_within_0p05=float(((cp - lvl).abs() <= 0.05).mean()), C1_pass=bool(((cp - lvl).abs() <= 0.05).mean() >= 0.75),
                        C2_mean_gap=float(cp.mean() - lvl), C2_pass=bool(abs(cp.mean() - lvl) <= 0.03),
                        C3_share_beats_gauss=float((pv["score"]["conformal_pooled"] < pv["score"]["gauss"]).mean()),
                        C3_pass=bool((pv["score"]["conformal_pooled"] < pv["score"]["gauss"]).mean() >= 0.75),
                        vol_beats_pooled_share=float((pv["score"]["conformal_vol"] < pv["score"]["conformal_pooled"]).mean()),
                        vol_share_within_0p05=float(((pv["coverage"]["conformal_vol"] - lvl).abs() <= 0.05).mean())))
    pd.DataFrame(ver).to_csv(d / "conformal_verdict.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "conformal.json", "J-CONFORMAL", ROOT, config="configs/journal_conformal.yaml",
                       data_files=["data/prices_2011_2026.csv"], status="success")
    print("conformal done ->", d, len(df), "rows")


def kurtosis_sim(reps: int = 2000, version: str = "v2") -> None:
    """Simulation checks S1-S3 of configs/journal_kurtosis.yaml (beta_hat estimated from each training window)."""
    from src.journal import kurtosis, sim_dynamic
    d = OUT / "simulations"
    scen = ["gaussian", "t5", "t8", "t15", "skew4", "skew1", "skewt8", "garch", "regime"]
    seed, rows, t0 = 20261104, [], time.time()
    H = 60
    for st in scenarios.STRUCTURES:
        for sc in scen:
            for N in (10, 30):
                for T in (60, 120, 250):
                    ss = np.random.SeedSequence([seed, scenarios.STRUCTURES.index(st), scen.index(sc), N, T])
                    rng = np.random.default_rng(ss)
                    Sigma = scenarios.make_sigma(st, N)
                    c, cq = forecast.c_gauss(N, T), forecast.c_qlike(N, T)
                    recs = []
                    for lo in range(0, reps, 250):
                        b = min(250, reps - lo)
                        if sc in ("garch", "regime"):
                            X, _ = sim_dynamic.simulate(sc, Sigma, b, T + H, rng)
                        else:
                            X = scenarios.draw(sc, b, T + H, Sigma, rng)
                        Xtr, Xte = X[:, :T], X[:, T:]
                        pred, true, real = gmv_stats(Xtr, Xte, Sigma)
                        bh = kurtosis.mardia_ratio_batch(Xtr) if version == "v1" else kurtosis.ew_kurtosis_ratio_batch(Xtr)
                        recs.append(pd.DataFrame(dict(pred=pred, true=true, real=real, beta=bh)))
                    g = pd.concat(recs, ignore_index=True)
                    g = g[(g.pred > 0) & (g.real > 0)]
                    m = kurtosis.multiplier(g.beta.to_numpy(), N, T)
                    o = dict(structure=st, scenario=sc, N=N, T=T, n=len(g), beta_mean=float(g.beta.mean()), m_mean=float(np.mean(m)))
                    for nm, f in (("c", c), ("cq", cq), ("kc", c * m), ("kcq", cq * m)):
                        o[f"bias_{nm}"] = float(((f * g.pred - g.true).mean()) / g.true.mean())
                        o[f"qlike_{nm}"] = float(metrics.qlike(g.real.to_numpy(), f * g.pred.to_numpy()).mean())
                        o[f"vr_{nm}"] = float(g.real.mean() / (f * g.pred).mean())
                    rows.append(o)
        print(f"  {st} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    sfx = "" if version == "v1" else "_v2"
    df.to_csv(d / f"kurtosis_validation{sfx}.csv", index=False)
    fam = df[df.scenario.isin(["t5", "t8", "t15"])]
    gz = df[df.scenario == "gaussian"]
    hs = df[df.scenario.isin(["skewt8", "garch", "regime"])]
    v = dict(S1_ratio_abs_bias=float(fam.bias_kc.abs().mean() / fam.bias_c.abs().mean()),
             S2_share_no_harm=float((gz.bias_kc.abs() <= gz.bias_c.abs() + 0.02).mean()),
             S3_share_lower_qlike=float((hs.qlike_kcq < hs.qlike_cq).mean()))
    v["S1_pass"], v["S2_pass"], v["S3_pass"] = v["S1_ratio_abs_bias"] <= 0.5, v["S2_share_no_harm"] >= 0.9, v["S3_share_lower_qlike"] >= 0.6
    pd.DataFrame([v]).to_csv(d / f"kurtosis_sim_verdict{sfx}.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / f"kurtosis_sim_{version}.json", f"J-KURTOSIS-SIM-{version}", ROOT, config="configs/journal_kurtosis.yaml", reps=reps, seed=seed, cells=len(df), status="success", **v)
    print("kurtosis_sim done ->", v)


def kurtosis_eval() -> None:
    """Market-data evaluation of the kurtosis-aware multiplier, once, as pre-specified in configs/journal_kurtosis.yaml."""
    from src.journal import kurtosis, metrics as M, regimes
    d = OUT / "kurtosis"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("ext_A", "ext_B", "ext_AB"):
        R = _sample_returns(name)
        X = R.to_numpy()
        N = X.shape[1]
        for H in (20, 60):
            fc = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            L = 5 if H == 20 else 3
            origins = sorted(fc.origin.unique())
            hv = regimes.label_origins(R, origins, H).set_index("origin").vol_regime
            for W in sorted(fc.window.unique()):
                g = fc[(fc.window == W) & (fc.forecast == "raw_corrected_qlike")].set_index("origin").loc[origins]
                gc = fc[(fc.window == W) & (fc.forecast == "raw_corrected")].set_index("origin").loc[origins]
                beta = np.array([kurtosis.ew_kurtosis_ratio(X[o - int(W):o]) for o in origins])      # v2 estimator
                m = kurtosis.multiplier(beta, N, int(W))
                v = g.v.to_numpy()
                q0 = M.qlike(v, g.h.to_numpy())
                q1 = M.qlike(v, m * g.h.to_numpy())
                qc = M.qlike(v, gc.h.to_numpy())
                qkc = M.qlike(v, m * gc.h.to_numpy())
                r = inference.paired_loss_test(q1, q0, block=L)
                diff = q1 - q0
                hi = (hv.loc[origins] == "high").to_numpy()
                keep = np.argsort(diff)[int(np.ceil(0.05 * len(diff))):]           # drop the 5% largest improvements (most negative)
                rows.append(dict(sample=name, H=H, W=int(W), n=len(origins), beta_mean=float(beta.mean()), m_mean=float(m.mean()),
                                 q_cq=float(q0.mean()), q_kcq=float(q1.mean()), delta=r.mean_diff, lo=r.lo, hi=r.hi, win=r.win_rate,
                                 delta_trim=float(diff[keep].mean()), delta_high=float(diff[hi].mean()) if hi.any() else np.nan, n_high=int(hi.sum()),
                                 delta_kc_vs_c=float((qkc - qc).mean()), vr_cq=float(v.mean() / g.h.mean()), vr_kcq=float(v.mean() / (m * g.h).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(d / "kurtosis_vs_cqlike.csv", index=False)
    lower = df.delta < 0
    ver = dict(cells=len(df), D1_share_lower=float(lower.mean()), D1_pass=bool(lower.mean() >= 0.75),
               D2_share_ci=float((df.hi < 0).mean()), D2_pass=bool((df.hi < 0).mean() >= 0.25),
               D3_median_high_vol=float(df.delta_high.median()), D3_pass=bool(df.delta_high.median() <= 0.02),
               D4_share_robust=float((df.delta_trim[lower] < 0).mean()), D4_pass=bool((df.delta_trim[lower] < 0).mean() >= 0.75))
    ver["supported"] = all(ver[k] for k in ("D1_pass", "D2_pass", "D3_pass", "D4_pass"))
    pd.DataFrame([ver]).to_csv(d / "kurtosis_verdict.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "kurtosis_eval.json", "J-KURTOSIS-EVAL", ROOT, config="configs/journal_kurtosis.yaml",
                       data_files=["data/prices_2011_2026.csv"], status="success", **{k: (v if not isinstance(v, (np.bool_,)) else bool(v)) for k, v in ver.items()})
    print("kurtosis_eval done ->", ver)


PAPER_ALTS = ["raw", "clipped", "mp", "ledoit_wolf", "nl", "bps_verbatim"]   # BPS = verbatim Bodnar-Parolya-Schmid (2018) intensity
EXT_CSV = ROOT / "data" / "prices_2011_2026.csv"


def _sample_returns(name: str) -> pd.DataFrame:
    """The five conference samples. dev = the cached 2021-26 file (as in the paper); others from data/prices_2011_2026.csv."""
    from src.data import clean_prices, log_returns
    if name == "dev":
        return _returns("P1_existing")
    p = pd.read_csv(EXT_CSV, index_col=0, parse_dates=True)
    tk = {"hist_A": config.TICKERS, "ext_A": config.TICKERS, "ext_B": config.NIFTY_NEXT_TICKERS,
          "ext_AB": config.TICKERS + config.NIFTY_NEXT_TICKERS}[name]
    hi = "2021-10-07" if name == "hist_A" else "2026-10-06"
    c, _ = clean_prices(p.loc["2011-06-01":hi, tk])
    return log_returns(c).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def extended() -> None:
    d = OUT / "extended"
    d.mkdir(parents=True, exist_ok=True)
    tests, tabs, meta = [], [], {}
    for name in ("dev", "hist_A", "ext_A", "ext_B", "ext_AB"):
        R = _sample_returns(name)
        N = R.shape[1]
        wins = [W for W in config.WINDOWS if name != "ext_AB" or W > 1.5 * N]
        meta[name] = dict(T=len(R), N=N, windows=wins, start=str(R.index[0].date()), end=str(R.index[-1].date()))
        for H in (20, 60):
            lt = forecast.loss_table(forecast.rolling_forecasts(R, windows=wins, H=H, extended=True))
            lt.to_csv(d / f"forecasts_{name}_H{H}.csv", index=False)
            L = 5 if H == 20 else 3                          # the conference paper's block lengths
            for W in wins:
                g = lt[lt.window == W]
                pv = g.pivot(index="origin", columns="forecast", values="qlike")
                vv = g.pivot(index="origin", columns="forecast", values="v")
                hh = g.pivot(index="origin", columns="forecast", values="h")
                vr = lambda f: float(vv[f].mean() / hh[f].mean())
                row = dict(sample=name, H=H, W=W, periods=len(pv), vr_raw=vr("raw"), vr_corr=vr("raw_corrected"), vr_nl=vr("nl"),
                           **{f"q_{k}": float(pv[f].mean()) for k, f in (("raw", "raw"), ("corr", "raw_corrected"), ("lw", "ledoit_wolf"),
                                                                       ("nl", "nl"), ("bps", "bps_verbatim"), ("bps_analogue", "bps"), ("clip", "clipped"), ("mp", "mp"),
                                                                       ("ew", "equal_weight"))})
                tabs.append(row)
                for alt in PAPER_ALTS:
                    r = inference.paired_loss_test(pv["raw_corrected"].to_numpy(), pv[alt].to_numpy(), block=L)
                    tests.append(dict(sample=name, H=H, W=W, alt=alt, n=r.n, block=L, mean_corr=float(pv["raw_corrected"].mean()),
                                      mean_alt=float(pv[alt].mean()), delta=r.mean_diff, lo=r.lo, hi=r.hi, win=r.win_rate,
                                      win_lo=r.win_lo, win_hi=r.win_hi, robust=bool(r.hi < 0 and r.win_lo > 0.5),
                                      loses=bool(r.win_hi < 0.5)))
        print(f"  {name} done", flush=True)
    pd.DataFrame(tabs).to_csv(d / "table2_all_rows.csv", index=False)
    pd.DataFrame(tests).to_csv(d / "paired_tests_extended.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "extended.json", "J-EXTENDED-5SAMPLES", ROOT, data_files=["data/prices.csv", "data/prices_2011_2026.csv"],
                       data_sha256={"prices.csv": runmeta.file_sha256(config.PRICES_CSV), "prices_2011_2026.csv": runmeta.file_sha256(EXT_CSV)},
                       samples=meta, block_rule="5 (H=20) / 3 (H=60), as in the conference paper", B=config.BOOT_N, seed=config.SEED, status="success")
    print("extended done ->", d)


def bps_check() -> None:
    """Compare the finite-sample BPS analogue with the verbatim Bodnar-Parolya-Schmid (2018) intensity (new output dir only)."""
    d = OUT / "bps_check"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("dev", "hist_A", "ext_A", "ext_B", "ext_AB"):
        R = _sample_returns(name)
        wins = [W for W in config.WINDOWS if name != "ext_AB" or W > 1.5 * R.shape[1]]
        for H in (20, 60):
            lt = forecast.loss_table(forecast.rolling_forecasts(R, windows=wins, H=H, extended=True))
            L = 5 if H == 20 else 3
            for W in wins:
                pv = lt[lt.window == W].pivot(index="origin", columns="forecast", values="qlike")
                for alt in ("bps", "bps_verbatim"):
                    r = inference.paired_loss_test(pv["raw_corrected"].to_numpy(), pv[alt].to_numpy(), block=L)
                    rows.append(dict(sample=name, H=H, W=W, alt=alt, n=r.n, mean_alt=float(pv[alt].mean()),
                                     mean_corr=float(pv["raw_corrected"].mean()), delta=r.mean_diff, lo=r.lo, hi=r.hi))
        print(f"  {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(d / "bps_variants.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "bps_check.json", "J-BPS-CHECK", ROOT, data_files=["data/prices_2011_2026.csv"], status="success")
    print("bps_check done ->", d)


def figures_stage() -> None:
    """Regenerate every journal figure from the saved CSVs only."""
    from src.journal import figures
    f = OUT / "figures"
    sim = pd.read_csv(OUT / "simulations" / "gaussian_t_validation.csv")
    figures.fig_gaussian_validation(sim, f)
    figures.fig_level_bias(sim, f)
    figures.fig_c_vs_cqlike(sim, f)
    figures.fig_dynamic(pd.read_csv(OUT / "simulations" / "dynamic_validation.csv"), f)
    figures.fig_surface(pd.read_csv(OUT / "empirical" / "calibration_surface.csv"),
                        pd.read_csv(OUT / "empirical" / "calibration_surface_fits.csv", index_col=0), f)
    figures.fig_second_stage(pd.read_csv(OUT / "second_stage" / "second_stage_summary.csv"), f)
    print("figures done ->", f)


def intervals_stage() -> None:
    """Empirical coverage of Gaussian-theory 80%/90% prediction intervals for realised variance (corrected raw forecast)."""
    from src.journal import intervals
    d = OUT / "intervals"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name in ("dev", "hist_A", "ext_A", "ext_B", "ext_AB"):
        N = _sample_returns(name).shape[1]
        for H in (20, 60):
            fc = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            fc = fc[fc.forecast == "raw_corrected"]
            L = 5 if H == 20 else 3
            for W in sorted(fc.window.unique()):
                g = fc[fc.window == W].sort_values("origin")
                v, h = g.v.to_numpy(), g.h.to_numpy()
                for lvl in (0.8, 0.9):
                    a = (1 - lvl) / 2
                    qlo, qhi = intervals.ratio_quantiles(N, int(W), H, [a, 1 - a], seed=config.SEED)
                    hit = ((v >= h * qlo) & (v <= h * qhi)).astype(float)
                    idx = inference.moving_block_indices(len(hit), L, 1000, rng)
                    boot = hit[idx[:, :len(hit)]].mean(axis=1)
                    rows.append(dict(sample=name, H=H, W=int(W), level=lvl, n=len(hit), coverage=float(hit.mean()),
                                     lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
                                     below=float((v < h * qlo).mean()), above=float((v > h * qhi).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(d / "interval_coverage.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "intervals.json", "J-INTERVALS", ROOT, data_files=["data/prices_2011_2026.csv"], status="success")
    print("intervals done ->", d)


def _dyn_cell(structure: str, kind: str, N: int, T: int, reps: int, seed: int, H: int = 60, batch: int = 250) -> list[dict]:
    from src.journal import sim_dynamic
    ss = np.random.SeedSequence([seed, scenarios.STRUCTURES.index(structure), ["iid", "garch", "regime"].index(kind), N, T])
    rng = np.random.default_rng(ss)
    Sigma = scenarios.make_sigma(structure, N)
    one = np.ones(N)
    V = 1.0 / float(one @ np.linalg.solve(Sigma, one))
    c, cq = forecast.c_gauss(N, T), forecast.c_qlike(N, T)
    recs = []
    for lo in range(0, reps, batch):
        b = min(batch, reps - lo)
        if kind == "iid":
            X = scenarios.draw("gaussian", b, T + H, Sigma, rng)
            aux = np.zeros((b, T + H))
        else:
            X, aux = sim_dynamic.simulate(kind, Sigma, b, T + H, rng)
        Xtr, Xte = X[:, :T], X[:, T:]
        pred, true, real = gmv_stats(Xtr, Xte, Sigma)
        w_ew = np.full(N, 1.0 / N)
        ew = Xtr @ w_ew
        rv20 = ew[:, -20:].var(axis=1, ddof=1) / ew.var(axis=1, ddof=1)       # causal: last 20 days vs whole window
        recs.append(pd.DataFrame(dict(pred=pred, true=true, real=real, rv20=rv20, state0=aux[:, T - 1],
                                      switched=(aux[:, T:] != aux[:, [T - 1]]).any(axis=1) if kind == "regime" else False)))
    d = pd.concat(recs, ignore_index=True)
    d = d[(d.pred > 0) & (d.real > 0)]
    def stats(g, tag):
        n = len(g)
        o = dict(structure=structure, kind=kind, N=N, T=T, group=tag, n=n,
                 level_bias_c=float(((c * g.pred - g.true).mean()) / g.true.mean()) if n else np.nan)
        for nm, f in (("raw", 1.0), ("c", c), ("cq", cq)):
            o[f"qlike_{nm}"] = float(metrics.qlike(g.real.to_numpy(), f * g.pred.to_numpy()).mean())
        o["vr_block_c"] = float(g.real.mean() / (c * g.pred).mean())
        o["median_ratio_block_c"] = float(np.median(g.real / (c * g.pred)))
        return o
    out = [stats(d, "all")]
    if kind == "regime":
        out += [stats(d[d.state0 == s], f"state0={'stress' if s else 'calm'}") for s in (0, 1)]
        out += [stats(d[d.switched], "switch_in_test"), stats(d[~d.switched], "no_switch_in_test")]
    elif kind == "garch":
        lo_, hi_ = d.rv20.quantile([1 / 3, 2 / 3])
        out += [stats(d[d.rv20 <= lo_], "recent_vol_low"), stats(d[(d.rv20 > lo_) & (d.rv20 <= hi_)], "recent_vol_mid"),
                stats(d[d.rv20 > hi_], "recent_vol_high")]
    return out


def dynamic(reps: int = 3000, smoke: bool = False) -> None:
    d = OUT / "simulations"
    d.mkdir(parents=True, exist_ok=True)
    seed = 20261103
    grid = dict(N=[10, 30], T=[60, 120, 250], structures=["equicorr", "hetero"], kinds=["iid", "garch", "regime"])
    if smoke:
        grid, reps = dict(N=[10], T=[60], structures=["hetero"], kinds=["iid", "garch", "regime"]), 300
    rows, t0 = [], time.time()
    for st in grid["structures"]:
        for kind in grid["kinds"]:
            for N in grid["N"]:
                for T in grid["T"]:
                    rows += _dyn_cell(st, kind, N, T, reps, seed)
        print(f"  {st} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    name = "dynamic_validation_smoke.csv" if smoke else "dynamic_validation.csv"
    df.to_csv(d / name, index=False)
    if not smoke:
        runmeta.write_meta(OUT / "metadata" / "dynamic.json", "J-DYNAMIC-MC", ROOT, reps=reps, seed=seed, rows=len(df),
                           garch=dict(a=0.10, b=0.85), regime=dict(p_stay_calm=0.99, p_stay_stress=0.95, m_stress=2.5), status="success")
    print("dynamic done ->", d / name)


def surface_stage() -> None:
    from src.journal import figures, surface
    d = OUT / "empirical"
    d.mkdir(parents=True, exist_ok=True)
    specs = {"dev": dict(Ns=[10, 15, 20, 25, 30], label="2021-26, N<=30"), "ext_AB": dict(Ns=[10, 20, 30, 40, 50], label="2011-26, N<=50")}
    tabs, fits = [], {}
    for name, sp in specs.items():
        R = _sample_returns(name)
        res = surface.surface_samples(R, sp["Ns"], config.WINDOWS, H=60, n_subsets=20, seed=config.SEED)
        t = surface.cell_table(res, H=60)
        t.insert(0, "sample", name)
        tabs.append(t)
        fits[name] = surface.fit_vs_ratio(t, res, H=60)
        print(f"  {name}: {fits[name]}", flush=True)
    tab = pd.concat(tabs)
    tab.to_csv(d / "calibration_surface.csv", index=False)
    pd.DataFrame(fits).T.to_csv(d / "calibration_surface_fits.csv")
    runmeta.write_meta(OUT / "metadata" / "surface.json", "J-SURFACE", ROOT, seed=config.SEED, subsets=20, H=60, block=3, B=1000, status="success")
    print("surface done ->", d)


def regimes_stage() -> None:
    from src.journal import regimes
    d = OUT / "empirical"
    tabs = []
    for name in ("dev", "ext_A", "ext_AB"):
        R = _sample_returns(name)
        for H in (20, 60):
            fc = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            lab = regimes.label_origins(R, sorted(fc.origin.unique()), H)
            lab.assign(sample=name, H=H).to_csv(d / f"regime_labels_{name}_H{H}.csv", index=False)
            for by, kind in (("vol_regime", "causal"), ("covid_overlap", "calendar (fixed a priori)"), ("expost_large_fall", "ex-post diagnostic only")):
                t = regimes.regime_table(fc, lab, by, block=5 if H == 20 else 3)
                t.insert(0, "by", by); t.insert(0, "kind", kind); t.insert(0, "H", H); t.insert(0, "sample", name)
                tabs.append(t)
    pd.concat(tabs).to_csv(d / "regime_calibration.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "regimes.json", "J-REGIMES", ROOT, thresholds="terciles of trailing 60d equal-weight vol over rows < 750",
                       covid_window=regimes.COVID, min_n=12, status="success")
    print("regimes done ->", d / "regime_calibration.csv")


def second_stage_stage() -> None:
    from src.journal import regimes, second_stage as ss, sim_dynamic
    d = OUT / "second_stage"
    d.mkdir(parents=True, exist_ok=True)
    models = ss.MODELS + ["ridge_l1", "ridge_l25"]

    def run(R: pd.DataFrame, fc: pd.DataFrame, H: int, tag: str) -> list[dict]:
        out = []
        origins = sorted(fc.origin.unique())
        lab = regimes.label_origins(R, origins, H)
        regime = lab.vol_regime.to_numpy()
        for W in sorted(fc.window.unique()):
            c = fc[(fc.window == W) & (fc.forecast == "raw_corrected")].sort_values("origin")
            feat = pd.DataFrame([ss.origin_features(R, int(o), int(W)) for o in c.origin])
            v, h = c.v.to_numpy(), c.h.to_numpy()
            g = ss.run_cell(v, h, feat, regime)
            g["ridge_l1"] = ss.run_cell(v, h, feat, regime, lam=1.0)["ridge"]
            g["ridge_l25"] = ss.run_cell(v, h, feat, regime, lam=25.0)["ridge"]
            g.insert(0, "origin", c.origin.to_numpy())
            g.assign(sample=tag, H=H, W=W).to_csv(d / f"multipliers_{tag}_H{H}_W{W}.csv", index=False)
            if g.evaluated.sum() < 4:
                continue
            for row in ss.evaluate_cell(v, h, g, regime == "high", models):
                out.append(dict(sample=tag, H=H, W=int(W), **row))
        return out

    rows = []
    for name in ("dev", "hist_A", "ext_A", "ext_B", "ext_AB"):
        R = _sample_returns(name)
        for H in (20, 60):
            fc = pd.read_csv(OUT / "extended" / f"forecasts_{name}_H{H}.csv")
            rows += run(R, fc, H, name)
        print(f"  {name} done", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(d / "second_stage_summary.csv", index=False)
    ver = ss.verdict(summ)
    ver.to_csv(d / "second_stage_verdict.csv", index=False)
    # controls: the same pipeline on simulated returns with the dev covariance (iid Gaussian = false-positive check;
    # regime / garch = positive controls where a state-dependent correction should help)
    Sigma = np.cov(_returns("P1_existing").to_numpy(), rowvar=False)
    ctrl = []
    for kind in ("iid", "garch", "regime"):
        rng = np.random.default_rng(np.random.SeedSequence([20261104, ["iid", "garch", "regime"].index(kind)]))
        if kind == "iid":
            X = scenarios.draw("gaussian", 1, 3784, Sigma, rng)[0]
        else:
            X = sim_dynamic.simulate(kind, Sigma, 1, 3784, rng)[0][0]
        R = pd.DataFrame(X, index=pd.bdate_range("2011-06-02", periods=3784))
        for H in (20,):
            fc = forecast.loss_table(forecast.rolling_forecasts(R, windows=[60, 250], H=H))
            ctrl += [dict(control=kind, **r) for r in run(R, fc, H, f"control_{kind}")]
    pd.DataFrame(ctrl).to_csv(d / "second_stage_controls.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "second_stage.json", "J-SECOND-STAGE", ROOT, config="configs/journal_second_stage.yaml",
                       warm_up=ss.WARM, clip=ss.CLIP, ridge_lambda=ss.LAMBDA, seed=config.SEED, controls_seed=20261104, status="success")
    print(ver.T.to_string())
    print("second_stage done ->", d)


def predictors_stage(n_sim: int = 2000) -> None:
    """Re-run the conference paper's pre-specified predictor test on the 2011-2021 sample (existing src/predictors code), plus the
    exploratory volatility control and a re-implemented signal-in-principle / power analysis (the paper's own script is not in the repo)."""
    from scipy import stats as st
    from src import multitest, predictors, simulate
    from src.covariance import sample_cov
    from src.linalg_tools import noise_exposure_score
    d = OUT / "predictors"
    d.mkdir(parents=True, exist_ok=True)
    R = _sample_returns("hist_A")
    parts, runs = [], {}
    for des in predictors.DESIGNS:
        runs[des] = predictors.add_predictors(predictors.build_runs(R, des), R)
        parts.append(predictors.run_tests(runs[des], des))
    tests = pd.concat(parts, ignore_index=True)
    ver = predictors.verdicts(tests)
    tests.to_csv(d / "predictor_tests_hist_A.csv", index=False)
    ver.to_csv(d / "predictor_verdicts_hist_A.csv", index=False)
    # --- exploratory: does training-period volatility explain the condition-type signals? (within-window percentile ranks)
    rows = []
    for des, r in runs.items():
        r = r.copy()
        r["vol"] = [float(R.iloc[o - W:o].mean(axis=1).std(ddof=1) * np.sqrt(252)) for W, o in zip(r.window, r.origin)]
        for cons in config.CONSTRAINTS:
            g = r[r.constraint == cons]
            for pred in ("log_cond", "eff_rank", "lam_ratio"):
                pr = {c: g.groupby("window")[c].rank(pct=True) for c in (pred, "risk_gap", "vol")}
                x, y, v = pr[pred].to_numpy(), pr["risk_gap"].to_numpy(), pr["vol"].to_numpy()
                res = lambda a: a - np.polyval(np.polyfit(v, a, 1), v)
                rows.append(dict(design=des, constraint=cons, predictor=pred, n=len(g), rho_gap=float(np.corrcoef(x, y)[0, 1]),
                                 rho_with_vol=float(np.corrcoef(x, v)[0, 1]), rho_gap_given_vol=float(np.corrcoef(res(x), res(y))[0, 1])))
    pd.DataFrame(rows).to_csv(d / "volatility_control_hist_A.csv", index=False)
    # --- signal in principle: iid Gaussian train/test pairs per window with the conference's simulation covariance; the effect is
    # window-specific (it decays with W) and is attenuated by the ratio real/Gaussian within-window gap dispersion.
    Rdev = _returns("P1_existing")
    Sigma = simulate.true_sigma(Rdev)
    N, H = Sigma.shape[0], 60
    L = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(config.SEED)
    u = runs["nonoverlap"][runs["nonoverlap"].constraint == "unconstrained"]
    nn = {int(W): int(n) for W, n in u.groupby("window").size().items() if n > 3}
    real_sd = u.groupby("window").risk_gap.std(ddof=1).to_dict()
    ideal, ratio, wrows = {k: {} for k in (1, 3, 5)}, {}, []
    for W in nn:
        sk, gap = {k: [] for k in (1, 3, 5)}, []
        for _ in range(n_sim):
            X = rng.standard_normal((W + H, N)) @ L.T
            S = sample_cov(X[:W])
            w = np.linalg.solve(S, np.ones(N))
            w /= w.sum()
            gap.append((np.sqrt(np.var(X[W:] @ w, ddof=1)) - np.sqrt(w @ S @ w)) * np.sqrt(config.TRADING_DAYS))
            for k in sk:
                sk[k].append(noise_exposure_score(S, w, k))
        gap = np.array(gap)
        ratio[W] = float(real_sd[W] / gap.std(ddof=1))
        for k in sk:
            ideal[k][W] = float(st.spearmanr(sk[k], gap)[0])
            wrows.append(dict(window=W, predictor=f"s{k}", n_runs=nn[W], ideal_rho=ideal[k][W], gap_dispersion_ratio=ratio[W],
                              attenuated_rho=ideal[k][W] / ratio[W]))
    pd.DataFrame(wrows).to_csv(d / "signal_by_window_hist_A.csv", index=False)
    sig = np.random.default_rng(config.SEED + 5)

    def power(rhos: dict, reps: int = 1000) -> float:
        """P(pre-specified non-overlapping test: pooled Fisher-z Spearman, Holm over 7 predictors, p < .05) when predictor 1 has the given
        window-specific Spearman rho and the other six are null; run lengths are the observed ones."""
        hit = 0
        for _ in range(reps):
            p = []
            for j in range(7):
                rs, ns = [], []
                for Wn, n in nn.items():
                    c = 2 * np.sin(np.pi * rhos[Wn] / 6.0) if j == 0 else 0.0
                    z = sig.standard_normal((n, 2))
                    z[:, 1] = c * z[:, 0] + np.sqrt(1 - c * c) * z[:, 1]
                    rs.append(st.spearmanr(z[:, 0], z[:, 1])[0]); ns.append(n)
                p.append(multitest.fisher_pool(np.array(rs), np.array(ns))["p"])
            hit += multitest.holm_adjust(np.array(p))[0] < 0.05
        return hit / reps

    prow = []
    for k in (1, 3, 5):
        att = {W: ideal[k][W] / ratio[W] for W in nn}
        prow.append(dict(predictor=f"s{k}", ideal_rho_W60=ideal[k][60], attenuated_rho_W60=att[60], power_ideal=power(ideal[k]),
                         power_attenuated=power(att),
                         observed_pooled_rho_nonoverlap=float(ver[(ver.constraint == "unconstrained") & (ver.predictor == f"exposure_k{k}")].rho_nonoverlap.iloc[0])))
    pd.DataFrame(prow).to_csv(d / "signal_and_power_hist_A.csv", index=False)
    runmeta.write_meta(OUT / "metadata" / "predictors.json", "J-PREDICTORS-HIST", ROOT, sample="hist_A", runs_per_window=nn, n_sim=n_sim,
                       power_reps=1000, window_specific_effect=True, sim_covariance="simulate.true_sigma(dev returns), shrink 0.5", seed=config.SEED, status="success")
    print(ver.round(3).to_string(index=False))
    print(pd.read_csv(d / "volatility_control_hist_A.csv").round(3).to_string(index=False))
    print(pd.DataFrame(prow).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "validate", "extended", "dynamic", "surface", "regimes", "second_stage", "predictors", "bps_check", "figures", "intervals", "tables", "validate_skew", "cqlike_check", "conformal", "kurtosis_sim", "kurtosis_eval"])
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "baseline":
        baseline()
    elif a.stage == "extended":
        extended()
    elif a.stage == "kurtosis_eval":
        kurtosis_eval()
    elif a.stage == "kurtosis_sim":
        kurtosis_sim(version=a.version)
    elif a.stage == "conformal":
        conformal_stage()
    elif a.stage == "cqlike_check":
        cqlike_check()
    elif a.stage == "validate_skew":
        validate_skew(a.reps if a.reps else 2000)
    elif a.stage == "tables":
        from src.journal import tables
        tables.build(OUT)
        print("tables done ->", OUT / "tables")
    elif a.stage == "intervals":
        intervals_stage()
    elif a.stage == "figures":
        figures_stage()
    elif a.stage == "bps_check":
        bps_check()
    elif a.stage == "dynamic":
        dynamic(smoke=a.smoke)
    elif a.stage == "surface":
        surface_stage()
    elif a.stage == "regimes":
        regimes_stage()
    elif a.stage == "second_stage":
        second_stage_stage()
    elif a.stage == "predictors":
        predictors_stage()
    else:
        smoke = dict(N=[5, 20], T=[60, 250], structures=["hetero"], scenarios=["gaussian", "t5"]) if a.smoke else None
        validate(a.reps if not a.smoke else 300, smoke)
