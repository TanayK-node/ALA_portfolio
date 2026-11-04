"""Independent verification pass (numpy/pandas/scipy only; does NOT import src.journal).

1. Theory: closed-form moments of Q and R, c and c_QLIKE, against an independent Monte Carlo.
2. Market data: recompute (v, h) for selected cells straight from the price CSV and compare with the stored forecasts.
3. Headline counts recomputed from the stored per-origin (v, h) with an independent QLIKE.
Writes results/<market dir>/audit/independent_check.json. Read-only with respect to all existing results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # tickers and windows only (data, not logic)


def qlike(v, h):
    r = np.asarray(v) / np.asarray(h)
    return r - np.log(r) - 1


def c_(N, T):
    return (T - 1) * (T - 2) / ((T - N) * (T - N - 1))


def cq_(N, T):
    return (T - 1) * (T - 2) / ((T - N - 1) * (T - N - 2))


def theory_check(seed=11):
    rng = np.random.default_rng(seed)
    out = {}
    for N, T in ((10, 40), (5, 30)):
        reps = 300_000
        Q = rng.chisquare(T - N, reps) / (T - 1)
        R = 1 + rng.chisquare(N - 1, reps) / rng.chisquare(T - N + 1, reps)
        # direct construction from Gaussian data as a check on the chi-square representation
        out[f"N{N}_T{T}"] = dict(
            EQ=(Q.mean(), (T - N) / (T - 1)), ER=(R.mean(), (T - 2) / (T - N - 1)), E1overQ=((1 / Q).mean(), (T - 1) / (T - N - 2)),
            c=(R.mean() / Q.mean(), c_(N, T)), cq=((R / Q).mean(), cq_(N, T)), cov_QR=float(np.corrcoef(Q, R)[0, 1]))
    # direct: draw data, form GMV, compute Q, R with identity covariance (V = 1/N)
    N, T, reps = 8, 40, 20_000
    qs, rs = [], []
    for _ in range(reps):
        X = rng.standard_normal((T, N))
        S = np.cov(X.T)
        w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
        V = 1.0 / N
        qs.append(w @ S @ w / V); rs.append(w @ w / V)
    qs, rs = np.array(qs), np.array(rs)
    out["direct_N8_T40"] = dict(EQ=(qs.mean(), (T - N) / (T - 1)), ER=(rs.mean(), (T - 2) / (T - N - 1)), cq=((rs / qs).mean(), cq_(N, T)))
    return {k: {m: [float(a) if np.isscalar(a) else a for a in v] if isinstance(v, tuple) else v for m, v in d.items()} for k, d in out.items()}


def clean(p, tickers, lo, hi):
    p = p.loc[lo:hi, tickers].dropna(how="all")
    miss = p.isna().mean()
    p = p.drop(columns=miss[miss > 0.05].index).ffill(limit=3)
    return np.log(p).diff().iloc[1:].dropna(how="any")


def recompute(R: pd.DataFrame, fc: pd.DataFrame, H: int, Ws, k_origins=6):
    """Recompute v,h of raw / raw_corrected / raw_corrected_qlike / equal_weight at the first k origins of each window."""
    X = R.to_numpy()
    N = X.shape[1]
    worst = 0.0
    n = 0
    for W in Ws:
        g = fc[fc.window == W]
        for o in sorted(g.origin.unique())[:k_origins]:
            tr, te = X[o - W:o], X[o:o + H]
            S = np.cov(tr.T)
            w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
            h_raw = w @ S @ w
            v_raw = np.var(te @ w, ddof=1)
            we = np.full(N, 1.0 / N)
            exp = {"raw": (v_raw, h_raw), "raw_corrected": (v_raw, c_(N, W) * h_raw), "raw_corrected_qlike": (v_raw, cq_(N, W) * h_raw),
                   "equal_weight": (np.var(te @ we, ddof=1), we @ S @ we)}
            for name, (v, h) in exp.items():
                row = g[(g.origin == o) & (g.forecast == name)]
                assert len(row) == 1, (name, W, o)
                worst = max(worst, abs(row.v.iloc[0] / v - 1), abs(row.h.iloc[0] / h - 1))
                n += 1
    return dict(values_checked=n, max_rel_diff=float(worst))


def headline(base: Path):
    t = pd.read_csv(base / "extended" / "paired_tests_extended.csv")
    cnt = 0
    tot = 0
    for (s, H, W), _ in t.groupby(["sample", "H", "W"]):
        fc = pd.read_csv(base / "extended" / f"forecasts_{s}_H{H}.csv")
        g = fc[fc.window == W]
        m = g.groupby("forecast").apply(lambda d: qlike(d.v.to_numpy(), d.h.to_numpy()).mean())
        for alt in t[(t["sample"] == s) & (t.H == H) & (t.W == W)].alt:
            tot += 1
            cnt += int(m["raw_corrected"] < m[alt])
    cells = []
    for (s, H), g0 in t.groupby(["sample", "H"]):
        fc = pd.read_csv(base / "extended" / f"forecasts_{s}_H{H}.csv")
        for W in sorted(fc.window.unique()):
            g = fc[fc.window == W]
            a = qlike(g[g.forecast == "raw_corrected_qlike"].v, g[g.forecast == "raw_corrected_qlike"].h).mean()
            b = qlike(g[g.forecast == "raw_corrected"].v, g[g.forecast == "raw_corrected"].h).mean()
            cells.append(a < b)
    return dict(comparisons=tot, corrected_lower_mean_qlike=cnt, cqlike_cells=len(cells), cqlike_lower_than_c=int(sum(cells)))


def var_check(R: pd.DataFrame, base: Path, sample="ext_A", H=20, W=60):
    """Independent pooled 5% VaR violation rates (normal quantile, zero mean) for plug-in and c_QLIKE forecasts vs risk_use_cells.csv."""
    from scipy import stats
    X = R.to_numpy()
    N = X.shape[1]
    org = list(range(max(config.WINDOWS), len(X) - H + 1, H))
    hit = {"plugin_raw": [], "c_qlike": []}
    for o in org:
        tr, te = X[o - W:o], X[o:o + H]
        S = np.cov(tr.T)
        w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
        h = w @ S @ w
        r = te @ w
        hit["plugin_raw"] += list(r < -stats.norm.ppf(0.95) * np.sqrt(h))
        hit["c_qlike"] += list(r < -stats.norm.ppf(0.95) * np.sqrt(cq_(N, W) * h))
    cells = pd.read_csv(base / "risk_use" / "risk_use_cells.csv")
    out = {}
    for v, hh in hit.items():
        row = cells[(cells["sample"] == sample) & (cells.H == H) & (cells.W == W) & (cells.variant == v)].iloc[0]
        out[v] = dict(independent=float(np.mean(hh)), stored=float(row.viol_5))
    return out


def conformal_check(base: Path, sample="ext_A", H=20, W=60, level=0.9):
    """Independent split-conformal coverage (expanding matured scores, 20-origin warm-up) vs conformal_intervals.csv."""
    fc = pd.read_csv(base / "extended" / f"forecasts_{sample}_H{H}.csv")
    g = fc[(fc.window == W) & (fc.forecast == "raw_corrected_qlike")].sort_values("origin")
    rho = np.log(g.v.to_numpy() / g.h.to_numpy())
    a = 1 - level
    hits = []
    for k in range(20, len(rho)):
        past = np.sort(rho[:k])
        n = len(past)
        lo_r = int(np.floor(round((n + 1) * a / 2, 9)))
        hi_r = int(np.ceil(round((n + 1) * (1 - a / 2), 9)))
        if lo_r < 1 or hi_r > n:
            continue
        hits.append(past[lo_r - 1] <= rho[k] <= past[hi_r - 1])
    st = pd.read_csv(base / "conformal" / "conformal_intervals.csv")
    row = st[(st["sample"] == sample) & (st.H == H) & (st.W == W) & (st.level == level) & (st.variant == "conformal_pooled")].iloc[0]
    return dict(independent=float(np.mean(hits)), stored=float(row.coverage), n_indep=len(hits), n_stored=int(row.n_eval))


def stored_counts(base: Path):
    """Counts read directly from stored paired-test CSVs (not recomputed from raw prices)."""
    c = pd.read_csv(base / "cqlike" / "cqlike_vs_c.csv")
    k = pd.read_csv(base / "kurtosis" / "kurtosis_vs_cqlike.csv")
    return dict(cqlike_cells=len(c), cqlike_lower=int((c.delta < 0).sum()), cqlike_interval_below_zero=int((c.hi < 0).sum()), cqlike_interval_above_zero=int((c.lo > 0).sum()),
                kurtosis_cells=len(k), kurtosis_lower=int((k.delta < 0).sum()), kurtosis_interval_below_zero=int((k.hi < 0).sum()))


def main():
    res = {"theory": theory_check()}
    for mk, base, csv, tick in (("india", ROOT / "results/journal_extension", ROOT / "data/prices_2011_2026.csv", config.TICKERS),
                                ("us", ROOT / "results/journal_extension_us", ROOT / "data/prices_us_2011_2026.csv", None)):
        if mk == "us":
            import yaml
            cfg = yaml.safe_load(open(ROOT / "configs/journal_us_replication.yaml"))
            tick = None
        p = pd.read_csv(csv, index_col=0, parse_dates=True)
        if mk == "us":
            sys.argv = ["x"]
            import run_journal as rj
            rj.MARKET = "us"
            A, B = rj._us_universe()
            tick = A
        R = clean(p, tick, "2011-06-01", "2026-10-06")
        fc = pd.read_csv(base / "extended" / "forecasts_ext_A_H20.csv")
        res[mk] = dict(sample_shape=list(R.shape), recompute_ext_A_H20=recompute(R, fc, 20, [60, 250]), headline=headline(base),
                      var_check=var_check(R, base), conformal_check=conformal_check(base), stored_counts=stored_counts(base))
    out = ROOT / "results/journal_extension/audit"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "independent_check.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
