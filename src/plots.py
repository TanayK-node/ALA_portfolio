"""All figures (PNG, 200 dpi) written to results/.

Style: validated categorical order (method colours are fixed per entity),
thin marks, recessive grid, text in ink tokens, a legend whenever there are
two or more series. Figures 1-4 are *illustrations of the mechanism* on one
training window (``config.DEMO_WINDOW``, the last non-overlapping origin); the
aggregate evidence is figures 5-7 and the CSV tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import config, evaluate  # noqa: E402
from .covariance import sample_cov  # noqa: E402
from .linalg_tools import (eig_decompose, mp_noise_edge,  # noqa: E402
                           min_variance_weights_spectral, normal_form_transform)
from .optimize import efficient_frontier, min_variance_closed_form  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
METHOD_COLOR = {"raw": "#2a78d6", "clipped": "#eb6834", "mp": "#1baf7a", "ledoit_wolf": "#eda100"}
METHOD_MARKER = {"raw": "o", "clipped": "s", "mp": "^", "ledoit_wolf": "D"}
METHOD_LABEL = {"raw": "raw", "clipped": "eig-clipped", "mp": "MP-filtered", "ledoit_wolf": "Ledoit-Wolf"}
WINDOW_MARKERS = dict(zip(config.WINDOWS, ["o", "s", "^", "D", "v", "P"]))
ANN = evaluate.ANN
SH = r"$\hat{\Sigma}$"  # mathtext; the combining-hat glyph renders badly


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 11,
        "axes.titleweight": "regular", "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def demo_window(returns: pd.DataFrame, W: int = config.DEMO_WINDOW
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for the last backtest origin of window length W."""
    o = max(range(W, len(returns) - config.HORIZON + 1, config.STEP))
    return returns.iloc[o - W:o], returns.iloc[o:o + config.HORIZON]


def _ctx(returns: pd.DataFrame) -> dict:
    train, test = demo_window(returns)
    covs = evaluate.estimate_covariances(train)
    return dict(train=train, test=test, covs=covs, W=len(train), N=train.shape[1],
                label=f"{train.index[0]:%Y-%m-%d} to {train.index[-1]:%Y-%m-%d}, W={len(train)}")


def fig_spectrum(returns: pd.DataFrame, out_dir: Path) -> Path:
    """1. Eigenvalue spectrum (log): raw vs clipped vs MP-filtered, MP edge marked."""
    c = _ctx(returns)
    S = c["covs"]["raw"][0]
    edge = mp_noise_edge(c["N"], c["W"], np.trace(S) / c["N"])
    full = np.linalg.eigvalsh(sample_cov(returns))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    idx = np.arange(1, c["N"] + 1)
    ax.plot(idx, full, color=INK2, lw=1.2, ls=":", label=f"raw, full sample (T={len(returns)})")
    for m in ("raw", "clipped", "mp"):
        ax.plot(idx, eig_decompose(c["covs"][m][0])[0], color=METHOD_COLOR[m],
                marker=METHOD_MARKER[m], label=f"{METHOD_LABEL[m]}, W={c['W']}", alpha=0.95)
    ax.axhline(edge, color=INK, lw=1, ls="--")
    ax.annotate(f"MP upper edge  λ+ = {edge:.2e}", (1, edge), xytext=(0, 5),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index (1 = smallest)")
    ax.set_ylabel(f"eigenvalue of {SH} (daily variance)")
    ax.set_title(f"Eigenvalue spectrum, training window {c['label']}")
    ax.legend(loc="lower right")
    return _save(fig, "fig1_eigenvalue_spectrum.png", out_dir)


def _oos(weights: np.ndarray, test: pd.DataFrame) -> tuple[float, float]:
    pr = test.to_numpy() @ weights
    return float(pr.std(ddof=1) * ANN * 100), float(pr.mean() * config.TRADING_DAYS * 100)


def fig_frontier(returns: pd.DataFrame, out_dir: Path) -> Path:
    """2. Long-only efficient frontier: raw vs repaired Σ (in-sample line, OOS dots)."""
    c = _ctx(returns)
    mu = c["train"].mean().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes, ("raw", config.REPAIR_METHOD)):
        S = c["covs"][m][0]
        risk, ret, W = efficient_frontier(mu, S, 25)
        ax.plot(risk * ANN * 100, ret * config.TRADING_DAYS * 100, color=METHOD_COLOR[m],
                label=f"in-sample frontier (training μ, {SH})")
        oos = np.array([_oos(w, c["test"]) for w in W])
        ax.scatter(oos[:, 0], oos[:, 1], s=22, color=METHOD_COLOR[m], alpha=0.8,
                   linewidth=0, zorder=3, label=f"same portfolios, next {config.HORIZON} days")
        ax.set_title(f"{SH} = {METHOD_LABEL[m]}")
        ax.set_xlabel("annualised risk (%)")
    axes[0].set_ylabel("annualised mean log return (%)")
    fig.suptitle(f"Long-only frontier, window {c['label']}", x=0.01, ha="left", fontsize=11)
    handles = [Line2D([], [], color=INK2, lw=2, label=f"in-sample frontier (training μ, {SH})"),
               Line2D([], [], color=INK2, lw=0, marker="o", ms=5, label=f"same portfolios, next {config.HORIZON} days")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return _save(fig, "fig2_efficient_frontier.png", out_dir)


def fig_weights(returns: pd.DataFrame, out_dir: Path) -> Path:
    """3. Unconstrained min-variance weights: raw vs repaired Σ."""
    c = _ctx(returns)
    m = config.REPAIR_METHOD
    w_raw = min_variance_closed_form(c["covs"]["raw"][0])
    w_rep = min_variance_closed_form(c["covs"][m][0])
    order = np.argsort(w_raw)
    names = [t.replace(".NS", "") for t in c["train"].columns[order]]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(y - 0.2, w_raw[order] * 100, height=0.36, color=METHOD_COLOR["raw"], label=f"raw {SH}")
    ax.barh(y + 0.2, w_rep[order] * 100, height=0.36, color=METHOD_COLOR[m],
            label=f"{METHOD_LABEL[m]} {SH}")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, names, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("weight (%)")
    ax.set_title(f"Min-variance weights (shorts allowed), window {c['label']}")
    ax.legend(loc="lower right")
    return _save(fig, "fig3_weight_comparison.png", out_dir)


def fig_risk_contrib(returns: pd.DataFrame, out_dir: Path) -> Path:
    """4. Risk share per principal direction, c_i = l_i y_i^2 / w'Σw."""
    c = _ctx(returns)
    S = c["covs"]["raw"][0]
    w_mv = min_variance_weights_spectral(S)
    w_ew = np.full(c["N"], 1 / c["N"])
    shares = {}
    for key, w in (("min", w_mv), ("ew", w_ew)):
        _, contrib = normal_form_transform(S, w)
        shares[key] = contrib / contrib.sum() * 100
    idx = np.arange(1, c["N"] + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(idx - 0.2, shares["min"], width=0.38, color=METHOD_COLOR["raw"], label="raw min-variance portfolio")
    ax.bar(idx + 0.2, shares["ew"], width=0.38, color="#9a9992", label="equal-weight portfolio (reference)")
    ax.set_xlabel("principal direction (1 = smallest eigenvalue)")
    ax.set_ylabel("share of portfolio variance (%)")
    ax.set_title(f"Risk contribution " + r"$\lambda_i y_i^2$, $y=Q^\top w$" + f", window {c['label']}")
    ax.legend(loc="upper center")
    return _save(fig, "fig4_risk_contribution.png", out_dir)


def fig_pred_vs_real(summary: pd.DataFrame, out_dir: Path) -> Path:
    """5. Predicted (dashed) vs realised (solid) annualised risk, by window and method."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    pos = {w: i for i, w in enumerate(config.WINDOWS)}
    for r, cons in enumerate(config.CONSTRAINTS):
        for c_, m in enumerate(config.METHODS):
            ax = axes[r, c_]
            d = summary[(summary.constraint == cons) & (summary.method == m)].sort_values("window")
            x = d.window.map(pos)
            ax.plot(x, d.realized_risk_mean * 100, color=METHOD_COLOR[m], marker=METHOD_MARKER[m])
            ax.plot(x, d.predicted_risk_mean * 100, color=METHOD_COLOR[m], ls="--",
                    marker=METHOD_MARKER[m], mfc=SURFACE)
            if r == 0:
                ax.set_title(METHOD_LABEL[m])
            if c_ == 0:
                ax.set_ylabel(f"{cons.replace('_', '-')}\nannualised risk (%)")
            if r == 1:
                ax.set_xlabel("training window (days)")
            ax.set_xticks(list(pos.values()), [str(w) for w in config.WINDOWS], fontsize=8)
    handles = [Line2D([], [], color=INK2, lw=2, label="realised (out-of-sample)"),
               Line2D([], [], color=INK2, lw=2, ls="--", marker="o", mfc=SURFACE, label=r"predicted $w^\top\hat{\Sigma}w$")]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Predicted vs realised risk of min-variance portfolios (mean over rolling runs)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig5_predicted_vs_realized.png", out_dir)


def _rho(table: pd.DataFrame, cons: str, k: int, window: str) -> float:
    r = table[(table.constraint == cons) & (table.k == k) & (table.window.astype(str) == window)]
    return float(r.spearman_rho.iloc[0])


def fig_exposure_scatter(runs: pd.DataFrame, table: pd.DataFrame, constraint: str,
                         out_dir: Path, k: int = config.PLOT_K) -> Path:
    """6. noise_exposure_score vs risk gap (raw Σ): pooled and within-window ranks."""
    col = f"exposure_k{k}"
    d = runs[(runs.method == "raw") & (runs.constraint == constraint)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for w, g in d.groupby("window"):
        ax.scatter(g[col], g.risk_gap * 100, marker=WINDOW_MARKERS[w], s=34, color=METHOD_COLOR["raw"],
                   edgecolor=SURFACE, linewidth=0.8, label=f"W={w}")
    ax.set_xlabel(f"noise exposure score (bottom {k} directions)")
    ax.set_ylabel("risk gap = realised − predicted (pts, annualised)")
    ax.set_title(f"pooled over windows: Spearman ρ = {_rho(table, constraint, k, 'all'):.2f}".replace("-", "−"))
    ax.legend(ncol=2, loc="best", fontsize=8)
    ax = axes[1]
    g = d.groupby("window")
    ax.scatter(g[col].rank(pct=True), g.risk_gap.rank(pct=True), s=26, color=METHOD_COLOR["raw"],
               edgecolor=SURFACE, linewidth=0.8)
    ax.set_xlabel("exposure score, percentile rank within window")
    ax.set_ylabel("risk gap, percentile rank within window")
    ax.set_title(f"within-window ranks: ρ = {_rho(table, constraint, k, 'within'):.2f}".replace("-", "−"))
    fig.suptitle(f"Raw {SH}, {constraint.replace('_', '-')} min-variance (n = {len(d)} runs)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, f"fig6_exposure_vs_gap_{constraint}.png", out_dir)


def fig_condition(runs: pd.DataFrame, out_dir: Path) -> Path:
    """7. Condition number of Sigma-hat vs training window (log y); median, raw IQR band."""
    d = runs[runs.constraint == config.CONSTRAINTS[0]]  # κ does not depend on constraint
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in config.METHODS:
        g = d[d.method == m].groupby("window").cond_number
        med = g.median()
        ax.plot(range(len(med)), med.values, color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                label=METHOD_LABEL[m])
        if m == "raw":
            ax.fill_between(range(len(med)), g.quantile(0.25).values, g.quantile(0.75).values,
                            color=METHOD_COLOR[m], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(config.WINDOWS)), [str(w) for w in config.WINDOWS])
    ax.set_xlabel("training window (days)")
    ax.set_ylabel("condition number λmax/λmin")
    ax.set_title(f"Condition number of {SH} (median over runs; band = raw IQR)")
    ax.legend()
    return _save(fig, "fig7_condition_number.png", out_dir)


def fig_common_oos_realized(summary_ci: pd.DataFrame, out_dir: Path,
                            name: str = "fig_common_oos_realized_risk.png") -> Path:
    """P5-A. Realised risk by window under the common-OOS design; 95% block-bootstrap bars."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    pos = {w: i for i, w in enumerate(config.WINDOWS)}
    n_methods = len(config.METHODS)
    for ax, cons in zip(axes, config.CONSTRAINTS):
        for j, m in enumerate(config.METHODS):
            d = summary_ci[(summary_ci.constraint == cons) & (summary_ci.method == m)].sort_values("window")
            x = d.window.map(pos).to_numpy() + (j - (n_methods - 1) / 2) * 0.08
            y = d.realized_risk_mean.to_numpy() * 100
            err = np.vstack([y - d.realized_risk_lo.to_numpy() * 100, d.realized_risk_hi.to_numpy() * 100 - y])
            ax.errorbar(x, y, yerr=err, color=METHOD_COLOR[m], marker=METHOD_MARKER[m], capsize=2,
                        elinewidth=1.0, label=METHOD_LABEL[m])
        ax.set_xticks(list(pos.values()), [str(w) for w in config.WINDOWS])
        ax.set_xlabel("training window (days)")
        ax.set_title(cons.replace("_", "-"))
    axes[0].set_ylabel("mean realised risk, annualised (%)")
    axes[0].legend(loc="upper right")
    n = int(summary_ci.n_periods.max())
    fig.suptitle(f"Common out-of-sample periods ({n} blocks); bars = 95% moving-block bootstrap",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def fig_theory_vs_empirical(table: pd.DataFrame, out_dir: Path,
                            name: str = "fig_theory_vs_empirical.png") -> Path:
    """P5-B. Left: Wishart theory vs Gaussian Monte Carlo (variance / true GMV variance V).
    Right: realised/predicted risk ratio: exact theory, three simulations, measured data."""
    _style()
    x = np.arange(len(table))
    T = table.window.to_numpy()
    blue, orange, aqua, yellow = (METHOD_COLOR[k] for k in ("raw", "clipped", "mp", "ledoit_wolf"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for col, mcol, ls, lab in (("th_true_var_over_V", "mc_gaussian_true_var_over_V", "-", "true variance of w-hat (out-of-sample)"),
                               ("th_pred_var_over_V", "mc_gaussian_pred_var_over_V", "--", "predicted variance (in-sample)")):
        ax.plot(x, table[col], color=INK, ls=ls, lw=1.6, label=f"theory: {lab}")
        ax.errorbar(x, table[mcol], yerr=table[mcol + "_se"] * 3, color=blue, marker="o", ls="none",
                    capsize=2, label=f"Gaussian MC ±3 SE ({'upper' if 'true' in lab else 'lower'} curve)")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(x, [str(int(t)) for t in T])
    ax.set_xlabel("training window T (days)")
    ax.set_ylabel("expected variance / true GMV variance V")
    ax.set_title("Wishart theory vs Gaussian simulation (N = %d)" % int(table.N.iloc[0]))
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    ax.plot(x, table.th_std_ratio, color=INK, lw=1.8, label="theory (Gaussian, exact representation)")
    for k, (scen, col, mk, lab, off) in enumerate((("gaussian", blue, "o", "MC Gaussian", -0.12),
                                                   ("student_t", orange, "s", f"MC Student-t (df={config.MC_T_DF})", -0.04),
                                                   ("bootstrap", aqua, "^", "MC bootstrap of real returns", 0.04))):
        ax.errorbar(x + off, table[f"mc_{scen}_std_ratio"], yerr=3 * table[f"mc_{scen}_std_ratio_se"],
                    color=col, marker=mk, ls="none", capsize=2, label=lab)
    err = np.vstack([table.emp_std_ratio - table.emp_std_ratio_lo, table.emp_std_ratio_hi - table.emp_std_ratio])
    ax.errorbar(x + 0.12, table.emp_std_ratio, yerr=err, color=yellow, marker="D", ls="none", capsize=2,
                label="measured (common OOS, raw, 95% block bootstrap)")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(x, [str(int(t)) for t in T])
    ax.set_xlabel("training window T (days)")
    ax.set_ylabel("realised / predicted risk (std)")
    ax.set_title("Realised/predicted risk ratio, unconstrained raw GMV")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return _save(fig, name, out_dir)


def fig_paired_differences(paired: pd.DataFrame, out_dir: Path, name: str = "fig_paired_differences.png") -> Path:
    """P5-E. Paired differences of the mean risk gap / realised risk (method - comparator) with 95% block bootstrap CIs."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True)
    pos = {w: i for i, w in enumerate(config.WINDOWS)}
    style = {"clipped - raw": (METHOD_COLOR["clipped"], "s", "eig-clipped − raw"),
             "mp - raw": (METHOD_COLOR["mp"], "^", "MP-filtered − raw"),
             "ledoit_wolf - raw": (METHOD_COLOR["ledoit_wolf"], "D", "Ledoit-Wolf − raw"),
             "ledoit_wolf - mp": (INK2, "o", "Ledoit-Wolf − MP-filtered")}
    for r, metric in enumerate(("risk_gap", "realized_risk")):
        for c, cons in enumerate(config.CONSTRAINTS):
            ax = axes[r, c]
            for j, (cmp_, (col, mk, lab)) in enumerate(style.items()):
                d = paired[(paired.constraint == cons) & (paired.metric == metric) & (paired.comparison == cmp_)].sort_values("window")
                x = d.window.map(pos).to_numpy() + (j - 1.5) * 0.09
                y = d.diff_mean.to_numpy() * 100
                err = np.vstack([y - d.diff_lo.to_numpy() * 100, d.diff_hi.to_numpy() * 100 - y])
                ax.errorbar(x, y, yerr=err, color=col, marker=mk, ls="none", capsize=2, elinewidth=1.1, label=lab)
            ax.axhline(0, color=INK2, lw=0.8)
            if r == 0:
                ax.set_title(cons.replace("_", "-"))
            if c == 0:
                ax.set_ylabel("Δ mean risk gap (pts)" if r == 0 else "Δ mean realised risk (pts)")
            ax.set_xticks(list(pos.values()), [str(w) for w in config.WINDOWS])
            if r == 1:
                ax.set_xlabel("training window (days)")
    n = int(paired.n_periods.max())
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Paired differences on common OOS periods (n = {n}); bars = 95% block bootstrap, unadjusted",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def fig_block_missing_heatmap(summary: pd.DataFrame, out_dir: Path, name: str = "fig_block_missing_heatmap.png") -> Path:
    """P5-F. Heatmap of the non-PSD rate of the pairwise-complete covariance over (m late stocks, fraction f)."""
    from matplotlib.colors import LinearSegmentedColormap

    _style()
    ms, fs = sorted(summary.m.unique()), sorted(summary.f.unique())
    rate = np.array([[summary[(summary.m == m) & (summary.f == f)].non_psd_rate.iloc[0] for f in fs] for m in ms])
    cnt = np.array([[summary[(summary.m == m) & (summary.f == f)].non_psd.iloc[0] for f in fs] for m in ms])
    runs = int(summary.runs.iloc[0])
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(len(ms)):
        for j in range(len(fs)):
            ax.text(j, i, f"{rate[i, j]:.0%}\n({cnt[i, j]}/{runs})", ha="center", va="center",
                    color="#ffffff" if rate[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(len(fs)), [f"{f:.0%}" for f in fs])
    ax.set_yticks(range(len(ms)), [str(m) for m in ms])
    ax.set_xlabel("fraction of the sample missing (late listing)")
    ax.set_ylabel("number of late-listed stocks m")
    ax.grid(False)
    ax.set_title("Non-PSD rate of pairwise-complete covariance")
    fig.colorbar(im, ax=ax, label="share of seeds with a negative eigenvalue")
    fig.tight_layout()
    return _save(fig, name, out_dir)


def _sweep_axes(title: str):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return fig, axes


def _finish_sweep(fig, axes, pos, name: str, out_dir: Path) -> Path:
    for r, ylabel in enumerate(("mean risk gap (pts, annualised)", "mean realised risk (%, annualised)")):
        axes[r, 0].set_ylabel(ylabel)
    for ax, cons in zip(axes[0], config.CONSTRAINTS):
        ax.set_title(cons.replace("_", "-"))
    for ax in axes[1]:
        ax.set_xticks(list(pos.values()), [str(w) for w in config.WINDOWS])
        ax.set_xlabel("training window (days)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return _save(fig, name, out_dir)


def fig_sweep_clip(sweep: pd.DataFrame, out_dir: Path, name: str = "fig_sweep_clip.png") -> Path:
    """P5-D. Clipping-floor sweep: gap and realised risk vs window, one line per floor fraction (raw dashed)."""
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # validated 5-step ordinal blue
    fig, axes = _sweep_axes("Eigenvalue-clipping floor sweep (common OOS periods); a-priori setting = 0.10")
    pos = {w: i for i, w in enumerate(config.WINDOWS)}
    fracs = sorted(f for f in sweep.eps_fraction.unique() if f > 0)
    for c, cons in enumerate(config.CONSTRAINTS):
        for r, (col, scale) in enumerate((("risk_gap_mean", 100), ("realized_risk_mean", 100))):
            ax = axes[r, c]
            b = sweep[(sweep.constraint == cons) & (sweep.eps_fraction == 0)].sort_values("window")
            ax.plot(b.window.map(pos), b[col] * scale, color=INK, ls=(0, (4, 3)), lw=1.3, zorder=5,
                    label="raw (= floor 0.01 where nothing is clipped)")
            for f, colr in zip(fracs, ramp):
                d = sweep[(sweep.constraint == cons) & (sweep.eps_fraction == f)].sort_values("window")
                ax.plot(d.window.map(pos), d[col] * scale, color=colr, marker="o", ms=4.5, label=f"floor = {f:g} · tr(S)/N")
    return _finish_sweep(fig, axes, pos, name, out_dir)


def fig_sweep_mp(sweep: pd.DataFrame, out_dir: Path, name: str = "fig_sweep_mp.png") -> Path:
    """P5-D. MP noise-variance sweep: sigma^2 = tr(S)/N (solid) vs median eigenvalue (dashed), raw in blue."""
    fig, axes = _sweep_axes("Marchenko-Pastur noise-variance sweep (common OOS periods)")
    pos = {w: i for i, w in enumerate(config.WINDOWS)}
    style = {"raw": (METHOD_COLOR["raw"], "-", "o", "raw"), "trace": (METHOD_COLOR["mp"], "-", "^", "MP, σ² = tr(S)/N (a priori)"),
             "median": (METHOD_COLOR["mp"], "--", "v", "MP, σ² = median eigenvalue")}
    for c, cons in enumerate(config.CONSTRAINTS):
        for r, col in enumerate(("risk_gap_mean", "realized_risk_mean")):
            ax = axes[r, c]
            for rule, (colr, ls, mk, lab) in style.items():
                d = sweep[(sweep.constraint == cons) & (sweep.sigma2_rule == rule)].sort_values("window")
                ax.plot(d.window.map(pos), d[col] * 100, color=colr, ls=ls, marker=mk,
                        mfc=colr if ls == "-" else SURFACE, label=lab)
    return _finish_sweep(fig, axes, pos, name, out_dir)


PRED_LABEL = {"exposure_k1": "exposure score k=1", "exposure_k3": "exposure score k=3",
              "exposure_k5": "exposure score k=5", "log_cond": "log condition number",
              "eff_rank": "effective rank", "lam_ratio": "λmin / mean λ", "max_abs_weight": "max |w|"}


def fig_predictor_forest(tests: pd.DataFrame, out_dir: Path, name: str = "fig_predictor_forest.png") -> Path:
    """P5-C. Forest plot of pooled within-window Spearman rho (95% CI) for the 7 pre-registered
    predictors, both designs; filled marker = Holm-adjusted p < 0.05."""
    from .predictors import PREDICTORS, DESIGNS

    _style()
    pooled = tests[tests.window.astype(str) == "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    style = {"nonoverlap": (METHOD_COLOR["raw"], "o", "non-overlapping windows (Fisher-z CI)", -0.14),
             "stride60_bootstrap": (METHOD_COLOR["clipped"], "s", "stride 60, block bootstrap CI", 0.14)}
    y = np.arange(len(PREDICTORS))[::-1]
    for ax, cons in zip(axes, config.CONSTRAINTS):
        for d in DESIGNS:
            col, mk, lab, off = style[d]
            g = pooled[(pooled.design == d) & (pooled.constraint == cons)].set_index("predictor").loc[list(PREDICTORS)]
            for yi, (_, r) in zip(y + off, g.iterrows()):
                sig = bool(r.p_holm < 0.05)
                if np.isnan(r.rho):
                    continue
                ax.errorbar(r.rho, yi, xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]] if not np.isnan(r.ci_lo) else None,
                            color=col, marker=mk, mfc=col if sig else SURFACE, capsize=2, elinewidth=1.2)
            ax.plot([], [], color=col, marker=mk, label=lab)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [PRED_LABEL[p] for p in PREDICTORS])
        ax.set_xlabel("pooled within-window Spearman ρ")
        ax.set_title(cons.replace("_", "-") + " raw min-variance")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Predictors of the risk gap (filled = Holm-adjusted p < 0.05; hollow = not)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def make_all(returns: pd.DataFrame, runs: pd.DataFrame, expo: pd.DataFrame,
             summary: pd.DataFrame, out_dir: Path = config.RESULTS_DIR) -> list[Path]:
    """Render all figures; returns the written paths."""
    _style()
    paths = [fig_spectrum(returns, out_dir), fig_frontier(returns, out_dir),
             fig_weights(returns, out_dir), fig_risk_contrib(returns, out_dir),
             fig_pred_vs_real(summary, out_dir)]
    paths += [fig_exposure_scatter(runs, expo, c, out_dir) for c in config.CONSTRAINTS]
    paths.append(fig_condition(runs, out_dir))
    return paths
