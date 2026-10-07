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
