"""Figures for the journal extension, generated ONLY from saved CSVs under results/journal_extension/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"gaussian": "#1f4e79", "t15": "#2a9d8f", "t8": "#e9a03b", "t5": "#c0392b"}


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gaussian_validation(df: pd.DataFrame, out: Path) -> Path:
    g = df[df.scenario == "gaussian"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, col, lab in ((ax[0], "z_EQ", "E[Q]"), (ax[1], "z_ER", "E[R]")):
        a.scatter(g.N / g["T"], g[col], s=14, c=COL["gaussian"], alpha=.7)
        a.axhline(0, c="k", lw=.6)
        for y in (-3, 3):
            a.axhline(y, c="r", ls="--", lw=.8)
        a.set_xlabel("N/T")
        a.set_ylabel(f"z = (MC - theory)/SE for {lab}")
        a.set_title(f"{lab}: {int((g[col].abs() <= 3).sum())}/{len(g)} cells within |z| <= 3")
    fig.tight_layout()
    return _save(fig, out, "fig_gaussian_validation.png")


def fig_level_bias(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, tgt, lab in ((ax[0], "level_bias_raw", "raw plug-in forecast"), (ax[1], "level_bias_c", "c(N,T)-corrected forecast")):
        for sc in ("gaussian", "t15", "t8", "t5"):
            d = df[df.scenario == sc].groupby(["N", "T"], as_index=False)[tgt].mean()
            a.scatter(d.N / d["T"], 100 * d[tgt], s=14, c=COL[sc], label=sc, alpha=.8)
        a.axhline(0, c="k", lw=.6)
        a.set_xlabel("N/T")
        a.set_title(lab)
    ax[0].set_ylabel("level bias vs true variance (%)")
    ax[1].legend(title="returns", fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig_level_bias_by_scenario.png")


def fig_c_vs_cqlike(df: pd.DataFrame, out: Path) -> Path:
    g = df[df.scenario == "gaussian"].groupby(["N", "T"], as_index=False).agg(m=("ratio_known_mean", "mean"), med=("ratio_known_median", "mean"))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    g = g.assign(th=(g["T"] - g.N) / (g["T"] - g.N - 2))
    ax.scatter(g.N / g["T"], g.th, s=30, facecolors="none", edgecolors="k", lw=.8, label="theory (T-N)/(T-N-2), same cell")
    ax.scatter(g.N / g["T"], g.m, s=14, c=COL["gaussian"], label="MC mean of R/(cQ)")
    ax.scatter(g.N / g["T"], g.med, s=14, c=COL["t8"], marker="s", label="MC median of R/(cQ)")
    ax.axhline(1, c="grey", lw=.6)
    ax.set_xlabel("N/T")
    ax.set_ylabel("true / corrected forecast")
    ax.legend(fontsize=6.5)
    ax.set_ylim(.95, 1.6)
    fig.tight_layout()
    return _save(fig, out, "fig_c_vs_cqlike.png")


def fig_dynamic(df: pd.DataFrame, out: Path) -> Path:
    """Calibration of the c-corrected forecast on the 60-day block, by process and by origin condition (ratio of means)."""
    g = df[df.group != "all"].groupby(["kind", "group"], as_index=False)["vr_block_c"].mean()
    a = df[df.group == "all"].groupby("kind", as_index=False)["vr_block_c"].mean().assign(group="all")
    t = pd.concat([a, g]).reset_index(drop=True)
    t["label"] = t.kind + ": " + t.group
    t = t.sort_values(["kind", "vr_block_c"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cols = {"iid": COL["gaussian"], "garch": COL["t8"], "regime": COL["t5"]}
    ax.barh(t.label, t.vr_block_c - 1, left=1, color=[cols[k] for k in t.kind])
    ax.axvline(1, c="k", lw=.8)
    ax.set_xlabel("realised / corrected forecast (ratio of means, 60-day block)")
    fig.tight_layout()
    return _save(fig, out, "fig_dynamic_calibration.png")


def fig_surface(tab: pd.DataFrame, fits: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, (name, lab) in zip(ax, (("dev", "2021-26, N <= 30"), ("ext_AB", "2011-26, N <= 50"))):
        g = tab[tab["sample"] == name]
        a.errorbar(g.N_over_T, g.ratio, yerr=[g.ratio - g.ratio_lo, g.ratio_hi - g.ratio], fmt="o", ms=3, lw=.7, color=COL["gaussian"])
        f = fits.loc[name]
        x = np.linspace(0, g.N_over_T.max(), 10)
        a.plot(x, f.intercept + f.slope * x, c=COL["t5"], lw=1)
        a.axhline(1, c="k", lw=.6)
        a.set_title(f"{lab}: r = {f.pearson_r:.2f}")
        a.set_xlabel("N/T")
    ax[0].set_ylabel("measured / Gaussian\n(95% origin-block CI)")
    fig.tight_layout()
    return _save(fig, out, "fig_calibration_surface.png")


def fig_second_stage(summ: pd.DataFrame, out: Path) -> Path:
    p = summ[summ["sample"].isin(["ext_A", "ext_B", "ext_AB"]) & summ.model.isin(["constant", "ridge"])].copy()
    p["cell"] = p["sample"] + " H" + p.H.astype(str) + " W" + p.W.astype(str)
    fig, ax = plt.subplots(1, 2, figsize=(10, 8.0), sharey=True)
    for a, m in zip(ax, ("constant", "ridge")):
        d = p[p.model == m].sort_values(["sample", "H", "W"]).reset_index(drop=True)
        y = np.arange(len(d))
        a.errorbar(d.d_gaussian, y, xerr=[d.d_gaussian - d.d_gaussian_lo, d.d_gaussian_hi - d.d_gaussian], fmt="o", ms=3, lw=.7,
                   color=COL["gaussian"])
        a.axvline(0, c="k", lw=.8)
        a.set_title(f"{m} multiplier minus Gaussian correction")
        a.set_xlabel("mean QLIKE difference (negative = second stage better)")
        a.set_yticks(y, d.cell, fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig_second_stage.png")
