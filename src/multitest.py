"""Multiple-comparison adjustments and Fisher-z pooling of correlations (no extra dependencies)."""
from __future__ import annotations

import numpy as np


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER). NaNs are ignored (stay NaN).

    Sort p_(1) <= ... <= p_(m); adjusted_(i) = max_{j<=i} min(1, (m-j+1) p_(j)).
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.minimum(1.0, (m - np.arange(m)) * p[ok][order])
    out[np.flatnonzero(ok)[order]] = np.maximum.accumulate(adj)
    return out


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR, valid under independence / positive dependence).

    adjusted_(i) = min_{j>=i} min(1, m p_(j) / j). NaNs are ignored (stay NaN).
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.minimum(1.0, p[ok][order] * m / (np.arange(m) + 1))
    out[np.flatnonzero(ok)[order]] = np.minimum.accumulate(adj[::-1])[::-1]
    return out


def fisher_pool(rhos: np.ndarray, ns: np.ndarray) -> dict:
    """Pool within-group correlations by weighted Fisher-z averaging.

    z_g = atanh(rho_g), weight w_g = n_g - 3 (groups with n_g <= 3 or NaN rho get weight 0),
    z_bar = sum w z / sum w, rho_bar = tanh(z_bar). For Spearman correlations Var(z_g) ~ 1.06/(n_g - 3)
    (Fieller et al.), so Var(z_bar) = 1.06 / sum w; the two-sided p-value is normal. Returns a dict
    with rho, z, se, p, total_weight, n_used (sum of n over groups with positive weight), k_used.
    """
    from scipy.stats import norm

    rhos, ns = np.asarray(rhos, float), np.asarray(ns, float)
    w = np.where(np.isnan(rhos) | (ns <= 3), 0.0, ns - 3.0)
    W = float(w.sum())
    if W == 0:
        return dict(rho=np.nan, z=np.nan, se=np.nan, p=np.nan, total_weight=0.0, n_used=0, k_used=0)
    z = float(np.sum(w * np.arctanh(np.clip(np.nan_to_num(rhos), -0.999999, 0.999999))) / W)
    se = float(np.sqrt(1.06 / W))
    return dict(rho=float(np.tanh(z)), z=z, se=se, p=float(2 * norm.sf(abs(z) / se)), total_weight=W,
                n_used=int(ns[w > 0].sum()), k_used=int((w > 0).sum()))
