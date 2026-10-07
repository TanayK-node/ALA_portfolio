"""Monte Carlo for the estimated GMV portfolio with a KNOWN true covariance.

Scenarios (the same true Sigma, N, windows and seeds):
  * ``gaussian``   i.i.d. N(0, Sigma)  -- the only scenario the theory in ``theory`` applies to;
  * ``student_t``  i.i.d. multivariate t with ``MC_T_DF`` dof, scaled to covariance Sigma
                   (fat tails; shows how far non-Gaussian data departs from the theory);
  * ``bootstrap``  rows resampled i.i.d. from the real return history (real marginals/fat
                   tails, but no volatility clustering); the population covariance is that of
                   the empirical distribution (divide by T_pool).
Each simulation draws a W-day training sample and an independent H-day test sample, builds the
unconstrained GMV weights from the raw sample covariance (same as ``optimize.min_variance_closed_form``,
done as one batched LU solve for speed) and records predicted variance, true variance and
realised (test-sample) variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config, theory
from .covariance import sample_cov
from .linalg_tools import symmetrize

SCENARIOS = ("gaussian", "student_t", "bootstrap")


def true_sigma(returns: pd.DataFrame, shrink: float = config.MC_SHRINK) -> np.ndarray:
    """Sigma_true = (1-d) S_full + d (tr S_full / N) I, a well-conditioned shrunk version of the empirical Sigma."""
    S = sample_cov(returns)
    N = S.shape[0]
    return symmetrize((1 - shrink) * S + shrink * np.trace(S) / N * np.eye(N))


def _draw(scenario: str, b: int, rows: int, L: np.ndarray, rng: np.random.Generator,
          pool: np.ndarray | None) -> np.ndarray:
    """(b, rows, N) draws for a scenario; L = Cholesky factor of the true covariance."""
    N = L.shape[0]
    if scenario == "bootstrap":
        return pool[rng.integers(0, len(pool), size=(b, rows))]
    z = rng.standard_normal((b, rows, N)) @ L.T
    if scenario == "student_t":
        df = config.MC_T_DF
        z *= np.sqrt((df - 2) / rng.chisquare(df, size=(b, rows, 1)))  # covariance stays Sigma
    elif scenario != "gaussian":
        raise ValueError(f"unknown scenario {scenario!r}")
    return z


def gmv_stats(Xtr: np.ndarray, Xte: np.ndarray, Sigma: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-simulation (predicted_var, true_var, realised_var) for batches Xtr (b,T,N), Xte (b,H,N).

    predicted = w'Sw = 1/(1'S^{-1}1); true = w'Sigma w; realised = sample variance (ddof=1) of w.x over
    the test block. S is the demeaned, divide-by-(T-1) covariance.
    """
    b, T, N = Xtr.shape
    Xc = Xtr - Xtr.mean(axis=1, keepdims=True)
    S = (Xc.transpose(0, 2, 1) @ Xc) / (T - 1)
    y = np.linalg.solve(S, np.ones((b, N, 1)))[..., 0]
    s = y.sum(axis=1)
    w = y / s[:, None]
    true_var = np.einsum("bi,ij,bj->b", w, Sigma, w)
    real_var = np.einsum("bhi,bi->bh", Xte, w).var(axis=1, ddof=1)
    return 1.0 / s, true_var, real_var


def simulate_gmv(Sigma: np.ndarray, W: int, scenario: str, n_sims: int = config.MC_SIMS,
                 H: int = config.HORIZON, seed: int = config.SEED, pool: np.ndarray | None = None,
                 batch: int = 250) -> pd.DataFrame:
    """Run ``n_sims`` simulations; returns a DataFrame of per-simulation outcomes plus constant V.

    For ``bootstrap`` pass ``pool`` (rows of real returns); Sigma is then replaced by the pool's
    population covariance (ddof = 0). Otherwise Sigma is the true covariance.
    """
    if scenario == "bootstrap":
        if pool is None:
            raise ValueError("bootstrap scenario needs pool")
        Sigma = symmetrize(np.cov(pool, rowvar=False, ddof=0))
    L = np.linalg.cholesky(Sigma)
    V = 1.0 / float(np.ones(len(Sigma)) @ np.linalg.solve(Sigma, np.ones(len(Sigma))))
    rng = np.random.default_rng(seed)
    out = []
    for lo in range(0, n_sims, batch):
        b = min(batch, n_sims - lo)
        Xtr = _draw(scenario, b, W, L, rng, pool)
        Xte = _draw(scenario, b, H, L, rng, pool)
        out.append(np.column_stack(gmv_stats(Xtr, Xte, Sigma)))
    a = np.vstack(out)
    return pd.DataFrame({"pred_var": a[:, 0], "true_var": a[:, 1], "real_var": a[:, 2], "V": V})


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    """Sample mean and its standard error sd/sqrt(n) (unreliable under very heavy tails)."""
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


def summarize_sim(sim: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """(mean, SE) of the four quantities compared with theory."""
    V = sim["V"].iloc[0]
    return {"pred_var_over_V": _mean_se(sim.pred_var / V),
            "true_var_over_V": _mean_se(sim.true_var / V),
            "var_ratio_per_run": _mean_se(sim.true_var / sim.pred_var),
            "std_ratio": _mean_se(np.sqrt(sim.real_var / sim.pred_var))}


def gaussian_checks(sim: pd.DataFrame, N: int, W: int, th_row: pd.Series) -> list[dict]:
    """Compare the Gaussian simulation with theory using the PRE-STATED tolerance (|z| <= MC_Z_TOL).

    Rows: four mean comparisons (z = (MC - theory)/SE) and two exact-distribution KS tests:
    (W-1) pred_var/V ~ chi2_{W-N}  [tier 1]  and  (R-1)(W-N+1)/(N-1) ~ F_{N-1, W-N+1}  [tier 2].
    The sampled-theory std-ratio comparison uses the combined SE of simulation and theory draws.
    """
    s = summarize_sim(sim)
    V = sim["V"].iloc[0]
    rows = []
    pairs = [("pred_var_over_V", th_row.th_pred_var_over_V, 0.0, "tier1"),
             ("true_var_over_V", th_row.th_true_var_over_V, 0.0, "tier2"),
             ("var_ratio_per_run", th_row.th_var_ratio_per_run, 0.0, "tier2"),
             ("std_ratio", th_row.th_std_ratio, th_row.th_std_ratio_se, "tier2")]
    for name, th_val, th_se, tier in pairs:
        m, se = s[name]
        z = (m - th_val) / np.hypot(se, th_se)
        rows.append(dict(window=W, quantity=name, tier=tier, theory=th_val, mc_mean=m, mc_se=se,
                         z=float(z), ok=bool(abs(z) <= config.MC_Z_TOL), ks_p=np.nan))
    q = (W - 1) * sim.pred_var / V
    p1 = stats.kstest(q, "chi2", args=(W - N,)).pvalue
    rows.append(dict(window=W, quantity="KS: (T-1)Q ~ chi2_{T-N}", tier="tier1", theory=np.nan,
                     mc_mean=np.nan, mc_se=np.nan, z=np.nan, ok=bool(p1 >= config.MC_KS_ALPHA), ks_p=float(p1)))
    f = (sim.true_var / V - 1) * (W - N + 1) / (N - 1)
    p2 = stats.kstest(f, "f", args=(N - 1, W - N + 1)).pvalue
    rows.append(dict(window=W, quantity="KS: (R-1)(T-N+1)/(N-1) ~ F", tier="tier2", theory=np.nan,
                     mc_mean=np.nan, mc_se=np.nan, z=np.nan, ok=bool(p2 >= config.MC_KS_ALPHA), ks_p=float(p2)))
    return rows
