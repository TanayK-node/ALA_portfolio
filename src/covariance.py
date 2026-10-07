"""Covariance estimators and "broken Sigma" scenario builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf  # benchmark only

from . import config
from .linalg_tools import inertia, symmetrize, Inertia


def sample_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Unbiased sample covariance S = X_c' X_c / (T-1) (rows = observations)."""
    return symmetrize(np.cov(np.asarray(returns, dtype=float), rowvar=False))


def pairwise_cov(returns_nan: pd.DataFrame) -> np.ndarray:
    """Pairwise-complete covariance: each (i,j) entry uses rows where both are
    observed. Different entries use different samples, so the result need not be
    PSD (it is not a Gram matrix of any single data set)."""
    return symmetrize(returns_nan.cov(min_periods=2).to_numpy())


def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity (sklearn; benchmark only)."""
    return symmetrize(LedoitWolf().fit(np.asarray(returns, dtype=float)).covariance_)


def short_window_cov(returns: pd.DataFrame, T: int, start: int = 0
                     ) -> tuple[np.ndarray, Inertia]:
    """Scenario (a): sample covariance from T rows. For T <= N the matrix has rank
    <= T-1 < N, hence is singular (n_zero >= N-T+1). Returns (S, inertia)."""
    S = sample_cov(returns.iloc[start:start + T])
    return S, inertia(S)


def masked_returns(returns: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Set a random fraction ``frac`` of entries to NaN (MCAR mask, seeded)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(returns.shape) < frac
    return returns.mask(mask)


def missing_data_scenarios(returns: pd.DataFrame,
                           fracs: list[float] = config.MASK_FRACS,
                           n_seeds: int = config.N_SEEDS,
                           base_seed: int = config.SEED) -> pd.DataFrame:
    """Scenario (b): mask frac of entries, pairwise-complete cov, record inertia.

    One row per (frac, seed) with n_pos/n_zero/n_neg, lambda_min and a
    ``non_psd`` flag (n_neg > 0). We *measure* how often this occurs; no
    outcome is assumed.
    """
    rows = []
    for frac in fracs:
        for s in range(n_seeds):
            seed = base_seed + s
            S = pairwise_cov(masked_returns(returns, frac, seed))
            p, z, n = inertia(S)
            rows.append(dict(mask_frac=frac, seed=seed, n_pos=p, n_zero=z,
                             n_neg=n, lambda_min=float(np.linalg.eigvalsh(S)[0]),
                             non_psd=n > 0))
    return pd.DataFrame(rows)
