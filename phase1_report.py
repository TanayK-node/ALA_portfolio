"""Phase 1 diagnostics on the full-sample covariance (real cached data)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config, covariance, data, linalg_tools as lt


def full_sample_report(returns: pd.DataFrame) -> dict:
    """Eigenvalue summary, inertia, condition number, PD check of sample Sigma."""
    S = covariance.sample_cov(returns)
    vals, _ = lt.eig_decompose(S)
    return dict(
        T=len(returns), N=returns.shape[1],
        lambda_min=vals[0], lambda_median=float(np.median(vals)), lambda_max=vals[-1],
        top1_variance_share=vals[-1] / vals.sum(),
        inertia=lt.inertia(S), condition_number=lt.condition_number(S),
        positive_definite=lt.is_positive_definite(S),
        mp_edge=lt.mp_noise_edge(returns.shape[1], len(returns),
                                 np.trace(S) / returns.shape[1]),
        n_above_mp_edge=int(np.sum(vals > lt.mp_noise_edge(
            returns.shape[1], len(returns), np.trace(S) / returns.shape[1]))),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    returns, _, dropped = data.get_returns()
    print(f"Dropped tickers: {dropped or 'none'}")
    for k, v in full_sample_report(returns).items():
        print(f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
