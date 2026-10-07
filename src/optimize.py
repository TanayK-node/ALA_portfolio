"""Portfolio optimisation: closed-form and long-only min-variance, frontier.

Min-variance problem:  min_w  w' S w   s.t.  1'w = 1.
Lagrangian stationarity gives  S w = lambda 1, so w* = S^{-1} 1 / (1' S^{-1} 1).
We use ``solve`` (LU) instead of forming S^{-1}.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from . import config
from .linalg_tools import symmetrize

log = logging.getLogger(__name__)


def min_variance_closed_form(S: np.ndarray) -> np.ndarray:
    """Unconstrained (shorts allowed) min-variance weights w = S^{-1}1 / 1'S^{-1}1.

    Raises ``np.linalg.LinAlgError`` if S is exactly singular. For nearly
    singular or indefinite S the solve succeeds numerically but the result is
    not a true minimiser; callers should check ``inertia`` first (that is the
    point of the project).
    """
    S = symmetrize(S)
    x = np.linalg.solve(S, np.ones(S.shape[0]))
    return x / x.sum()


def _equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance_long_only(S: np.ndarray, w0: np.ndarray | None = None
                           ) -> tuple[np.ndarray, bool]:
    """Long-only min-variance via SLSQP: w >= 0, 1'w = 1.

    The objective w'Sw has gradient 2Sw. For indefinite S the problem is
    non-convex and SLSQP returns a local solution. On optimiser failure a
    warning is logged and the best feasible iterate (or equal weight) is
    returned. Returns (weights, converged).
    """
    S = symmetrize(S)
    n = S.shape[0]
    # Rescale so the objective is O(1); SLSQP's tolerances are absolute.
    scale = float(np.max(np.abs(np.diag(S)))) or 1.0
    Ss = S / scale
    x0 = _equal_weight(n) if w0 is None else w0
    res = minimize(lambda w: w @ Ss @ w, x0, jac=lambda w: 2 * Ss @ w,
                   method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones(n)}],
                   options={"maxiter": config.LONG_ONLY_MAX_ITER, "ftol": 1e-12})
    return _finalize(res, n, "min_variance_long_only")


def _finalize(res, n: int, name: str) -> tuple[np.ndarray, bool]:
    """Validate an SLSQP result; project to the simplex-feasible set if needed."""
    w = np.clip(res.x, 0.0, None)
    ok = bool(res.success) and np.all(np.isfinite(w)) and w.sum() > 0
    if not ok:
        log.warning("%s: optimiser failed (%s); using projected iterate", name, res.message)
        if not (np.all(np.isfinite(w)) and w.sum() > 0):
            return _equal_weight(n), False
    return w / w.sum(), bool(ok)


def efficient_frontier(mu: np.ndarray, S: np.ndarray, n_points: int = 30
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long-only efficient frontier.

    For target returns r on a grid between the min-variance portfolio's return
    and max(mu), solve  min w'Sw  s.t. 1'w=1, mu'w = r, w>=0.
    Returns (risk, return, weights) with shapes (m,), (m,), (m, N); infeasible
    or failed targets are skipped (with a warning). Risk is sqrt(w'Sw) in the
    same (daily) units as S and mu.
    """
    S, mu = symmetrize(S), np.asarray(mu, dtype=float)
    n = S.shape[0]
    w_mv, _ = min_variance_long_only(S)
    targets = np.linspace(float(mu @ w_mv), float(mu.max()), n_points)
    scale = float(np.max(np.abs(np.diag(S)))) or 1.0
    Ss = S / scale
    risks, rets, ws = [], [], []
    w_prev = w_mv
    for r in targets:
        res = minimize(lambda w: w @ Ss @ w, w_prev, jac=lambda w: 2 * Ss @ w,
                       method="SLSQP", bounds=[(0.0, 1.0)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                                    {"type": "eq", "fun": lambda w, r=r: mu @ w - r}],
                       options={"maxiter": config.LONG_ONLY_MAX_ITER, "ftol": 1e-12})
        if not res.success or abs(mu @ res.x - r) > 1e-6:
            log.warning("efficient_frontier: target %.3e failed (%s)", r, res.message)
            continue
        w = np.clip(res.x, 0.0, None)
        w /= w.sum()
        w_prev = w  # warm start along the frontier
        risks.append(float(np.sqrt(max(w @ S @ w, 0.0))))
        rets.append(float(mu @ w))
        ws.append(w)
    return np.array(risks), np.array(rets), np.array(ws).reshape(len(ws), n)
