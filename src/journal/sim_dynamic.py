"""Non-i.i.d. simulators whose UNCONDITIONAL covariance equals Sigma exactly.

* ``garch``  : common-volatility GARCH(1,1). x_t = sqrt(h_t) L z_t, z_t ~ N(0, I) i.i.d.; h is driven by the squared
               equal-weight portfolio return: h_{t+1} = (1-a-b) + a * (w_ew'x_t)^2 / (w_ew' Sigma w_ew) + b h_t,
               E[h] = 1 so Cov(x) = Sigma. Finite 4th moment needs 3a^2 + 2ab + b^2 < 1.
* ``regime`` : two-state Markov chain of volatility scale, x_t = sqrt(m_{s_t}) L z_t, with m chosen so that
               pi_calm m_calm + pi_stress m_stress = 1 (stationary mixture over time, nonstationary within a window).
Both keep the correlation structure fixed; only the volatility level moves. They are not claimed to be realistic
multivariate models (no dynamic correlations), only controlled tests of the i.i.d. assumption.
"""
from __future__ import annotations

import numpy as np

GARCH = dict(a=0.10, b=0.85)
REGIME = dict(p_stay_calm=0.99, p_stay_stress=0.95, m_stress=2.5)


def regime_scales(p_stay_calm: float, p_stay_stress: float, m_stress: float) -> tuple[float, float, float]:
    """(m_calm, m_stress, pi_stress) with the stationary mean scale equal to 1."""
    pi_s = (1 - p_stay_calm) / ((1 - p_stay_calm) + (1 - p_stay_stress))
    m_c = (1 - pi_s * m_stress) / (1 - pi_s)
    if m_c <= 0:
        raise ValueError("m_stress too large for these persistences (calm scale would be non-positive)")
    return m_c, m_stress, pi_s


def simulate(kind: str, Sigma: np.ndarray, reps: int, length: int, rng: np.random.Generator, burn: int = 300,
             params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(X, aux): X has shape (reps, length, N); aux is h_t (garch) or the state in {0 calm, 1 stress} (regime), shape (reps, length)."""
    N = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    steps = burn + length
    Z = rng.standard_normal((reps, steps, N)) @ L.T
    X = np.empty((reps, steps, N))
    aux = np.empty((reps, steps))
    if kind == "garch":
        p = {**GARCH, **(params or {})}
        a, b = p["a"], p["b"]
        if 3 * a * a + 2 * a * b + b * b >= 1:
            raise ValueError("GARCH parameters violate the 4th-moment condition")
        w = np.full(N, 1.0 / N)
        var_ew = float(w @ Sigma @ w)
        h = np.ones(reps)
        for t in range(steps):
            aux[:, t] = h
            X[:, t] = np.sqrt(h)[:, None] * Z[:, t]
            q = (X[:, t] @ w) ** 2 / var_ew
            h = (1 - a - b) + a * q + b * h
    elif kind == "regime":
        p = {**REGIME, **(params or {})}
        m_c, m_s, pi_s = regime_scales(p["p_stay_calm"], p["p_stay_stress"], p["m_stress"])
        s = (rng.random(reps) < pi_s).astype(int)
        u = rng.random((reps, steps))
        for t in range(steps):
            aux[:, t] = s
            X[:, t] = np.sqrt(np.where(s == 1, m_s, m_c))[:, None] * Z[:, t]
            stay = np.where(s == 1, p["p_stay_stress"], p["p_stay_calm"])
            s = np.where(u[:, t] < stay, s, 1 - s)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return X[:, burn:], aux[:, burn:]
