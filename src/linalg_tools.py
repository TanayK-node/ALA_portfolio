"""Core linear algebra for portfolio risk as a quadratic form.

Notation: Sigma = S is a real symmetric N x N covariance matrix, w a weight
vector, risk(w) = w' S w. Spectral theorem: S = Q diag(l) Q' with Q orthogonal.
With y = Q'w (coordinates in the eigenbasis), w' S w = sum_i l_i y_i^2 --
the *normal form* of the quadratic form under an orthogonal (hence congruent)
change of variables.

Only ``np.linalg.eigh`` (and Cholesky, for the PD test) is used as a solver;
inertia, definiteness, clipping, Marchenko-Pastur filtering, risk
decomposition and the exposure score are implemented here.
"""
from __future__ import annotations

import numpy as np

from . import config

Inertia = tuple[int, int, int]


def symmetrize(S: np.ndarray) -> np.ndarray:
    """Return (S + S')/2, removing floating-point asymmetry."""
    S = np.asarray(S, dtype=float)
    return 0.5 * (S + S.T)


def eig_decompose(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectral decomposition S = Q diag(l) Q'.

    S is symmetrized first. ``eigh`` returns eigenvalues in ascending order and
    orthonormal eigenvectors as columns of Q (Q'Q = I).
    """
    vals, Q = np.linalg.eigh(symmetrize(S))
    return vals, Q


def inertia(S: np.ndarray, tol: float = config.INERTIA_TOL) -> Inertia:
    """Inertia (n_pos, n_zero, n_neg) = signature of the quadratic form x'Sx.

    Eigenvalue l_i counts as zero when |l_i| <= tol * max|l|. A *relative*
    tolerance is used because the eigenvalue scale of a covariance matrix
    (~1e-4 for daily returns) is arbitrary; an absolute cutoff would misclassify
    after rescaling. The default 1e-10 is ~1e6 x machine epsilon, which leaves
    room for the O(N * eps * max|l|) error of eigh.
    """
    vals, _ = eig_decompose(S)
    cut = tol * float(np.max(np.abs(vals))) if vals.size else 0.0
    return int(np.sum(vals > cut)), int(np.sum(np.abs(vals) <= cut)), int(np.sum(vals < -cut))


def is_positive_definite(S: np.ndarray) -> bool:
    """True iff S = L L' exists with L having positive diagonal (attempted Cholesky).

    Cholesky succeeds exactly when S is symmetric positive definite, and is
    cheaper than an eigendecomposition. Note: borderline matrices (l_min ~ eps)
    may be classified either way by rounding; use ``inertia`` for a toleranced view.
    """
    try:
        np.linalg.cholesky(symmetrize(S))
        return True
    except np.linalg.LinAlgError:
        return False


def is_psd(S: np.ndarray, tol: float = config.INERTIA_TOL) -> bool:
    """True iff l_min >= -tol * max|l| (x'Sx >= 0 up to numerical tolerance)."""
    vals, _ = eig_decompose(S)
    return bool(vals[0] >= -tol * np.max(np.abs(vals)))


def condition_number(S: np.ndarray) -> float:
    """kappa(S) = l_max / l_min for symmetric S; +inf if l_min <= 0."""
    vals, _ = eig_decompose(S)
    return float("inf") if vals[0] <= 0 else float(vals[-1] / vals[0])


def normal_form_transform(S: np.ndarray, w: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate w into the eigenbasis and split risk by principal direction.

    y = Q'w, contributions c_i = l_i y_i^2, and w'Sw = sum_i c_i. A negative
    l_i (indefinite S) gives a negative contribution. Returns (y, c), both
    ordered by ascending eigenvalue. Asserts the identity to numerical precision.
    """
    w = np.asarray(w, dtype=float)
    vals, Q = eig_decompose(S)
    y = Q.T @ w
    contrib = vals * y**2
    direct = float(w @ symmetrize(S) @ w)
    # Scale: sum|c_i| bounds the rounding error of both sides.
    assert abs(contrib.sum() - direct) <= 1e-9 * max(1.0, float(np.abs(contrib).sum())), \
        "normal form does not reproduce w'Sw"
    return y, contrib


def random_invertible(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random matrix, redrawn until well-conditioned (kappa < 1e6)."""
    while True:
        P = rng.standard_normal((n, n))
        if np.linalg.cond(P) < 1e6:
            return P


def verify_sylvester(S: np.ndarray, P: np.ndarray | None = None,
                     seed: int = config.SEED,
                     tol: float = config.INERTIA_TOL) -> tuple[bool, Inertia, Inertia]:
    """Check Sylvester's law of inertia: inertia(P'SP) == inertia(S), P invertible.

    The congruence S -> P'SP changes eigenvalues but not the counts of positive,
    zero and negative ones. If P is None a random invertible P is drawn.
    Returns (holds, inertia(S), inertia(P'SP)).
    """
    S = symmetrize(S)
    if P is None:
        P = random_invertible(S.shape[0], np.random.default_rng(seed))
    i0, i1 = inertia(S, tol), inertia(P.T @ S @ P, tol)
    return i0 == i1, i0, i1


def clip_eigenvalues(S: np.ndarray, eps: float = config.CLIP_EPS
                     ) -> tuple[np.ndarray, int]:
    """Spectral clipping: S' = Q diag(max(l_i, eps)) Q'.

    S' is symmetric with every eigenvalue >= eps, hence positive definite for
    eps > 0. (It is also the Frobenius-nearest matrix with spectrum >= eps, but
    we build it directly from the spectrum.) Returns (S', number clipped).
    """
    vals, Q = eig_decompose(S)
    clipped = vals < eps
    new = np.where(clipped, eps, vals)
    return symmetrize((Q * new) @ Q.T), int(clipped.sum())


def mp_noise_edge(N: int, T: int, sigma2: float = 1.0) -> float:
    """Marchenko-Pastur upper edge  l+ = sigma^2 (1 + sqrt(N/T))^2.

    For N x N sample covariance of T iid observations with variance sigma^2,
    all eigenvalues of pure noise fall in [sigma^2(1-sqrt(q))^2, l+], q = N/T.
    """
    return float(sigma2 * (1.0 + np.sqrt(N / T)) ** 2)


def clip_by_mp(S: np.ndarray, N: int, T: int, sigma2: float | None = None
               ) -> tuple[np.ndarray, int, float]:
    """Marchenko-Pastur noise filtering.

    Eigenvalues l_i <= l+ are deemed noise and replaced by their common mean
    (this preserves the trace). Eigenvalues above l+ are kept ("signal").
    ``sigma2`` is the noise variance; the default is the average variance
    tr(S)/N, which is the standard correlation-matrix-style choice and slightly
    conservative (overstates sigma^2 when signal is strong, so the edge is high).
    Returns (S', number of eigenvalues treated as noise, edge l+).
    """
    vals, Q = eig_decompose(S)
    if sigma2 is None:
        sigma2 = float(np.trace(symmetrize(S)) / N)
    edge = mp_noise_edge(N, T, sigma2)
    noise = vals <= edge
    new = vals.copy()
    if noise.any():
        new[noise] = vals[noise].mean()  # trace-preserving flattening
    return symmetrize((Q * new) @ Q.T), int(noise.sum()), edge


def min_variance_weights_spectral(S: np.ndarray,
                                  rcond: float = config.RANK_RCOND) -> np.ndarray:
    """Minimum-variance weights via the spectrum.

    Solves min w'Sw s.t. 1'w = 1, whose solution is w ~ S^{-1} 1. In the
    eigenbasis  S^{-1} 1 = sum_i (1/l_i) q_i (q_i' 1), then normalized to sum 1.
    This shows 1/l_i amplification: the smallest eigenvalues dominate.

    For singular / indefinite S: only directions with l_i > rcond * l_max are
    inverted (truncated pseudo-inverse S^+); the null/negative directions are
    dropped, so the result is the minimum-variance portfolio *within the
    retained subspace*, not a true minimizer (which does not exist/is unbounded).
    """
    vals, Q = eig_decompose(S)
    ones_coords = Q.T @ np.ones(S.shape[0])
    keep = vals > rcond * vals[-1]
    inv = np.zeros_like(vals)
    inv[keep] = 1.0 / vals[keep]
    x = Q @ (inv * ones_coords)
    total = x.sum()
    if abs(total) < 1e-300:
        raise ValueError("S^+ 1 sums to zero; cannot normalize weights.")
    return x / total


def noise_exposure_score(S: np.ndarray, w: np.ndarray, k: int) -> float:
    """Fraction of ||w||^2 lying in the k smallest-eigenvalue directions.

    With a_i = q_i'w (eigenvalues ascending):
        score_k = sum_{i<=k} a_i^2 / ||w||^2  in [0, 1]
    because Q is orthogonal, sum_i a_i^2 = ||w||^2. Score = 1 means w lies
    entirely in the span of the k least-variance directions.
    """
    w = np.asarray(w, dtype=float)
    nrm2 = float(w @ w)
    if nrm2 == 0.0:
        return 0.0
    _, Q = eig_decompose(S)
    a = Q.T @ w
    return float(np.clip(np.sum(a[:k] ** 2) / nrm2, 0.0, 1.0))


def effective_rank(S: np.ndarray) -> float:
    """Participation-ratio effective rank  (sum_i l_i)^2 / sum_i l_i^2  =  tr(S)^2 / ||S||_F^2.

    Equals N for a multiple of the identity (all directions equally used) and 1 for a rank-one S;
    it measures how many eigen-directions carry the variance. Uses the eigenvalues of S.
    """
    vals, _ = eig_decompose(S)
    return float(vals.sum() ** 2 / np.sum(vals**2))


def min_eig_ratio(S: np.ndarray) -> float:
    """lambda_min / mean(lambda) = N lambda_min / tr(S) in (-inf, 1]; small means a near-null direction."""
    vals, _ = eig_decompose(S)
    return float(vals[0] / vals.mean())
