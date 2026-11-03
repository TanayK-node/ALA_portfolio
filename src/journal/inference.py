"""Dependence-aware inference for paired forecast-loss differences."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config


def block_length_rule(n: int) -> int:
    """Pre-specified primary block length L = max(1, round(n^(1/3))) (fixed before the journal runs)."""
    return max(1, int(round(n ** (1.0 / 3.0))))


def moving_block_indices(n: int, L: int, B: int, rng: np.random.Generator) -> np.ndarray:
    L = max(1, min(L, n))
    nb = -(-n // L)
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    return (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]


def stationary_indices(n: int, mean_len: float, B: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths, circular)."""
    p = 1.0 / max(mean_len, 1.0)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    jump = rng.random((B, n)) < p
    new = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], new[:, t], (idx[:, t - 1] + 1) % n)
    return idx


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_diff: float
    lo: float
    hi: float
    win_rate: float
    win_lo: float
    win_hi: float
    p_two_sided: float
    block: int
    method: str

    @property
    def robust_win(self) -> bool:
        """Conference-paper rule: both the mean-difference interval (below 0) and the win-rate interval (above 50%) favour A."""
        return bool(self.hi < 0 and self.win_lo > 0.5)


def paired_loss_test(loss_a: np.ndarray, loss_b: np.ndarray, block: int | None = None, B: int = config.BOOT_N,
                     alpha: float = config.BOOT_ALPHA, seed: int = config.SEED, method: str = "moving") -> PairedResult:
    """Bootstrap the mean of d = loss_a - loss_b (negative favours A) and A's win rate on ALIGNED origins.

    The p-value is the two-sided bootstrap-sign p-value from the centred distribution (a heuristic for few periods).
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("paired series must be aligned")
    d, n = a - b, len(a)
    L = block_length_rule(n) if block is None else block
    rng = np.random.default_rng(seed)
    idx = moving_block_indices(n, L, B, rng) if method == "moving" else stationary_indices(n, L, B, rng)
    m = d[idx].mean(axis=1)
    win = (d[idx] < 0).mean(axis=1)
    lo, hi = np.quantile(m, [alpha / 2, 1 - alpha / 2])
    wlo, whi = np.quantile(win, [alpha / 2, 1 - alpha / 2])
    centred = m - m.mean()
    p = float(np.mean(np.abs(centred) >= abs(d.mean()))) if n > 1 else float("nan")
    return PairedResult(n, float(d.mean()), float(lo), float(hi), float((d < 0).mean()), float(wlo), float(whi),
                        max(p, 1.0 / B), L, method)
