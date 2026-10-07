"""Central configuration: tickers, dates, windows, seeds, thresholds.

Every tunable choice in the project lives here so the README can point to a
single place. Nothing in this file is a *result*; all result numbers come from
running the pipeline.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
RESULTS_DIR: Path = ROOT / "results"
PRICES_CSV: Path = DATA_DIR / "prices.csv"

# --------------------------------------------------------------- universe
# 30 liquid Nifty 50 large caps (Yahoo Finance NSE symbols).
TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "TATASTEEL.NS",
    "M&M.NS", "TECHM.NS", "COALINDIA.NS", "DRREDDY.NS", "CIPLA.NS",
]

# ------------------------------------------------------------------ dates
# Start ~5 years before today (a *default*: once data/prices.csv exists the
# cache is used as-is, so later runs are reproducible regardless of today's date).
END_DATE: date = date.today()
START_DATE: date = END_DATE - timedelta(days=5 * 365)

# --------------------------------------------------------- data cleaning
MAX_MISSING_FRAC: float = 0.05   # drop a ticker if > 5% of its prices are NaN
FFILL_LIMIT: int = 3             # forward-fill gaps of at most 3 trading days
MASK_FRACS: list[float] = [0.05, 0.10, 0.20]  # missing-data experiment levels

# ---------------------------------------------------- linear-algebra knobs
INERTIA_TOL: float = 1e-10       # relative: eigenvalue "zero" if |l| <= tol*max|l|
CLIP_EPS: float = 1e-6           # absolute floor for clip_eigenvalues (daily-return units)
# Backtest clipping floor is *relative*: eps = CLIP_REL * tr(S)/N (10% of the average
# eigenvalue). An absolute 1e-6 is below every sample eigenvalue once T > N, so it would
# make "clipped" identical to "raw". Chosen a priori, not tuned.
CLIP_REL: float = 0.10
RANK_RCOND: float = 1e-10        # rank cutoff for pseudo-inverse (relative to lmax)
BOTTOM_K: list[int] = [1, 3, 5]  # k for noise_exposure_score

# ---------------------------------------------------------- backtest setup
WINDOWS: list[int] = [60, 90, 120, 250, 500, 750]
HORIZON: int = 60                # out-of-sample days after each training window
STEP: int = 60                   # roll step; = HORIZON gives non-overlapping OOS blocks
TRADING_DAYS: int = 252          # annualisation: std * sqrt(252) (log returns)
LONG_ONLY_MAX_ITER: int = 500
METHODS: tuple[str, ...] = ("raw", "clipped", "mp", "ledoit_wolf")
CONSTRAINTS: tuple[str, ...] = ("unconstrained", "long_only")

# -------------------------------------------------------------- randomness
SEED: int = 42
N_SEEDS: int = 20                # seeds for broken-Sigma scenarios

# ------------------------------------------------------------------- plots
DEMO_WINDOW: int = 60            # training window for the illustrative figures 1-4
REPAIR_METHOD: str = "mp"        # "repaired" Sigma in figures 2-3 (mp | clipped | ledoit_wolf)
PLOT_K: int = 3                  # bottom-k used in the exposure scatter
DPI: int = 200

# ----------------------------------------------------------------- phase 5
PHASE5_DIR: Path = RESULTS_DIR / "phase5"
BOOT_BLOCK: int = 3              # moving-block length, in OOS periods
BOOT_N: int = 2000               # bootstrap replications
BOOT_ALPHA: float = 0.05         # 95% percentile intervals

# --- 5B theory / Monte Carlo. Tolerances are fixed BEFORE any simulation is run. ---
MC_SIMS: int = 10000             # simulated (train, test) pairs per window and scenario
MC_SHRINK: float = 0.5           # true Sigma = (1-d)*S_full + d*tr(S_full)/N*I
MC_T_DF: int = 5                 # degrees of freedom of the Student-t scenario
MC_Z_TOL: float = 3.0            # "matches" = |MC mean - theory| <= 3 Monte Carlo standard errors
MC_KS_ALPHA: float = 0.01        # exact-distribution KS check flags a mismatch if p < 0.01
THEORY_DRAWS: int = 400_000      # draws from the exact chi-square representation

# 5D sensitivity sweeps
CLIP_SWEEP: list[float] = [0.01, 0.05, 0.10, 0.25, 0.50]   # eps = fraction * tr(S)/N
MP_SIGMA2_RULES: tuple[str, ...] = ("trace", "median")     # MP noise variance: tr(S)/N or median eigenvalue

# 5E bootstrap diagnostics
BOOT_BLOCK_SENSITIVITY: tuple[int, ...] = (1, 2, 3, 4)   # block lengths for the sensitivity table
BOOT_COVER_REPS: int = 1000                              # replications in the coverage check
BOOT_COVER_B: int = 400                                  # bootstrap replications per coverage rep
