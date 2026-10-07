"""Phase 1 diagnostics on the full-sample covariance (real cached data)."""
from __future__ import annotations

import logging

from src import data
from src.diagnostics import full_sample_report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    returns, _, dropped = data.get_returns()
    print(f"Dropped tickers: {dropped or 'none'}")
    for k, v in full_sample_report(returns).items():
        print(f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
