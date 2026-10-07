"""Run the whole pipeline for each universe (and optionally sub-period) and write ONE comparison table.

    python compare_universes.py                       # every configured universe, full sample
    python compare_universes.py --phase5              # include the Phase 5 extensions (slower)
    python compare_universes.py --periods full first_half second_half
    python compare_universes.py --universes nifty30 nifty_next

Each run is a separate process (`python run_all.py --universe U --period P`), so runs cannot contaminate
each other. A universe whose price cache is missing is skipped with instructions (nothing is downloaded).
Output: results/universe_comparison.csv (metric x run) and the same table printed to the terminal.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from src import config, universes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universes", nargs="+", default=sorted(config.UNIVERSES), choices=sorted(config.UNIVERSES))
    ap.add_argument("--periods", nargs="+", default=["full"])
    ap.add_argument("--phase5", action="store_true")
    a = ap.parse_args(argv)

    dirs: dict[str, object] = {}
    for u in a.universes:
        for p in a.periods:
            spec = universes.resolve_run(u, p)
            name = u if p == "full" else f"{u}:{p}"
            if not spec.prices_csv.exists():
                print(f"-- skipping {name}\n{universes.missing_data_message(spec)}\n", file=sys.stderr)
                continue
            cmd = [sys.executable, "run_all.py", "--universe", u, "--period", p] + (["--phase5"] if a.phase5 else [])
            print(f"== {name}: {' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=config.ROOT)
            if r.returncode != 0:
                print(f"-- {name} failed (exit {r.returncode}); not included", file=sys.stderr)
                continue
            dirs[name] = spec.results_dir
    if not dirs:
        print("nothing to compare (no universe had data).", file=sys.stderr)
        return 1
    table = universes.compare(dirs)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(config.RESULTS_DIR / "universe_comparison.csv")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print("\nHeadline comparison (results/universe_comparison.csv):")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
