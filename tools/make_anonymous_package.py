"""Build an anonymised review copy of the code in ../ALA_portfolio_anonymous_package/ (never touches the repo itself).

Included: src/, tests/, configs/, runner scripts, requirements, derived tables/figures/metadata, a price-download script and a
data manifest (hashes + shapes, NOT the Yahoo price files: Yahoo's terms restrict redistribution). Excluded: .git history,
notebooks, research_docs (internal audit notes with the remote name), raw price CSVs, per-origin forecast CSVs.
After building, the script scans every text file for identifying strings and fails if one is found.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "ALA_portfolio_anonymous_package"   # outside the repo so pytest/git never see it
IDENT = re.compile(r"tanay|vanshi|spit\.ac|/Users/|gathani", re.I)

KEEP_DIRS = ["src", "tests", "configs"]
KEEP_FILES = ["run_journal.py", "run_all.py", "compare_universes.py", "phase1_report.py", "phase2_report.py", "phase3_report.py",
              "requirements.txt", "requirements-dev.txt"]
RESULT_DIRS = ["results/journal_extension/tables", "results/journal_extension/figures", "results/journal_extension/metadata",
               "results/journal_extension/audit", "results/journal_extension_us/tables", "results/journal_extension_us/metadata",
               "results/journal_extension_us/audit"]
KEEP_TOOLS = ["fetch_us_prices.py", "independent_check.py"]

FETCH = '''"""Download the price files used in the study from Yahoo Finance (you must accept Yahoo's terms of use yourself).

Writes data/prices.csv (2021-10-08..2026-10-06, 30 stocks) and data/prices_2011_2026.csv (2011-05-25..2026-10-06, 60 tickers),
then compares shapes and hashes with data/MANIFEST.json. Vendor data can be revised, so a hash mismatch is reported, not fatal;
shape and ticker-count mismatches are fatal.
"""
import hashlib
import json
from pathlib import Path

from src import config
from src.data import download_prices

ROOT = Path(__file__).resolve().parent
man = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
tickers = config.TICKERS + config.NIFTY_NEXT_TICKERS
(ROOT / "data").mkdir(exist_ok=True)
for fname, spec in man["files"].items():
    if fname.startswith("prices_us"):
        continue                                  # US file: python tools/fetch_us_prices.py
    px = download_prices(config.TICKERS if fname == "prices.csv" else tickers, spec["start"], spec["end_exclusive"])
    px.to_csv(ROOT / "data" / fname)
    ok_shape = list(px.shape) == spec["shape"]
    h = hashlib.sha256((ROOT / "data" / fname).read_bytes()).hexdigest()
    print(fname, "shape", px.shape, "expected", spec["shape"], "| sha256 match:", h == spec["sha256"])
    if px.shape[1] != spec["shape"][1]:
        raise SystemExit(f"{fname}: ticker count differs (Yahoo dropped/renamed symbols); see README_ANON.md")
    if not ok_shape:
        print("  WARNING: row count differs; results may differ slightly")
'''

README = """# Anonymised code package (review copy)

Code, tests, configurations and derived tables/figures for the manuscript. Author names, history and raw vendor data are removed.

## Environment
Python 3.13.9; numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.9.0 (the versions used; `requirements.txt` gives minimum versions).
Random seeds are fixed in the configuration files and in `src/config.py` (SEED = 42 for bootstraps; stage seeds are written to each run's metadata).

## Reproduce
1. `pip install -r requirements.txt`
2. `python fetch_prices.py` (India) and `python tools/fetch_us_prices.py` (US): download prices from Yahoo Finance; see data/MANIFEST.json for shapes and hashes. You must accept Yahoo's terms yourself; vendor revisions can change results slightly.
3. India (`python run_journal.py <stage>`), in this order: `baseline`, `extended`, `bps_check`, `validate`, `validate_skew`, `dynamic`, `surface`, `regimes`,
   `second_stage`, `predictors`, `intervals`, `cqlike_check`, `conformal`, `kurtosis_sim`, `kurtosis_eval`, `risk_use`, `same_target`,
   `interval_check`, `zero_sens`.
   US (`--market us`): `extended`, `intervals`, `cqlike_check`, `conformal`, `kurtosis_eval`, `risk_use`, `same_target`, `interval_check`; then `replication_summary`.
4. `python run_journal.py tables` and `python run_journal.py figures` build every table and figure from the saved result files.
5. `python tools/independent_check.py` recomputes headline numbers with separate code. `python -m pytest` runs the tests (some need the price files).

Pre-specified configurations are in `configs/` (written before the runs they govern). Run metadata (data hashes, seeds, package versions) is in `results/*/metadata/`.
Price data are not redistributed because Yahoo's terms restrict reproduction and distribution.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    managed = KEEP_DIRS + KEEP_FILES + ["tools", "results", "data", "fetch_prices.py", "README_ANON.md"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in managed:                         # remove only what this script creates; never touch other folders a user put here
        t = OUT / name
        shutil.rmtree(t) if t.is_dir() else t.unlink(missing_ok=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for d in KEEP_DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=ign)
    for f in KEEP_FILES:
        shutil.copy2(ROOT / f, OUT / f)
    (OUT / "tools").mkdir()
    for f in KEEP_TOOLS:
        shutil.copy2(ROOT / "tools" / f, OUT / "tools" / f)
    for d in RESULT_DIRS:
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, OUT / d)
    import pandas as pd
    files = {}
    for name, (start, end) in {"prices.csv": ("2021-10-08", "2026-10-07"), "prices_2011_2026.csv": ("2011-05-25", "2026-10-07")}.items():
        p = ROOT / "data" / name
        files[name] = dict(sha256=sha(p), shape=list(pd.read_csv(p, index_col=0).shape), start=start, end_exclusive=end)
    pu = ROOT / "data" / "prices_us_2011_2026.csv"
    if pu.exists():
        files["prices_us_2011_2026.csv"] = dict(sha256=sha(pu), shape=list(pd.read_csv(pu, index_col=0).shape), start="2011-05-25", end_exclusive="2026-10-07",
                                                note="fetched by tools/fetch_us_prices.py")
    (OUT / "data").mkdir()
    (OUT / "data" / "MANIFEST.json").write_text(json.dumps(dict(source="Yahoo Finance via yfinance, adjusted close", files=files), indent=2))
    (OUT / "fetch_prices.py").write_text(FETCH)
    (OUT / "README_ANON.md").write_text(README)
    bad = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml"}:
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if IDENT.search(line):
                    bad.append(f"{p.relative_to(OUT)}:{i}: {line[:100]}")
    if bad:
        print("IDENTIFYING STRINGS FOUND:\n" + "\n".join(bad[:30]))
        sys.exit(1)
    print("anonymous package ->", OUT, sum(1 for _ in OUT.rglob('*') if _.is_file()), "files, no identifying strings found")


if __name__ == "__main__":
    main()
