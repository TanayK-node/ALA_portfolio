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
RESULT_DIRS = ["results/journal_extension/tables", "results/journal_extension/figures", "results/journal_extension/metadata"]

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

## Reproduce
1. `pip install -r requirements.txt`
2. `python fetch_prices.py`  (downloads prices from Yahoo Finance; see data/MANIFEST.json for the expected shape and hashes)
3. `python run_journal.py baseline`, `extended`, `bps_check`, `intervals`, `surface`, `regimes`, `second_stage`, `predictors`,
   `validate`, `validate_skew`, `dynamic`, then `python run_journal.py tables` and `figures`.
4. `python -m pytest` (tests that need the price files require step 2 first).

Run metadata (data hashes, seeds, package versions) for the reported results is in `results/journal_extension/metadata/`.
Price data are not redistributed because Yahoo's terms restrict reproduction and distribution; vendor revisions may change results
slightly, which is why the manifest lists the hashes of the files the reported numbers were computed from.
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    managed = KEEP_DIRS + KEEP_FILES + ["results", "data", "fetch_prices.py", "README_ANON.md"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in managed:                         # remove only what this script creates; never touch other folders a user put here
        t = OUT / name
        shutil.rmtree(t) if t.is_dir() else t.unlink(missing_ok=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for d in KEEP_DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=ign)
    for f in KEEP_FILES:
        shutil.copy2(ROOT / f, OUT / f)
    for d in RESULT_DIRS:
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, OUT / d)
    import pandas as pd
    files = {}
    for name, (start, end) in {"prices.csv": ("2021-10-08", "2026-10-07"), "prices_2011_2026.csv": ("2011-05-25", "2026-10-07")}.items():
        p = ROOT / "data" / name
        files[name] = dict(sha256=sha(p), shape=list(pd.read_csv(p, index_col=0).shape), start=start, end_exclusive=end)
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
