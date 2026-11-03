"""Download the US replication universe (configs/journal_us_replication.yaml) from Yahoo Finance into data/prices_us_2011_2026.csv."""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import download_prices  # noqa: E402

cfg = yaml.safe_load((ROOT / "configs" / "journal_us_replication.yaml").read_text())
tk = cfg["universe_A_us"] + cfg["universe_B_us"]
px = download_prices(tk, "2011-05-25", "2026-10-07")
miss = [t for t in tk if px[t].isna().all()]
for t in miss:                                   # Yahoo is flaky in bulk calls: retry singly
    one = download_prices([t], "2011-05-25", "2026-10-07")
    px[t] = one[t]
print(px.shape, px.index[0].date(), px.index[-1].date(), "all-NaN tickers:", [t for t in tk if px[t].isna().all()])
print("missing share per ticker > 1%:", {t: round(float(px[t].isna().mean()), 3) for t in tk if px[t].isna().mean() > 0.01})
px.to_csv(ROOT / "data" / "prices_us_2011_2026.csv")
