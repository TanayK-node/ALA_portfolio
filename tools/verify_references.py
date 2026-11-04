"""Look up every entry of manuscript/latex/sn-bibliography.bib in Crossref and compare title, year, journal, volume, issue, pages.
Writes the raw comparison to manuscript/reference_audit_crossref.csv. A mismatch or a missing record is a flag for the authors, not a verdict."""
import csv
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bib = (ROOT / "manuscript/latex/sn-bibliography.bib").read_text()


def field(e, name):
    m = re.search(rf"{name}\s*=\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}", e)
    return re.sub(r"[{}\\]", "", m.group(1)).strip() if m else ""


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower().replace("&", "and"))


rows = []
for e in re.split(r"\n(?=@)", bib):
    k = re.match(r"@(\w+)\{([^,]+),", e)
    if not k:
        continue
    key, title, year = k.group(2), field(e, "title"), field(e, "year")
    author = field(e, "author").split(" and ")[0].split(",")[0]
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author} {year}", "rows": 3})
    try:
        req = urllib.request.Request(f"https://api.crossref.org/works?{q}", headers={"User-Agent": "ref-audit/1.0 (mailto:none@example.org)"})
        items = json.load(urllib.request.urlopen(req, timeout=30))["message"]["items"]
    except Exception as ex:
        items = []
    best, score = None, 0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, norm(t), norm(title)).ratio()
        if s > score:
            best, score = it, s
    row = dict(key=key, bib_title=title, bib_year=year, bib_journal=field(e, "journal"), bib_vol=field(e, "volume"), bib_num=field(e, "number"),
               bib_pages=field(e, "pages").replace("--", "-"), title_similarity=round(score, 3))
    if best:
        issued = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
        row.update(doi=best.get("DOI", ""), cr_title=(best.get("title") or [""])[0], cr_year=issued,
                   cr_journal=(best.get("container-title") or [""])[0], cr_vol=best.get("volume", ""), cr_num=best.get("issue", ""),
                   cr_pages=best.get("page", ""), cr_first_author=(best.get("author") or [{}])[0].get("family", ""))
    rows.append(row)
    time.sleep(0.3)
cols = ["key", "bib_title", "cr_title", "title_similarity", "bib_year", "cr_year", "bib_journal", "cr_journal", "bib_vol", "cr_vol", "bib_num", "cr_num",
        "bib_pages", "cr_pages", "cr_first_author", "doi"]
with open(ROOT / "manuscript/reference_audit_crossref.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(len(rows), "entries written")
