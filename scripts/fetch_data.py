"""Download Apple's public filings from SEC EDGAR into data/raw/.

Why EDGAR rather than investor.apple.com: Apple's investor site sits behind bot
protection and returns 403 to any programmatic client. EDGAR is the authoritative
origin of the same documents and explicitly permits automated access provided the
User-Agent identifies the requester.

Two format conversions happen here, both from official sources:

1. EDGAR serves filings as HTML, but the assignment requires PDF ingestion, so
   each filing is rendered to PDF locally with Edge headless.
2. SEC published a ready-made Financial_Report.xlsx for the FY2023 and FY2024
   filings but stopped for filings from 2025 onward. For those, the statement
   tables are rebuilt from EDGAR's own R*.htm XBRL renderings — the same source
   SEC used to generate the spreadsheet itself.

Neither conversion invents data; both are re-encodings of the official filing.

Run:  python scripts/fetch_data.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

# SEC asks that automated clients identify themselves with a contact address.
# Read from the environment rather than hardcoded: a personal email does not
# belong in a repository that gets shared. The corpus is already committed, so
# this script only needs to run if you want to refresh the data.
UA = os.environ.get(
    "SEC_USER_AGENT",
    "azentio-assignment financial-agent (set SEC_USER_AGENT to your contact)",
)
CIK = 320193  # Apple Inc.
RAW = Path("data/raw")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# Three annual reports gives three fiscal years of narrative; four quarters gives
# enough structured Excel to be worth indexing. One proxy carries the executive
# compensation tables that the CTO role must not be able to read.
WANTED = {"10-K": 3, "10-Q": 4, "DEF 14A": 1}

# SEC asks for no more than 10 requests/second.
REQUEST_DELAY_S = 0.2

# The R*.htm files are EDGAR's rendered XBRL tables. Their count varies by
# filing (a 10-K has ~70, a 10-Q ~40), so the list is read from the filing's
# index rather than guessed at — an earlier fixed range silently captured only
# the first ten and dropped sixty detail tables.
R_FILE_PATTERN = re.compile(r"^R(\d+)\.htm$")

# Apple's fiscal year ends in late September, so a quarter ending in Oct-Dec
# belongs to the *next* fiscal year. Q1 FY2026 ends 27 Dec 2025.
FY_ROLLOVER_MONTH = 10


def fetch(url: str, retries: int = 3) -> bytes:
    """GET with the identifying User-Agent EDGAR requires.

    EDGAR returns transient 503s under load. A 404 means the file genuinely is
    not there and is raised immediately; anything else is retried with backoff.
    """
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            time.sleep(REQUEST_DELAY_S)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fiscal_year_of(report_date: str) -> int:
    year, month = int(report_date[:4]), int(report_date[5:7])
    return year + 1 if month >= FY_ROLLOVER_MONTH else year


def recent_filings() -> list[dict]:
    raw = json.loads(fetch(f"https://data.sec.gov/submissions/CIK{CIK:010d}.json"))
    r = raw["filings"]["recent"]
    out, taken = [], {k: 0 for k in WANTED}
    for form, fdate, acc, doc, rdate in zip(
        r["form"], r["filingDate"], r["accessionNumber"],
        r["primaryDocument"], r["reportDate"]
    ):
        if form not in WANTED or taken[form] >= WANTED[form]:
            continue
        taken[form] += 1
        fy = fiscal_year_of(rdate)
        # Slugs must be unique: two 10-Qs can share a fiscal year, so quarterly
        # filings are keyed by period end date rather than by year alone.
        slug = {"10-K": f"10-K_FY{fy}",
                "10-Q": f"10-Q_{rdate}",
                "DEF 14A": f"DEF14A_{fdate[:4]}"}[form]
        out.append({
            "form": form, "slug": slug, "filing_date": fdate,
            "report_date": rdate, "fiscal_year": fy,
            "base": f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc.replace('-', '')}",
            "doc": doc,
        })
    return out


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Render a local HTML filing to PDF with Edge headless.

    An isolated --user-data-dir keeps this from clashing with a running Edge.
    """
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={profile}",
             f"--print-to-pdf={pdf_path.resolve()}",
             html_path.resolve().as_uri()],
            check=True, capture_output=True, timeout=300,
        )


def verify_pdf(path: Path) -> tuple[int, int]:
    """Return (page_count, chars_on_first_page).

    Zero characters means no text layer, which would make the PDF useless to the
    ingestion pipeline. Checked rather than assumed.
    """
    import fitz
    with fitz.open(path) as doc:
        return doc.page_count, len(doc[0].get_text())


def r_file_numbers(base: str) -> list[int]:
    """Read the filing's index to learn which R*.htm files actually exist."""
    index = json.loads(fetch(f"{base}/index.json"))
    nums = []
    for item in index["directory"]["item"]:
        m = R_FILE_PATTERN.match(item["name"])
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def unique_sheet_name(title: str, taken: set[str]) -> str:
    """Excel caps sheet names at 31 chars and forbids []:*?/\\ — and truncation
    makes collisions likely across seventy statement tables, so names are
    de-duplicated with a numeric suffix.
    """
    safe = "".join(c for c in title if c not in "[]:*?/\\").strip() or "Sheet"
    name, n = safe[:31], 2
    while name in taken:
        suffix = f"_{n}"
        name = safe[:31 - len(suffix)] + suffix
        n += 1
    taken.add(name)
    return name


def build_xlsx_from_r_files(base: str, out_path: Path) -> int:
    """Rebuild statement tables from EDGAR's R*.htm XBRL renderings.

    Used only where SEC did not publish Financial_Report.xlsx. Returns the number
    of sheets written.
    """
    sheets: dict[str, pd.DataFrame] = {}
    taken: set[str] = set()
    for n in r_file_numbers(base):
        try:
            html = fetch(f"{base}/R{n}.htm").decode("utf-8", errors="replace")
        except urllib.error.HTTPError:
            continue
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:  # no table in this R file
            continue
        if not tables:
            continue
        df = tables[0]
        # read_html gives MultiIndex columns for these nested statement headers,
        # which openpyxl cannot write. Flatten to single strings first.
        df.columns = [
            " ".join(str(p) for p in col if "Unnamed" not in str(p)).strip()
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
        # The first column header carries the statement name, e.g.
        # "CONSOLIDATED STATEMENTS OF OPERATIONS - USD ($) $ in Millions".
        title = str(df.columns[0]).split(" - ")[0].strip() or f"R{n}"
        sheets[unique_sheet_name(title, taken)] = df

    if not sheets:
        return 0
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return len(sheets)


def main() -> int:
    if not EDGE.exists():
        print(f"ERROR: Edge not found at {EDGE}", file=sys.stderr)
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []

    for f in recent_filings():
        print(f"\n{f['form']}  FY{f['fiscal_year']}  period {f['report_date']}")

        html_path = RAW / f"{f['slug']}.htm"
        pdf_path = RAW / f"{f['slug']}.pdf"
        html_path.write_bytes(fetch(f"{f['base']}/{f['doc']}"))
        html_to_pdf(html_path, pdf_path)
        html_path.unlink()  # the PDF is the deliverable; the HTML was scaffolding

        pages, chars = verify_pdf(pdf_path)
        if chars == 0:
            print(f"  ERROR: {pdf_path.name} has no text layer", file=sys.stderr)
            return 1
        print(f"  {pdf_path.name}: {pages} pages, {chars} chars on p.1")
        entry = {**f, "pdf": pdf_path.name, "pages": pages}

        if f["form"] in ("10-K", "10-Q"):
            xlsx_path = RAW / f"{f['slug']}_financials.xlsx"
            try:
                xlsx_path.write_bytes(fetch(f"{f['base']}/Financial_Report.xlsx"))
                print(f"  {xlsx_path.name}: SEC-published, "
                      f"{xlsx_path.stat().st_size:,} bytes")
                entry["xlsx"] = xlsx_path.name
                entry["xlsx_origin"] = "sec_published"
            except urllib.error.HTTPError:
                sheets = build_xlsx_from_r_files(f["base"], xlsx_path)
                if sheets:
                    print(f"  {xlsx_path.name}: rebuilt from R*.htm, "
                          f"{sheets} sheets, {xlsx_path.stat().st_size:,} bytes")
                    entry["xlsx"] = xlsx_path.name
                    entry["xlsx_origin"] = "rebuilt_from_xbrl_rendering"
                else:
                    print("  WARNING: no Excel data available for this filing")

        manifest.append(entry)

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pdfs = sum(1 for m in manifest if "pdf" in m)
    xlsxs = sum(1 for m in manifest if "xlsx" in m)
    print(f"\nDone. {len(manifest)} filings: {pdfs} PDFs, {xlsxs} spreadsheets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
