"""Extract stated figures from narrative text.

Not every number a filing reports lives in a spreadsheet. Apple states its
employee count in a sentence on page 8 of each 10-K:

    "As of September 30, 2023, the Company had approximately 161,000
     full-time equivalent employees."

That is the authoritative company-wide headcount — Apple's own statement, in
the filing, with a page to cite. It never appears in the statement workbooks,
so a pipeline that only reads spreadsheets holds the sentence as prose and
cannot answer "how many employees were there in FY2023" numerically.

This module closes that gap for figures the filings state in words. It is
deliberately narrow: a small set of explicit patterns, each producing a Fact
with a real citation. It is not a general "find numbers in text" pass, which
would produce confident nonsense at scale.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.access.model import Tag
from src.ingest.pdf_loader import load_pdf
from src.understanding.facts import Fact

log = logging.getLogger(__name__)

# "approximately 161,000 full-time equivalent employees"
#
# The hyphen is optional and may carry a space, because PDF extraction splits
# "full-time" across a line break as "full- time" — the same artifact that made
# an exact-string probe report this sentence as missing from a document that
# plainly contains it.
_HEADCOUNT = re.compile(
    r"approximately\s+([\d,]+)\s+full-?\s*time\s+equivalent\s+employees",
    re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")


def extract_headcount(path: Path, fiscal_year: int) -> list[Fact]:
    """Read the company-wide employee count stated in a 10-K.

    Returns at most one Fact. Several pages may repeat the figure; the first
    occurrence is the Human Capital statement and the rest are references.
    """
    for page_number, text in load_pdf(path):
        match = _HEADCOUNT.search(_WHITESPACE.sub(" ", text))
        if match is None:
            continue
        value = float(match.group(1).replace(",", ""))
        log.info("headcount %s FY%d: %s (p.%d)", path.name, fiscal_year,
                 f"{value:,.0f}", page_number)
        return [Fact(
            metric="headcount", value=value, unit="people",
            period=f"FY{fiscal_year}", fiscal_year=fiscal_year,
            source=path.name, locator=f"p.{page_number}",
            tag=Tag.HR_HEADCOUNT,
        )]
    return []


def extract_from_corpus(raw_dir: Path) -> list[Fact]:
    """Run every narrative extractor over the annual reports.

    Only 10-Ks: the employee count is an annual disclosure, and reading it from
    a 10-Q would attach a stale figure to a quarter that never stated one.
    """
    import json

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    facts: list[Fact] = []
    for entry in manifest:
        if entry["form"] != "10-K":
            continue
        facts += extract_headcount(raw_dir / entry["pdf"], entry["fiscal_year"])
    return facts
