"""PDF ingestion, tested against the real committed filings.

The RBAC demonstration depends entirely on this layer tagging correctly. If the
Human Capital section of the 10-K is not tagged hr.headcount, the CTO refusal
is meaningless — there would be nothing to withhold.
"""

from pathlib import Path

import pytest

from src.access.model import Tag
from src.ingest.chunker import chunk_document, load_config
from src.ingest.pdf_loader import load_pdf
from src.ingest.tagger import default_tag_for_document, normalise, tag_for_heading

RAW = Path("data/raw")
CFG = load_config(Path("config/sources.yaml"))


# -- normalisation --------------------------------------------------------

def test_normalise_collapses_pdf_layout_artifacts():
    """Extraction splits Apple's "full-time" across the hyphen as "full- time".
    Every comparison normalises first, or the headcount section goes untagged."""
    assert normalise("full- time  equivalent\nemployees") == \
        "full- time equivalent employees"
    assert normalise("  Human   Capital  ") == "human capital"


# -- heading -> tag -------------------------------------------------------

@pytest.mark.parametrize("heading,expected", [
    ("Item 1A. Risk Factors", Tag.NARR_RISK),
    ("Human Capital", Tag.HR_HEADCOUNT),
    ("Summary Compensation Table", Tag.HR_COMPENSATION),
    ("Compensation Discussion and Analysis", Tag.HR_COMPENSATION),
    ("Corporate Governance", Tag.GOVERNANCE),
    ("Segment Information", Tag.FIN_SEGMENT),
    ("Item 7. Management's Discussion and Analysis", Tag.NARR_MDNA),
])
def test_headings_map_to_tags(heading, expected):
    assert tag_for_heading(heading, CFG) is expected


def test_ordinary_prose_is_not_a_heading():
    assert tag_for_heading("The Company designs and markets smartphones.",
                           CFG) is None


@pytest.mark.parametrize("fragment", [
    # Real lines from the proxy that were wrongly promoted to headings and then
    # governed pages of unrelated prose.
    "appropriate, other employees",
    "guidance to our employees, executive officers, and directors.",
    "Review and make recommendations to the Board regarding the compensation",
    "including employees and",
])
def test_prose_fragments_are_not_promoted_to_headings(fragment):
    """A heading governs every chunk until the next one, so a false positive is
    expensive: one bad match mis-tagged pages of governance prose as restricted
    HR data."""
    assert tag_for_heading(fragment, CFG) is None


def test_numbered_item_headings_are_always_headings():
    """SEC filings number their sections; "Item 7." is a heading however much
    trailing text it carries."""
    assert tag_for_heading(
        "Item 7. Management's Discussion and Analysis of Financial Condition",
        CFG) is Tag.NARR_MDNA


def test_document_defaults_differ_by_filing_type():
    """A proxy statement is governance material by default; a 10-K is
    narrative. Content before the first heading has to land somewhere."""
    assert default_tag_for_document("DEF14A_2026.pdf", CFG) is Tag.GOVERNANCE
    assert default_tag_for_document("10-K_FY2025.pdf", CFG) is Tag.NARR_MDNA


# -- whole documents ------------------------------------------------------

def test_10k_produces_chunks_with_page_provenance():
    chunks = chunk_document(RAW / "10-K_FY2025.pdf", fiscal_year=2025, cfg=CFG)

    assert len(chunks) > 60
    assert all(c.locator.startswith("p.") for c in chunks)
    assert all(c.source == "10-K_FY2025.pdf" for c in chunks)
    assert all(len(c.text) <= CFG["chunking"]["max_chars"] for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"


def test_10k_tags_the_headcount_section():
    """Without this, the CTO refusal has nothing to withhold."""
    chunks = chunk_document(RAW / "10-K_FY2025.pdf", fiscal_year=2025, cfg=CFG)
    headcount = [c for c in chunks if c.tag is Tag.HR_HEADCOUNT]

    assert headcount, "Human Capital section was not tagged"
    joined = " ".join(c.text for c in headcount).lower()
    assert "employees" in joined
    # The actual restricted figure must land in a restricted chunk. If it leaks
    # into an unrestricted one, the whole access model is decorative.
    assert "166,000" in joined


def test_headcount_chunks_come_only_from_annual_reports():
    """The employee count is stated in the 10-K, not the proxy. An earlier
    rule matching the bare word "employees" promoted wrapped sentence endings
    to headings and tagged pages of executive compensation prose as headcount."""
    from src.ingest.chunker import chunk_corpus

    chunks = chunk_corpus(RAW, CFG)
    headcount = [c for c in chunks if c.tag is Tag.HR_HEADCOUNT]

    assert headcount
    assert all(c.source.startswith("10-K") for c in headcount), \
        {c.source for c in headcount}
    # One per filing year, carrying the figure that year reports.
    joined = " ".join(c.text for c in headcount)
    for expected in ("166,000", "164,000", "161,000"):
        assert expected in joined


def test_10k_tags_risk_factors():
    chunks = chunk_document(RAW / "10-K_FY2025.pdf", fiscal_year=2025, cfg=CFG)
    assert any(c.tag is Tag.NARR_RISK for c in chunks)


def test_proxy_tags_executive_compensation():
    """The DEF 14A carries the real compensation tables the CTO must not see."""
    chunks = chunk_document(RAW / "DEF14A_2026.pdf", fiscal_year=2026, cfg=CFG)
    comp = [c for c in chunks if c.tag is Tag.HR_COMPENSATION]

    assert comp, "no compensation section found in the proxy"
    assert len(comp) >= 5


def test_chunks_carry_the_fiscal_year_they_describe():
    chunks = chunk_document(RAW / "10-K_FY2023.pdf", fiscal_year=2023, cfg=CFG)
    assert all(c.fiscal_year == 2023 for c in chunks)


def test_nothing_is_quarantined_by_default():
    """Quarantine is set by the injection scan, not by ingestion."""
    chunks = chunk_document(RAW / "10-K_FY2025.pdf", fiscal_year=2025, cfg=CFG)
    assert not any(c.quarantined for c in chunks)


def test_load_pdf_returns_page_numbers_from_one():
    pages = load_pdf(RAW / "10-K_FY2025.pdf")
    assert pages[0][0] == 1
    assert len(pages) == 77
    assert all(isinstance(text, str) for _, text in pages)
