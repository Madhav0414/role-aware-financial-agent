"""Excel ingestion, tested against the real committed workbooks.

Two shapes exist in the corpus and both must work:

  SEC-published   header split across two rows — "12 Months Ended" on one,
                  the period end dates on the next
  rebuilt         header combined into one row — "12 Months Ended Sep. 27, 2025"

Values arrive as bare integers, as "$ 416,161", and as "(24)" for negatives.
"""

from pathlib import Path

import pytest

from src.access.model import Tag
from src.ingest.excel_loader import (
    load_headcount_workbook,
    load_statement_workbook,
    parse_period_header,
    parse_value,
)

RAW = Path("data/raw")


# -- value parsing --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (391035, 391035.0),
    ("$ 416,161", 416161.0),
    ("(24)", -24.0),
    ("$ (1,234)", -1234.0),
    ("2.5", 2.5),
])
def test_values_parse(raw, expected):
    assert parse_value(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "Operating expenses:", None, "n/a"])
def test_non_numeric_values_return_none(raw):
    """Section headers and blanks are rows without a value, not errors."""
    assert parse_value(raw) is None


# -- period headers -------------------------------------------------------

@pytest.mark.parametrize("header,expected", [
    ("12 Months Ended Sep. 27, 2025", "FY2025"),
    ("12 Months Ended Sep. 28, 2024", "FY2024"),
    # Apple's fiscal year ends in late September, so a quarter ending in
    # December belongs to the NEXT fiscal year.
    ("3 Months Ended Dec. 27, 2025", "Q1FY2026"),
    ("3 Months Ended Jun. 27, 2026", "Q3FY2026"),
    ("3 Months Ended Mar. 28, 2026", "Q2FY2026"),
])
def test_period_headers_parse(header, expected):
    parsed = parse_period_header(header)
    assert parsed is not None
    assert parsed.label == expected


def test_year_to_date_columns_are_skipped():
    """A "9 Months Ended" column is cumulative. Admitting it alongside
    quarterly figures would put two different meanings under one label."""
    assert parse_period_header("9 Months Ended Jun. 27, 2026") is None


def test_header_without_a_date_is_not_a_period():
    assert parse_period_header("CONSOLIDATED STATEMENTS OF OPERATIONS") is None


# -- whole workbooks ------------------------------------------------------

def test_sec_published_workbook_yields_correct_net_sales():
    facts = load_statement_workbook(RAW / "10-K_FY2024_financials.xlsx")
    hits = [f for f in facts if f.metric == "net_sales" and f.period == "FY2024"]

    assert hits, "net_sales FY2024 not found"
    assert hits[0].value == 391_035.0
    assert hits[0].tag is Tag.FIN_STATEMENTS
    assert hits[0].source == "10-K_FY2024_financials.xlsx"
    assert "!" in hits[0].locator  # sheet!cell provenance


def test_rebuilt_workbook_yields_correct_net_sales():
    facts = load_statement_workbook(RAW / "10-K_FY2025_financials.xlsx")
    hits = [f for f in facts if f.metric == "net_sales" and f.period == "FY2025"]

    assert hits
    assert hits[0].value == 416_161.0


def test_prior_year_columns_are_captured_with_their_own_period():
    """Each statement restates two prior years. Those columns are real data and
    must be tagged with the year they describe, not the year of the filing."""
    facts = load_statement_workbook(RAW / "10-K_FY2025_financials.xlsx")
    periods = {f.period for f in facts if f.metric == "net_sales"}
    assert {"FY2025", "FY2024", "FY2023"} <= periods


def test_quarterly_workbook_produces_quarterly_labels():
    facts = load_statement_workbook(RAW / "10-Q_2026-06-27_financials.xlsx")
    assert any(f.period.startswith("Q") for f in facts)
    assert all(f.fiscal_year >= 2025 for f in facts)


def test_synthetic_headcount_is_tagged_hr_and_skips_the_banner():
    facts = load_headcount_workbook(
        RAW / "_synthetic" / "headcount_by_department.xlsx")

    assert facts
    assert all(f.tag is Tag.HR_HEADCOUNT for f in facts)
    assert all(f.unit == "people" for f in facts)
    # Row 1 is the SYNTHETIC banner and must not become a fact.
    assert all("SYNTHETIC" not in f.metric.upper() for f in facts)

    fy2025 = sum(f.value for f in facts if f.fiscal_year == 2025)
    assert fy2025 == 166_000  # reconciles with the figure the 10-K states


def test_nested_categories_do_not_collapse_into_one_metric():
    """XBRL renderings nest breakdowns under category headers: a row reading
    "iPhone" with no figures scopes the "Net sales" row beneath it. Flattening
    that hierarchy would give twenty different values all called `net_sales`,
    and a question about net sales would return whichever came first."""
    facts = load_statement_workbook(RAW / "10-K_FY2025_financials.xlsx")

    net_sales = {f.value for f in facts
                 if f.metric == "net_sales" and f.period == "FY2025"}
    assert net_sales == {416_161.0}, "net_sales must resolve to one value"

    by_metric = {f.metric: f.value for f in facts if f.period == "FY2025"}
    assert by_metric["iphone_net_sales"] == 209_586.0
    assert by_metric["services_net_sales"] == 109_158.0


def test_product_breakdown_reconciles_with_the_total():
    """A structural check on the parser, not on Apple: if scoping were wrong,
    these parts would not add up to the whole."""
    facts = load_statement_workbook(RAW / "10-K_FY2025_financials.xlsx")
    by_metric = {f.metric: f.value for f in facts if f.period == "FY2025"}

    parts = ["iphone_net_sales", "mac_net_sales", "ipad_net_sales",
             "wearables_home_and_accessories_net_sales", "services_net_sales"]
    assert sum(by_metric[p] for p in parts) == by_metric["net_sales"]


def test_xbrl_scaffolding_rows_never_become_metrics():
    """"[Line Items]" and "[Abstract]" rows sit between a category header and
    its figures. Left in, they would hijack the scope."""
    facts = load_statement_workbook(RAW / "10-K_FY2025_financials.xlsx")
    for f in facts:
        assert "line_items" not in f.metric
        assert "abstract" not in f.metric


def test_malformed_workbook_is_skipped_not_fatal(tmp_path):
    """One unreadable sheet must never abort ingestion of the rest."""
    import pandas as pd
    path = tmp_path / "messy.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({0: ["nothing", "useful", "here"]}).to_excel(
            writer, sheet_name="Junk", index=False, header=False)

    assert load_statement_workbook(path) == []
