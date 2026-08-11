"""Read financial statement workbooks into typed Facts.

Two workbook shapes exist in the corpus, and the loader handles both without
branching on filename:

  SEC-published   "12 Months Ended" sits on one header row and the period end
                  dates on the next, with the duration spanning merged cells.
  rebuilt         Duration and date are combined: "12 Months Ended Sep. 27, 2025".

The unifying trick is to scan the first few rows as one header *block*, carry
each duration rightwards across the columns it spans, and treat any column that
resolves to a date as a period column. Both shapes then collapse to the same
thing.

Failures are logged and skipped, never raised: one unreadable sheet must not
abort ingestion of the other seventy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.access.model import Tag
from src.understanding.facts import Fact

log = logging.getLogger(__name__)

# How many leading rows may form the header block. SEC-published sheets use
# two; allowing four tolerates a title row and a blank without over-reaching
# into data.
HEADER_SCAN_ROWS = 4

# Apple's fiscal year ends in late September.
FY_ROLLOVER_MONTH = 10

# Only whole-year and single-quarter columns are ingested. A "9 Months Ended"
# column is cumulative year-to-date; admitting it alongside quarterly figures
# would file two different meanings under one label.
ACCEPTED_DURATIONS = {3, 12}

_DURATION = re.compile(r"(\d+)\s+Months?\s+Ended", re.IGNORECASE)
_DATE = re.compile(r"([A-Z][a-z]{2})\.?\s+(\d{1,2}),\s*(\d{4})")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# Sheets whose name says they break results down by segment or geography carry
# a different sensitivity than the headline statements.
_SEGMENT_HINTS = ("SEGMENT", "GEOGRAPHIC", "DISAGGREGAT")

# XBRL renderings carry structural marker rows that are not data and must not
# become a scope: "Disaggregation of Revenue [Line Items]" sits between the
# category header and its figures.
_XBRL_SCAFFOLD = ("[line items]", "[abstract]", "[member]", "[domain]",
                  "[axis]", "[table]", "[roll forward]")

# A scope applies until a TOTAL closes it out. Without this, every row after
# the last category would inherit that category's name.
#
# Only totals belong here. An earlier version also listed "net sales", which
# made every "Net sales" row close its own scope — so iPhone's net sales and
# the consolidated figure both came out as plain `net_sales` and collided.
_SCOPE_CLOSERS = ("total", "grand total")

# Labels that carry no meaning on their own. A footnote table may hold several
# rows called just "Total" under different sub-headings, so these keep their
# scope as a qualifier even though they also close it.
_GENERIC_LABELS = {"total", "totals", "subtotal", "other", "net"}


@dataclass(frozen=True)
class Period:
    label: str          # "FY2025" or "Q3FY2026"
    fiscal_year: int


def parse_value(raw: object) -> float | None:
    """Coerce a cell to a number, or None if it does not hold one.

    Accounting notation puts negatives in parentheses and prefixes currency, so
    "$ (1,234)" means -1234. A None result is ordinary — section headers and
    spacer rows have no value and are not errors.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    # Strip currency and separators BEFORE testing for parentheses: accounting
    # notation writes negatives as "$ (1,234)", where the sign marker is not at
    # the start of the string.
    cleaned = text.replace("$", "").replace(",", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def parse_period_header(text: object) -> Period | None:
    """Read a column header into a period label, or None if it is not one.

    Returns None for anything lacking a recognisable date, and for durations
    outside ACCEPTED_DURATIONS.
    """
    if text is None:
        return None
    header = " ".join(str(text).split())

    date_match = _DATE.search(header)
    if date_match is None:
        return None

    month = _MONTHS.get(date_match.group(1))
    year = int(date_match.group(3))
    if month is None:
        return None

    fiscal_year = year + 1 if month >= FY_ROLLOVER_MONTH else year

    duration_match = _DURATION.search(header)
    months = int(duration_match.group(1)) if duration_match else 12
    if months not in ACCEPTED_DURATIONS:
        return None

    if months == 12:
        return Period(f"FY{fiscal_year}", fiscal_year)

    # Apple's Q1 is Oct-Dec, so the quarter is offset from the calendar.
    quarter = ((month - FY_ROLLOVER_MONTH) % 12) // 3 + 1
    return Period(f"Q{quarter}FY{fiscal_year}", fiscal_year)


def _period_columns(df: pd.DataFrame) -> tuple[dict[int, Period], int]:
    """Locate the period columns and the row where data begins.

    Durations are carried rightwards across the columns they span, which is
    what reconciles the two-row SEC layout with the one-row rebuilt layout.
    """
    scan = min(HEADER_SCAN_ROWS, len(df))
    durations: dict[int, str] = {}
    dates: dict[int, str] = {}
    last_header_row = -1

    for row in range(scan):
        carried = ""
        for col in range(df.shape[1]):
            cell = " ".join(str(df.iat[row, col]).split())
            if cell in ("nan", ""):
                # Merged cells leave blanks to the right of the label they span.
                pass
            elif _DURATION.search(cell):
                carried = _DURATION.search(cell).group(0)
                durations[col] = carried
            if carried and col not in durations:
                durations[col] = carried
            if cell not in ("nan", "") and _DATE.search(cell):
                dates[col] = cell
                last_header_row = max(last_header_row, row)

    periods: dict[int, Period] = {}
    for col, date_text in dates.items():
        combined = date_text if _DURATION.search(date_text) else \
            f"{durations.get(col, '12 Months Ended')} {date_text}"
        period = parse_period_header(combined)
        if period is not None:
            periods[col] = period

    return periods, last_header_row + 1


# Statement figures are reported in millions, but a workbook mixes units freely:
# earnings per share is dollars, share counts are thousands, rates are percents.
# Assuming millions everywhere renders "$6.08 per share" as "$6 million".
#
# The unit is inferred from the metric name because XBRL renderings encode it
# there — "..._in_dollars_per_share", "..._in_shares", "..._percentage".
_UNIT_HINTS: tuple[tuple[str, str], ...] = (
    ("_in_dollars_per_share", "USD_PER_SHARE"),
    ("per_share", "USD_PER_SHARE"),
    ("_in_shares", "SHARES"),
    ("shares_outstanding", "SHARES"),
    ("percentage", "PERCENT"),
    ("_rate", "PERCENT"),
)


def infer_unit(metric: str) -> str:
    """Unit for a metric, defaulting to millions of USD."""
    for suffix, unit in _UNIT_HINTS:
        if suffix in metric:
            return unit
    return "USD_M"


def _normalise_metric(label: str) -> str:
    """'Net sales' -> 'net_sales'. Stable keys matter more than pretty ones:
    the planner looks metrics up by this name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return slug.strip("_")


def _tag_for_sheet(sheet_name: str) -> Tag:
    upper = sheet_name.upper()
    if any(hint in upper for hint in _SEGMENT_HINTS):
        return Tag.FIN_SEGMENT
    return Tag.FIN_STATEMENTS


def _read_tabular(path: Path) -> dict[str, pd.DataFrame]:
    """Load a workbook or a CSV into the same {sheet_name: frame} shape.

    SEC publishes statement data as .xlsx, so that is what the committed corpus
    contains. CSV is supported because the assignment names it alongside Excel
    and a flat export is the obvious thing a client would hand over — the
    header block, period columns and category scoping all work identically
    once the file is a frame.
    """
    if path.suffix.lower() == ".csv":
        return {path.stem: pd.read_csv(path, header=None, dtype=object)}
    return pd.read_excel(path, sheet_name=None, header=None)


def load_statement_workbook(path: Path) -> list[Fact]:
    """Read every sheet of a statement workbook — .xlsx or .csv — into Facts."""
    try:
        sheets = _read_tabular(path)
    except Exception as exc:  # noqa: BLE001 — an unreadable file is not fatal
        log.warning("could not open %s: %s", path.name, exc)
        return []

    facts: list[Fact] = []
    for sheet_name, df in sheets.items():
        if df.empty or df.shape[1] < 2:
            continue

        periods, first_data_row = _period_columns(df)
        if not periods:
            log.debug("no period columns in %s!%s — skipped", path.name, sheet_name)
            continue

        tag = _tag_for_sheet(sheet_name)
        scope = ""  # current category, e.g. "iphone" or "products"

        for row in range(first_data_row, len(df)):
            label = " ".join(str(df.iat[row, 0]).split())
            if label in ("nan", ""):
                continue
            lowered = label.lower()

            # Structural XBRL rows are neither data nor scope.
            if any(marker in lowered for marker in _XBRL_SCAFFOLD):
                continue

            values = {col: parse_value(df.iat[row, col])
                      for col, period in periods.items()}
            has_values = any(v is not None for v in values.values())

            if not has_values:
                # A labelled row with no figures is a category header. It
                # qualifies everything beneath it until a total closes it.
                scope = _normalise_metric(label.rstrip(":"))
                continue

            base = _normalise_metric(label)
            if not base:
                continue

            closes_scope = any(lowered.startswith(c) for c in _SCOPE_CLOSERS)
            is_generic = base in _GENERIC_LABELS

            # A descriptive total ("Total net sales") is the parent figure, so
            # it stands unqualified and ends the category. A bare "Total" means
            # nothing on its own and keeps the scope as its qualifier —
            # otherwise every sub-table in a footnote produces a metric called
            # "total" and they collide.
            metric = f"{scope}_{base}" if scope and (is_generic or not closes_scope) \
                else base
            if closes_scope and scope:
                scope = ""

            for col, period in periods.items():
                value = values[col]
                if value is None:
                    continue
                facts.append(Fact(
                    metric=metric, value=value, unit=infer_unit(metric),
                    period=period.label, fiscal_year=period.fiscal_year,
                    source=path.name,
                    locator=f"{sheet_name}!R{row + 1}C{col + 1}",
                    tag=tag,
                ))
    return facts


def load_headcount_workbook(path: Path) -> list[Fact]:
    """Read the synthetic departmental headcount file.

    A separate reader rather than a branch inside the statement loader: this is
    a genuinely different shape — a plain Department/FiscalYear/Headcount table
    behind a disclaimer banner — and pretending one parser covers both would
    make the statement loader harder to reason about.
    """
    try:
        # Row 0 is the SYNTHETIC banner; the real header is on row 1.
        df = pd.read_excel(path, sheet_name="Headcount", header=1)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open %s: %s", path.name, exc)
        return []

    facts: list[Fact] = []
    for row_number, row in df.iterrows():
        value = parse_value(row.get("Headcount"))
        fiscal_year = parse_value(row.get("FiscalYear"))
        department = str(row.get("Department", "")).strip()
        if value is None or fiscal_year is None or not department:
            log.warning("unparseable headcount row %s in %s — skipped",
                        row_number, path.name)
            continue
        facts.append(Fact(
            metric=f"headcount_{_normalise_metric(department)}",
            value=value, unit="people",
            period=f"FY{int(fiscal_year)}", fiscal_year=int(fiscal_year),
            source=path.name,
            locator=f"Headcount!R{row_number + 3}",
            tag=Tag.HR_HEADCOUNT,
        ))

    # A company-wide `headcount` total, derived by summing the departments.
    #
    # Without it there is no metric a question like "revenue per employee" can
    # resolve to — only per-department names — so the derivation demo would
    # silently answer with revenue alone. The sum reconciles by construction
    # with the figure each 10-K states on page 8 (161,000 / 164,000 / 166,000),
    # which `scripts/make_synthetic_hr.py` asserts at generation time.
    by_year: dict[int, float] = {}
    for fact in facts:
        by_year[fact.fiscal_year] = by_year.get(fact.fiscal_year, 0.0) + fact.value

    for fiscal_year, total in sorted(by_year.items()):
        facts.append(Fact(
            metric="headcount", value=total, unit="people",
            period=f"FY{fiscal_year}", fiscal_year=fiscal_year,
            source=path.name, locator="Headcount!total",
            tag=Tag.HR_HEADCOUNT,
        ))
    return facts
