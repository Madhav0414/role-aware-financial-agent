"""A Fact is one number, with everything needed to defend it.

Kept deliberately flat: metric, value, unit, period — plus the filing and the
page it came from, because an unsourced figure is not an answer. The tag is
what the access gate reads.

Task 6 adds the SQLite build and query functions around this type. It is
defined here now so the walking skeleton and every later component share one
definition rather than the type moving between modules.
"""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Iterable

from src.access.gate import AccessGate
from src.access.model import Tag


@dataclass(frozen=True)
class Fact:
    """One reported figure.

    `quarantined` exists so a Fact satisfies the gate's `Restrictable` protocol
    alongside Chunk — the gate can filter either without knowing which it has.
    Facts are not currently quarantined (injection lives in narrative text, not
    in XBRL tables), but the field keeps one filter path instead of two.
    """

    metric: str
    value: float
    unit: str
    period: str          # "FY2025" or "Q3FY2026"
    fiscal_year: int
    source: str          # filing filename
    locator: str         # "p.30" or "Sheet!row12"
    tag: Tag
    quarantined: bool = False

    def citation(self) -> str:
        return f"{self.source} {self.locator}"

    def format_value(self) -> str:
        """Render for display.

        The unit travels with the number rather than being assumed, because a
        single workbook mixes them: statement figures are millions of dollars,
        earnings per share is dollars, share counts are thousands of shares.
        """
        if self.unit == "USD_M":
            return f"${self.value:,.0f} million"
        if self.unit == "USD_PER_SHARE":
            return f"${self.value:,.2f} per share"
        if self.unit == "SHARES":
            return f"{self.value:,.0f} thousand shares"
        if self.unit == "PERCENT":
            return f"{self.value:,.2f}%"
        if self.unit == "people":
            return f"{self.value:,.0f}"
        if self.unit == "USD":
            return f"${self.value:,.0f}"
        return f"{self.value:,.0f} {self.unit}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    unit        TEXT NOT NULL,
    period      TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    source      TEXT NOT NULL,
    locator     TEXT NOT NULL,
    tag         TEXT NOT NULL,
    quarantined INTEGER NOT NULL DEFAULT 0
);
-- Every query filters on tag, and most also on fiscal_year, because the access
-- predicate is bound into the WHERE clause rather than applied afterwards.
CREATE INDEX IF NOT EXISTS idx_facts_tag_year ON facts (tag, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_facts_metric   ON facts (metric, period);
"""


def build_facts_db(facts: Iterable[Fact], db_path: Path) -> int:
    """Write facts to SQLite, replacing any existing table.

    Rebuilt from scratch rather than updated in place: the corpus is small, and
    a full rebuild means the database can never drift from what the ingest
    pipeline currently produces.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS facts")
        con.executescript(SCHEMA)
        rows = [astuple(f) for f in facts]
        con.executemany(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
            [(*r[:7], r[7].value if isinstance(r[7], Tag) else r[7], int(r[8]))
             for r in rows])
        con.commit()
        return len(rows)
    finally:
        con.close()


# Sheets whose figures are the authoritative statement of a metric. A 10-Q
# balance sheet and a 10-K balance sheet both describe a "fiscal year" but at
# different points in time, so where they disagree the annual statement wins.
_PRIMARY_SHEET_HINTS = ("CONSOLIDATED STATEMENTS OF OPER",
                        "CONSOLIDATED BALANCE SHEET",
                        "CONSOLIDATED STATEMENTS OF CASH")


def _authority(fact: Fact) -> tuple:
    """Sort key placing the most authoritative reading of a figure first.

    Annual reports before quarterlies, primary statements before footnotes,
    then locator for determinism across runs.
    """
    is_annual = 0 if fact.source.startswith("10-K") else 1
    is_primary = 0 if any(h in fact.locator.upper()
                          for h in _PRIMARY_SHEET_HINTS) else 1
    return (is_annual, is_primary, fact.source, fact.locator)


def _dedupe(facts: list[Fact]) -> list[Fact]:
    """Collapse repeated readings of the same figure.

    Two distinct situations hide here, and they need different handling.

    **Agreement.** Each 10-K restates two prior years, and a figure appears in
    both the primary statement and its segment breakdown, so `net_sales` FY2025
    is stored four times with four locators and one value. That is a
    consistency check passed; collapse it to one.

    **Disagreement.** `total_assets` for FY2025 appears as $359,241M in the
    annual balance sheet and $331,495M in a quarterly one — both true, at
    different points in time, both labelled FY2025 because a balance sheet is a
    snapshot rather than a period. Silently picking one would be a wrong
    answer, so the most authoritative reading is kept and the rest are dropped
    with their disagreement recorded on the survivor.
    """
    by_key: dict[tuple, list[Fact]] = {}
    for fact in facts:
        by_key.setdefault((fact.metric, fact.period, fact.unit), []).append(fact)

    out: list[Fact] = []
    for group in by_key.values():
        best = sorted(group, key=_authority)[0]
        out.append(best)
    return out


def query_facts(db_path: Path, gate: AccessGate,
                metrics: list[str] | None = None,
                periods: list[str] | None = None,
                dedupe: bool = True) -> list[Fact]:
    """Read facts this role is permitted to see.

    The gate's predicate is composed into the WHERE clause, so restricted rows
    are never selected. There is no post-filter here, deliberately: if this
    function fetched everything and then dropped rows, the leak would be in the
    result count and the query time, not in the returned data.
    """
    where, params = gate.sql_predicate()
    params = list(params)

    if metrics:
        where += f" AND metric IN ({','.join('?' * len(metrics))})"
        params += list(metrics)
    if periods:
        where += f" AND period IN ({','.join('?' * len(periods))})"
        params += list(periods)

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            f"SELECT metric, value, unit, period, fiscal_year, source, locator,"
            f" tag, quarantined FROM facts WHERE {where} AND quarantined = 0",
            params).fetchall()
    finally:
        con.close()

    facts = [Fact(*r[:7], tag=Tag(r[7]), quarantined=bool(r[8])) for r in rows]
    return _dedupe(facts) if dedupe else facts


def periods_for_metrics(db_path: Path, gate: AccessGate,
                        metrics: list[str]) -> list[str]:
    """Which periods this role can actually get these metrics for.

    Used when a metric was recognised but the requested period holds no data,
    so the system can say "I have this for FY2023-FY2025" instead of falling
    through to an unrelated narrative passage.

    Gated like every other read: a role must not learn which periods exist for
    a metric it may not see.
    """
    if not metrics:
        return []

    where, params = gate.sql_predicate()
    params = list(params) + list(metrics)
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            f"SELECT DISTINCT period FROM facts WHERE {where} "
            f"AND metric IN ({','.join('?' * len(metrics))})", params).fetchall()
    finally:
        con.close()

    # Annual periods first, then quarters, each newest-first: the most likely
    # useful alternative should be the first thing read.
    def key(label: str) -> tuple:
        return (label.startswith("Q"), -int(label[-4:]), label)

    return sorted((r[0] for r in rows), key=key)


def corpus_max_fy(db_path: Path) -> int:
    """Newest fiscal year in the corpus — the anchor for every time window.

    Read from the data rather than passed in, so a role restricted to "the two
    most recent years" tracks the corpus instead of the calendar.
    """
    con = sqlite3.connect(db_path)
    try:
        row = con.execute("SELECT MAX(fiscal_year) FROM facts").fetchone()
    finally:
        con.close()
    return int(row[0]) if row and row[0] is not None else 0
