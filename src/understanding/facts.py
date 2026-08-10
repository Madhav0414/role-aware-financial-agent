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
        """Render for display. Financial statements are reported in millions,
        so the unit travels with the number rather than being assumed."""
        if self.unit == "USD_M":
            return f"${self.value:,.0f} million"
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


def _dedupe(facts: list[Fact]) -> list[Fact]:
    """Collapse the same figure reported in several places.

    Each 10-K restates the two prior years, and a figure appears in both the
    primary statement and its segment breakdown, so `net_sales` for FY2025 is
    stored four times with four different locators. They agree — that is a
    consistency check passed, not a conflict — but an answer should cite one
    source, not repeat itself four times. The earliest-sorted locator wins so
    the choice is deterministic across runs.
    """
    seen: dict[tuple, Fact] = {}
    for fact in sorted(facts, key=lambda f: (f.source, f.locator)):
        seen.setdefault((fact.metric, fact.period, fact.value, fact.unit), fact)
    return list(seen.values())


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
