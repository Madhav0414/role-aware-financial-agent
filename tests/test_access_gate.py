"""Tests for the single read choke point.

The gate is deliberately ignorant of what a chunk or a fact *is* — it only
requires `.tag`, `.fiscal_year` and `.quarantined`. These tests use a minimal
stub to prove that, which is why the access layer needs no import from the
ingest layer.
"""

from dataclasses import dataclass
from pathlib import Path

from src.access.gate import AccessGate
from src.access.model import Tag, load_roles

ROLES = load_roles(Path("config/roles.yaml"))
CORPUS_MAX_FY = 2026  # newest fiscal year in the shipped corpus


@dataclass(frozen=True)
class StubRecord:
    tag: Tag
    fiscal_year: int
    quarantined: bool = False


def test_cto_may_read_financials_but_not_compensation():
    gate = AccessGate(ROLES["CTO"], CORPUS_MAX_FY)
    assert gate.check_tag(Tag.FIN_STATEMENTS).allowed

    denial = gate.check_tag(Tag.HR_COMPENSATION)
    assert not denial.allowed
    assert denial.denied_tags == (Tag.HR_COMPENSATION,)
    assert "CTO" in denial.reason


def test_analyst_time_window_is_relative_to_the_corpus():
    """Two most recent fiscal years of a corpus ending FY2026 means 2025-2026.
    Anchored to the corpus, not to today, so the tests do not rot."""
    gate = AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY)
    assert gate.min_permitted_fy() == 2025
    assert gate.check_period(2026).allowed
    assert gate.check_period(2025).allowed
    assert not gate.check_period(2024).allowed
    assert gate.check_period(2024).denied_periods == ("FY2024",)


def test_unrestricted_roles_have_no_time_floor():
    for name in ("CEO", "CTO"):
        gate = AccessGate(ROLES[name], CORPUS_MAX_FY)
        assert gate.min_permitted_fy() is None
        assert gate.check_period(2023).allowed


def test_sql_predicate_binds_only_permitted_tags():
    """The filter must travel into the database, not run in Python afterwards —
    so restricted rows are never selected in the first place."""
    sql, params = AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY).sql_predicate()

    assert "tag IN (?,?)" in sql
    assert "fiscal_year >= ?" in sql
    assert set(params) == {"financials.statements", "financials.segment", 2025}
    assert Tag.HR_COMPENSATION.value not in params


def test_sql_predicate_omits_the_year_clause_when_unrestricted():
    sql, params = AccessGate(ROLES["CEO"], CORPUS_MAX_FY).sql_predicate()
    assert "fiscal_year" not in sql
    assert len(params) == len(Tag)


def test_filter_chunks_drops_restricted_stale_and_quarantined():
    gate = AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY)
    records = [
        StubRecord(Tag.FIN_STATEMENTS, 2026),                    # keep
        StubRecord(Tag.HR_COMPENSATION, 2026),                   # wrong tag
        StubRecord(Tag.FIN_STATEMENTS, 2023),                    # too old
        StubRecord(Tag.FIN_STATEMENTS, 2026, quarantined=True),  # poisoned
    ]
    kept = gate.filter_chunks(records)
    assert kept == [records[0]]


def test_ceo_filter_keeps_everything_except_quarantine():
    """Quarantine is not an access rule — a document carrying an injection
    attempt is withheld from every role, including the one that may read all."""
    gate = AccessGate(ROLES["CEO"], CORPUS_MAX_FY)
    records = [
        StubRecord(Tag.HR_COMPENSATION, 2023),
        StubRecord(Tag.FIN_STATEMENTS, 2026, quarantined=True),
    ]
    assert gate.filter_chunks(records) == [records[0]]
