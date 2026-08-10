"""The single read choke point.

Every read of every fact and every chunk passes through an AccessGate. There is
no second path to the data — that is the whole design, and it is what makes the
claim "restricted data never reaches the model" testable rather than aspirational.

Two enforcement surfaces, because there are two stores:

  SQL path     sql_predicate() returns a WHERE fragment bound into the query,
               so the database never returns a restricted row.
  Index path   filter_chunks() runs before BM25 scores anything, so a
               restricted chunk never competes for a slot in the results.

Both filter *before* retrieval. Filtering afterwards would still leak through
result counts and response latency.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from src.access.model import Decision, Role, Tag


@runtime_checkable
class Restrictable(Protocol):
    """What the gate needs in order to judge a record.

    Deliberately narrow: the access layer does not import Chunk or Fact, so it
    never depends on the ingest or understanding layers. Anything carrying
    these three attributes can be filtered.
    """

    tag: Tag
    fiscal_year: int
    quarantined: bool


R = TypeVar("R", bound=Restrictable)


class AccessGate:
    """Resolves one role's permissions against one corpus.

    `corpus_max_fy` is the newest fiscal year present in the ingested data. Time
    windows are measured from it rather than from today's date, so behaviour is
    a function of the corpus and the tests do not rot as the calendar moves.
    """

    def __init__(self, role: Role, corpus_max_fy: int) -> None:
        self.role = role
        self.corpus_max_fy = corpus_max_fy

    # -- period logic ------------------------------------------------------

    def min_permitted_fy(self) -> int | None:
        """Oldest fiscal year this role may read, or None if unrestricted.

        A role limited to the 2 most recent years of a corpus ending FY2026 may
        read FY2025 onward: 2026 - 2 + 1.
        """
        if self.role.recent_years_only is None:
            return None
        return self.corpus_max_fy - self.role.recent_years_only + 1

    def check_period(self, fiscal_year: int) -> Decision:
        floor = self.min_permitted_fy()
        if floor is None or fiscal_year >= floor:
            return Decision(True, f"FY{fiscal_year} is within {self.role.name}'s window")
        return Decision(
            False,
            f"{self.role.name} may only read FY{floor} onward",
            denied_periods=(f"FY{fiscal_year}",),
        )

    # -- tag logic ---------------------------------------------------------

    def check_tag(self, tag: Tag) -> Decision:
        if tag in self.role.allowed_tags:
            return Decision(True, f"{self.role.name} may read {tag.value}")
        return Decision(
            False,
            f"{self.role.name} is not permitted to read {tag.value}",
            denied_tags=(tag,),
        )

    # -- enforcement surfaces ---------------------------------------------

    def sql_predicate(self) -> tuple[str, list[str | int]]:
        """A parameterised WHERE fragment and its bound values.

        Returned as a fragment rather than a finished query so the caller can
        add its own conditions, and parameterised rather than formatted so a
        tag value can never carry SQL.
        """
        tags = sorted(t.value for t in self.role.allowed_tags)
        sql = f"tag IN ({','.join('?' * len(tags))})"
        params: list[str | int] = list(tags)

        floor = self.min_permitted_fy()
        if floor is not None:
            sql += " AND fiscal_year >= ?"
            params.append(floor)
        return sql, params

    def filter_chunks(self, records: list[R]) -> list[R]:
        """Drop everything this role may not see, before scoring.

        Quarantine is checked first and applies to every role including CEO: a
        document carrying an injection attempt is unsafe for everyone, so it is
        a safety rule rather than a permission.
        """
        return [
            r for r in records
            if not r.quarantined
            and self.check_tag(r.tag).allowed
            and self.check_period(r.fiscal_year).allowed
        ]
