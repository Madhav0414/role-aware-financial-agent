"""A Fact is one number, with everything needed to defend it.

Kept deliberately flat: metric, value, unit, period — plus the filing and the
page it came from, because an unsourced figure is not an answer. The tag is
what the access gate reads.

Task 6 adds the SQLite build and query functions around this type. It is
defined here now so the walking skeleton and every later component share one
definition rather than the type moving between modules.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        return f"{self.value:,.0f} {self.unit}"
