"""The derivation guard.

Tag filtering alone is not enough. A user denied one operand can still recover
it from a permitted one plus a derived result:

    revenue per employee  =  revenue (permitted)  /  headcount (restricted)

Nothing prints the headcount, and the headcount is disclosed anyway. Hiding a
value while publishing a function of it is not hiding it.

So the agent must declare what it intends to touch *before* it touches
anything. `guard_plan` validates that declaration in ordinary Python. The model
writes the plan; it does not get a vote on the verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.access.gate import AccessGate
from src.access.model import Decision, Tag

# "FY2025" or "Q3FY2026" — a quarterly label carries a prefix, so the fiscal
# year cannot simply be read off the last four characters.
_PERIOD = re.compile(r"^(?:Q[1-4])?FY(\d{4})$", re.IGNORECASE)


class UnparseablePeriod(ValueError):
    """Raised for a period label the guard does not recognise."""


def fiscal_year_of_period(period: str) -> int:
    match = _PERIOD.match(period.strip())
    if match is None:
        raise UnparseablePeriod(period)
    return int(match.group(1))


@dataclass(frozen=True)
class QueryPlan:
    """What an answer would require, declared before anything is fetched.

    Frozen so that nothing can edit the plan between the guard approving it and
    the tools executing it — an approved plan and an executed plan must be the
    same object.
    """

    intent: str                    # "metric" | "narrative" | "mixed"
    metrics: tuple[str, ...]
    periods: tuple[str, ...]
    tags: tuple[Tag, ...]


def guard_plan(plan: QueryPlan, gate: AccessGate) -> Decision:
    """Approve or refuse a plan in full.

    Refusal is all-or-nothing. Executing the permitted half of a mixed plan and
    withholding the rest is what produces the leak: the user gets the ratio and
    reverses it. The whole plan fails.

    Every violation is collected rather than returning on the first one, so a
    role restricted on two dimensions gets told about both. A refusal that
    reports one cause invites a retry that trips over the next.
    """
    denied_tags = tuple(t for t in plan.tags if not gate.check_tag(t).allowed)

    denied_periods: list[str] = []
    for period in plan.periods:
        try:
            fiscal_year = fiscal_year_of_period(period)
        except UnparseablePeriod:
            # Fail closed. Skipping an unrecognised label would let a malformed
            # plan bypass the time window entirely.
            denied_periods.append(period)
            continue
        if not gate.check_period(fiscal_year).allowed:
            denied_periods.append(period)

    if not denied_tags and not denied_periods:
        return Decision(True, f"Plan permitted for {gate.role.name}")

    causes = []
    if denied_tags:
        causes.append("restricted data: " + ", ".join(t.value for t in denied_tags))
    if denied_periods:
        causes.append("outside permitted period: " + ", ".join(denied_periods))

    return Decision(
        False,
        f"Refused for {gate.role.name} — {'; '.join(causes)}. "
        "A value derived from restricted data discloses that data, so the "
        "whole request is refused rather than partially answered.",
        denied_tags=denied_tags,
        denied_periods=tuple(denied_periods),
    )
