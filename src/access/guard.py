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
# "FY2025", "Q3FY2026", and the cumulative "9MFY2025" the corpus does not
# hold. The cumulative form must PARSE so the time window can judge it —
# otherwise it fails closed and reports "outside permitted period" to a role
# that has no time restriction, which misdescribes the refusal. Whether the
# figure exists is a separate question, answered by the facts lookup.
_PERIOD = re.compile(r"^(?:Q[1-4]|[369]M)?FY(\d{4})$", re.IGNORECASE)


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

    intent: str                    # "metric" | "narrative" | "mixed" | "unknown"
    metrics: tuple[str, ...]
    periods: tuple[str, ...]
    tags: tuple[Tag, ...]

    # Words the question used to narrow the request that the chosen metric does
    # NOT account for — asking for "gross margin for products" when only the
    # consolidated gross margin is stored. Carried so the answer can say the
    # qualifier was dropped instead of silently answering a broader question.
    ignored_qualifiers: tuple[str, ...] = ()


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

    # The explanation must match the actual cause. The derivation argument
    # applies to a denied tag combined with a permitted one; attaching it to a
    # date that simply falls outside the role's window would misdescribe the
    # refusal to whoever reads the audit log.
    if denied_tags and len(plan.tags) > len(denied_tags):
        rationale = (" A value derived from restricted data discloses that "
                     "data, so the whole request is refused rather than "
                     "partially answered.")
    elif denied_tags:
        rationale = " This role has no access to that category of data."
    else:
        rationale = " This role may only read the most recent fiscal years."

    return Decision(
        False,
        f"Refused for {gate.role.name} — {'; '.join(causes)}.{rationale}",
        denied_tags=denied_tags,
        denied_periods=tuple(denied_periods),
    )
