"""The hardest clause in the brief:

    "The agent must not leak restricted data to a user who shouldn't see it —
     including when the answer would require combining restricted and
     permitted sources."

A CTO asking for revenue per employee is never shown the headcount. They
recover it by dividing. The guard runs on the *plan*, before any tool executes,
so the restricted operand is never fetched at all.
"""

from pathlib import Path

import pytest

from src.access.gate import AccessGate
from src.access.guard import QueryPlan, guard_plan
from src.access.model import Tag, load_roles

ROLES = load_roles(Path("config/roles.yaml"))
CORPUS_MAX_FY = 2026

# revenue (permitted for CTO) / headcount (denied for CTO) -> discloses headcount
REVENUE_PER_EMPLOYEE = QueryPlan(
    intent="mixed",
    metrics=("net_sales", "headcount"),
    periods=("FY2025",),
    tags=(Tag.FIN_STATEMENTS, Tag.HR_HEADCOUNT),
)


def test_cto_is_refused_revenue_per_employee():
    decision = guard_plan(REVENUE_PER_EMPLOYEE, AccessGate(ROLES["CTO"], CORPUS_MAX_FY))

    assert not decision.allowed
    assert Tag.HR_HEADCOUNT in decision.denied_tags
    # The permitted half must not appear as a denial — the refusal has to name
    # the actual cause, or the audit log misleads whoever reads it.
    assert Tag.FIN_STATEMENTS not in decision.denied_tags
    assert "derived" in decision.reason.lower()


def test_ceo_gets_the_same_plan_executed():
    assert guard_plan(REVENUE_PER_EMPLOYEE,
                      AccessGate(ROLES["CEO"], CORPUS_MAX_FY)).allowed


def test_period_only_refusal_does_not_invoke_the_derivation_argument():
    """A refusal must explain its actual cause. A date outside the role's
    window has nothing to do with deriving a restricted value, and saying so
    would misdescribe the refusal to whoever reads the audit log."""
    plan = QueryPlan(intent="metric", metrics=("net_sales",), periods=("FY2023",),
                     tags=(Tag.FIN_STATEMENTS,))
    decision = guard_plan(plan, AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY))

    assert not decision.allowed
    assert "derived" not in decision.reason.lower()
    assert "most recent fiscal years" in decision.reason


def test_wholly_restricted_plan_is_not_described_as_derivation():
    """Nothing is being combined when every tag is denied, so the refusal
    should say the category is off-limits rather than invoke arithmetic."""
    plan = QueryPlan(intent="metric", metrics=("headcount",), periods=("FY2025",),
                     tags=(Tag.HR_HEADCOUNT,))
    decision = guard_plan(plan, AccessGate(ROLES["CTO"], CORPUS_MAX_FY))

    assert not decision.allowed
    assert "derived" not in decision.reason.lower()
    assert "no access to that category" in decision.reason


def test_analyst_refusal_names_both_causes():
    """ANALYST is restricted on tag and on time, so a plan violating both must
    report both rather than stopping at the first failure."""
    plan = QueryPlan(intent="narrative", metrics=(), periods=("FY2022",),
                     tags=(Tag.NARR_RISK,))
    decision = guard_plan(plan, AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY))

    assert not decision.allowed
    assert Tag.NARR_RISK in decision.denied_tags
    assert "FY2022" in decision.denied_periods


def test_wholly_permitted_plan_passes():
    plan = QueryPlan(intent="metric", metrics=("net_sales",), periods=("FY2026",),
                     tags=(Tag.FIN_STATEMENTS,))
    assert guard_plan(plan, AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY)).allowed


@pytest.mark.parametrize("period,expected_fy", [
    ("FY2025", 2025), ("Q3FY2026", 2026), ("FY2023", 2023),
])
def test_period_labels_parse_to_fiscal_years(period, expected_fy):
    """Quarterly labels carry a prefix, so the fiscal year cannot be read off
    the last four characters without checking the format."""
    from src.access.guard import fiscal_year_of_period
    assert fiscal_year_of_period(period) == expected_fy


def test_unparseable_period_is_refused_not_ignored():
    """An unrecognised period label must fail closed. Skipping it would let a
    malformed plan slip past the time window entirely."""
    plan = QueryPlan(intent="metric", metrics=("net_sales",),
                     periods=("last quarter",), tags=(Tag.FIN_STATEMENTS,))
    decision = guard_plan(plan, AccessGate(ROLES["ANALYST"], CORPUS_MAX_FY))

    assert not decision.allowed
    assert "last quarter" in decision.denied_periods
