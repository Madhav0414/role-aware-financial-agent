"""The answering path: question in, guarded answer out.

WALKING SKELETON. Three stubs remain, each replaced in a later task:

  SKELETON_FACTS   -> facts.db, built from the real filings   (Task 6)
  plan_query()     -> the real planner with a metric alias map (Task 9)
  _compose()       -> the LLM adapter, with this as the keyless fallback (Task 8)

The order of operations below is NOT a stub. It is the architecture, and it
does not change when the stubs are replaced:

    plan -> guard -> refuse-or-fetch -> compose -> audit

The guard runs before anything is fetched. That is what makes "restricted data
never reaches the model" true rather than merely intended.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.access.audit import audit
from src.access.gate import AccessGate
from src.access.guard import QueryPlan, guard_plan
from src.access.model import Tag
from src.understanding.facts import Fact

# Real figures from the committed corpus, so the skeleton answers truthfully
# even before ingestion exists. Replaced wholesale by facts.db in Task 6.
SKELETON_FACTS: list[Fact] = [
    Fact("net_sales", 416_161, "USD_M", "FY2025", 2025,
         "10-K_FY2025.pdf", "p.30", Tag.FIN_STATEMENTS),
    Fact("net_sales", 391_035, "USD_M", "FY2024", 2024,
         "10-K_FY2024.pdf", "p.29", Tag.FIN_STATEMENTS),
    Fact("net_sales", 383_285, "USD_M", "FY2023", 2023,
         "10-K_FY2023.pdf", "p.28", Tag.FIN_STATEMENTS),
    Fact("headcount", 166_000, "people", "FY2025", 2025,
         "10-K_FY2025.pdf", "p.8", Tag.HR_HEADCOUNT),
    Fact("headcount", 164_000, "people", "FY2024", 2024,
         "10-K_FY2024.pdf", "p.8", Tag.HR_HEADCOUNT),
]

# Which tag governs which metric. The planner needs this to declare a plan's
# tags before any data is touched — the guard cannot check what was not
# declared, so the mapping lives in code the model cannot influence.
METRIC_TAGS: dict[str, Tag] = {
    "net_sales": Tag.FIN_STATEMENTS,
    "headcount": Tag.HR_HEADCOUNT,
}

_PERIOD_IN_QUESTION = re.compile(r"\bF?Y?(20\d{2})\b")
_REVENUE_WORDS = ("revenue", "net sales", "sales", "turnover")
_HEADCOUNT_WORDS = ("employee", "headcount", "staff", "workforce")


def plan_query(question: str, default_fy: int) -> QueryPlan:
    """Turn a question into a declaration of what answering it would require.

    Keyword matching in the skeleton. What matters is that the plan is built
    *before* any fetch, and that its tags come from METRIC_TAGS rather than
    from the question — a plan that under-declares its tags would slip past the
    guard, so the mapping must not be attacker-influenced.
    """
    text = question.lower()

    wants_revenue = any(w in text for w in _REVENUE_WORDS)
    wants_headcount = any(w in text for w in _HEADCOUNT_WORDS)

    metrics: tuple[str, ...] = ()
    if wants_revenue and wants_headcount:
        metrics = ("net_sales", "headcount")   # e.g. revenue per employee
    elif wants_revenue:
        metrics = ("net_sales",)
    elif wants_headcount:
        metrics = ("headcount",)

    years = _PERIOD_IN_QUESTION.findall(question)
    periods = tuple(f"FY{y}" for y in years) or (f"FY{default_fy}",)

    intent = "mixed" if len(metrics) > 1 else "metric" if metrics else "narrative"
    tags = tuple(dict.fromkeys(METRIC_TAGS[m] for m in metrics))

    return QueryPlan(intent=intent, metrics=metrics, periods=periods, tags=tags)


def _compose(facts: list[Fact], plan: QueryPlan) -> str:
    """Word the answer deterministically.

    In Task 8 the LLM phrases this instead — but only phrases it. The numbers
    are computed here either way, so a model outage degrades the prose and
    never the arithmetic.
    """
    if not facts:
        return "No matching figures were found in the corpus."

    by_metric = {f.metric: f for f in facts}
    period = plan.periods[0]

    if plan.intent == "mixed" and {"net_sales", "headcount"} <= by_metric.keys():
        revenue, people = by_metric["net_sales"], by_metric["headcount"]
        per_employee = revenue.value / people.value
        return (
            f"For {period}, net sales were {revenue.format_value()} and "
            f"headcount was {people.format_value()}, giving revenue per "
            f"employee of ${per_employee:,.2f} million."
        )

    # Label-and-value rather than a sentence: "net sales" is plural and
    # "headcount" is singular, and no single verb agrees with both.
    return " ".join(
        f"{f.metric.replace('_', ' ').capitalize()} for {f.period}: "
        f"{f.format_value()}."
        for f in facts
    )


def answer(question: str, gate: AccessGate,
           audit_path: Path | None = None) -> dict:
    """Answer a question under one role's permissions.

    Returns rather than raises on refusal: the caller must be able to tell
    "you may not see this" from "there is no such data".
    """
    plan = plan_query(question, default_fy=gate.corpus_max_fy)
    decision = guard_plan(plan, gate)

    # Logged before the branch, so a refusal is recorded exactly like an allow.
    audit(decision, gate.role.name,
          context=f"plan[{plan.intent}] metrics={','.join(plan.metrics) or '-'} "
                  f"periods={','.join(plan.periods)}",
          path=audit_path)

    if not decision.allowed:
        return {
            "allowed": False,
            "reason": decision.reason,
            "answer": f"Request refused. {decision.reason}",
            "citations": [],
        }

    # Only now is anything read — and it is read through the gate, so the
    # permission check applies even if the plan were somehow wrong.
    matching = [f for f in SKELETON_FACTS
                if f.metric in plan.metrics and f.period in plan.periods]
    permitted = gate.filter_chunks(matching)

    return {
        "allowed": True,
        "reason": decision.reason,
        "answer": _compose(permitted, plan),
        "citations": [f.citation() for f in permitted],
    }
