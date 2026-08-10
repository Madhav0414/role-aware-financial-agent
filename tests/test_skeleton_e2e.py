"""End-to-end tests for the thinnest complete path through the system.

These run against hardcoded facts and a keyword planner. Every stub behind them
is replaced by a real component in later tasks — the ingest pipeline, the facts
database, the BM25 index, the LLM adapter — but `answer()` keeps this exact
signature and these tests keep passing throughout.

That is the point of the skeleton: from here on, the system always runs.
"""

from pathlib import Path

import pytest

from src.access.gate import AccessGate
from src.access.model import load_roles
from src.agent.loop import answer

ROLES = load_roles(Path("config/roles.yaml"))
CORPUS_MAX_FY = 2026

# use_llm=False throughout: these assert the DETERMINISTIC guarantee, which is
# what has to hold when no key is configured. Model phrasing is exercised
# separately and must never be what makes an answer correct.
pytestmark = pytest.mark.skipif(
    not Path("data/understanding/metric_tags.json").exists(),
    reason="run scripts/build_understanding.py first")


def gate_for(role: str) -> AccessGate:
    return AccessGate(ROLES[role], CORPUS_MAX_FY)


def ask(question: str, role: str, **kwargs) -> dict:
    return answer(question, gate_for(role), use_llm=False, **kwargs)


def test_ceo_gets_a_figure_with_a_citation():
    result = ask("What was net sales in FY2025?", "CEO")

    assert result["allowed"] is True
    assert "416,161" in result["answer"]
    # An unsourced number is not an answer — every figure names its filing
    # and the page it came from.
    assert result["citations"]
    assert any("FY2025" in c for c in result["citations"])


def test_cto_is_refused_revenue_per_employee():
    """The derivation leak, end to end: revenue is permitted, headcount is not,
    and the ratio would disclose headcount without printing it."""
    result = ask("What is revenue per employee for FY2025?", "CTO")

    assert result["allowed"] is False
    assert "hr.headcount" in result["reason"]
    assert result["citations"] == []
    # The refusal must not smuggle the restricted figure into its explanation.
    assert "166,000" not in result["answer"]
    assert "166000" not in result["answer"]


def test_ceo_may_ask_the_same_question():
    """The same question, the same system, a different role — and now an
    answer. That contrast is the whole demonstration."""
    result = ask("What is revenue per employee for FY2025?", "CEO")
    assert result["allowed"] is True
    assert "166,000" in result["answer"]
    assert "416,161" in result["answer"]


def test_analyst_is_refused_an_out_of_window_year():
    result = ask("What was net sales in FY2023?", "ANALYST")

    assert result["allowed"] is False
    assert "FY2023" in result["reason"]


def test_analyst_may_read_recent_financials():
    result = ask("What was net sales in FY2025?", "ANALYST")
    assert result["allowed"] is True
    assert "416,161" in result["answer"]


def test_every_request_is_audited(tmp_path):
    """Allows and denials both. A log holding only refusals cannot answer
    'what did this role actually read', which is the first audit question.

    The plan decision is logged, and so is every tool call it authorised, so a
    permitted request produces more than one line.
    """
    from src.access import audit as audit_module

    log = tmp_path / "audit.log"
    ask("What was net sales in FY2025?", "CEO", audit_path=log)
    ask("What is revenue per employee for FY2025?", "CTO", audit_path=log)

    entries = audit_module.read_audit(log)
    assert len(entries) >= 2

    plan_entries = [e for e in entries if e["context"].startswith("plan[")]
    assert [e["allowed"] for e in plan_entries] == [True, False]
    assert plan_entries[1]["denied_tags"] == ["hr.headcount"]

    # The log records the decision, never the content it protects.
    text = log.read_text(encoding="utf-8")
    assert "166,000" not in text
    assert "416,161" not in text
