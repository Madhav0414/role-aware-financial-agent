"""End-to-end tests for the thinnest complete path through the system.

These run against hardcoded facts and a keyword planner. Every stub behind them
is replaced by a real component in later tasks — the ingest pipeline, the facts
database, the BM25 index, the LLM adapter — but `answer()` keeps this exact
signature and these tests keep passing throughout.

That is the point of the skeleton: from here on, the system always runs.
"""

from pathlib import Path

from src.access.gate import AccessGate
from src.access.model import load_roles
from src.agent.loop import answer

ROLES = load_roles(Path("config/roles.yaml"))
CORPUS_MAX_FY = 2026


def gate_for(role: str) -> AccessGate:
    return AccessGate(ROLES[role], CORPUS_MAX_FY)


def test_ceo_gets_a_figure_with_a_citation():
    result = answer("What was net sales in FY2025?", gate_for("CEO"))

    assert result["allowed"] is True
    assert "416,161" in result["answer"]
    # An unsourced number is not an answer — every figure names its filing
    # and the page it came from.
    assert result["citations"]
    assert "10-K_FY2025.pdf" in result["citations"][0]


def test_cto_is_refused_revenue_per_employee():
    """The derivation leak, end to end: revenue is permitted, headcount is not,
    and the ratio would disclose headcount without printing it."""
    result = answer("What is revenue per employee for FY2025?", gate_for("CTO"))

    assert result["allowed"] is False
    assert "hr.headcount" in result["reason"]
    assert result["citations"] == []
    # The refusal must not smuggle the restricted figure into its explanation.
    assert "166,000" not in result["answer"]
    assert "166000" not in result["answer"]


def test_ceo_may_ask_the_same_question():
    result = answer("What is revenue per employee for FY2025?", gate_for("CEO"))
    assert result["allowed"] is True
    assert "166,000" in result["answer"]


def test_analyst_is_refused_an_out_of_window_year():
    result = answer("What was net sales in FY2023?", gate_for("ANALYST"))

    assert result["allowed"] is False
    assert "FY2023" in result["reason"]


def test_analyst_may_read_recent_financials():
    result = answer("What was net sales in FY2025?", gate_for("ANALYST"))
    assert result["allowed"] is True
    assert "416,161" in result["answer"]


def test_every_request_is_audited(tmp_path):
    """Allows and denials both. A log holding only refusals cannot answer
    'what did this role actually read', which is the first audit question."""
    from src.access import audit as audit_module

    log = tmp_path / "audit.log"
    answer("What was net sales in FY2025?", gate_for("CEO"), audit_path=log)
    answer("What is revenue per employee for FY2025?", gate_for("CTO"),
           audit_path=log)

    entries = audit_module.read_audit(log)
    assert len(entries) == 2
    assert [e["allowed"] for e in entries] == [True, False]
    assert entries[1]["denied_tags"] == ["hr.headcount"]
    # The log records the decision, never the content it protects.
    assert "166,000" not in log.read_text(encoding="utf-8")
