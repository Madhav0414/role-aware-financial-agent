"""The HTTP layer.

The API holds no access logic of its own — it builds a gate from the asserted
role and calls the same `answer()` the CLI does. These tests exist to prove
that no enforcement was accidentally re-implemented (or lost) at the boundary.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not Path("data/understanding/facts.db").exists(),
    reason="run scripts/build_understanding.py first")

client = TestClient(app)


def test_context_exposes_each_role_with_its_permissions():
    body = client.get("/api/context").json()
    names = {r["name"] for r in body["roles"]}
    assert names == {"CEO", "CTO", "ANALYST"}

    cto = next(r for r in body["roles"] if r["name"] == "CTO")
    assert set(cto["denied_tags"]) == {"hr.headcount", "hr.compensation"}

    analyst = next(r for r in body["roles"] if r["name"] == "ANALYST")
    assert analyst["min_fiscal_year"] == body["corpus_max_fy"] - 1


def test_the_api_enforces_the_same_rules_as_the_cli():
    refused = client.post("/api/ask", json={
        "role": "CTO", "question": "What is revenue per employee for FY2025?",
        "use_llm": False}).json()

    assert refused["allowed"] is False
    assert "hr.headcount" in refused["denied_tags"]
    assert "166,000" not in refused["answer"]


def test_a_refusal_reports_that_retrieval_never_ran():
    """The console draws its pipeline from these flags, and the claim it makes
    visually — that nothing was fetched — has to be true."""
    body = client.post("/api/ask", json={
        "role": "CTO", "question": "What is executive compensation?",
        "use_llm": False}).json()

    assert body["stages"]["guard"] is True
    assert body["stages"]["retrieve"] is False
    assert body["stages"]["compose"] is False


def test_a_permitted_question_returns_cited_evidence():
    body = client.post("/api/ask", json={
        "role": "CEO", "question": "What was net sales in FY2025?",
        "use_llm": False}).json()

    assert body["allowed"] is True
    assert "416,161" in body["answer"]
    assert body["citations"]
    assert body["stages"]["retrieve"] is True


def test_an_unknown_role_is_rejected():
    assert client.post("/api/ask", json={
        "role": "ADMIN", "question": "anything", "use_llm": False
    }).status_code == 400


def test_role_cannot_be_smuggled_through_casing():
    """Role lookup is exact. A near-miss must fail closed rather than fall
    back to a default."""
    assert client.post("/api/ask", json={
        "role": "ceo", "question": "net sales", "use_llm": False
    }).status_code == 400


def test_audit_endpoint_returns_decisions_without_document_text():
    client.post("/api/ask", json={"role": "CEO",
                                  "question": "What was net sales in FY2025?",
                                  "use_llm": False})
    body = client.get("/api/audit?limit=10").json()

    assert body["rows"]
    for row in body["rows"]:
        assert {"ts", "role", "context", "allowed", "reason"} <= set(row)
        assert "416,161" not in row["reason"]


def test_feedback_round_trips():
    posted = client.post("/api/feedback", json={
        "role": "CEO", "question": "q", "answer": "a", "verdict": "up",
        "chunk_ids": ["x"], "correction": None}).json()
    assert posted["id"] > 0

    listed = client.get("/api/feedback?limit=5").json()
    assert any(r["id"] == posted["id"] for r in listed["rows"])


def test_an_invalid_verdict_is_rejected():
    assert client.post("/api/feedback", json={
        "role": "CEO", "question": "q", "answer": "a", "verdict": "sideways",
        "chunk_ids": []}).status_code == 400


def test_the_console_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Decision pipeline" in response.text
