"""LLM-assisted question interpretation.

The model is stubbed throughout. These tests are about the CONTRACT — what the
system accepts from a model and what it refuses to accept — not about any
particular model's output. That contract is the security-relevant part: a model
can be wrong here without being dangerous.
"""

from pathlib import Path

import pytest

from src.access.gate import AccessGate
from src.access.model import load_roles
from src.agent import interpreter
from src.agent.loop import answer
from src.understanding.facts import corpus_max_fy

UND = Path("data/understanding")
ROLES = load_roles(Path("config/roles.yaml"))

VOCABULARY = ["net_sales", "net_income", "gross_margin", "headcount",
              "iphone_net_sales", "total_assets"]


def stub_model(monkeypatch, reply: str | None):
    monkeypatch.setattr(interpreter.llm, "active_provider", lambda: "stub")
    monkeypatch.setattr(interpreter.llm, "complete",
                        lambda *a, **k: reply)


# -- what it accepts ------------------------------------------------------

def test_a_paraphrase_maps_onto_a_stored_metric(monkeypatch):
    """"How profitable were we" names no metric and no alias list ever will."""
    stub_model(monkeypatch, '{"metrics": ["net_income"]}')
    assert interpreter.propose_metrics("how profitable were we last year",
                                       VOCABULARY) == ("net_income",)


def test_prose_around_the_json_is_tolerated(monkeypatch):
    stub_model(monkeypatch,
               'Sure! Here is the mapping:\n{"metrics": ["gross_margin"]}\nHope '
               'that helps.')
    assert interpreter.propose_metrics("what was our margin",
                                       VOCABULARY) == ("gross_margin",)


def test_at_most_three_names_are_taken(monkeypatch):
    stub_model(monkeypatch,
               '{"metrics": ["net_sales", "net_income", "gross_margin", '
               '"total_assets"]}')
    assert len(interpreter.propose_metrics("everything", VOCABULARY)) == 3


# -- what it refuses ------------------------------------------------------

def test_an_invented_metric_is_discarded(monkeypatch):
    """The model cannot conjure data. A name absent from the real vocabulary is
    dropped, so a hallucination never becomes a query."""
    stub_model(monkeypatch, '{"metrics": ["profit_margin_2026_forecast"]}')
    assert interpreter.propose_metrics("what will we earn", VOCABULARY) == ()


def test_a_partly_invented_reply_keeps_only_the_real_names(monkeypatch):
    stub_model(monkeypatch,
               '{"metrics": ["net_income", "made_up_metric"]}')
    assert interpreter.propose_metrics("profit", VOCABULARY) == ("net_income",)


@pytest.mark.parametrize("reply", [
    None,                      # model unavailable
    "",                        # empty reply
    "I cannot help with that", # no JSON at all
    '{"metrics": [1, 2, 3]}',  # wrong types
    '{"metrics": "net_sales"}',# wrong shape
    "{broken json",            # unparseable
])
def test_an_unusable_reply_yields_nothing(monkeypatch, reply):
    """Every failure resolves the same way: return nothing, and the
    deterministic planner's answer stands."""
    stub_model(monkeypatch, reply)
    assert interpreter.propose_metrics("anything", VOCABULARY) == ()


def test_no_model_means_no_call(monkeypatch):
    monkeypatch.setattr(interpreter.llm, "active_provider", lambda: None)
    monkeypatch.setattr(interpreter.llm, "complete",
                        lambda *a, **k: pytest.fail("model must not be called"))
    assert interpreter.propose_metrics("anything", VOCABULARY) == ()


# -- the security boundary is untouched -----------------------------------

@pytest.mark.skipif(not (UND / "facts.db").exists(),
                    reason="run scripts/build_understanding.py first")
def test_a_model_proposal_still_faces_the_guard(monkeypatch):
    """The point of the whole design. Even if the model proposes a restricted
    metric — through error, or because a user talked it into one — the guard
    refuses it in ordinary Python, exactly as it refuses a keyword match.
    """
    stub_model(monkeypatch, '{"metrics": ["headcount"]}')
    gate = AccessGate(ROLES["CTO"], corpus_max_fy(UND / "facts.db"))

    result = answer("how big is the team these days", gate,
                    understanding_dir=UND, use_llm=True, use_feedback=False)

    assert result["allowed"] is False
    assert "hr.headcount" in result["denied_tags"]


@pytest.mark.skipif(not (UND / "facts.db").exists(),
                    reason="run scripts/build_understanding.py first")
def test_the_deterministic_path_is_not_overridden(monkeypatch):
    """The model is consulted only when keyword matching found nothing. A
    certain match must never be second-guessed by a slower, less predictable
    one."""
    import src.agent.planner as planner_module

    monkeypatch.setattr(
        planner_module.interpreter, "propose_metrics",
        lambda *a, **k: pytest.fail("interpreter consulted despite a match"))

    gate = AccessGate(ROLES["CEO"], corpus_max_fy(UND / "facts.db"))
    result = answer("What was net sales in FY2025?", gate,
                    understanding_dir=UND, use_llm=True, use_feedback=False)

    assert "416,161" in result["answer"]
