"""The agent layer: planner, tools, and the assembled context.

These run against the real built artifacts in data/understanding/, so they fail
if the pipeline that produces them regresses.
"""

from pathlib import Path

import pytest

from src.access.gate import AccessGate
from src.access.model import Tag, load_roles
from src.agent import llm, tools
from src.agent.loop import answer, build_context
from src.agent.planner import Planner
from src.ingest.chunker import load_config
from src.understanding.facts import corpus_max_fy

UND = Path("data/understanding")
ROLES = load_roles(Path("config/roles.yaml"))
CFG = load_config(Path("config/sources.yaml"))
MAX_FY = corpus_max_fy(UND / "facts.db")

pytestmark = pytest.mark.skipif(
    not (UND / "metric_tags.json").exists(),
    reason="run scripts/build_understanding.py first")


def gate(role: str) -> AccessGate:
    return AccessGate(ROLES[role], MAX_FY)


@pytest.fixture(scope="module")
def planner() -> Planner:
    return Planner.from_artifacts(CFG, UND, MAX_FY)


# -- planner --------------------------------------------------------------

def test_plain_wording_maps_onto_stored_metrics(planner):
    plan = planner.plan("What was revenue in FY2025?")
    assert "net_sales" in plan.metrics
    assert plan.periods == ("FY2025",)
    assert Tag.FIN_STATEMENTS in plan.tags


def test_longer_aliases_win_over_shorter_ones(planner):
    """"services revenue" must not also match the bare "revenue" alias and
    produce two metrics where one was asked for."""
    plan = planner.plan("what were services revenue in FY2025")
    assert "services_net_sales" in plan.metrics
    assert "net_sales" not in plan.metrics


def test_a_question_with_no_year_defaults_to_the_newest_in_the_corpus(planner):
    plan = planner.plan("what was net income")
    assert plan.periods == (f"FY{MAX_FY}",)


def test_quarterly_labels_are_understood(planner):
    plan = planner.plan("net sales for Q3 FY2026")
    assert "Q3FY2026" in plan.periods


def test_topic_tags_are_declared_even_with_no_metric(planner):
    """A narrative question naming no stored metric must still declare what it
    is about, or the guard has nothing to check."""
    plan = planner.plan("what do we pay the chief executive?")
    assert Tag.HR_COMPENSATION in plan.tags


def test_headcount_question_declares_the_hr_tag(planner):
    plan = planner.plan("how many employees are there?")
    assert Tag.HR_HEADCOUNT in plan.tags


def test_derivation_question_declares_both_tags(planner):
    """The leak case: the plan must name both operands before anything runs."""
    plan = planner.plan("what is revenue per employee in FY2025?")
    assert Tag.FIN_STATEMENTS in plan.tags
    assert Tag.HR_HEADCOUNT in plan.tags


# -- tools ----------------------------------------------------------------

def test_tools_enforce_rbac_regardless_of_caller(tmp_path):
    """RBAC lives inside the tool, so it holds for agent, CLI and test alike."""
    result = tools.query_metrics(["headcount_retail"], ["FY2025"],
                                 gate=gate("CTO"), understanding_dir=UND,
                                 audit_path=tmp_path / "a.log")
    assert result["allowed"] is False
    assert result["rows"] == []


def test_ceo_reads_the_same_metric(tmp_path):
    result = tools.query_metrics(["headcount_retail"], ["FY2025"],
                                 gate=gate("CEO"), understanding_dir=UND,
                                 audit_path=tmp_path / "a.log")
    assert result["allowed"] is True
    assert result["rows"]


def test_unknown_metric_is_treated_as_most_restricted(tmp_path):
    """Failing open on an unrecognised name is how a typo becomes a bypass."""
    result = tools.query_metrics(["not_a_real_metric"], ["FY2025"],
                                 gate=gate("ANALYST"), understanding_dir=UND,
                                 audit_path=tmp_path / "a.log")
    assert result["allowed"] is False


def test_schema_is_filtered_by_role(tmp_path):
    """Listing metric names a role may not read would disclose that the data
    exists and what it is called."""
    ceo = tools.get_schema(gate=gate("CEO"), understanding_dir=UND,
                           audit_path=tmp_path / "a.log")
    analyst = tools.get_schema(gate=gate("ANALYST"), understanding_dir=UND,
                               audit_path=tmp_path / "a.log")
    assert ceo["total_visible"] > analyst["total_visible"]
    assert not any("headcount" in m for m in analyst["rows"])


def test_list_sources_hides_documents_a_role_cannot_read(tmp_path):
    analyst = tools.list_sources(gate=gate("ANALYST"), understanding_dir=UND,
                                 audit_path=tmp_path / "a.log")
    assert not any("DEF14A" in row["source"] for row in analyst["rows"])


def test_llm_absence_is_not_an_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.active_provider() is None
    assert llm.complete("system", "user") is None


# -- context isolation: the proof ----------------------------------------

def test_restricted_text_never_enters_the_assembled_prompt():
    """THE strongest claim in this system.

    Not "the model declined to say it" but "the model never had it". This
    inspects the context window rather than the output, which is the whole
    difference between a prompt instruction and an access control.
    """
    context = build_context("executive compensation and salary of the CEO",
                            gate("CTO"), understanding_dir=UND).lower()

    for forbidden in ("summary compensation table", "stock awards",
                      "166,000", "headcount"):
        assert forbidden not in context, f"{forbidden!r} leaked into the prompt"


def test_ceo_context_does_contain_it():
    """The negative test above is only meaningful if the material is reachable
    at all — otherwise it would pass on an empty corpus."""
    context = build_context("executive compensation summary table",
                            gate("CEO"), understanding_dir=UND).lower()
    assert "compensation" in context


def test_analyst_context_excludes_out_of_window_years():
    context = build_context("net sales and risk factors", gate("ANALYST"),
                            understanding_dir=UND)
    assert "10-K_FY2023.pdf" not in context


# -- end to end -----------------------------------------------------------

def test_ceo_gets_a_cited_figure(tmp_path):
    result = answer("What was net sales in FY2025?", gate("CEO"),
                    understanding_dir=UND, audit_path=tmp_path / "a.log",
                    use_llm=False)
    assert result["allowed"] is True
    assert "416,161" in result["answer"]
    assert result["citations"]


def test_cto_refused_revenue_per_employee(tmp_path):
    result = answer("What is revenue per employee for FY2025?", gate("CTO"),
                    understanding_dir=UND, audit_path=tmp_path / "a.log",
                    use_llm=False)
    assert result["allowed"] is False
    assert "hr.headcount" in result["denied_tags"]
    assert "166,000" not in result["answer"]


def test_analyst_refused_an_out_of_window_year(tmp_path):
    result = answer("What was net sales in FY2023?", gate("ANALYST"),
                    understanding_dir=UND, audit_path=tmp_path / "a.log",
                    use_llm=False)
    assert result["allowed"] is False
    assert "FY2023" in result["denied_periods"]


def test_the_plan_is_returned_for_inspection(tmp_path):
    """The plan is what the guard judged, so it is shown alongside the answer.
    An access decision nobody can inspect is not auditable."""
    result = answer("What was net sales in FY2025?", gate("CEO"),
                    understanding_dir=UND, audit_path=tmp_path / "a.log",
                    use_llm=False)
    assert result["plan"]["intent"] == "metric"
    assert "net_sales" in result["plan"]["metrics"]
