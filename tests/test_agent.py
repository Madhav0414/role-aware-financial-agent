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


# -- reaching every stored metric -----------------------------------------

def test_headcount_comes_from_the_filing_not_the_synthetic_file():
    """Apple states its employee count in prose on page 8 of each 10-K and in
    no workbook. A spreadsheet-only pipeline held that sentence as text and
    could not answer it numerically — FY2023 was missing entirely, because the
    synthetic departmental file only covers FY2024 and FY2025."""
    result = ask_ceo("how many employees were there as of sep 2023")

    assert result["allowed"] is True
    assert "161,000" in result["answer"]
    assert any("10-K_FY2023" in c for c in result["citations"])


def test_headcount_is_available_for_every_annual_report():
    result = ask_ceo("How did headcount change over the years?")
    for figure in ("161,000", "164,000", "166,000"):
        assert figure in result["answer"]


@pytest.mark.parametrize("question,expected_metric", [
    ("What was accounts payable in FY2025?", "accounts_payable"),
    ("What were inventories in FY2025?", "inventories"),
    ("What were term debt levels in FY2025?", "term_debt"),
    ("What were commercial paper balances in FY2025?", "commercial_paper"),
    ("What was depreciation and amortization in FY2024?", "depreciation"),
])
def test_metrics_the_alias_list_never_named_are_reachable(question,
                                                          expected_metric):
    """561 metrics are stored and roughly 30 were hand-aliased. The rest have
    to be reachable by matching the question against the stored vocabulary, or
    most of the ingested data is unusable."""
    result = ask_ceo(question)
    assert result["plan"]["metrics"], f"nothing matched for {question!r}"
    assert expected_metric in result["plan"]["metrics"][0]
    assert result["figures"]


def test_a_specific_request_beats_a_loose_alias():
    """"Deferred revenue" contains "revenue", so the alias fires and answers
    with consolidated net sales — a confidently wrong answer to a question the
    corpus can actually answer."""
    result = ask_ceo("What was deferred revenue in FY2024?")
    assert "deferred_revenue" in result["plan"]["metrics"][0]
    assert "net_sales" not in result["plan"]["metrics"]


def test_a_curated_multi_word_alias_is_not_overridden():
    """"Share repurchases" is a deliberate mapping to the cash flow line, not
    to whichever metric happens to contain both words."""
    result = ask_ceo("How much was spent on share repurchases in FY2025?")
    assert "financing_activities" in result["plan"]["metrics"][0]


def test_a_bare_alias_still_wins_on_a_one_word_question():
    """"Revenue" must stay consolidated net sales rather than wandering into
    one of the forty footnote metrics whose name contains the word."""
    assert ask_ceo("What was revenue in FY2025?")["plan"]["metrics"] \
        == ["net_sales"]


def test_singular_and_plural_forms_match():
    """The filing says "headquarters"; a user types "headquarter". Exact
    matching scored that term at zero and the search fell through to whatever
    else the question mentioned."""
    result = ask_ceo("where was companys headquarter in 2023")

    assert "Cupertino" in result["answer"]
    assert "10-K_FY2023" in result["citations"][0]


def test_the_excerpt_shows_the_matching_sentence():
    """Chunks run to ~1,800 characters and the answer can sit anywhere inside.
    Showing the opening returned "Board of Directors, and the Company's
    share..." while the Cupertino sentence sat 560 characters further down —
    retrieval was right and the excerpt hid it."""
    result = ask_ceo("Where is the company headquarters?")
    assert "headquarters is located in Cupertino" in result["answer"]


def test_a_who_question_does_not_return_a_number():
    """"Who is the auditor" matched `auditor_location_auditor_firm_id` and
    answered "$42 million" — the audit firm's registration number, rendered as
    dollars."""
    result = ask_ceo("who is the auditor")

    assert result["plan"]["metrics"] == []
    assert "$42" not in result["answer"]


def test_a_recognised_narrative_question_is_not_hedged():
    """It asked for prose and said so. Answering correctly and then apologising
    for it reads as a failure."""
    result = ask_ceo("Where is the company headquarters?")
    assert result["answer"].startswith("From the filings")


def test_an_unrecognised_question_is_still_hedged():
    """The control for the rule above — "market valuation" names no metric, no
    topic and no narrative form, so whatever comes closest must be labelled as
    a guess."""
    result = ask_ceo("What is the market valuation?")
    assert "does not match" in result["answer"]


def test_narrative_questions_do_not_hunt_for_metrics():
    """"Risk factors" reaches a concentration-risk footnote if the vocabulary
    is searched, and answering with that number is worse than answering with
    the passage the question actually wanted."""
    result = ask_ceo("What are the main risk factors?")
    assert result["plan"]["metrics"] == []
    assert result["answer"].startswith("From the filings")


# -- trends and comparisons ----------------------------------------------

def test_a_comparison_question_returns_several_years(planner):
    """"How did revenue change?" answered with one number is not an answer to
    the question that was asked."""
    plan = planner.plan("How did revenue change over the years?")
    assert len(plan.periods) == 3
    assert plan.metrics == ("net_sales",)


def test_an_explicit_range_is_respected(planner):
    plan = planner.plan("Compare net sales in FY2023 and FY2025")
    assert set(plan.periods) == {"FY2023", "FY2025"}


def test_a_plain_question_still_gets_one_period(planner):
    """Comparison handling must not turn every question into a trend."""
    assert planner.plan("What was net sales in FY2025?").periods == ("FY2025",)


def test_a_trend_answer_computes_the_change():
    result = answer("How did revenue change over the years?", gate("CEO"),
                    understanding_dir=UND, use_llm=False, use_feedback=False)

    assert "FY2023" in result["answer"] and "FY2025" in result["answer"]
    assert "%" in result["answer"]
    assert "up" in result["answer"] or "down" in result["answer"]


def test_access_still_applies_across_every_period_of_a_trend():
    """A multi-period question must not become a way around the time window."""
    result = answer("How did revenue change over the years?", gate("ANALYST"),
                    understanding_dir=UND, use_llm=False, use_feedback=False)

    assert result["allowed"] is False
    assert "FY2023" in result["denied_periods"]


def test_a_restricted_trend_is_refused():
    result = answer("How did headcount change over the years?", gate("CTO"),
                    understanding_dir=UND, use_llm=False, use_feedback=False)

    assert result["allowed"] is False
    assert "hr.headcount" in result["denied_tags"]


# -- readable labels ------------------------------------------------------

def test_metrics_are_shown_with_human_names():
    """Internal names encode where a figure sits in the filing.
    `operating_expenses_research_and_development` is precise and unreadable."""
    result = answer("How much did research and development cost in FY2025?",
                    gate("CEO"), understanding_dir=UND, use_llm=False,
                    use_feedback=False)

    assert "Research and development" in result["answer"]
    assert "operating_expenses" not in result["answer"]


def test_units_are_not_assumed_to_be_millions():
    """A workbook mixes units. Earnings per share rendered as "$6 million"
    instead of "$6.08 per share" is a wrong answer, not a formatting nit."""
    result = answer("What was diluted earnings per share in FY2024?",
                    gate("CEO"), understanding_dir=UND, use_llm=False,
                    use_feedback=False)

    assert "6.08" in result["answer"]
    assert "per share" in result["answer"]
    assert "million" not in result["answer"]


def test_expenses_are_reported_positive():
    """The unscoped `research_and_development` metric comes from a segment
    reconciliation table where the figure is a deduction, so aliasing to it
    reported R&D as -$34,550 million."""
    result = answer("research and development in FY2025", gate("CEO"),
                    understanding_dir=UND, use_llm=False, use_feedback=False)

    assert "34,550" in result["answer"]
    assert "-34,550" not in result["answer"]


# -- honest failure -------------------------------------------------------

def ask_ceo(question: str) -> dict:
    return answer(question, gate("CEO"), understanding_dir=UND,
                  use_llm=False, use_feedback=False)


def test_an_unanswerable_question_says_so():
    """Found by using the console: "what was evaluation of company in FY2024"
    returned generic risk-factor prose presented as an answer. A fluent
    irrelevance is worse than an honest dead end — it reads like an answer, so
    nobody checks whether it was one."""
    result = ask_ceo("what was the market valuation of the company")

    assert result["allowed"] is True
    lowered = result["answer"].lower()
    assert "does not match" in lowered or "could not find" in lowered


def test_a_recognised_metric_with_no_data_says_which_periods_exist():
    """The most misleading failure in the system, found by using the console.

    "What was profit in 2026" parsed correctly — net_income, FY2026 — but
    FY2026 is a partial year covered only by quarterly filings, so there is no
    annual figure. With zero figures it fell through to narrative search and
    returned unrelated tariff prose under a confident heading. The question was
    UNDERSTOOD, which makes the wrong answer far more convincing.
    """
    result = ask_ceo("What was profit in 2026")

    assert result["allowed"] is True
    assert result["figures"] == []
    assert "do not hold" in result["answer"]
    assert "Net income" in result["answer"]
    # It must point at periods that actually exist, including the quarters.
    assert "FY2025" in result["answer"]
    assert "tariff" not in result["answer"].lower()


def test_a_defaulted_period_falls_back_to_one_with_data():
    """Asking just "revenue" defaulted to the newest fiscal year in the corpus
    — which is a PARTIAL year covered only by quarterly filings, so it held no
    annual figure and the answer was "I do not hold Net sales for FY2026"."""
    result = ask_ceo("revenue")

    assert result["figures"], "a bare metric question should still answer"
    assert "416,161" in result["answer"]
    assert result["plan"]["periods"] == ["FY2025"]


def test_an_explicitly_named_period_is_never_substituted():
    """The fallback applies only to a period the planner chose. Silently
    answering about FY2025 when the user asked about FY2026 would answer a
    different question than the one asked."""
    result = ask_ceo("What was net sales in FY2026?")

    assert result["figures"] == []
    assert "do not hold" in result["answer"]
    assert "FY2026" in result["answer"]


def test_negative_amounts_use_accounting_notation():
    """Cash-flow lines carry the sign the statement uses, so share repurchases
    are negative. "$-90,711 million" reads as a typo."""
    result = ask_ceo("How much was spent on share repurchases in FY2025?")

    assert "$(90,711) million" in result["answer"]
    assert "$-90,711" not in result["answer"]


@pytest.mark.parametrize("question,expected_metric", [
    ("What was Wearables revenue in FY2025?", "wearables"),
    ("What was Asia Pacific revenue in FY2024?", "asia_pacific"),
    ("What was Greater China revenue in FY2024?", "greater_china"),
])
def test_region_and_product_questions_match_one_metric(question,
                                                       expected_metric):
    """Consuming "wearables" left "revenue" behind, which then matched
    net_sales — so one question produced two metrics and the answer reported a
    ratio nobody asked for."""
    result = ask_ceo(question)

    assert len(result["plan"]["metrics"]) == 1
    assert expected_metric in result["plan"]["metrics"][0]
    assert "ratio" not in result["answer"]


def test_the_quarter_that_does_exist_still_answers():
    """The control: FY2026 has no annual figure, but its quarters do."""
    result = ask_ceo("What was net income in Q3FY2026?")

    assert result["figures"]
    assert "29,789" in result["answer"]


def test_a_year_before_the_corpus_is_reported_honestly():
    result = ask_ceo("What was net sales in FY2019?")
    assert "do not hold" in result["answer"]


def test_a_restricted_metric_is_refused_before_availability_is_revealed():
    """Availability is itself information. A role must not learn which periods
    exist for a metric it may not read."""
    result = answer("What was headcount in FY2026?", gate("CTO"),
                    understanding_dir=UND, use_llm=False, use_feedback=False)

    assert result["allowed"] is False
    assert "Available periods" not in result["answer"]


def test_nonsense_gets_guidance_not_a_passage():
    result = ask_ceo("asdkjfh qwerty zzz")

    assert "could not find anything" in result["answer"].lower()
    # A dead end should tell the user what WOULD work.
    assert "net sales" in result["answer"].lower()


def test_a_legitimate_narrative_question_is_not_treated_as_a_failure():
    """Narrative questions are first-class here, not failed metric lookups.
    An earlier version prefixed them with "no reported figure matched", which
    made a correct answer read like an apology."""
    result = ask_ceo("what were the main risk factors")

    assert result["allowed"] is True
    assert result["answer"].startswith("From the filings")
    assert "could not find" not in result["answer"].lower()
    assert "does not match" not in result["answer"].lower()


def test_a_metric_question_still_answers_exactly():
    """The control. Fixing vague questions must not touch the precise ones."""
    result = ask_ceo("what was net sales in FY2024")

    assert "391,035" in result["answer"]
    assert "could not find" not in result["answer"].lower()


def test_weak_matches_are_not_presented_as_answers():
    """Below the relevance floor, nothing is shown at all."""
    from src.agent.loop import MIN_RELEVANCE, _compose_deterministic
    from src.access.guard import QueryPlan

    plan = QueryPlan(intent="narrative", metrics=(), periods=("FY2025",),
                     tags=())
    weak = [{"text": "irrelevant text", "citation": "x p.1",
             "score": MIN_RELEVANCE - 1}]

    assert "could not find" in _compose_deterministic([], weak, plan).lower()


def test_the_plan_is_returned_for_inspection(tmp_path):
    """The plan is what the guard judged, so it is shown alongside the answer.
    An access decision nobody can inspect is not auditable."""
    result = answer("What was net sales in FY2025?", gate("CEO"),
                    understanding_dir=UND, audit_path=tmp_path / "a.log",
                    use_llm=False)
    assert result["plan"]["intent"] == "metric"
    assert "net_sales" in result["plan"]["metrics"]
