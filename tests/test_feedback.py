"""Feedback must demonstrably change behaviour.

Storing votes is not learning from them, so these tests assert the *effect* on
retrieval rather than the contents of a table.
"""

from pathlib import Path

import pytest

from src.access.gate import AccessGate
from src.access.model import Tag, load_roles
from src.agent.loop import answer
from src.feedback import rerank, store
from src.ingest.chunker import Chunk
from src.understanding.facts import corpus_max_fy

UND = Path("data/understanding")
ROLES = load_roles(Path("config/roles.yaml"))

pytestmark = pytest.mark.skipif(
    not (UND / "facts.db").exists(),
    reason="run scripts/build_understanding.py first")


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "feedback.db"
    store.init(path)
    return path


def gate(role: str) -> AccessGate:
    return AccessGate(ROLES[role], corpus_max_fy(UND / "facts.db"))


# -- similarity scoping ---------------------------------------------------

def test_rephrasings_are_recognised_as_the_same_question():
    assert store.similarity("what were the risk factors",
                            "what are the main risk factors") >= 0.3


def test_unrelated_questions_are_not():
    """A correction about risk factors must not reorder a revenue question."""
    assert store.similarity("what were the risk factors",
                            "what was net sales in FY2025") < 0.3


def test_stopwords_do_not_make_questions_look_similar():
    assert store.similarity("what was the revenue", "what were the employees") < 0.3


# -- storage --------------------------------------------------------------

def test_a_vote_is_recorded_with_its_chunks(db):
    store.record("CEO", "risk factors?", "an answer", "down",
                 ["chunk_a"], correction="prefer supply chain", db_path=db)
    records = store.all_records(db)

    assert len(records) == 1
    assert records[0].verdict == "down"
    assert records[0].chunk_ids == ("chunk_a",)
    assert records[0].correction == "prefer supply chain"


def test_an_invalid_verdict_is_rejected(db):
    with pytest.raises(ValueError):
        store.record("CEO", "q", "a", "maybe", [], db_path=db)


# -- the effect on ranking ------------------------------------------------

def test_a_downvote_demotes_the_chunk(db):
    store.record("CEO", "what are the risk factors", "answer", "down",
                 ["chunk_a"], db_path=db)
    weights = rerank.adjustments("what are the main risk factors", db_path=db)

    assert weights["chunk_a"] == rerank.DOWN_WEIGHT


def test_an_upvote_promotes_it(db):
    store.record("CEO", "what are the risk factors", "answer", "up",
                 ["chunk_b"], db_path=db)
    weights = rerank.adjustments("what are the risk factors", db_path=db)

    assert weights["chunk_b"] == rerank.UP_WEIGHT


def test_repeated_votes_compound(db):
    """Correcting the same mistake twice should push further than once."""
    for _ in range(2):
        store.record("CEO", "risk factors", "answer", "down", ["chunk_c"],
                     db_path=db)
    weights = rerank.adjustments("risk factors", db_path=db)

    assert weights["chunk_c"] == pytest.approx(rerank.DOWN_WEIGHT ** 2)


def test_feedback_does_not_cross_between_unrelated_questions(db):
    store.record("CEO", "what are the risk factors", "answer", "down",
                 ["chunk_a"], db_path=db)
    assert rerank.adjustments("what was net sales in FY2025", db_path=db) == {}


def test_apply_reorders_the_result_list(db):
    chunks = [
        Chunk("a", "risk one", "s", "p.1", 2025, Tag.NARR_RISK),
        Chunk("b", "risk two", "s", "p.2", 2025, Tag.NARR_RISK),
    ]
    scored = [(chunks[0], 10.0), (chunks[1], 9.0)]

    store.record("CEO", "risk factors", "answer", "down", ["a"], db_path=db)
    reordered = rerank.apply(scored, "risk factors", db_path=db)

    assert [c.id for c, _ in reordered] == ["b", "a"]


# -- the security property ------------------------------------------------

def test_upvoting_restricted_material_does_not_unlock_the_question(db):
    """Twenty up-votes must not turn a refusal into an answer.

    The guard runs on the plan, before retrieval and therefore before any
    re-ranking, so feedback cannot reach the decision at all.
    """
    forbidden = "DEF14A_2026.pdf:p52:1"
    for _ in range(20):
        store.record("CTO", "executive compensation", "answer", "up",
                     [forbidden], db_path=db)

    result = answer("executive compensation salary", gate("CTO"),
                    understanding_dir=UND, use_llm=False, feedback_db=db)

    assert result["allowed"] is False
    assert "hr.compensation" in result["denied_tags"]


def test_upvoted_restricted_chunks_cannot_surface_in_a_permitted_query(db):
    """The other half: on a question the role IS allowed to ask, an up-voted
    restricted chunk still cannot appear.

    Re-ranking runs strictly after the gate, on a list from which restricted
    material has already been removed — so the up-vote has nothing to act on.
    """
    forbidden = "DEF14A_2026.pdf:p52:1"
    for _ in range(20):
        store.record("CTO", "what are the main risk factors", "answer", "up",
                     [forbidden], db_path=db)

    result = answer("What are the main risk factors?", gate("CTO"),
                    understanding_dir=UND, use_llm=False, feedback_db=db)

    assert result["allowed"] is True
    assert result["passages"]
    assert all(row["tag"] != Tag.HR_COMPENSATION.value
               for row in result["passages"])
    assert all(row["id"] != forbidden for row in result["passages"])


# -- corrections ----------------------------------------------------------

def test_corrections_reach_the_prompt(db):
    store.record("CEO", "what are the risk factors", "answer", "down",
                 ["chunk_a"], correction="Prefer supply chain risk.",
                 db_path=db)
    notes = rerank.corrections_for("what are the main risk factors", db_path=db)

    assert notes == ["Prefer supply chain risk."]


def test_end_to_end_ranking_changes_after_a_correction(db):
    """The property the brief asks to be shown: the same question, asked twice
    either side of a correction, produces a different ordering."""
    question = "What are the main risk factors facing the business?"

    before = answer(question, gate("CEO"), understanding_dir=UND,
                    use_llm=False, feedback_db=db)
    demoted = [row["id"] for row in before["passages"][:2]]
    store.record("CEO", question, before["answer"], "down", demoted,
                 correction="Prioritise supply chain risk.", db_path=db)

    after = answer(question, gate("CEO"), understanding_dir=UND,
                   use_llm=False, feedback_db=db)

    assert after["reranked_by_feedback"] is True
    assert [r["citation"] for r in before["passages"]] != \
           [r["citation"] for r in after["passages"]]
    assert after["corrections_applied"] == ["Prioritise supply chain risk."]
