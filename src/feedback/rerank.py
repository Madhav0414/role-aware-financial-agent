"""Where feedback actually changes behaviour.

Two mechanisms, deliberately different in kind:

1. **Retrieval re-ranking** — chunks that appeared in a down-voted answer are
   demoted for similar questions; up-voted ones are promoted. This changes
   *what the model is shown*.

2. **Correction memory** — a correction written against a similar question is
   injected into the prompt as guidance. This changes *how the model reads it*.

Both are scoped by question similarity. A correction about risk factors must
not reorder results for a question about revenue.

WHAT THIS DELIBERATELY DOES NOT DO
Feedback never widens access. A user cannot up-vote their way into restricted
data, because re-ranking happens strictly after the gate has already removed
everything the role may not see. Learning operates on the permitted set only.
"""

from __future__ import annotations

from src.feedback.store import (
    SIMILARITY_THRESHOLD,
    FeedbackRecord,
    similar,
)

# A down-vote halves a chunk's score, an up-vote adds half again. Multiplicative
# so the adjustment scales with how relevant the chunk was to begin with — a
# weak match that was down-voted does not need to be pushed to zero, and a
# strong match should not be dislodged by a single vote.
DOWN_WEIGHT = 0.5
UP_WEIGHT = 1.5


def adjustments(question: str, threshold: float = SIMILARITY_THRESHOLD,
                db_path=None) -> dict[str, float]:
    """Per-chunk score multipliers implied by past feedback on similar questions.

    Multipliers compound: a chunk down-voted twice is demoted further than one
    down-voted once, which is the behaviour a user expects after correcting the
    same mistake repeatedly.
    """
    weights: dict[str, float] = {}
    for record, _score in similar(question, threshold, db_path):
        factor = DOWN_WEIGHT if record.verdict == "down" else UP_WEIGHT
        for chunk_id in record.chunk_ids:
            weights[chunk_id] = weights.get(chunk_id, 1.0) * factor
    return weights


def apply(scored: list[tuple], question: str,
          threshold: float = SIMILARITY_THRESHOLD,
          db_path=None) -> list[tuple]:
    """Re-rank `[(chunk, score), ...]` using feedback on similar questions.

    Takes the already-gated result list. Nothing here can add a chunk — only
    reorder what the role was already permitted to see.
    """
    weights = adjustments(question, threshold, db_path)
    if not weights:
        return scored

    adjusted = [(chunk, score * weights.get(chunk.id, 1.0))
                for chunk, score in scored]
    adjusted.sort(key=lambda pair: pair[1], reverse=True)
    return adjusted


def corrections_for(question: str, threshold: float = SIMILARITY_THRESHOLD,
                    db_path=None, limit: int = 3) -> list[str]:
    """Corrections a user wrote against similar questions.

    Injected into the prompt as guidance, not as fact: a correction is one
    person's opinion about a past answer, and it must never outrank a figure
    read from a filing.
    """
    seen: list[str] = []
    for record, _score in similar(question, threshold, db_path):
        if record.correction and record.correction not in seen:
            seen.append(record.correction)
        if len(seen) >= limit:
            break
    return seen


def summarise(record: FeedbackRecord) -> str:
    """One-line description, for the CLI and the audit view."""
    mark = "down" if record.verdict == "down" else "up"
    tail = f" — {record.correction}" if record.correction else ""
    return f"[{mark}] {record.role}: {record.question}{tail}"
