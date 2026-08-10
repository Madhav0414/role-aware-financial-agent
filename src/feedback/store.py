"""Where user reactions are kept.

Storing feedback is not learning from it. This module only records; `rerank.py`
is what changes behaviour, and `scripts/demo.py` is what proves the change is
visible. A feedback table nobody can observe affecting an answer is decoration.

Similarity between questions is token overlap (Jaccard), not embeddings — the
same reasoning as BM25: it needs no key, no download, and at this corpus size
it is entirely adequate for deciding whether two questions are "about the same
thing".
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/feedback.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    role        TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    verdict     TEXT NOT NULL CHECK (verdict IN ('up', 'down')),
    correction  TEXT,
    chunk_ids   TEXT NOT NULL
);
"""

# Two questions count as "the same" above this token overlap. Set by hand
# rather than tuned: at 0.3 a rephrasing still matches while an unrelated
# question does not, and tuning it against a handful of examples would fit
# noise.
SIMILARITY_THRESHOLD = 0.3

_WORD = re.compile(r"[a-z0-9]+")

# Words carried by almost every question, which would make unrelated questions
# look similar. Kept deliberately short.
_STOPWORDS = frozenset("""
what was is are the a an of in for to and or how much many did does do we our
""".split())


@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    ts: str
    role: str
    question: str
    answer: str
    verdict: str
    correction: str | None
    chunk_ids: tuple[str, ...]


def init(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        con.commit()
    finally:
        con.close()


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of significant tokens. 1.0 is identical, 0.0 disjoint."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def record(role: str, question: str, answer: str, verdict: str,
           chunk_ids: list[str], correction: str | None = None,
           db_path: Path | None = None) -> int:
    """Store one reaction. Returns its row id."""
    if verdict not in ("up", "down"):
        raise ValueError(f"verdict must be 'up' or 'down', got {verdict!r}")

    path = db_path or DB_PATH
    init(path)
    con = sqlite3.connect(path)
    try:
        cursor = con.execute(
            "INSERT INTO feedback (ts, role, question, answer, verdict,"
            " correction, chunk_ids) VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), role, question, answer,
             verdict, correction, json.dumps(chunk_ids)))
        con.commit()
        return int(cursor.lastrowid)
    finally:
        con.close()


def all_records(db_path: Path | None = None) -> list[FeedbackRecord]:
    path = db_path or DB_PATH
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT id, ts, role, question, answer, verdict, correction,"
            " chunk_ids FROM feedback ORDER BY id").fetchall()
    finally:
        con.close()
    return [FeedbackRecord(r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                           tuple(json.loads(r[7]))) for r in rows]


def similar(question: str, threshold: float = SIMILARITY_THRESHOLD,
            db_path: Path | None = None) -> list[tuple[FeedbackRecord, float]]:
    """Past feedback on questions close enough to this one to be relevant.

    Scoped by similarity rather than applied globally: a correction about risk
    factors should not reorder results for a question about revenue.
    """
    scored = [(rec, similarity(question, rec.question))
              for rec in all_records(db_path)]
    matches = [(rec, score) for rec, score in scored if score >= threshold]
    matches.sort(key=lambda pair: pair[1], reverse=True)
    return matches
