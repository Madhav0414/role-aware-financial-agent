"""The four tools the agent may call.

RBAC lives INSIDE each tool, not around it. The gate is a required argument of
every one, so enforcement holds identically whether the caller is the agent
loop, the CLI, the web API, or a test. Telling an agent not to read something
is a request; not giving it a tool that can is a guarantee.

Every tool returns the same envelope — `{allowed, reason, rows, ...}` — and
never raises on a denial. A caller must be able to tell "you may not see this"
from "there is no such data", and an exception collapses that distinction the
moment something upstream catches it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.access.audit import audit
from src.access.gate import AccessGate
from src.access.model import Decision, Tag
from src.understanding.facts import query_facts
from src.understanding.index import BM25Index

UNDERSTANDING = Path("data/understanding")


def _envelope(allowed: bool, reason: str, rows: list | None = None,
              **extra) -> dict:
    return {"allowed": allowed, "reason": reason, "rows": rows or [], **extra}


def query_metrics(metrics: list[str], periods: list[str], *,
                  gate: AccessGate,
                  understanding_dir: Path = UNDERSTANDING,
                  audit_path: Path | None = None) -> dict:
    """Exact figures from facts.db, filtered inside the database.

    Returns an empty `rows` with `allowed=False` when the role may not read the
    requested data, rather than an empty list that looks like absence.
    """
    facts = query_facts(understanding_dir / "facts.db", gate,
                        metrics=metrics or None, periods=periods or None)

    # An empty result is ambiguous on its own, so the tool reports whether the
    # role could have seen these metrics at all. A metric spanning several tags
    # is blocked if ANY of them is denied — fail closed.
    blocked = [m for m in metrics
               if any(not gate.check_tag(t).allowed
                      for t in _tags_of(m, understanding_dir))]
    if blocked:
        denied = tuple({t for m in blocked
                        for t in _tags_of(m, understanding_dir)
                        if not gate.check_tag(t).allowed})
        decision = Decision(False,
                            f"{gate.role.name} may not read: {', '.join(blocked)}",
                            denied_tags=denied)
        audit(decision, gate.role.name,
              f"query_metrics({','.join(metrics)})", path=audit_path)
        return _envelope(False, decision.reason)

    decision = Decision(True, f"{len(facts)} figures returned")
    audit(decision, gate.role.name,
          f"query_metrics({','.join(metrics) or '*'})", path=audit_path)
    return _envelope(True, decision.reason,
                     rows=[{"metric": f.metric, "value": f.value,
                            "display": f.format_value(), "period": f.period,
                            "unit": f.unit,
                            "citation": f.citation(), "tag": f.tag.value}
                           for f in facts])


def search_filings(question: str, *, gate: AccessGate, k: int = 5,
                   understanding_dir: Path = UNDERSTANDING,
                   audit_path: Path | None = None,
                   feedback_db: Path | None = None,
                   use_feedback: bool = True) -> dict:
    """Narrative passages from the BM25 index, filtered before scoring.

    Feedback re-ranking runs *after* the gate, on a list from which restricted
    material has already been removed. A user cannot up-vote their way into
    data their role may not read.
    """
    index = BM25Index.load(understanding_dir / "index" / "bm25.json")
    # Over-fetch before re-ranking, or a chunk demoted out of the top k could
    # never be replaced by one promoted into it.
    hits = index.search(question, gate, k=k * 3 if use_feedback else k)

    reranked = False
    if use_feedback:
        from src.feedback import rerank

        adjusted = rerank.apply(hits, question, db_path=feedback_db)
        reranked = [c.id for c, _ in adjusted] != [c.id for c, _ in hits]
        hits = adjusted[:k]

    decision = Decision(True, f"{len(hits)} passages returned")
    audit(decision, gate.role.name, f"search_filings({question[:60]!r})",
          path=audit_path)
    return _envelope(True, decision.reason,
                     rows=[{"id": chunk.id, "text": chunk.text,
                            # The snippet is computed here because this is
                            # where both the index (which knows term rarity)
                            # and the question are available.
                            "snippet": index.snippet(chunk, question),
                            "citation": f"{chunk.source} {chunk.locator}",
                            "tag": chunk.tag.value, "score": round(score, 3)}
                           for chunk, score in hits],
                     reranked_by_feedback=reranked)


def get_schema(*, gate: AccessGate,
               understanding_dir: Path = UNDERSTANDING,
               audit_path: Path | None = None) -> dict:
    """What this role can ask about.

    The schema is filtered too. Listing metric names a role may not read would
    disclose that the data exists and what it is called — a small leak, but the
    kind that makes a permission model look decorative.
    """
    mapping = _metric_tags(understanding_dir)
    visible = sorted(m for m, tags in mapping.items()
                     if all(gate.check_tag(t).allowed for t in tags))

    decision = Decision(True, f"{len(visible)} metrics visible to "
                              f"{gate.role.name}")
    audit(decision, gate.role.name, "get_schema", path=audit_path)
    return _envelope(True, decision.reason,
                     rows=visible[:200],
                     total_visible=len(visible),
                     total_in_corpus=len(mapping))


def list_sources(*, gate: AccessGate,
                 understanding_dir: Path = UNDERSTANDING,
                 audit_path: Path | None = None) -> dict:
    """Which documents this role can reach, and what each contains."""
    summaries_dir = understanding_dir / "summaries"
    rows = []
    for path in sorted(summaries_dir.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        readable = [t for t in summary["tags"]
                    if gate.check_tag(Tag(t)).allowed]
        if not readable:
            continue
        rows.append({"source": summary["source"],
                     "fiscal_years": summary["fiscal_years"],
                     "readable_tags": readable})

    decision = Decision(True, f"{len(rows)} documents visible")
    audit(decision, gate.role.name, "list_sources", path=audit_path)
    return _envelope(True, decision.reason, rows=rows)


# -- helpers --------------------------------------------------------------

_METRIC_TAG_CACHE: dict[Path, dict[str, tuple[Tag, ...]]] = {}


def _metric_tags(understanding_dir: Path) -> dict[str, tuple[Tag, ...]]:
    if understanding_dir not in _METRIC_TAG_CACHE:
        raw = json.loads((understanding_dir / "metric_tags.json")
                         .read_text(encoding="utf-8"))
        _METRIC_TAG_CACHE[understanding_dir] = {
            m: tuple(Tag(t) for t in tags) for m, tags in raw.items()}
    return _METRIC_TAG_CACHE[understanding_dir]


def _tags_of(metric: str, understanding_dir: Path) -> tuple[Tag, ...]:
    """Every tag governing a metric.

    An unknown metric is treated as the most restricted category rather than
    the least. Failing open on a name the system does not recognise is how a
    typo becomes a bypass.
    """
    return _metric_tags(understanding_dir).get(metric, (Tag.HR_COMPENSATION,))


TOOLS = {
    "query_metrics": query_metrics,
    "search_filings": search_filings,
    "get_schema": get_schema,
    "list_sources": list_sources,
}
