"""Turn a question into a declaration of what answering it would require.

The plan is the thing the guard checks, so how it is built matters as much as
what it contains. Two rules hold throughout:

1. **Tags come from configuration, never from the question.** A metric's tag is
   looked up in `metric_tags.json`, derived at ingest time from the data
   itself. If the question could influence its own tags, a request could
   under-declare and walk past the guard.

2. **Being wrong is safe in one direction only.** Declaring a tag the question
   does not need causes an unnecessary refusal. Failing to declare one lets a
   request through unchecked. So topic matching is deliberately broad.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.access.guard import QueryPlan
from src.access.model import Tag

# "FY2025", "Q3FY2026", "fiscal 2024", or a bare "2024".
_EXPLICIT_PERIOD = re.compile(r"\b(Q[1-4])?\s*(?:FY|fiscal(?:\s+year)?\s*)?(20\d{2})\b",
                              re.IGNORECASE)

_NARRATIVE_CUES = ("why", "explain", "describe", "what does", "what did",
                   "management say", "discuss", "outlook", "how does",
                   "tell me about", "summar")


class Planner:
    """Builds plans from questions using the precomputed artifacts."""

    def __init__(self, cfg: dict, metric_tags: dict[str, tuple[Tag, ...]],
                 corpus_max_fy: int) -> None:
        self.cfg = cfg
        self.metric_tags = metric_tags
        self.corpus_max_fy = corpus_max_fy

        # Longest aliases first, so "services revenue" wins over "revenue".
        self.aliases: list[tuple[str, str]] = sorted(
            ((alias.lower(), metric)
             for metric, aliases in cfg.get("metric_aliases", {}).items()
             for alias in aliases),
            key=lambda pair: -len(pair[0]),
        )

    @classmethod
    def from_artifacts(cls, cfg: dict, understanding_dir: Path,
                       corpus_max_fy: int) -> "Planner":
        raw = json.loads((understanding_dir / "metric_tags.json")
                         .read_text(encoding="utf-8"))
        return cls(cfg,
                   {m: tuple(Tag(t) for t in tags) for m, tags in raw.items()},
                   corpus_max_fy)

    # -- components --------------------------------------------------------

    def find_metrics(self, question: str) -> tuple[str, ...]:
        """Match question wording onto stored metric names.

        Two rules make this safe:

        - **Word boundaries.** A substring match would let the alias "eps" fire
          on "steps" and "r&d" on unrelated punctuation. Short aliases are the
          useful ones, so they have to be matched precisely.
        - **Consumption.** A matched alias is blanked out, so "services
          revenue" cannot also match the bare "revenue" alias afterwards and
          produce two metrics where the user asked for one. Aliases are tried
          longest-first for the same reason.
        """
        text = f" {question.lower()} "
        found: list[str] = []
        for alias, metric in self.aliases:
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            if pattern.search(text) and metric not in found:
                if metric in self.metric_tags:
                    found.append(metric)
                text = pattern.sub(" ", text)
        return tuple(found)

    def suggest_metrics(self, question: str, limit: int = 6) -> list[str]:
        """Stored metric names closest to the words in the question.

        Used when nothing matched, so the system can say "I don't hold that,
        here is what I do hold" instead of silently falling back to a weak
        narrative search. Scored by token overlap — no dependency, and good
        enough to turn a dead end into a next step.
        """
        words = {w for w in re.findall(r"[a-z]+", question.lower())
                 if len(w) > 3}
        if not words:
            return []

        scored: list[tuple[int, str]] = []
        for metric in self.metric_tags:
            parts = set(metric.split("_"))
            overlap = len(words & parts)
            if overlap:
                # Shorter names first at equal overlap: `net_sales` is a more
                # useful suggestion than `products_net_sales_deferred_revenue`.
                scored.append((-overlap * 100 + len(metric), metric))
        scored.sort()
        return [metric for _, metric in scored[:limit]]

    def has_explicit_period(self, question: str) -> bool:
        """Did the user actually name a period, or did the planner default one?

        A defaulted period may be wrong — the newest fiscal year in the corpus
        can be a partial year covered only by quarterly filings — so the caller
        needs to know whether it is safe to substitute a better one.
        """
        return bool(_EXPLICIT_PERIOD.search(question))

    def wants_comparison(self, question: str) -> bool:
        """Does the question ask how something CHANGED rather than what it was?"""
        text = f" {question.lower()} "
        return any(cue in text for cue in self.cfg.get("comparison_cues", []))

    def find_periods(self, question: str) -> tuple[str, ...]:
        """Extract periods, defaulting to the newest year in the corpus.

        Defaulting to the corpus rather than to today's date keeps behaviour
        stable as the calendar moves.

        A comparison question that names no years gets the three most recent
        annual periods — asking "how has revenue grown?" and receiving one
        number is not an answer to the question that was asked.
        """
        periods = []
        for quarter, year in _EXPLICIT_PERIOD.findall(question):
            label = f"{quarter.upper().replace(' ', '')}FY{year}" if quarter \
                else f"FY{year}"
            if label not in periods:
                periods.append(label)

        if periods:
            return tuple(periods)

        if self.wants_comparison(question):
            # The newest annual period in the corpus may be a partial year
            # covered only by quarterlies, so step back from the newest year
            # that actually carries annual figures.
            newest = self.corpus_max_fy - 1
            return tuple(f"FY{newest - offset}" for offset in range(3))

        return (f"FY{self.corpus_max_fy}",)

    def find_topic_tags(self, question: str) -> tuple[Tag, ...]:
        """Tags implied by what the question is *about*.

        This is what lets a narrative question — one naming no metric at all —
        still be refused. "What do we pay the CEO?" declares hr.compensation
        even though no stored metric matched.
        """
        text = question.lower()
        tags: list[Tag] = []
        for rule in self.cfg.get("question_topic_tags", []):
            if any(keyword in text for keyword in rule["match"]):
                tag = Tag(rule["tag"])
                if tag not in tags:
                    tags.append(tag)
        return tuple(tags)

    # -- the plan ----------------------------------------------------------

    def plan(self, question: str) -> QueryPlan:
        metrics = self.find_metrics(question)
        periods = self.find_periods(question)

        # Union of tags from matched metrics and from the question's topic.
        # A union, not an intersection: every tag either source suggests must
        # be declared, or the guard checks less than the request touches.
        tags: list[Tag] = []
        from_metrics = [t for m in metrics for t in self.metric_tags[m]]
        for tag in from_metrics + list(self.find_topic_tags(question)):
            if tag not in tags:
                tags.append(tag)

        lowered = question.lower()
        wants_narrative = any(cue in lowered for cue in _NARRATIVE_CUES)

        if len(metrics) > 1:
            intent = "mixed"
        elif metrics and not wants_narrative:
            intent = "metric"
        else:
            intent = "narrative"

        return QueryPlan(intent=intent, metrics=metrics, periods=periods,
                         tags=tuple(tags))
