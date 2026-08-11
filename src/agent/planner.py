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
from src.agent import interpreter

# "FY2025", "Q3FY2026", "fiscal 2024", or a bare "2024".
_EXPLICIT_PERIOD = re.compile(r"\b(Q[1-4])?\s*(?:FY|fiscal(?:\s+year)?\s*)?(20\d{2})\b",
                              re.IGNORECASE)

# "June 2025", "ending June 28, 2025", "quarter ended Jun 2025".
_MONTH_YEAR = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"(?:\s+\d{1,2})?,?\s+(20\d{2})\b", re.IGNORECASE)

_MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# Phrases that mean a single quarter rather than a year.
_QUARTER_CUES = ("3 months", "three months", "3-month", "three-month",
                 "quarter", "quarterly", "q1", "q2", "q3", "q4")

# Cumulative year-to-date periods. These are published in the 10-Qs but are
# deliberately NOT ingested (a "9 Months Ended" column would file two different
# meanings under one label), so a question asking for one must be told plainly
# rather than answered with the annual figure.
_YTD_CUES = ("6 months", "six months", "9 months", "nine months",
             "year to date", "year-to-date", "ytd", "half year", "first half")

# Apple's fiscal year ends in late September, so Q1 is Oct-Dec.
_FY_ROLLOVER_MONTH = 10

# Words describing WHEN rather than WHAT. Excluded from ignored-qualifier
# detection, since the period parser already accounts for them.
_PERIOD_WORDS = frozenset("""
month months ending ended end quarter quarterly annual year years fiscal
january february march april may june july august september october november
december jan feb mar apr jun jul aug sep sept oct nov dec half date
""".split())

# Question shapes that ask for prose, never for a figure. "Who is the auditor"
# once matched `auditor_location_auditor_firm_id` and answered "$42 million" —
# the audit firm's registration number, rendered as dollars.
_NARRATIVE_CUES = ("why", "explain", "describe", "what does", "what did",
                   "management say", "discuss", "outlook", "how does",
                   "tell me about", "summar",
                   "who is", "who are", "who was", "who were",
                   "where is", "where are", "where was", "where were",
                   "headquarter", "located", "location")

# Words carried by almost every question. Left in, they would let a metric
# whose name contains "total" or "the" match anything.
_QUESTION_NOISE = frozenset("""
what was were is are the a an of in for to and or how much many did does do we
our me tell show give please can you value amount figure number total company
apple fiscal year quarter period report filing about with from at as on it its
""".split())

# Structural words inside metric names that carry no distinguishing meaning.
_METRIC_NOISE = frozenset("""
the and of in for to a an total net gross value amount
""".split())

# A relationship between two metrics is only wanted when the question asks for
# one. "Revenue per employee" is a ratio; "other comprehensive income" merely
# matches several metrics, and dividing them would be arithmetic nobody asked
# for.
_RATIO_CUES = (" per ", "ratio", "divided by", "per employee", "per share",
               "compared to each", "relative to")


# How much of a metric name may be left unaccounted for when the question
# supplied only one word. Two allows a section prefix like
# `current_assets_inventories`; more admits footnote metrics that merely happen
# to contain the word.
_MAX_EXTRA_FOR_ONE_WORD = 2


def _coverage(metric: str, words: set[str]) -> int:
    """How many of the question's words this metric's name actually contains.

    The test for whether a vocabulary match beats a curated alias. "Deferred
    revenue" is covered twice by `current_liabilities_deferred_revenue` and not
    at all by `net_sales`, so the specific metric wins. A bare "revenue" is
    covered once by dozens of footnote metrics, which is why the caller also
    requires the question to name at least two things before switching.
    """
    return len({p for p in metric.split("_") if len(p) > 2} & words)


class Planner:
    """Builds plans from questions using the precomputed artifacts."""

    def __init__(self, cfg: dict, metric_tags: dict[str, tuple[Tag, ...]],
                 corpus_max_fy: int,
                 metric_counts: dict[str, int] | None = None) -> None:
        self.cfg = cfg
        self.metric_tags = metric_tags
        self.corpus_max_fy = corpus_max_fy
        self.metric_counts = metric_counts or {}
        self._vocabulary: frozenset[str] | None = None

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
        counts_path = understanding_dir / "metric_counts.json"
        counts = json.loads(counts_path.read_text(encoding="utf-8")) \
            if counts_path.exists() else {}
        return cls(cfg,
                   {m: tuple(Tag(t) for t in tags) for m, tags in raw.items()},
                   corpus_max_fy, counts)

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
        return tuple(metric for metric, _alias in self._alias_hits(question))

    def _alias_hits(self, question: str) -> list[tuple[str, str]]:
        """Matched `(metric, alias)` pairs, ordered as they appear in the
        question.

        The order matters for ratios. "Revenue per employee" matched
        `headcount` before `net_sales` — aliases are tried longest-first, not
        in reading order — and the composer divided the first by the second,
        reporting 0.40 instead of $2.51 million per employee. An inverted ratio
        is a wrong answer that looks like a right one.

        Callers also use the alias text to see HOW specific a match was: a
        two-word alias like "share repurchases" is a deliberate human mapping,
        while a one-word alias like "revenue" is a convenience that a more
        specific request may legitimately override.
        """
        text = f" {question.lower()} "
        hits: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for alias, metric in self.aliases:
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            match = pattern.search(text)
            if match and metric not in seen:
                if metric in self.metric_tags:
                    hits.append((match.start(), metric, alias))
                    seen.add(metric)
                text = pattern.sub(" ", text)

        hits.sort()
        return [(metric, alias) for _, metric, alias in hits]

    @property
    def vocabulary(self) -> frozenset[str]:
        """Every word that appears in any stored metric name.

        Used to discard question words the corpus has no concept of. "What were
        term debt levels" contains "levels", which names nothing — demanding it
        match would reject `total_term_debt`, the metric being asked for.
        """
        if self._vocabulary is None:
            self._vocabulary = frozenset(
                part for metric in self.metric_tags
                for part in metric.split("_") if len(part) > 2)
        return self._vocabulary

    def content_words(self, question: str) -> set[str]:
        """The words in a question that actually name something in the corpus.

        Filtered against the stored vocabulary, so a phrasing word the user
        added ("levels", "balances", "figures") cannot block a match.
        """
        words = {w for w in re.findall(r"[a-z]+", question.lower())
                 if w not in _QUESTION_NOISE and len(w) > 2}
        return {w for w in words if w in self.vocabulary}

    def match_vocabulary(self, question: str) -> tuple[str, ...]:
        """Match a question against EVERY stored metric name, not just aliases.

        The curated alias list resolves ambiguity for metrics people phrase
        loosely — "revenue" must mean consolidated net sales, not one of forty
        revenue-ish footnote lines. But a hand-written list can only ever cover
        what was written down, which left 530 of 561 stored metrics unreachable
        in plain English.

        The rule here is that **every content word of the question must appear
        in the metric name** — not the reverse. Category scoping renamed
        `accounts_payable` to `current_liabilities_accounts_payable`, so
        requiring the metric's words to appear in the question would reject the
        very metric being asked for. Asking for "accounts payable" should find
        it despite the section prefix the user could not know about.

        Ranked by fewest extra words, so a question about "deferred revenue"
        gets `current_liabilities_deferred_revenue` rather than
        `deferred_tax_assets_deferred_revenue`.
        """
        words = self.content_words(question)
        if not words:
            return ()

        # A word can exist in the vocabulary and still be the wrong one to
        # insist on: "balances" appears inside `..._beginning_balances`, so it
        # survives filtering and then blocks "commercial paper balances" from
        # reaching `commercial_paper`. If the full set matches nothing, drop
        # the least distinctive word and try again — the rarest words carry the
        # meaning.
        exact = self._metrics_containing(words)

        # A single word is thin evidence. "How big is the team these days"
        # reduces to {"days"} and matched
        # `maturities_greater_than_90_days_repayments_of_commercial_paper` —
        # one incidental word producing a confident, unrelated figure.
        #
        # So a one-word question must name essentially the WHOLE metric:
        # "inventories" reaching `current_assets_inventories` is fine, because
        # only the section prefix is left over.
        if len(words) < 2:
            return tuple(m for m in exact
                         if len({p for p in m.split("_")
                                 if len(p) > 2 and p not in _METRIC_NOISE}
                                - words) <= _MAX_EXTRA_FOR_ONE_WORD)

        if exact:
            return exact

        # A word can exist in the vocabulary and still be the wrong one to
        # insist on: "balances" appears inside `..._beginning_balances`, so it
        # blocks "commercial paper balances" from reaching `commercial_paper`.
        #
        # Try dropping each single word and keep the best result. Exactly one
        # word is dropped, never more — relaxing further finds a match for
        # anything, which is how "headcount change over the years" once landed
        # on an unrecognised tax-benefit metric.
        # A relaxed match must still account for at least two of the question's
        # words. One is not evidence: "market valuation" would find any metric
        # mentioning "market", and "risk factors" would find a concentration
        # risk footnote — both confidently wrong answers to questions the
        # corpus should decline.
        best: tuple[str, ...] = ()
        best_cover = 1
        for dropped in words:
            found = self._metrics_containing(words - {dropped})
            if found and _coverage(found[0], words) > best_cover:
                best, best_cover = found, _coverage(found[0], words)
        return best

    def _commonness(self, word: str) -> int:
        """How many metric names contain this word. Higher means less useful."""
        return sum(1 for metric in self.metric_tags
                   if word in metric.split("_"))

    def _metrics_containing(self, required: set[str]) -> tuple[str, ...]:
        """Metrics whose name contains every required word, best first.

        Ranked by how often the metric is actually reported, then by fewest
        unrelated extra words, then name length.

        Frequency leads because every candidate already contains all the
        question's words — the remaining choice is between a figure restated in
        every balance sheet and a name appearing once in a footnote. Ranking on
        name brevity instead returned `total_deferred_revenue` ($13 million,
        a timing-table row) over `current_liabilities_deferred_revenue`
        ($8,249 million, the balance sheet line the question meant).
        """
        if not required:
            return ()

        candidates: list[tuple[int, int, int, str]] = []
        for metric in self.metric_tags:
            parts = {p for p in metric.split("_") if len(p) > 2}
            if not required <= parts:
                continue
            extra = len({p for p in parts if p not in _METRIC_NOISE} - required)
            candidates.append((-self.metric_counts.get(metric, 0), extra,
                               len(metric), metric))

        candidates.sort()
        return tuple(metric for *_, metric in candidates[:3])

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
        text = question.lower()
        periods: list[str] = []

        # A month named with a year identifies a QUARTER, not a year:
        # "3 months ending June 2025" is Q3 FY2025, not FY2025. Reading only
        # the year answered a quarterly question with the annual figure —
        # $112,010M instead of $23,434M — which is a wrong answer that looks
        # entirely right.
        # Only when the question actually says "quarter" or "3 months". A bare
        # month and year — "as of Sep 2023" — is usually a point in time within
        # a fiscal year, and headcount is stated annually; converting it to
        # Q4FY2023 asked for a figure that does not exist.
        wants_quarter = any(cue in text for cue in _QUARTER_CUES)
        for month, year in (_MONTH_YEAR.findall(question) if wants_quarter else []):
            month_num = _MONTH_NUM[month.lower()[:3]]
            fiscal_year = int(year) + 1 if month_num >= _FY_ROLLOVER_MONTH \
                else int(year)
            quarter = ((month_num - _FY_ROLLOVER_MONTH) % 12) // 3 + 1
            label = f"Q{quarter}FY{fiscal_year}"
            if label not in periods:
                periods.append(label)

        # A cumulative period the corpus does not hold. Naming it explicitly
        # means the answer says so instead of silently substituting the year.
        if any(cue in text for cue in _YTD_CUES) and not wants_quarter:
            span = "9M" if ("9 month" in text or "nine month" in text) else "6M"
            years = [y for _, y in _EXPLICIT_PERIOD.findall(question)]
            return (f"{span}FY{years[0]}",) if years \
                else (f"{span}FY{self.corpus_max_fy}",)

        if periods:
            return tuple(periods)

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

    def plan(self, question: str, use_llm: bool = False) -> QueryPlan:
        alias_hits = self._alias_hits(question)
        aliased = tuple(metric for metric, _ in alias_hits)
        words = self.content_words(question)
        topic = self.find_topic_tags(question)

        # How specific the curated match was. A multi-word alias is a decision
        # somebody made deliberately — "share repurchases" maps to the cash
        # flow line, not to whichever metric happens to contain both words.
        alias_specificity = max((len(alias.split()) for _, alias in alias_hits),
                                default=0)

        # A question about risk, governance or management commentary is asking
        # for prose, not a figure. Searching the metric vocabulary for it finds
        # something — "risk factors" reaches a concentration-risk footnote —
        # and answering with that number is worse than answering with the
        # passage the question actually wanted.
        narrative_topic = bool(topic) and all(
            t in (Tag.NARR_RISK, Tag.NARR_MDNA, Tag.GOVERNANCE) for t in topic)
        wants_narrative_prose = narrative_topic or any(
            cue in question.lower() for cue in _NARRATIVE_CUES)

        vocabulary = () if wants_narrative_prose \
            else self.match_vocabulary(question)

        # A vocabulary match beats an alias only when it is STRICTLY more
        # specific — that is, its name still contains what the alias matched,
        # plus more. "Deferred revenue" contains "revenue", so the alias fires
        # and answers with consolidated net sales; `current_liabilities_
        # deferred_revenue` contains "revenue" too and is what was asked for.
        #
        # Without the containment check, any extra word in the question was
        # enough to abandon a correct alias: "how did headcount change over the
        # years" wandered off to an unrecognised tax-benefit metric.
        metrics = aliased or vocabulary
        if aliased and vocabulary and len(words) >= 2 \
                and alias_specificity <= 1 \
                and _coverage(vocabulary[0], words) > _coverage(aliased[0], words):
            metrics = vocabulary

        # Keyword matching cannot cover paraphrase: "how profitable were we"
        # names no metric, and no alias list ever will. Ask the model only when
        # the deterministic path found nothing — it is the slow, optional step,
        # and it must never override a match that was already certain.
        #
        # Its proposals are filtered against the real vocabulary, their tags
        # still come from config, and the guard still runs. The model widens
        # understanding without touching the security boundary.
        if use_llm and not metrics and not wants_narrative_prose:
            # Ordered by how often each metric is reported, so when the
            # question shares no words with any name the model is still shown
            # the headline figures rather than an alphabetical slice of
            # footnotes.
            by_importance = sorted(
                self.metric_tags,
                key=lambda m: (-self.metric_counts.get(m, 0), len(m), m))
            metrics = interpreter.propose_metrics(question, by_importance)

        # Several metrics only make sense together when the question asks for a
        # RELATIONSHIP. Without a cue like "per", reporting a ratio between two
        # loosely-matched metrics produces arithmetic nobody asked for.
        wants_ratio = any(cue in f" {question.lower()} " for cue in _RATIO_CUES)
        if len(metrics) > 1 and not wants_ratio:
            metrics = metrics[:1]

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

        if len(metrics) > 1 and wants_ratio:
            intent = "mixed"
        elif metrics and not wants_narrative:
            intent = "metric"
        elif wants_narrative_prose or topic:
            # The question asked for prose and said so — "where is", "who is",
            # "what does management say". Its answer is a passage, and should
            # not be presented with an apology attached.
            intent = "narrative"
        else:
            # Nothing recognised: no metric, no topic, no narrative cue. The
            # answer is whatever came closest, and the caller should say so.
            intent = "unknown"

        # Words the question used to narrow the request that the chosen metric
        # does not account for. Asking "gross margin for products" matches the
        # two-word alias and returns the CONSOLIDATED gross margin — the split
        # by product line lives in the 10-K's narrative table and was never
        # extracted as a figure. Answering with the total and saying nothing is
        # the confidently-wrong failure this system exists to avoid.
        consumed = {w for _, alias in alias_hits for w in alias.split()}
        covered = consumed | {p for m in metrics for p in m.split("_")}
        # Words that describe WHEN, not WHAT. They are already handled by the
        # period parser, so treating them as dropped qualifiers attached a
        # spurious "this is the consolidated figure" note to every quarterly
        # question.
        ignored = tuple(sorted(words - covered - _PERIOD_WORDS)) if metrics else ()

        return QueryPlan(intent=intent, metrics=metrics, periods=periods,
                         tags=tuple(tags), ignored_qualifiers=ignored)
