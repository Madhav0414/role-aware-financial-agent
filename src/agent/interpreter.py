"""Let the language model map a question onto stored metric names.

WHY THIS EXISTS
Keyword matching cannot cover paraphrase. Someone will ask "how profitable were
we last year" or "what did we make on iPhones", and no alias list survives that
— the filings say "net income" and "iPhone net sales". Extending the list is the
same fix repeated forever.

Understanding phrasing is what a language model is genuinely good at, so this
is the one place it earns its keep.

WHAT IT IS NOT ALLOWED TO DO
The model **proposes metric names and nothing else**. It cannot decide access,
cannot invent a figure, and cannot widen a role's permissions:

- Every name it returns is checked against the real vocabulary and dropped if
  it is not there, so it cannot conjure a metric.
- Tags still come from `metric_tags.json`, never from the model, so a proposal
  cannot under-declare what it touches.
- The derivation guard still runs on the resulting plan, in ordinary Python.

So the worst a compromised or confused model can do is propose a metric the
user is not allowed to see — which the guard then refuses, exactly as it
refuses a keyword-matched one. Understanding is widened; the security boundary
is untouched.

Unavailable model, missing key, bad JSON, or a name that does not exist all
resolve the same way: return nothing, and the deterministic planner's answer
stands.
"""

from __future__ import annotations

import json
import logging
import re

from src.agent import llm

log = logging.getLogger(__name__)

SYSTEM = """\
You map financial questions onto exact metric names from a fixed list.

Rules:
- Reply with JSON only: {"metrics": ["exact_name", ...]}
- Every name MUST be copied exactly from the candidate list you are given.
- Never invent a name. If nothing in the list fits, reply {"metrics": []}.
- Choose the most specific match. Prefer a consolidated figure over a footnote
  unless the question clearly asks for the detail.
- Return at most three names, best first.
"""

# How many candidate names to put in front of the model. The full vocabulary is
# ~560 names; sending all of them wastes tokens and buries the plausible ones.
MAX_CANDIDATES = 60

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _candidates(question: str, vocabulary: list[str], limit: int) -> list[str]:
    """Shortlist the names to put in front of the model.

    A model asked to pick from 560 names does worse than one asked to pick from
    60 plausible ones, and a short prompt stays cheap.

    Word overlap ranks the list but never *filters* it. That distinction is the
    whole point: "how profitable were we" shares no word with `net_income`, and
    a shortlist built only from overlap would be empty for exactly the
    paraphrases this function exists to handle. The remainder is filled from
    `vocabulary` order, which the caller sorts by how often each metric is
    reported — so the headline figures are always on offer.
    """
    words = {w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 2}

    overlapping, rest = [], []
    for name in vocabulary:
        overlap = len(words & set(name.split("_")))
        (overlapping if overlap else rest).append((-overlap, len(name), name))

    overlapping.sort()
    shortlist = [name for *_, name in overlapping[:limit]]
    for *_, name in rest:
        if len(shortlist) >= limit:
            break
        shortlist.append(name)
    return shortlist


def propose_metrics(question: str, vocabulary: list[str],
                    limit: int = MAX_CANDIDATES) -> tuple[str, ...]:
    """Ask the model which stored metrics the question means.

    Returns an empty tuple whenever the model is unavailable or its answer
    cannot be trusted, so the caller's deterministic result stands.
    """
    if llm.active_provider() is None:
        return ()

    shortlist = _candidates(question, vocabulary, limit)
    if not shortlist:
        return ()

    reply = llm.complete(
        SYSTEM,
        f"Question: {question}\n\nCandidate metric names:\n"
        + "\n".join(shortlist))
    if not reply:
        return ()

    match = _JSON.search(reply)
    if match is None:
        log.warning("interpreter: model reply was not JSON")
        return ()

    try:
        proposed = json.loads(match.group(0)).get("metrics", [])
    except json.JSONDecodeError:
        log.warning("interpreter: model reply was not valid JSON")
        return ()

    # Anything not in the real vocabulary is discarded. This is what stops a
    # hallucinated name from becoming a query, and it is why the model can be
    # wrong without being dangerous.
    known = set(vocabulary)
    return tuple(name for name in proposed
                 if isinstance(name, str) and name in known)[:3]
