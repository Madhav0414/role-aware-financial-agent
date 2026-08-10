"""Assign sensitivity tags to document text.

Deterministic and declarative: every rule lives in config/sources.yaml, and no
model is consulted. This is the foundation the access gate stands on, so it has
to be reproducible and inspectable — an access decision that depends on an
LLM's classification is not an access control.
"""

from __future__ import annotations

import re

from src.access.model import Tag

# A heading is short. Body prose in these filings runs far longer, so length is
# a cheap first filter before any pattern matching.
MAX_HEADING_CHARS = 90

# Length alone is not enough. "appropriate, other employees" is short and
# contains "employees", and treating it as a heading tagged pages of governance
# prose as restricted HR data. A real heading is mostly composed of its own
# title, so the matched pattern must account for at least this share of the
# line. "Summary Compensation Table" scores 1.0; the fragment above scores 0.32.
MIN_MATCH_RATIO = 0.5

# SEC filings number their sections. A line opening "Item 7." is a heading
# regardless of how much trailing text it carries.
_ITEM_HEADING = re.compile(r"^item\s+\d+[a-z]?\b", re.IGNORECASE)

# Prose punctuation. A line ending mid-clause is a wrapped sentence, not a
# title — with the exception of numbered Item headings, which legitimately
# contain a full stop after the number.
_PROSE_ENDINGS = (",", ";", ":", "and", "or", "the", "of", "to")

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace before any comparison.

    PDF extraction introduces layout artifacts — Apple's 10-K reads "full-time
    equivalent employees" and extracts as "full- time equivalent employees".
    An exact-string match reported that phrase as missing from a document that
    plainly contains it, which is how this function came to exist.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


def _looks_like_a_heading(normalised: str) -> bool:
    """Structural test, applied before any pattern is considered.

    A section heading governs every chunk after it until the next one, so a
    false positive is expensive: one wrongly-detected heading in a 106-page
    proxy mis-tagged pages of prose as restricted compensation data.
    """
    if _ITEM_HEADING.match(normalised):
        return True
    if normalised.endswith(_PROSE_ENDINGS):
        return False
    # A full stop mid-line means at least one complete sentence, so this is
    # prose. Trailing full stops are fine — plenty of titles carry one.
    return ". " not in normalised


def tag_for_heading(line: str, cfg: dict) -> Tag | None:
    """Return the tag this line introduces, or None if it is not a heading.

    Rules are evaluated in file order and the first match wins, so
    config/sources.yaml lists the most specific patterns first — "summary
    compensation table" must be tested before "employees".
    """
    if len(line) > MAX_HEADING_CHARS:
        return None

    normalised = normalise(line)
    if not normalised or not _looks_like_a_heading(normalised):
        return None

    for rule in cfg["section_tags"]:
        pattern = rule["match"]
        if pattern not in normalised:
            continue
        # The pattern must be most of the line. A heading is its own title;
        # a sentence that merely mentions the words is not.
        if _ITEM_HEADING.match(normalised) or \
                len(pattern) / len(normalised) >= MIN_MATCH_RATIO:
            return Tag(rule["tag"])
    return None


def default_tag_for_document(filename: str, cfg: dict) -> Tag:
    """Where a document's content lands before the first heading is seen.

    A proxy statement is governance material by default; a 10-K is narrative.
    Content ahead of any heading has to be tagged as something, and guessing
    per-filing beats one global default that is wrong half the time.
    """
    for rule in cfg["document_defaults"]:
        if rule["match"] in filename:
            return Tag(rule["tag"])
    return Tag.NARR_MDNA
