"""BM25 retrieval over narrative chunks.

WHY BM25 RATHER THAN EMBEDDINGS
Financial questions are keyword-heavy — "net sales FY2024", "supply chain
risk" — which is exactly what lexical scoring is good at. Dense embeddings
would need either an API key or a multi-hundred-megabyte model download, and
they are actively worse on this corpus: passages that differ only by fiscal
year are near-identical in embedding space, so a question about FY2023 happily
retrieves FY2022. BM25 also runs offline, which is what lets the whole system
answer with no key at all.

Roughly forty lines of scoring, no dependency beyond numpy, and the index is
plain JSON that can be opened and read during a walkthrough.

ACCESS CONTROL
`search` filters through the gate BEFORE scoring anything. A restricted chunk
never competes for a slot, so it cannot influence which results come back, how
many there are, or how long the query took.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.access.gate import AccessGate
from src.access.model import Tag
from src.ingest.chunker import Chunk

# Okapi BM25 defaults. k1 controls how fast term frequency saturates; b controls
# how much document length is penalised. These are the standard values and were
# not tuned — tuning them against a corpus this small would fit noise.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that carry question grammar rather than meaning. They are kept for
# SCORING — BM25's IDF already discounts them — but excluded when choosing
# which part of a passage to show. "Where" appears in only 39 of 737 chunks, so
# IDF rates it as informative, and a window containing "where/was/company/2023"
# then outscored the one containing "headquarter". Grammar should not decide
# which sentence a reader is shown.
_SNIPPET_STOPWORDS = frozenset("""
what where when who why which how was were is are be been the a an of in for to
and or that this these those it its their our we you they i do does did has
have had can could would should may might will shall about with from as at by
on into over under more most much many any some all such other than then there
here company companys apple year years fiscal quarter tell show give me my
""".split())


def _fold(token: str) -> str:
    """Normalise a token so singular and plural forms match.

    Found by asking "where was the company's headquarter in 2023". The filing
    says "headquarters"; the question said "headquarter". Exact matching scored
    that term at zero, so the search fell through to whatever else the question
    mentioned and returned a revenue table.

    ONE rule, not a stemmer. Porter stemming would fold "capitalised" to
    "capital" and "reserves" to "reserve", and in financial text those are
    different things. This handles the plural/possessive mismatch that actually
    breaks queries and leaves everything else alone.

    It does mangle some words — "analysis" becomes "analysi". That is harmless
    because the same fold is applied when building the index and when searching
    it: both sides agree, and the folded form is an internal key nobody sees.
    Retrieval needs consistency, not linguistic correctness.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs, with plurals folded.

    Digits are kept: "2025" and "166" are exactly the tokens that make a
    financial query specific, so stripping numbers would remove the most
    discriminating terms in the corpus.
    """
    return [_fold(token) for token in _TOKEN.findall(text.lower())]


class BM25Index:
    def __init__(self, chunks: list[Chunk], term_freqs: list[dict[str, int]],
                 doc_lens: list[int], doc_freq: dict[str, int]) -> None:
        self.chunks = chunks
        self.term_freqs = term_freqs
        self.doc_lens = doc_lens
        self.doc_freq = doc_freq
        self.avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        term_freqs, doc_lens, doc_freq = [], [], Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            term_freqs.append(dict(counts))
            doc_lens.append(len(tokens))
            doc_freq.update(counts.keys())
        return cls(chunks, term_freqs, doc_lens, dict(doc_freq))

    def _idf(self, term: str) -> float:
        """Inverse document frequency, with the +1 smoothing that keeps a term
        appearing in every document from going negative."""
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def snippet(self, chunk: Chunk, query: str, width: int = 460) -> str:
        """The part of a passage that actually answers the question.

        Chunks run to ~1,800 characters and the matching sentence can sit
        anywhere inside, so showing the opening returned "Board of Directors,
        and the Company's share..." for a question about the headquarters while
        the Cupertino sentence sat 560 characters further down.

        Windows are scored by the summed IDF of the distinct query terms they
        contain — not by how many. Counting terms lets "company", "was" and
        "2023" outvote "headquarter", which is the one word that matters:
        "company" appears in 519 of 737 chunks and "headquarter" in 3.
        """
        flat = " ".join(chunk.text.split())
        if len(flat) <= width:
            return flat

        terms = {t for t in tokenize(query)
                 if len(t) > 2 and t not in _SNIPPET_STOPWORDS}
        if not terms:
            return flat[:width]

        # Centre the excerpt on the RAREST matching term rather than sliding a
        # window and taking the best-scoring one. Sliding produced a window
        # that technically contained "headquarters" — in its final 48
        # characters, where a reader never sees it. The rare term is the reason
        # this passage was retrieved, so it belongs in the middle.
        anchor, best_idf = None, 0.0
        lowered = flat.lower()
        for term in terms:
            position = lowered.find(term)
            if position == -1:
                continue
            idf = self._idf(term)
            if idf > best_idf:
                anchor, best_idf = position, idf

        if anchor is None:
            return flat[:width]

        start = max(0, anchor - width // 3)
        text = flat[start:start + width]
        if start > 0:
            cut = text.find(" ")          # open at a word boundary
            text = text[cut + 1:] if cut != -1 else text
            return f"…{text}"
        return text

    def search(self, query: str, gate: AccessGate,
               k: int = 5) -> list[tuple[Chunk, float]]:
        """Score only what this role may read.

        The gate runs first. Everything after it operates on a corpus from
        which restricted material has already been removed.
        """
        permitted = {id(c) for c in gate.filter_chunks(self.chunks)}
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            if id(chunk) not in permitted:
                continue
            freqs = self.term_freqs[i]
            length = self.doc_lens[i] or 1
            score = 0.0
            for term in query_terms:
                f = freqs.get(term, 0)
                if not f:
                    continue
                denominator = f + K1 * (1 - B + B * length / self.avg_len)
                score += self._idf(term) * (f * (K1 + 1)) / denominator
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Plain JSON, so the index can be opened and read during a
        walkthrough rather than being an opaque binary."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [{**asdict(c), "tag": c.tag.value} for c in self.chunks],
            "term_freqs": self.term_freqs,
            "doc_lens": self.doc_lens,
            "doc_freq": self.doc_freq,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**{**c, "tag": Tag(c["tag"])}) for c in payload["chunks"]]
        return cls(chunks, payload["term_freqs"], payload["doc_lens"],
                   payload["doc_freq"])
