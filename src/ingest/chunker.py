"""Turn pages of extracted text into tagged, citable chunks.

A Chunk is the unit the BM25 index scores and the access gate filters. Every
one carries where it came from, which fiscal year it describes, and what kind
of data it is — because an answer that cannot name its source is not an answer,
and a chunk that cannot state its sensitivity cannot be governed.

Section state carries ACROSS pages: a heading on page 8 governs the prose that
continues onto page 9. Resetting per page would leave most of every section
untagged and quietly visible to roles that should not see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.access.model import Tag
from src.ingest.pdf_loader import load_pdf
from src.ingest.tagger import default_tag_for_document, tag_for_heading


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    Shares `.tag`, `.fiscal_year` and `.quarantined` with Fact, which is the
    whole reason the access gate can filter either without importing either.
    """

    id: str
    text: str
    source: str
    locator: str          # "p.42"
    fiscal_year: int
    tag: Tag
    quarantined: bool = False


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _split_long(line: str, max_chars: int) -> list[str]:
    """Break an over-long line into pieces that fit.

    Statement tables in these filings extract as a single line hundreds of
    characters wide, so the cap cannot be enforced by line boundaries alone.
    """
    if len(line) <= max_chars:
        return [line]
    return [line[i:i + max_chars] for i in range(0, len(line), max_chars)]


def _flush(buffer: list[str], min_chars: int) -> str | None:
    """Join a pending buffer into chunk text, or None if too thin to be useful.

    Page furniture — headers, footers, page numbers — arrives as tiny fragments
    that would otherwise become chunks of their own and dilute retrieval.
    """
    text = "\n".join(buffer).strip()
    return text if len(text) >= min_chars else None


def chunk_document(path: Path, fiscal_year: int, cfg: dict) -> list[Chunk]:
    """Read one PDF into tagged chunks."""
    max_chars = cfg["chunking"]["max_chars"]
    min_chars = cfg["chunking"]["min_chars"]

    current_tag = default_tag_for_document(path.name, cfg)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_chars = 0
    buffer_page = 1
    sequence = 0

    def emit(page: int) -> None:
        nonlocal buffer, buffer_chars, sequence
        text = _flush(buffer, min_chars)
        if text is not None:
            sequence += 1
            chunks.append(Chunk(
                id=f"{path.name}:p{page}:{sequence}",
                text=text,
                source=path.name,
                locator=f"p.{page}",
                fiscal_year=fiscal_year,
                tag=current_tag,
            ))
        buffer = []
        buffer_chars = 0

    for page_number, page_text in load_pdf(path):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            heading_tag = tag_for_heading(stripped, cfg)
            if heading_tag is not None:
                # A section boundary closes the current chunk. Letting text
                # straddle it would put two sensitivities in one chunk, and the
                # gate can only make one decision per chunk.
                emit(buffer_page)
                current_tag = heading_tag
                buffer_page = page_number

            # A single extracted line can exceed the cap on its own — tables
            # flatten into one very long line. Split it rather than emitting an
            # oversized chunk.
            for piece in _split_long(stripped, max_chars):
                # Close the buffer BEFORE it would overflow, not after, or the
                # line that trips the limit is already inside the chunk.
                if buffer and buffer_chars + len(piece) + 1 > max_chars:
                    emit(buffer_page)
                    buffer_page = page_number
                if not buffer:
                    buffer_page = page_number
                buffer.append(piece)
                buffer_chars += len(piece) + 1

    emit(buffer_page)
    return chunks


def chunk_corpus(raw_dir: Path, cfg: dict) -> list[Chunk]:
    """Chunk every PDF listed in the download manifest."""
    import json

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for entry in manifest:
        chunks += chunk_document(raw_dir / entry["pdf"],
                                 fiscal_year=entry["fiscal_year"], cfg=cfg)
    return chunks
