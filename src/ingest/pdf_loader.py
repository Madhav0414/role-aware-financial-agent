"""Extract text from PDF filings, one page at a time.

PyMuPDF rather than pdfplumber or PyPDF2: it needs no system-level dependency
(pdfplumber's rendering path wants poppler), it is markedly faster across 467
pages, and its page-level text extraction preserves reading order well enough
that section headings survive as their own lines — which is what the tagger
depends on.

Page numbers are 1-based because they end up in citations a human reads.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

log = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page.

    A page that fails to extract is logged and yielded as empty rather than
    aborting the document: one bad page must not cost the other seventy-six.
    """
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            try:
                pages.append((index, page.get_text()))
            except Exception as exc:  # noqa: BLE001
                log.warning("could not extract %s p.%d: %s", path.name, index, exc)
                pages.append((index, ""))
    return pages
