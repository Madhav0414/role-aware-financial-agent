"""Regenerate every understanding artifact from data/raw/.

    python scripts/build_understanding.py

Safe to re-run: the facts database is rebuilt from scratch each time, so it can
never drift from what the ingest pipeline currently produces.
"""

from __future__ import annotations

import sys
import time

import _bootstrap  # noqa: F401  -- fixes sys.path and the working directory

from src.understanding.build import OUT, build_all


def main() -> int:
    started = time.time()
    print("Building understanding artifacts from data/raw/ ...")
    report = build_all()
    elapsed = time.time() - started

    print(f"\n  facts.db          {report['facts']:>6,} facts")
    print(f"  index/bm25.json   {report['chunks']:>6,} chunks")
    print(f"  summaries/        {report['documents']:>6,} documents")
    print("  schema_notes.md   generated")
    print(f"\n  newest fiscal year in corpus: FY{report['corpus_max_fy']}")

    print("\n  sensitivity tags:")
    for tag, count in sorted(report["tags"].items(),
                             key=lambda kv: -kv[1]):
        print(f"    {tag:<26}{count:>7,}")

    print(f"\nDone in {elapsed:.1f}s. Artifacts in {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
