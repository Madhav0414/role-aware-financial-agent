"""Append-only log of every access decision.

Both allows and denials are recorded. A log containing only refusals cannot
answer "what did this role actually read", which is the question an auditor
asks first.

JSON Lines rather than a database: appending is atomic enough for a single
process, the file stays readable with any text editor during a walkthrough, and
it carries no schema migration burden.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.access.model import Decision

AUDIT_PATH = Path("data/audit.log")


def audit(decision: Decision, role: str, context: str,
          path: Path | None = None) -> None:
    """Record one decision.

    `context` says what was being attempted — a tool name, or the query plan
    that was refused — so a line in the log is meaningful on its own.
    """
    target = path or AUDIT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "context": context,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "denied_tags": [t.value for t in decision.denied_tags],
        "denied_periods": list(decision.denied_periods),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def read_audit(path: Path | None = None) -> list[dict]:
    """Read the log back. Used by the CLI to show why an answer was refused."""
    target = path or AUDIT_PATH
    if not target.exists():
        return []
    return [json.loads(line) for line in
            target.read_text(encoding="utf-8").splitlines() if line.strip()]
