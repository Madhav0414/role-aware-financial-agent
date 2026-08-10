"""The vocabulary of access control: what data is, who may read it, and what
came of asking.

Nothing here performs enforcement — these are the types the gate and the guard
reason over. Keeping them separate means the enforcement logic can be read on
one screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class Tag(str, Enum):
    """The sensitivity label carried by every chunk and every fact.

    Tags are assigned during ingestion by a declarative map, never by a model.
    An access decision that depends on an LLM's classification is not an access
    control; it is a suggestion.

    Subclassing `str` means a Tag compares equal to its own value, which lets it
    be bound directly as a SQL parameter without an unwrapping step.
    """

    FIN_STATEMENTS = "financials.statements"
    FIN_SEGMENT = "financials.segment"
    NARR_MDNA = "narrative.mdna"
    NARR_RISK = "narrative.risk"
    HR_HEADCOUNT = "hr.headcount"
    HR_COMPENSATION = "hr.compensation"
    GOVERNANCE = "governance"


@dataclass(frozen=True)
class Role:
    """A resolved permission set.

    Frozen because a role is handed to every gate and tool in a request; if a
    caller could mutate it, an access decision would become shared mutable
    state and the audit log would stop meaning anything.

    `allowed_tags` is already net of any denials — see `load_roles`.
    `recent_years_only` is a count of fiscal years, resolved against the corpus
    by the gate, or None for no time restriction.
    """

    name: str
    allowed_tags: frozenset[Tag]
    recent_years_only: int | None
    description: str = ""


@dataclass(frozen=True)
class Decision:
    """The outcome of an access check.

    Always returned, never raised, and never expressed as an empty result: a
    caller must be able to tell "you may not see this" apart from "there is no
    such data". The reason is written for a human reading the audit log.
    """

    allowed: bool
    reason: str
    denied_tags: tuple[Tag, ...] = ()
    denied_periods: tuple[str, ...] = ()


def load_roles(path: Path) -> dict[str, Role]:
    """Read the permission model from YAML.

    Denials are subtracted from allowances here rather than being checked
    separately at request time. Resolving to a single set at load time means
    there is exactly one thing for the gate to test, and no ordering bug
    between an allow rule and a deny rule.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["roles"]

    roles: dict[str, Role] = {}
    for name, cfg in config.items():
        declared = cfg["allowed_tags"]
        allowed = frozenset(Tag) if declared == ["*"] else frozenset(
            Tag(t) for t in declared
        )
        denied = frozenset(Tag(t) for t in cfg.get("denied_tags", []))
        roles[name] = Role(
            name=name,
            allowed_tags=allowed - denied,
            recent_years_only=cfg.get("recent_years_only"),
            description=" ".join(cfg.get("description", "").split()),
        )
    return roles
