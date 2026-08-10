"""Command-line interface.

    python -m src.cli --role CTO --ask "What is revenue per employee?"
    python -m src.cli --role CEO            # interactive

Role is asserted at session start rather than authenticated. That is a
deliberate scope decision: the brief asks for enforcement at the data layer,
so the 24 hours went into making that layer unbypassable rather than into a
login screen. Swapping in real authentication changes only which Role object
gets constructed here — nothing downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.access.gate import AccessGate
from src.access.model import Role, load_roles
from src.agent.loop import answer

ROLES_PATH = Path("config/roles.yaml")

# The newest fiscal year in the committed corpus. Time windows are measured
# from this rather than from today's date. Task 6 reads it from facts.db.
CORPUS_MAX_FY = 2026


def render(result: dict) -> str:
    lines = []
    if result["allowed"]:
        lines.append(result["answer"])
        if result["citations"]:
            lines.append("")
            lines.append("Sources: " + "; ".join(result["citations"]))
    else:
        # A refusal states its reason. Silence, or an empty result, would be
        # indistinguishable from the data not existing.
        lines.append("REFUSED")
        lines.append(result["reason"])
    return "\n".join(lines)


def describe_roles(roles: dict[str, Role]) -> str:
    out = []
    for name, role in roles.items():
        window = ("all periods" if role.recent_years_only is None
                  else f"most recent {role.recent_years_only} fiscal years")
        out.append(f"  {name:<9} {len(role.allowed_tags)} tags, {window}")
        if role.description:
            out.append(f"            {role.description}")
    return "\n".join(out)


def repl(gate: AccessGate, roles: dict[str, Role]) -> None:
    print(f"Role: {gate.role.name}. Ask a question, or :help. :quit to exit.")
    while True:
        try:
            line = input(f"\n[{gate.role.name}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":quit", ":q"):
            return
        if line == ":help":
            print("  :role <NAME>   switch role (see :roles)\n"
                  "  :roles         list roles and their permissions\n"
                  "  :quit          exit")
            continue
        if line == ":roles":
            print(describe_roles(roles))
            continue
        if line.startswith(":role "):
            name = line.split(maxsplit=1)[1].strip().upper()
            if name not in roles:
                print(f"Unknown role {name}. Known: {', '.join(roles)}")
                continue
            gate = AccessGate(roles[name], CORPUS_MAX_FY)
            print(f"Switched to {name}.")
            continue
        print()
        print(render(answer(line, gate)))


def main(argv: list[str] | None = None) -> int:
    roles = load_roles(ROLES_PATH)

    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Ask questions of Apple's public filings under a role.")
    parser.add_argument("--role", default="CEO", choices=sorted(roles),
                        help="identity asserted for this session")
    parser.add_argument("--ask", help="one question, then exit")
    parser.add_argument("--roles", action="store_true",
                        help="list roles and their permissions, then exit")
    args = parser.parse_args(argv)

    if args.roles:
        print(describe_roles(roles))
        return 0

    gate = AccessGate(roles[args.role], CORPUS_MAX_FY)

    if args.ask:
        result = answer(args.ask, gate)
        print(render(result))
        # Non-zero exit on refusal so the demo script can assert on it.
        return 0 if result["allowed"] else 3

    repl(gate, roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
