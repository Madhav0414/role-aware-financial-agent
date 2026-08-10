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
UNDERSTANDING = Path("data/understanding")


def corpus_anchor() -> int:
    """Newest fiscal year in the corpus — the anchor for every time window.

    Read from the data rather than hardcoded, so a role limited to "the two
    most recent years" tracks the corpus instead of drifting as time passes.
    """
    from src.understanding.facts import corpus_max_fy
    return corpus_max_fy(UNDERSTANDING / "facts.db")


def render(result: dict, show_plan: bool = False) -> str:
    lines = []

    if show_plan:
        plan = result["plan"]
        lines.append(f"PLAN  intent={plan['intent']} "
                     f"metrics={','.join(plan['metrics']) or '-'} "
                     f"periods={','.join(plan['periods'])} "
                     f"tags={','.join(plan['tags']) or '-'}")
        lines.append("")

    if result["allowed"]:
        lines.append(result["answer"])
        if result["citations"]:
            lines.append("")
            lines.append("Sources: " + "; ".join(result["citations"][:6]))
        if result.get("source") == "llm":
            lines.append("(phrased by the model; figures computed "
                         "deterministically)")
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


def repl(gate: AccessGate, roles: dict[str, Role],
         show_plan: bool = False, use_llm: bool = True) -> None:
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
            gate = AccessGate(roles[name], gate.corpus_max_fy)
            print(f"Switched to {name}.")
            continue
        print()
        print(render(answer(line, gate, use_llm=use_llm), show_plan=show_plan))


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
    parser.add_argument("--plan", action="store_true",
                        help="show the query plan the guard judged")
    parser.add_argument("--no-llm", action="store_true",
                        help="deterministic answers only, no model call")
    args = parser.parse_args(argv)

    if args.roles:
        print(describe_roles(roles))
        return 0

    gate = AccessGate(roles[args.role], corpus_anchor())

    if args.ask:
        result = answer(args.ask, gate, use_llm=not args.no_llm)
        print(render(result, show_plan=args.plan))
        # Non-zero exit on refusal so the demo script can assert on it.
        return 0 if result["allowed"] else 3

    repl(gate, roles, show_plan=args.plan, use_llm=not args.no_llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
