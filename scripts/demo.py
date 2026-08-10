"""End-to-end demonstration. Run this to see the whole system in ninety seconds.

    python scripts/demo.py

Four acts:

  1. The same question, three roles, three different outcomes.
  2. The derivation leak — refused before anything is fetched.
  3. Context isolation — what the model was actually given.
  4. Feedback changing behaviour, shown as a before/after diff.

Act 4 is the one the brief asks to be *shown*: storing feedback is not learning
from it, so the script asks the same question either side of a correction and
prints what changed.

Uses a scratch feedback database so the demo is repeatable and leaves no state.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from src.access.gate import AccessGate
from src.access.model import load_roles
from src.agent.loop import answer, build_context
from src.feedback import store
from src.understanding.facts import corpus_max_fy

UND = Path("data/understanding")
ROLES = load_roles(Path("config/roles.yaml"))

WIDTH = 78


def rule(title: str = "") -> None:
    print("\n" + "=" * WIDTH)
    if title:
        print(title)
        print("=" * WIDTH)


def gate(role: str, max_fy: int) -> AccessGate:
    return AccessGate(ROLES[role], max_fy)


def show(result: dict, indent: str = "    ") -> None:
    plan = result["plan"]
    print(f"{indent}plan   : intent={plan['intent']} "
          f"metrics={','.join(plan['metrics']) or '-'} "
          f"periods={','.join(plan['periods'])}")
    print(f"{indent}tags   : {','.join(plan['tags']) or '-'}")
    if result["allowed"]:
        print(f"{indent}ANSWER : {result['answer'][:300]}")
        if result["citations"]:
            print(f"{indent}source : {result['citations'][0]}")
    else:
        print(f"{indent}REFUSED: {result['reason'][:260]}")


def act_one(max_fy: int) -> None:
    rule("ACT 1  One question, three roles")
    question = "What was net sales in FY2023?"
    print(f"\nQuestion: {question!r}\n")
    for role in ("CEO", "CTO", "ANALYST"):
        print(f"  [{role}]")
        show(answer(question, gate(role, max_fy), use_llm=False,
                    use_feedback=False))
        print()
    print("  ANALYST is refused on TIME, not on subject: financial statements")
    print("  are permitted, but only for the two most recent fiscal years.")


def act_two(max_fy: int) -> None:
    rule("ACT 2  The derivation leak")
    question = "What is revenue per employee for FY2025?"
    print(f"\nQuestion: {question!r}\n")
    for role in ("CEO", "CTO"):
        print(f"  [{role}]")
        show(answer(question, gate(role, max_fy), use_llm=False,
                    use_feedback=False))
        print()
    print("  The CTO may read revenue. The CTO may not read headcount.")
    print("  The RATIO would disclose headcount by division without ever")
    print("  printing it, so the plan is refused before a tool runs.")


def act_three(max_fy: int) -> None:
    rule("ACT 3  Context isolation — what the model was actually given")
    question = "executive compensation and salary of the chief executive"
    print(f"\nQuestion: {question!r}\n")

    for role in ("CEO", "CTO"):
        context = build_context(question, gate(role, max_fy),
                               understanding_dir=UND, use_feedback=False)
        leaked = [term for term in ("summary compensation", "stock awards",
                                    "salary")
                  if term in context.lower()]
        print(f"  [{role}] context = {len(context):,} chars; "
              f"restricted terms present: {leaked or 'NONE'}")

    print("\n  This inspects the PROMPT, not the answer. The difference between")
    print("  'the model declined to say it' and 'the model never had it'.")


def act_four(max_fy: int, feedback_db: Path) -> None:
    rule("ACT 4  Feedback changing behaviour")
    question = "What are the main risk factors facing the business?"
    role = "CEO"
    print(f"\nQuestion: {question!r}\n")

    before = answer(question, gate(role, max_fy), use_llm=False,
                    feedback_db=feedback_db)
    print("  BEFORE feedback — top passages:")
    for row in before["passages"][:3]:
        print(f"    {row['score']:>7.2f}  {row['citation']}")

    demoted = [row["id"] for row in before["passages"][:2]]
    store.record(role=role, question=question,
                 answer=before["answer"], verdict="down",
                 chunk_ids=demoted,
                 correction="Prioritise supply chain concentration and "
                            "manufacturing risk over generic legal language.",
                 db_path=feedback_db)
    print(f"\n  -> user down-votes the top {len(demoted)} passages "
          f"and writes a correction")

    after = answer(question, gate(role, max_fy), use_llm=False,
                   feedback_db=feedback_db)
    print("\n  AFTER feedback — top passages:")
    for row in after["passages"][:3]:
        marker = "  (demoted)" if row["id"] in demoted else ""
        print(f"    {row['score']:>7.2f}  {row['citation']}{marker}")

    changed = [r["citation"] for r in before["passages"][:3]] != \
              [r["citation"] for r in after["passages"][:3]]
    print(f"\n  ranking changed : {changed}")
    print(f"  reranked flag   : {after['reranked_by_feedback']}")
    print(f"  corrections now in prompt : {after['corrections_applied']}")
    print("\n  Storing feedback is not learning from it. This is the diff that")
    print("  shows the difference.")


def main() -> int:
    if not (UND / "facts.db").exists():
        print("Run `python scripts/build_understanding.py` first.",
              file=sys.stderr)
        return 1

    max_fy = corpus_max_fy(UND / "facts.db")
    scratch = Path(tempfile.mkdtemp(prefix="demo-feedback-"))
    feedback_db = scratch / "feedback.db"

    try:
        print("=" * WIDTH)
        print("RBAC-ENFORCED FINANCIAL DATA AGENT — DEMONSTRATION")
        print(f"corpus: Apple SEC filings, newest fiscal year FY{max_fy}")
        print("=" * WIDTH)

        act_one(max_fy)
        act_two(max_fy)
        act_three(max_fy)
        act_four(max_fy, feedback_db)

        rule("DONE")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
