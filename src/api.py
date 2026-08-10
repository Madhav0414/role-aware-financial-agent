"""HTTP API behind the web interface.

The API is a thin shell over the same functions the CLI calls. It holds no
access logic of its own — the gate is constructed per request from the asserted
role and handed to `answer()`, exactly as the CLI does. Adding a second
enforcement point here would mean two things to keep in agreement, and the one
that drifts is the one that leaks.

Role is asserted by the client, not authenticated. That is the same scope
decision the CLI makes and is stated plainly in the README: the brief asks for
enforcement at the data layer, so the effort went there rather than into a
login screen. Swapping in real authentication changes only how `Role` is
chosen here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.access.audit import read_audit
from src.access.gate import AccessGate
from src.access.model import Tag, load_roles
from src.agent import llm
from src.agent.loop import answer
from src.feedback import rerank, store
from src.understanding.facts import corpus_max_fy

ROOT = Path(__file__).resolve().parent.parent
UNDERSTANDING = ROOT / "data" / "understanding"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="RBAC Financial Data Agent", docs_url="/api/docs")

ROLES = load_roles(ROOT / "config" / "roles.yaml")


def _max_fy() -> int:
    return corpus_max_fy(UNDERSTANDING / "facts.db")


def _gate(role: str) -> AccessGate:
    if role not in ROLES:
        raise HTTPException(400, f"unknown role {role!r}")
    return AccessGate(ROLES[role], _max_fy())


class AskRequest(BaseModel):
    role: str
    question: str = Field(min_length=1, max_length=500)
    use_llm: bool = True


class FeedbackRequest(BaseModel):
    role: str
    question: str
    answer: str
    verdict: str
    chunk_ids: list[str] = []
    correction: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/context")
def context() -> dict:
    """Everything the page needs to render before a question is asked."""
    max_fy = _max_fy()
    every_tag = frozenset(Tag)
    roles = []
    for name, role in ROLES.items():
        gate = AccessGate(role, max_fy)
        roles.append({
            "name": name,
            "description": role.description,
            "allowed_tags": sorted(t.value for t in role.allowed_tags),
            "denied_tags": sorted(t.value for t in every_tag - role.allowed_tags),
            "min_fiscal_year": gate.min_permitted_fy(),
            "recent_years_only": role.recent_years_only,
        })

    return {
        "roles": roles,
        "corpus_max_fy": max_fy,
        "llm_provider": llm.active_provider(),
        "feedback_count": len(store.all_records()),
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    gate = _gate(request.role)
    result = answer(request.question, gate,
                    understanding_dir=UNDERSTANDING,
                    config_path=ROOT / "config" / "sources.yaml",
                    use_llm=request.use_llm)

    # The page renders the pipeline from this, so it needs to know which
    # stages actually ran. A refusal stops before retrieval, and showing that
    # visually is the whole point of the interface.
    result["stages"] = {
        "plan": True,
        "guard": True,
        "retrieve": result["allowed"],
        "compose": result["allowed"],
        "audit": True,
    }
    return result


@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest) -> dict:
    if request.verdict not in ("up", "down"):
        raise HTTPException(400, "verdict must be 'up' or 'down'")
    row_id = store.record(role=request.role, question=request.question,
                          answer=request.answer, verdict=request.verdict,
                          chunk_ids=request.chunk_ids,
                          correction=request.correction)
    return {"id": row_id, "total": len(store.all_records())}


@app.get("/api/feedback")
def list_feedback(limit: int = 20) -> dict:
    records = store.all_records()[-limit:]
    return {"rows": [{"id": r.id, "role": r.role, "verdict": r.verdict,
                      "question": r.question, "correction": r.correction,
                      "summary": rerank.summarise(r)}
                     for r in reversed(records)]}


@app.get("/api/audit")
def audit(limit: int = 30) -> dict:
    """The decision log. Contains verdicts and reasons, never document text."""
    return {"rows": list(reversed(read_audit()[-limit:]))}
