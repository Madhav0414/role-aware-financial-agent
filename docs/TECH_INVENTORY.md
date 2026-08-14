# Tech Inventory

Every dependency, why it is here, and what breaks without it. The rule: if a
library cannot earn a line in this table, it does not go in `requirements.txt`.

---

## Runtime dependencies

| Library | Why this one | What breaks without it |
|---|---|---|
| **PyMuPDF** (`fitz`) | PDF text with page-level provenance. Chosen over pdfplumber because it needs **no system dependency** — pdfplumber's rendering path wants poppler installed, which would make `pip install -r requirements.txt` insufficient on a clean machine. Also markedly faster across 467 pages, and its reading order keeps section headings on their own lines, which the tagger depends on. | No PDF ingestion at all. Roughly 737 chunks and the entire narrative corpus disappear; only spreadsheet figures remain. |
| **pandas** | Reads both spreadsheet layouts through one interface, and `read_html` parses EDGAR's `R*.htm` XBRL renderings for the filings where SEC no longer publishes a workbook. | No Excel ingestion, and the FY2025 and 10-Q statement tables cannot be rebuilt. 4,685 facts drop to zero. |
| **openpyxl** | pandas' `.xlsx` engine. Also writes the rebuilt workbooks and the synthetic headcount file. | pandas raises on every `.xlsx` read; ingestion fails at the first workbook. |
| **lxml** | The HTML table parser behind `pandas.read_html`. | `read_html` raises `ImportError`; the FY2025+ spreadsheets cannot be rebuilt from XBRL renderings. |
| **numpy** | BM25 scoring arithmetic. | Retrieval over narrative text fails; only exact numeric questions can be answered. |
| **PyYAML** | Reads `config/roles.yaml` and `config/sources.yaml` — the permission model and every tagging rule. | The system cannot start. Roles and tags are configuration, and there is no fallback in code by design. |
| **FastAPI** | The web console's HTTP layer. Type-validated request bodies via pydantic mean a malformed role or question is rejected at the boundary rather than deeper in. | The console cannot start. The CLI is unaffected — it is the primary interface and shares all logic. |
| **uvicorn** | ASGI server for FastAPI. | Same as above: console only. |

---

## Test dependencies

| Library | Why | What breaks without it |
|---|---|---|
| **pytest** | 221 tests, parameterised heavily — value parsing, period labels, injection patterns. | No test suite. Every claim in the docs becomes an assertion rather than a demonstration. |
| **httpx** | Required by FastAPI's `TestClient`. | `tests/test_api.py` is skipped; the other 211 tests still run. |

---

## Optional dependencies

Neither is installed by default. **The system answers every factual question
without them** — they change only how the final prose is worded.

| Library | Why | What breaks without it |
|---|---|---|
| **openai** | Used when `OPENAI_API_KEY` is set. | Nothing. `llm.complete()` catches `ImportError` and returns `None`; answers come from deterministic formatting. |
| **anthropic** | Used when `ANTHROPIC_API_KEY` is set. | Nothing, for the same reason. |

---

## Standard library, load-bearing

Worth naming, because these are deliberate choices and not incidental imports.

| Module | Role | Why not a library |
|---|---|---|
| `sqlite3` | `facts.db` and `feedback.db` | Ships with Python, needs no server, and the whole corpus is 4,685 rows. A real database is the 100× answer, not this one. |
| `re` | Tagging, period parsing, injection detection, tokenisation | The patterns are small and inspectable. A parser generator would obscure rules that must stay readable, because they decide sensitivity. |
| `dataclasses` | `Tag`, `Role`, `Decision`, `Chunk`, `Fact`, `QueryPlan` | `frozen=True` makes `Role` immutable, which is a security property: a role is passed to every tool in a request, and a mutable one would make an access decision shared mutable state. |
| `json` | Audit log (JSON Lines), BM25 index, summaries | The index stays inspectable with a text editor instead of being an opaque binary, which matters when a retrieval result needs explaining. |
| `subprocess` | Drives Edge headless for HTML→PDF | Avoids a heavyweight PDF-generation dependency for something the OS already ships. |

---

## Deliberately absent

Often the more interesting half of an inventory.

| Not used | Why not |
|---|---|
| **sentence-transformers / any embedding model** | A multi-hundred-megabyte download to make exact-figure retrieval *worse*. Passages differing only by fiscal year are near-identical in embedding space, so a question about FY2023 retrieves FY2022. BM25 keeps digits as tokens, which is what discriminates in this corpus — and it runs with no key. |
| **A vector database** (Chroma, FAISS, pgvector) | 737 chunks. An in-memory index with a JSON file is honest at this scale; naming the migration path is worth more than performing it. |
| **LangChain / LlamaIndex** | The interesting logic here is the access gate and the derivation guard. A framework would hide the retrieval path behind abstractions, and "explain every part of what you submit" is a stated ground rule of the assignment. |
| **An ORM** | Two tables. `sqlite3` with parameterised queries is less code and makes the pushed-down permission predicate visible in the SQL, which is the point. |
| **A frontend framework** | One page. React would add a build step to a repository whose brief says the interface may be minimal, for no gain the reviewer would see. |
| **An LLM-based classifier for tagging or injection** | It would put a model inside the security path — and a model is exactly what the attacker is targeting. Both are declarative rules in `config/sources.yaml` and `sanitize.py`. |
| **python-dotenv** | Three lines of `os.environ.get` cover it, and one fewer dependency touching secrets is the right trade. |

---

## Environment

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | no | Enables model-worded prose |
| `ANTHROPIC_API_KEY` | no | Same; checked second |
| `OPENAI_MODEL` | no | Override, default `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | no | Override, default `claude-sonnet-5` |
| `SEC_USER_AGENT` | no | Only for re-downloading filings; SEC asks automated clients to identify themselves |

No secret is read from anywhere but the environment, written to disk, logged, or
placed in the audit trail. `.env` is gitignored; `.env.example` ships the
variable names with empty values.

---

## Platform notes

- **Python 3.11+**. Uses `X | None` type syntax and `frozenset[Tag]` generics.
  Developed on 3.13.7.
- **Edge headless** is required only by `scripts/fetch_data.py` and
  `scripts/make_injection_fixture.py` — both regenerate committed artifacts. The
  system runs without it.
- **No poppler, no system packages.** `pip install -r requirements.txt` on a
  clean machine is sufficient to run everything.
