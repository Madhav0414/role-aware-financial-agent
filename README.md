# RBAC-Enforced Financial Data Agent

Ask questions in natural language about Apple's public SEC filings. Answers
respect **who is asking**, enforced at the data layer — and the system refuses
questions whose answer would leak restricted data by arithmetic, even when the
restricted value is never printed.

Built for the Azentio AI Agent Developer assignment.

```
CEO      "What was net sales in FY2025?"      →  $416,161 million  [10-K_FY2025 p.30]
CTO      "What is revenue per employee?"      →  REFUSED · hr.headcount
ANALYST  "What was net sales in FY2023?"      →  REFUSED · outside permitted period
```

The same question. The same corpus. Different answers, because the **data
layer** decided — not the prompt.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_understanding.py     # ~3 seconds
python scripts/demo.py                    # the whole system in five acts
```

Then either interface:

```bash
python scripts/serve.py                   # web console at http://127.0.0.1:8000
python -m src.cli --role CTO              # interactive terminal
python -m src.cli --role CEO --ask "What was net sales in FY2025?"
```

**No API key is required.** Every figure and every access decision is computed
in Python. A language model, if configured, only rewords the final answer — see
[Running without a key](#running-without-a-key).

Run the tests:

```bash
pytest                                    # 221 tests, ~8 seconds
```

---

## What it does

| Requirement | Where it lives |
|---|---|
| Ingest `.xlsx`/`.csv` and `.pdf` | `src/ingest/` — 4,685 facts, 737 chunks from 467 pages |
| Understanding files | `data/understanding/` — facts DB, BM25 index, summaries, schema notes |
| RBAC at the data layer | `src/access/gate.py` — one choke point, two enforcement surfaces |
| No leakage when combining sources | `src/access/guard.py` — the derivation guard |
| Feedback that changes behaviour | `src/feedback/` — re-ranking + correction memory |
| Natural-language interface | `src/cli.py`, `src/web/index.html` |
| **Bonus:** prompt injection | `src/agent/sanitize.py` — quarantine at ingest **and** sanitised user input |

---

## What you can ask

Roughly 560 metrics across 16 periods (FY2021–FY2026 and quarters), plus the
narrative sections of every filing.

| Category | Example questions |
|---|---|
| Income statement | net sales · gross margin · operating income · net income · cost of sales · R&D · SG&A · income taxes |
| Per share | `What was diluted EPS in FY2024?` |
| Balance sheet | total assets · total liabilities · cash · inventories · accounts payable · term debt |
| Cash flow | share repurchases · dividends · depreciation and amortization |
| Product lines | `What was iPhone revenue in FY2025?` · Mac · iPad · Services · Wearables |
| Geography | `What was Greater China revenue in FY2024?` · Americas · Europe · Japan |
| Quarters | `What was net sales in Q3FY2026?` |
| **Trends** | `How did revenue change over the years?` → three years **and the computed change** |
| Narrative | risk factors · governance · management's discussion · `Where is the company headquarters?` |
| Restricted | headcount · executive compensation — *answer depends on your role* |

Questions the corpus cannot answer are **declined rather than guessed at**:

```
"What was profit in 2026"        → I do not hold Net income for FY2026.
                                    Available periods: FY2025, FY2024, … Q3FY2026.
"What is the market valuation?"  → does not match a reported figure
"gross margin for products"      → gives the consolidated figure AND says the
                                    products split is not in the statement tables
```

That last one matters: the question was *narrowed* and the system says so
instead of silently answering the broader one.

---

## The architecture in one line

> **Deterministic core, LLM at the edge.**

Every figure, every access decision and every refusal is produced by ordinary
Python. The model interprets phrasing and words the answer. That is what makes
the system testable, reproducible, and able to run with no key at all.

```
  PDF + XLSX ──► ingest ──► understanding ──► ACCESS GATE ──► agent ──► CLI / web
                (tagged)     facts.db            ▲              tools
                             bm25.json           │
                                            feedback store
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
execution path: [`docs/FLOW.md`](docs/FLOW.md) ·
every choice and its rejected alternative: [`docs/DECISIONS.md`](docs/DECISIONS.md)

---

## The three roles

Defined in [`config/roles.yaml`](config/roles.yaml) — configuration, not code.

| Role | May read | Restricted by |
|---|---|---|
| **CEO** | everything | — |
| **CTO** | everything except `hr.headcount`, `hr.compensation` | tag |
| **ANALYST** | `financials.*` only, newest two fiscal years only | tag **and** time |

ANALYST is restricted on two orthogonal dimensions deliberately: it demonstrates
a real permission model rather than a hard-coded conditional, and it produces a
structurally different refusal from the CTO's.

---

## The interesting problem

Tag filtering alone is not enough. A user denied one operand can still recover
it from a permitted one plus a derived result:

```
revenue per employee  =  revenue (permitted)  ÷  headcount (restricted)
```

Nothing prints the headcount, and the headcount is disclosed anyway.

So the agent **declares a query plan before touching anything**, and plain
Python validates that declaration:

```
PLAN   intent=mixed  metrics=headcount,net_sales  periods=FY2025
       tags=hr.headcount, financials.statements, financials.segment
GUARD  denied: hr.headcount
       → refused before a single tool ran
```

Refusal is all-or-nothing. Executing the permitted half and withholding the
rest is precisely what leaks.

---

## Running without a key

The system answers every factual question with no API key configured. Set one
only if you want the prose reworded by a model:

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-5   # optional override
```

`complete()` returns `None` on *any* failure — missing key, missing SDK,
network error, rate limit, wrong model id — and the caller always falls back to
deterministic formatting.

This stopped being hypothetical during the build: the configured key turned out
to have no credit balance, every call returned HTTP 400, and the system answered
every question correctly throughout. The only visible difference was plainer
prose. See `D47` in [`docs/DECISIONS.md`](docs/DECISIONS.md).

Use `--no-llm` to force the deterministic path explicitly.

---

## Data

All from **SEC EDGAR**, the authoritative source. Apple's own investor site
returns 403 to programmatic clients; EDGAR permits automated access with an
identifying user agent.

| Document | Count | Purpose |
|---|---|---|
| 10-K (FY2023–FY2025) | 3 | Narrative, risk factors, human capital |
| 10-Q | 4 | Quarterly statements |
| DEF 14A proxy | 1 | **Real** executive compensation tables |
| Statement workbooks | 7 | Structured figures |

Everything is committed under `data/raw/`, so the repository runs without
network access. To refresh:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python scripts/fetch_data.py
```

EDGAR serves filings as HTML, so `fetch_data.py` renders them to PDF locally
with Edge headless, then verifies each has a real text layer. SEC stopped
publishing `Financial_Report.xlsx` for filings from 2025 onward, so those
statement tables are rebuilt from EDGAR's own `R*.htm` XBRL renderings — the
same source SEC generated the spreadsheet from, validated against FY2024 where
both versions exist.

### Synthetic data — full disclosure

Two files under `data/raw/_synthetic/` are fabricated. Nothing else is.

**`headcount_by_department.xlsx`** — a departmental headcount split. Apple
publishes only a company-wide figure, which is too coarse to demonstrate the
derivation leak. The departments sum exactly to the number each 10-K states on
page 8 (161,000 / 164,000 / 166,000), and the generator *raises* rather than
writing a file that fails to reconcile. Row 1 carries a banner declaring it
fabricated.

**`poisoned_supplier_report.pdf`** — a deliberately malicious document for
testing prompt-injection defences, labelled on its own first page. Generated by
`scripts/make_injection_fixture.py`.

Executive compensation is **not** synthetic — it comes from the real proxy
statement.

---

## Scope decisions

Stated plainly rather than left to be discovered:

- **Identity is asserted, not authenticated.** The brief asks for enforcement
  at the data layer, so the time went into making that layer unbypassable
  rather than into a login screen. Swapping in real auth changes only which
  `Role` object gets constructed. The server binds `127.0.0.1` for this reason.
- **No persistence beyond SQLite**, no multi-tenancy, no multi-company.
- **Two known data-quality limits**, both recorded with their fixes:
  306 of 2,561 metric/period keys remain ambiguous inside footnote tables
  (`D37`), and contents-page entries mis-tag two proxy pages (`D42`, superseded
  by `D43`).

---

## Where it breaks at 100×

The in-memory index and SQLite's single writer go first; both become a
server-backed store with the permission predicate pushed down into the
database, because post-retrieval filtering leaks through result counts and
latency. Whole-corpus re-embedding must become incremental and content-hash
keyed. Feedback re-ranking is a linear scan needing its own index, and per-user
correction memory needs TTL and decay. PDF parsing is the throughput ceiling at
seconds per page, so it becomes a queue with workers.

The quiet one is the tag taxonomy: it is hand-maintained, and at 100× the
documents its coverage rots silently while untagged data defaults into
*visible*.

---

## Repository layout

```
config/          roles.yaml, sources.yaml — the permission model and tag rules
data/raw/        committed filings (+ _synthetic/)
data/understanding/   generated artifacts (gitignored, rebuilt in 3s)
src/access/      model, gate, guard, audit  ← the differentiator
src/ingest/      pdf_loader, excel_loader, tagger, chunker
src/understanding/    facts, index, build
src/agent/       planner, tools, loop, llm, sanitize
src/feedback/    store, rerank
src/web/         the console
scripts/         fetch_data, build_understanding, demo, serve, make_*
tests/           221 tests
docs/            ARCHITECTURE, FLOW, DECISIONS, TECH_INVENTORY
```

---

## The test that matters most

`tests/test_agent.py::test_restricted_text_never_enters_the_assembled_prompt`

It inspects the **assembled prompt**, not the answer. That is the difference
between *"the model declined to say it"* and *"the model never had it"* — and
it is the entire claim of this system. A companion test asserts the same
material *is* reachable as CEO, so the negative test cannot pass vacuously.
