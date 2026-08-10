# Flow

How execution actually travels, file by file. Bugs live in the gaps between
files, so this traces the two paths that matter: **build time** (once) and
**query time** (per question).

---

## Build time

`python scripts/build_understanding.py`

```
scripts/build_understanding.py
   └─► src/understanding/build.py :: build_all()
         │
         ├─► load_config("config/sources.yaml")
         │
         ├─► collect_facts()
         │     ├─► ingest/excel_loader.py :: load_statement_workbook()   ×7
         │     │     ├─ _period_columns()   header block → period columns
         │     │     ├─ parse_value()       "$ (1,234)" → -1234.0
         │     │     └─ scope tracking      "iPhone" → iphone_net_sales
         │     └─► ingest/excel_loader.py :: load_headcount_workbook()   ×1
         │           └─ also emits the company-wide `headcount` total
         │
         ├─► ingest/chunker.py :: chunk_corpus()
         │     └─► chunk_document()  ×9  (8 filings + 1 poisoned fixture)
         │           ├─ pdf_loader.load_pdf()          page text, 1-based
         │           ├─ tagger.tag_for_heading()       heading → Tag
         │           ├─ tagger.allowed_tags_for_document()   bounds the tag space
         │           └─ sanitize.detect()              → quarantined=True
         │
         ├─► understanding/facts.py :: build_facts_db()      → facts.db
         ├─► understanding/index.py :: BM25Index.build()     → index/bm25.json
         ├─► write_summaries()                                → summaries/*.json
         ├─► write_metric_tags()                              → metric_tags.json
         └─► write_schema_notes()                             → schema_notes.md
```

**Where it can fail:** an unreadable workbook is logged and skipped; a page that
will not extract yields empty text rather than aborting the document; a malformed
row is skipped with its locator logged. Nothing here raises. The one place that
*does* raise is `scripts/make_synthetic_hr.py`, which refuses to write synthetic
totals that disagree with the figures the 10-K states.

---

## Query time

`answer("What is revenue per employee for FY2025?", gate)`

```
src/agent/loop.py :: answer()
   │
   │ ── 1. PLAN ─────────────────────────────────────────────────────
   ├─► planner.Planner.plan(question)
   │     ├─ find_metrics()      aliases longest-first, consumed as matched
   │     ├─ find_periods()      "FY2025" / "Q3FY2026" / default to corpus max
   │     ├─ find_topic_tags()   what the question is ABOUT
   │     └─ tags = union(metric tags, topic tags)     ← never from the question
   │        returns QueryPlan(intent, metrics, periods, tags)
   │
   │ ── 2. GUARD ────────────────────────────────────────────────────
   ├─► access/guard.py :: guard_plan(plan, gate)
   │     ├─ gate.check_tag()     for every declared tag
   │     ├─ gate.check_period()  for every declared period
   │     └─ Decision(allowed, reason, denied_tags, denied_periods)
   │
   ├─► access/audit.py :: audit(decision, ...)        ← BEFORE the branch,
   │                                                    so denials are logged
   │                                                    exactly like allows
   │
   │   ┌── if NOT allowed ──────────────────────────────────────────┐
   │   │  return {allowed: False, reason, denied_tags, plan}        │
   │   │  NOTHING BELOW THIS LINE RUNS. No query, no retrieval,     │
   │   │  no prompt. The restricted operand is never read.          │
   │   └────────────────────────────────────────────────────────────┘
   │
   │ ── 3. RETRIEVE ─────────────────────────────────────────────────
   ├─► agent/tools.py :: query_metrics(metrics, periods, gate=gate)
   │     ├─ _tags_of()                unknown metric → maximally restricted
   │     ├─ understanding/facts.py :: query_facts()
   │     │     └─ gate.sql_predicate() → bound into WHERE   ← filter in the DB
   │     ├─ _dedupe()                  same figure, several locators
   │     └─ audit()
   │
   ├─► agent/tools.py :: search_filings(question, gate=gate)
   │     ├─ BM25Index.load()
   │     ├─ index.search(query, gate, k*3)
   │     │     └─ gate.filter_chunks()  ← BEFORE scoring; drops quarantined
   │     │                                 material for every role
   │     ├─ feedback/rerank.py :: apply()   ← AFTER the gate; can only reorder
   │     │     └─ feedback/store.py :: similar()   scoped by token overlap
   │     ├─ truncate to k
   │     └─ audit()
   │
   │ ── 4. COMPOSE ──────────────────────────────────────────────────
   ├─► _compose_deterministic(figures, passages, plan)   ← the arithmetic
   │
   └─► if use_llm:
         ├─► build_context()      the ONLY place a prompt is assembled
         │     ├─ <figure> blocks from gated query_metrics
         │     ├─ <document> blocks from gated search_filings
         │     └─ <user_correction> blocks, labelled, placed last
         └─► llm.complete(SYSTEM_PROMPT, context)
               └─ returns None on ANY failure → deterministic text stands
```

**The load-bearing property:** the guard runs at step 2, before any of step 3.
Restricted data is not fetched-then-filtered — it is never fetched. Everything
downstream operates on a corpus the gate has already reduced.

---

## Where the two interfaces join

```
src/cli.py :: main()                    src/api.py :: ask()
   ├─ load_roles()                         ├─ load_roles()  (module level)
   ├─ corpus_anchor()                      ├─ _max_fy()
   ├─ AccessGate(role, max_fy)             ├─ _gate(role)
   └─► loop.answer()  ◄────────────────────┘
                                           └─ adds `stages` for the pipeline UI
```

Neither interface holds access logic. The API adds exactly one thing: a
`stages` dict so the console can draw which steps ran.

---

## The feedback cycle

```
  answer()  ──►  passages[].id
                     │
   user votes        ▼
  ──────────►  feedback/store.py :: record()      → data/feedback.db
                     │
   next similar      ▼
   question    feedback/store.py :: similar()      token overlap ≥ 0.3
                     │
                     ├─► rerank.adjustments()      chunk_id → multiplier
                     │      down ×0.5   up ×1.5    compounding
                     │
                     └─► rerank.corrections_for()  → <user_correction> in prompt
```

Feedback enters at retrieval, **after** the gate. It reorders what a role may
already see and can never add to it.

---

## Reading the audit log

`data/audit.log`, one JSON object per line:

```json
{"ts": "...", "role": "CTO", "context": "plan[mixed] metrics=headcount,net_sales
 periods=FY2025 tags=hr.headcount,...", "allowed": false,
 "reason": "Refused for CTO — restricted data: hr.headcount. ...",
 "denied_tags": ["hr.headcount"], "denied_periods": []}
```

A permitted request produces several lines — one for the plan decision, then one
per tool call it authorised. A refusal produces exactly one, which is itself the
evidence that nothing downstream ran.
