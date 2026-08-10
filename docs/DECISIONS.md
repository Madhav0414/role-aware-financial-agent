# Decisions

Every meaningful choice made while building this system, with the reasoning
behind it. Code shows *what* changed; this file shows *why*.

Read the **REJECTED** line in each box first — the alternative not taken is
usually what an interviewer asks about.

Model used throughout: **Claude Opus 5**. Recorded because behaviour differs
between models, and six months from now that matters when tracing a decision.

---

## Part 1 — Data acquisition

### D1 · Take the filings from SEC EDGAR, not Apple's investor site

**WHY** — `investor.apple.com` returns **403 Forbidden** to every programmatic
client, including one sending a full browser user-agent. It sits behind bot
protection. EDGAR is the authoritative origin of the identical documents and
explicitly permits automated access when the request identifies the requester.

**HOW** — `data.sec.gov/submissions/CIK0000320193.json` lists every filing;
each accession number resolves to a document directory under `/Archives/`.
Requests carry `User-Agent: Madhav Sharma <email>` as SEC requires, throttled
to one every 0.2s against their 10-per-second limit.

**REJECTED** — Scraping Apple's site with a headless browser to defeat the bot
check. It would work, and it would be the wrong instinct: the data has an
official machine-readable source, and defeating an access control to obtain
public data is a poor thing to demonstrate in an assignment about access
control.

---

### D2 · Render the HTML filings to PDF locally

**WHY** — The brief requires PDF ingestion. EDGAR serves filings as HTML.

**HOW** — Edge headless (`--headless=new --print-to-pdf`) against a local
`file:///` copy, with an isolated `--user-data-dir` so it never collides with a
running Edge. Every output is then opened with PyMuPDF and checked for a text
layer; **zero extracted characters fails the run** rather than silently
producing an empty index.

**REJECTED** — Hunting for pre-made PDFs elsewhere on the web. Provenance would
be unverifiable, and a scanned copy would have no text layer at all.

---

### D3 · Rebuild the missing spreadsheets from EDGAR's XBRL renderings

**WHY** — SEC published `Financial_Report.xlsx` for the FY2023 and FY2024
filings but **stopped for filings from 2025 onward** — those URLs 404. Without
a fix, the Excel side of the corpus would have covered only old years.

**HOW** — Each filing directory still contains `R*.htm`: EDGAR's own rendered
XBRL statement tables, the same source SEC generated the spreadsheet from.
These are parsed with `pandas.read_html` and written to `.xlsx`. Validated by
comparing against FY2024, where both the published and rebuilt versions exist —
same sheet structure, same figures.

**REJECTED** — Shipping only the two real spreadsheets. It would have left the
most recent year, the one most likely to be queried, without structured data.

---

### D4 · Read the R-file list from the filing index instead of guessing a range

**WHY** — A first pass hardcoded `R2`–`R11`, which captured **10 tables where a
10-K publishes about 70**. The corpus looked complete and was not. Caught only
by auditing sheet counts against the SEC-published years.

**HOW** — `index.json` in each filing directory lists every file; R-files are
matched by regex and sorted numerically. FY2025 went from 10 sheets to 70,
against SEC's own 73 for FY2024.

**REJECTED** — Widening the hardcoded range to `R2`–`R80`. It would have worked
today and broken silently on any filing shaped differently. Guessing a bound is
how the bug happened in the first place.

---

### D5 · Make the synthetic headcount reconcile with the real filings

**WHY** — Exactly one file in this project is fabricated: the departmental
headcount split, which exists only so a *derived* figure can leak a restricted
value. Fabricated data that contradicts the real filings beside it would be a
trap for anyone reading the repo.

**HOW** — Departmental totals sum to the company-wide figures Apple actually
states on page 8 of each 10-K: **161,000 (FY2023) · 164,000 (FY2024) ·
166,000 (FY2025)**. The generator **raises** rather than writing a file that
fails to reconcile. Row 1 of the sheet carries a banner declaring it
fabricated.

**REJECTED** — Fabricating compensation data too. Unnecessary — the DEF 14A
proxy contains real, public executive compensation tables, so only the
departmental split had to be invented.

---

## Part 2 — Architecture

### D6 · Deterministic core, LLM at the edge

**WHY** — Every figure, every access decision and every refusal must be
reproducible and testable. A regulator, or an evaluator, cannot audit a
number that a model might phrase differently on the next run.

**HOW** — Ordinary Python computes facts and decides access. The model only
interprets the question and phrases the answer. A consequence worth stating:
**the system answers every factual question with no API key at all.**

**REJECTED** — Letting the agent reason freely over raw documents. More
impressive to demo once, impossible to test, and it would place the access
decision inside a model's judgement.

---

### D7 · Retrieval defaults to BM25, not neural embeddings

**WHY** — Financial questions are keyword-heavy ("net income FY2024"), which is
BM25's strength. Dense embeddings would need either an API key or a model
download — the single largest overnight dependency risk — and vector search is
weak at exactly the thing this corpus is made of: near-identical rows that
differ only by fiscal year.

**HOW** — Okapi BM25 (`k1=1.5`, `b=0.75`) over lowercased tokens, ~40 lines of
numpy, persisted as JSON. Dense embeddings remain an optional upgrade when
`OPENAI_API_KEY` is present.

**REJECTED** — `sentence-transformers`. A multi-hundred-megabyte download at
2 a.m. on a 24-hour clock, to make exact-figure retrieval *worse*.

---

### D8 · Exact figures come from SQL, narrative comes from the index

**WHY** — Ask a vector index for "FY2023 net income" and it will happily return
the FY2022 row, because those two passages are nearly identical text. Numbers
must be exact and citable.

**HOW** — Two retrieval paths. Structured metrics land in a typed SQLite facts
table queried by SQL; narrative passages land in the BM25 index. A planner
routes each question to the right one.

**REJECTED** — One unified vector store for everything. Simpler to build, and
it would fail the first question an evaluator asks.

---

### D9 · Build a walking skeleton before deepening any layer

**WHY** — 50% of the grade is "does it run end to end". Building layer by layer
means the system answers nothing until the final hours; if anything slips,
there is nothing to submit.

**HOW** — Tasks 1–3 (pure logic) then a thin end-to-end path — three hardcoded
facts, a keyword planner, string formatting — wired through the real gate and
guard. **From that commit onward there is always something submittable**, and
every later task improves a working system rather than building toward one.

**REJECTED** — Finishing ingestion properly first. Cleanest code, worst risk
profile against the criterion that carries half the marks.

---

## Part 3 — The access layer

### D10 · Roles live in YAML, not in code

**WHY** — A permission model is configuration. Changing who may see payroll
should not require editing enforcement logic, and the two have completely
different review requirements.

**HOW** — `config/roles.yaml` declares tags and time windows per role;
`load_roles()` resolves them into frozen `Role` objects. The gate reads roles
and never hardcodes a name.

**REJECTED** — An `if role == "CTO"` chain. It works for three roles and rots
at ten, and it makes the permission model unreadable to anyone who is not a
Python programmer.

---

### D11 · Subtract denials at load time, producing one resolved set

**WHY** — Holding `allowed` and `denied` separately means every check must
consult both, in the right order. That ordering is a bug waiting to happen, and
it doubles what the tests have to cover.

**HOW** — `allowed_tags = allowed - denied`, computed once in `load_roles`.
The gate then tests one set membership. `CTO` is declared as `["*"]` minus the
two HR tags, so it stays broad by default and the denial is surgical.

**REJECTED** — Evaluating deny rules at request time. More flexible, more
expensive, more ways to be wrong.

---

### D12 · `Role` is frozen

**WHY** — A role is passed to every gate and every tool in a request. If a
caller could mutate it, an access decision would become shared mutable state
and the audit log would stop being trustworthy.

**HOW** — `@dataclass(frozen=True)`, with `frozenset` for the tag set. A test
asserts that assignment raises `FrozenInstanceError`.

**REJECTED** — A plain dict. Convenient, and it would let any code anywhere
grant itself a permission.

---

### D13 · `Tag` subclasses `str`

**WHY** — Tags cross into SQL as bound parameters and into JSON for the audit
log. A plain `Enum` would need unwrapping at every boundary, and one forgotten
`.value` is a silent mismatch that fails open.

**HOW** — `class Tag(str, Enum)`, so a Tag compares equal to its own string and
binds directly.

**REJECTED** — Bare string constants. No typo protection — `"hr.headcout"`
would parse fine and quietly match nothing.

---

### D14 · A denial is a returned `Decision`, never an exception or an empty list

**WHY** — The caller must be able to distinguish **"you may not see this"** from
**"there is no such data"**. Collapsing the two is how a system leaks by
implication, and it makes refusals impossible to audit.

**HOW** — Every check returns `Decision(allowed, reason, denied_tags,
denied_periods)`. The reason is written for a human reading the log. Every
decision, allow or deny, is appended to `data/audit.log`.

**REJECTED** — Raising `PermissionError`. Exceptions get caught and swallowed
by a retry somewhere upstream, and an empty result looks identical to no data.

---

### D15 · The gate does not know what a chunk is

**WHY** — The access layer must not depend on the ingest layer. If it did, the
dependency would eventually run both ways, and the one module that must stay
readable on a single screen would become the one that imports everything.

**HOW** — `filter_chunks` requires only `.tag`, `.fiscal_year` and
`.quarantined`. Both `Chunk` and `Fact` satisfy that without being imported.
The tests prove it by passing a four-line stub.

**REJECTED** — Importing `Chunk` into `gate.py`. One line today, a circular
import by Task 6.

---

### D16 · The permission filter is pushed into SQL, not applied afterwards

**WHY** — Filtering after retrieval means the restricted rows were already
selected. Even if they are dropped before display, the system has leaked
through **result counts and response latency** — both observable side channels.

**HOW** — `sql_predicate()` returns a parameterised `WHERE` fragment plus its
bound values, composed into the query so the database never returns a
restricted row. Parameterised rather than formatted, so a tag can never carry
SQL.

**REJECTED** — `[r for r in rows if allowed(r)]`. Reads fine, and it is the
difference between real enforcement and a display filter.

---

### D17 · The time window is anchored to the corpus, not the calendar

**WHY** — "The two most recent fiscal years" measured against today's date
means the tests change behaviour as time passes and the suite rots.

**HOW** — `AccessGate(role, corpus_max_fy)`; the floor is
`corpus_max_fy - recent_years_only + 1`. With a corpus ending FY2026, ANALYST
sees FY2025–FY2026 and is refused FY2024.

**REJECTED** — `datetime.now().year`. A test that passes in August and fails in
October is worse than no test.

---

### D18 · Quarantine is not an access rule

**WHY** — A document carrying a prompt-injection attempt is unsafe for
*everyone*. Treating it as a permission would imply some role is senior enough
to be attacked.

**HOW** — `filter_chunks` drops `quarantined` records before any tag or period
check, for every role including CEO. A test asserts CEO is denied a quarantined
chunk while still reading restricted ones.

**REJECTED** — Giving CEO an override. It reads like seniority and it is a
vulnerability.

---

### D19 · Normalise whitespace before matching text pulled from a PDF

**WHY** — Found the hard way: Apple's 10-K says "full-time equivalent
employees", but PDF extraction splits it across the hyphen as **`full- time`**.
An exact-string probe reported the phrase as missing from a document that
plainly contains it.

**HOW** — All text matching in the tagger collapses runs of whitespace before
comparing. Layout artifacts are the norm in extracted PDF text, not the
exception.

**REJECTED** — Trusting the extracted string. The corpus check would have
quietly mis-tagged the headcount section, and the RBAC demo depends on that
section being tagged correctly.

---

## Part 4 — The derivation guard

### D20 · The agent declares a plan before it touches anything

**WHY** — The guard has to run somewhere. Checking the *output* is too late —
the restricted value has already been fetched, and by then you are asking a
model to redact its own answer. Checking the *plan* means the restricted
operand is never read at all.

**HOW** — `QueryPlan(intent, metrics, periods, tags)` is produced first and
validated by `guard_plan()` in ordinary Python. The model writes the plan; it
gets no vote on the verdict. The plan is frozen, so nothing can edit it between
approval and execution — an approved plan and an executed plan must be the
same object.

**REJECTED** — Scanning the finished answer for restricted terms. It catches
the word and misses the arithmetic, which is the entire attack.

---

### D21 · Refusal is all-or-nothing

**WHY** — This *is* the leak. Executing the permitted half of a mixed plan and
withholding the rest hands the user a ratio they can reverse. Revenue is
permitted, headcount is not, and revenue-per-employee discloses headcount by
division without ever printing it.

**HOW** — If any tag or period in the plan is denied, the whole plan is
refused. No partial execution, no "here's what I can tell you" that quietly
completes the equation.

**REJECTED** — Answering with the permitted portion and noting the omission.
Friendlier, and it defeats the control it appears to respect. This was one of
the four questions emailed to Azentio; hard refusal is the stated default.

---

### D22 · Collect every violation instead of returning on the first

**WHY** — ANALYST is restricted on two dimensions at once. A refusal naming
only the tag invites a retry that then trips over the time window — two
round-trips to learn what one message could have said.

**HOW** — Tag violations and period violations are gathered independently and
reported together. `test_analyst_refusal_names_both_causes` asserts it.

**REJECTED** — Early return. One line shorter, and it makes the refusal
message actively misleading about why the request failed.

---

### D23 · An unrecognised period label fails closed

**WHY** — A period the guard cannot parse is a period it cannot check. Skipping
it would let a malformed plan bypass the time window entirely — the exact shape
of a bug that only shows up when someone is probing for one.

**HOW** — `fiscal_year_of_period` raises `UnparseablePeriod`; the guard catches
it and adds the label to `denied_periods` rather than continuing. Fail closed,
always.

**REJECTED** — `continue` on a parse failure. Silent, permissive, and
untestable from the outside.

---

### D24 · Parse the fiscal year with a regex, not string slicing

**WHY** — The plan originally read the year off the last four characters.
That works for `FY2025` and quietly mis-parses nothing — until `Q3FY2026`
arrives, where the last four characters are still a valid-looking year but the
prefix changes what the label means.

**HOW** — `^(?:Q[1-4])?FY(\d{4})$`, tested against annual and quarterly labels
and against a junk string. Anything that does not match is refused per D23.

**REJECTED** — `int(period[-4:])`. Correct for today's corpus, wrong the moment
a quarterly label appears — which it does, since four 10-Qs are in the data.

---

## Part 5 — Secrets and disclosure

### D25 · No credential is ever read from anywhere but the environment

**WHY** — This repository gets shared with an employer and may end up public.
A key committed once is a key leaked permanently: rewriting history does not
un-publish it, and the only real remedy is rotation.

**HOW** — Keys are read from `os.environ` at the point of use and never
written to disk, never printed, and never included in the audit log. `.env` is
gitignored; `.env.example` ships the variable *names* with empty values so the
evaluator knows what is available without any secret existing in the tree. A
regex scan for `sk-*`, AWS keys, Slack tokens and PEM private-key headers over
all tracked files returns clean.

**REJECTED** — A `config.py` holding keys with a "don't commit this" comment.
That comment has never once worked.

---

### D26 · The SEC contact address comes from the environment too

**WHY** — SEC asks automated clients to identify themselves with a contact
address, so `fetch_data.py` needs one. It was hardcoded to a personal email —
not a credential, but personal data that does not belong in a repository sent
to a company.

**HOW** — `SEC_USER_AGENT` is read from the environment with a neutral,
non-identifying default. The corpus is already committed, so the evaluator
never needs to run the fetch script at all; the variable only matters when
refreshing the data.

**REJECTED** — Leaving it in as "just an email". Personal data in a shared
repository is a disclosure whether or not it unlocks anything.

---

### D27 · The audit log records decisions, not content

**WHY** — An audit log is written on every request and is the file most likely
to be shared while debugging. If it captured prompts or retrieved passages, it
would become a second, unguarded copy of exactly the restricted data the gate
exists to protect — and it would defeat the RBAC model from inside.

**HOW** — Each line holds timestamp, role, a short context label, the verdict,
the reason, and which tags or periods were denied. **No document text, no
prompt, no answer body.** `data/audit.log` is gitignored regardless.

**REJECTED** — Logging the full prompt for debuggability. It is genuinely
useful and it would make the log the most sensitive file in the project.

---

## Part 6 — The walking skeleton

### D28 · The order of operations is architecture, not scaffolding

**WHY** — The skeleton contains three stubs (hardcoded facts, a keyword
planner, string formatting) that later tasks replace entirely. What it must
*not* contain is a temporary control flow, because that is the thing later
tasks would inherit.

**HOW** — `plan → guard → refuse-or-fetch → compose → audit` is fixed from this
commit. Every stub is swapped out beneath it without the sequence changing, and
`answer()` keeps its signature, so `test_skeleton_e2e.py` keeps passing as the
real components land.

**REJECTED** — Fetching first and filtering after, "just for now". The whole
claim of this system is that restricted data is never fetched; a skeleton that
violated it would have to be rewritten rather than filled in.

---

### D29 · A refusal must explain its actual cause

**WHY** — Caught by running the demo, not by a test. Every refusal was emitting
the same sentence — *"a value derived from restricted data discloses that
data"* — including when ANALYST was refused purely because FY2023 falls outside
its window. That has nothing to do with derivation. A refusal that misstates
its own cause misleads whoever reads the audit log, which is the one artifact
that has to be trustworthy.

**HOW** — Three distinct rationales: derivation (some tags denied, some
permitted — the leak case), category (all tags denied — nothing is being
combined), and window (period only). Two tests now assert that a period-only or
wholly-restricted refusal never mentions derivation.

**REJECTED** — One generic message covering every case. Shorter, and it turns
the audit log into something you cannot reason from.

---

### D30 · The skeleton answers with real figures

**WHY** — Placeholder values like `revenue = 100` make a demo that proves
nothing and hide unit and formatting bugs until much later.

**HOW** — `SKELETON_FACTS` carries the actual numbers from the committed
filings — net sales $416,161M / $391,035M / $383,285M, headcount 166,000 /
164,000, each with its real page. Every answer the skeleton gives is true, and
`format_value()` had to handle millions versus people from the first commit.

**REJECTED** — Dummy data with a "replace me" comment. It would have deferred
the unit problem to Task 6 and made the skeleton undemonstrable in the
meantime.

---

### D31 · The CLI exits non-zero on a refusal

**WHY** — The demo script and any future CI check need to assert that a
refusal actually happened. Parsing stdout for the word "REFUSED" is brittle.

**HOW** — Exit code `3` when `allowed` is false, `0` otherwise. A refusal is a
correct outcome, so it is distinguished from a crash (`1`) as well as from
success.

**REJECTED** — Always exiting 0. It would make "the CTO was refused" untestable
from outside the process.

---

### D32 · Values are rendered as label-and-value, not as sentences

**WHY** — "Net sales for FY2025 **was** $416,161 million" — no single verb
agrees with both "net sales" (plural) and "headcount" (singular), and the
mismatch is visible in the demo.

**HOW** — `Net sales for FY2025: $416,161 million.` The unit travels with the
number via `format_value()`, since statements are reported in millions and
headcount in people.

**REJECTED** — Per-metric grammar rules. Real work, zero marks, and the LLM
replaces this phrasing in Task 8 anyway.

---

## Part 7 — Excel ingestion

### D33 · One loader handles both workbook shapes, without branching on filename

**WHY** — The corpus contains two layouts. SEC-published sheets put
"12 Months Ended" on one header row and the period end dates on the next, with
the duration spanning merged cells. The rebuilt sheets combine them into a
single header. Special-casing by filename would break the moment either format
changed.

**HOW** — Scan the first four rows as one header *block*, carry each duration
rightwards across the columns it spans, and treat any column resolving to a
date as a period column. Both layouts then collapse to the same structure.

**REJECTED** — Two loaders keyed on `xlsx_origin` from the manifest. It would
tie the parser to how the file was obtained rather than to how it is shaped.

---

### D34 · Only 12-month and 3-month columns are ingested

**WHY** — A 10-Q publishes "3 Months Ended" *and* "9 Months Ended" side by
side. The nine-month column is cumulative year-to-date. Ingesting both would
file two different meanings under one period label, and a comparison between
quarters would silently mix them.

**HOW** — `ACCEPTED_DURATIONS = {3, 12}`; anything else returns None from
`parse_period_header` and is skipped. A test asserts a nine-month header is
rejected.

**REJECTED** — Taking every duration and disambiguating later. The ambiguity
would already be baked into the stored data by then.

---

### D35 · Accounting notation is stripped before the sign is read

**WHY** — Caught by a failing test. Negatives are written `(24)`, and with a
currency prefix that becomes `$ (1,234)` — where the parenthesis is no longer
the first character. Testing for the sign first silently returned None, so
every negative figure with a currency symbol would have vanished from the
corpus rather than erroring.

**HOW** — Strip `$` and thousands separators first, *then* test for enclosing
parentheses. Parameterised tests cover bare integers, `$ 416,161`, `(24)` and
`$ (1,234)`.

**REJECTED** — A regex for the whole number format. Harder to read, and it
would have hidden the ordering bug rather than exposing it.

---

### D36 · Category headers scope the rows beneath them ⭐

**WHY** — This is the most consequential ingestion decision. XBRL renderings
nest breakdowns: a row reading **iPhone** with no figures, then a row reading
**Net sales** carrying 209,586. Flattened, `net_sales` for FY2025 resolved to
**twenty different values** — the real total of 416,161 alongside every product
line and region. "What was net sales in FY2025?" would have returned whichever
one happened to come first.

**HOW** — A labelled row with no value in any period column is a category
header and qualifies everything below it: `iphone_net_sales`,
`services_net_sales`. XBRL scaffolding rows (`[Line Items]`, `[Abstract]`) are
skipped so they cannot hijack the scope. A descriptive total closes the
category; a bare "Total" keeps the scope as a qualifier, because a footnote may
hold several sub-tables each ending in one.

Two tests lock it in: `net_sales` must resolve to exactly one value, and the
five product lines must sum to the consolidated total. If the scoping breaks,
the parts stop adding up to the whole.

**REJECTED** — Preferring the consolidated statement sheet and discarding
duplicates. It would have produced a correct headline figure and thrown away
the entire segment breakdown, which is real data the ANALYST role is meant to
read.

---

### D37 · Known limitation: ~12% of deep footnote metrics remain ambiguous

**WHY RECORD IT** — 306 of 2,561 metric/period keys still carry more than one
value, all inside footnote tables such as income-tax provisions and derivative
fair values, where several sub-tables repeat identical row labels. None of the
headline metrics are affected. Stating this is more useful than implying the
parser is perfect.

**HOW IT WOULD BE FIXED** — The rendered `R*.htm` tables are a *presentation*
of the underlying XBRL, and the XBRL element name (`us-gaap:RevenueFromContract
WithCustomerExcludingAssessedTax`) is globally unique by construction. Parsing
the XBRL instance document instead of its rendering removes label collisions
entirely. That is a larger change than the remaining time allowed, and it is
the first thing to do with more of it.

**REJECTED** — Hand-curating an alias map for the footnote metrics. It would
paper over the 306 known cases and do nothing for the next filing.

---

## Part 8 — PDF ingestion and tagging

### D38 · PyMuPDF for extraction

**WHY** — It needs no system-level dependency, where pdfplumber's rendering
path wants poppler installed. It is markedly faster across 467 pages. And its
reading order preserves section headings as their own lines, which is exactly
what the tagger depends on.

**HOW** — `load_pdf` yields `(page_number, text)` with 1-based pages, because
those numbers end up in citations a human reads. A page that fails to extract
is logged and yielded empty rather than aborting the document.

**REJECTED** — pdfplumber (system dependency, slower) and PyPDF2 (weaker text
extraction, effectively unmaintained).

---

### D39 · Section state carries across pages

**WHY** — A heading on page 8 governs prose that continues onto page 9.
Resetting per page would leave most of every section untagged and therefore
visible to roles that should not see it — the failure would be silent and would
default *open*.

**HOW** — The current tag persists across page boundaries until the next
heading. A heading also closes the pending chunk, so text never straddles a
section boundary: the gate makes one decision per chunk, and a chunk holding
two sensitivities cannot be governed.

**REJECTED** — Per-page tagging. Simpler, and it fails in the dangerous
direction.

---

### D40 · A heading must look like a heading ⭐

**WHY** — Found by reading the output, not by a failing test. Matching any
short line containing a pattern promoted prose fragments to headings, and
because a heading governs everything until the next one, a single false
positive mis-tagged pages of governance prose as restricted HR data.
`"appropriate, other employees"` became an `hr.headcount` section.

**HOW** — Three structural tests before any pattern is considered: the line
must not end mid-clause, must not contain a completed sentence, and the matched
pattern must account for at least half the line. "Summary Compensation Table"
scores 1.0; the fragment above scores 0.32. Numbered `Item 7.` headings bypass
the ratio, since SEC filings number their sections and carry long titles.

**REJECTED** — Requiring exact heading matches. Too brittle against extraction
artifacts, where a heading can arrive with stray spacing or a trailing page
number.

---

### D41 · The bare "employees" rule was deleted

**WHY** — A wrapped sentence ending `"employees."` is ten characters, so the
pattern accounted for 90% of the line and passed the ratio test. It became a
heading in the middle of the proxy's compensation discussion and tagged pages
of salary detail as `hr.headcount`.

**HOW** — Removed. `"human capital"` is the heading Apple actually uses, and
the employee count lives inside that section. After the change, `hr.headcount`
holds 7 chunks — all from 10-Ks, all carrying the real figures 166,000 /
164,000 / 161,000. A test asserts headcount chunks come only from annual
reports.

**REJECTED** — Raising the ratio threshold. It would have suppressed this case
and broken legitimate short headings like "Human Capital" at 1.0.

---

### D42 · Known limitation: table-of-contents entries mis-tag

**WHY RECORD IT** — A contents page lists "Executive Compensation" on a line of
its own, which is structurally identical to the real heading. Those two proxy
pages therefore carry section tags derived from the contents rather than the
content.

**IMPACT** — Small and in the safe direction: contents entries are tagged as
the restricted section they name, so they are *over*-restricted rather than
leaked. Real headings later in the document re-tag the body correctly.

**HOW IT WOULD BE FIXED** — Detect contents pages by density (many heading-like
lines and bare page numbers in close succession) and suppress heading updates
while inside one.

**REJECTED FOR NOW** — Not worth the remaining time when the failure mode
restricts too much rather than too little.

> **Superseded by D43.** The claim above — that heading mis-detection only
> over-restricts — turned out to be wrong, and running live retrieval proved
> it. See below.

---

## Part 9 — The understanding layer

### D43 · Document type constrains the tag space ⭐

**WHY** — Live retrieval showed ANALYST pulling DEF 14A pages tagged
`financials.statements`: the Ernst & Young ratification, security ownership
tables. The proxy mentions "financial statements" in its audit committee
report, a heading rule matched it, and proxy governance content landed under a
tag ANALYST is permitted to read. **That is a leak, not an over-restriction**,
and it directly contradicts what D42 assumed.

**HOW** — `config/sources.yaml` now declares `allowed_tags` per document type.
A heading may only move the tag *within* that document's legitimate range, so a
proxy statement can emit `governance`, `hr.compensation` or `narrative.risk`
and nothing else. Two tests: the proxy can never produce a financial tag, and
ANALYST retrieves **zero** chunks from it.

The principle generalises: heading matching is a heuristic and will sometimes
be wrong, so constrain the blast radius rather than trying to make the
heuristic perfect. A wrong tag now costs precision inside one document instead
of crossing a role boundary.

**REJECTED** — Adding more heading patterns to disambiguate. It would have
fixed this instance and left the class of bug intact.

---

### D44 · Repeated identical figures are collapsed at query time

**WHY** — Each 10-K restates two prior years, and a figure appears in both the
primary statement and its segment breakdown, so `net_sales` FY2025 is stored
four times with four locators. They agree — that is a consistency check passed,
not a conflict — but an answer citing the same number four times reads as a
bug.

**HOW** — `query_facts` collapses on `(metric, period, value, unit)`, keeping
the earliest-sorted locator so the choice is deterministic across runs. Passing
`dedupe=False` returns every occurrence, which is what an audit view wants.

**REJECTED** — Deduplicating during ingestion. The duplicates are evidence that
independent parts of the corpus agree; discarding them at write time would
throw away a signal worth keeping.

---

### D45 · BM25 keeps digits as tokens

**WHY** — Standard text pipelines strip numbers as noise. In this corpus they
are the most discriminating terms available: "2025", "166", "416" are exactly
what separates one otherwise near-identical passage from another.

**HOW** — `tokenize` matches `[a-z0-9]+`, keeping alphanumeric runs. No
stemming and no stopword list either — both would cost more than they return on
741 chunks, and BM25's IDF already discounts terms that appear everywhere.

**REJECTED** — A conventional NLP pipeline with stemming and stopwords. More
machinery, worse results on the queries this system actually receives.

---

### D46 · The facts database is rebuilt from scratch, never updated in place

**WHY** — An incrementally-updated database can drift from what the ingest
pipeline currently produces, and the drift is invisible: the data looks fine,
it is simply stale. On a corpus this size the rebuild costs three seconds.

**HOW** — `build_facts_db` drops and recreates the table. `scripts/build_
understanding.py` is safe to re-run at any time, which also means the evaluator
can regenerate every artifact from the committed raw files.

**REJECTED** — Incremental upserts keyed on a content hash. The right answer at
100× the data, and premature at this one.
