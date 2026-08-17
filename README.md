# AFIP — Enterprise Agentic Fraud Investigation Platform (v0.4)

A real, running multi-agent fraud investigation platform with a human-in-the-loop
approval gate before any SAR escalation, a ticketing workflow, and an analyst
console. **Live and verified end-to-end on Render + a real Neo4j Aura + Postgres
instance** — see "What's verified" below for exactly what's been tested against
real infrastructure, not just in-memory fallbacks.

## What's real here

- **ML detection**: XGBoost classifier on transaction features. `app/ml/fraud_model.py`
- **GraphRAG fraud ring detection**: real Neo4j (Aura, verified live) with an
  in-memory NetworkX fallback for local dev without a database. Detects mule rings
  via shared device/IP + circular transfer chains, with guardrails against
  shared-infrastructure false positives. Writes are batched (`UNWIND`), not
  one-row-per-query — this matters on a real cloud DB with network latency.
  `app/graph/graph_store.py`
- **Multi-agent pipeline** (LangGraph): Transaction → Customer → GraphRAG → Compliance →
  Supervisor, converging on a risk-scored decision with full evidence trail.
  `app/agents/investigation_graph.py`
- **Checkpointed, resumable state**: real Postgres (verified — survives an actual
  process kill/restart, not just a code review) via a persistent `ConnectionPool`,
  SQLite fallback for local dev.
- **Human-in-the-loop approval gate**: any case the Supervisor proposes to escalate to
  `ESCALATE_SAR` pauses the graph (`langgraph.types.interrupt`) and waits for an analyst
  to call `/case/{id}/approve`. Nothing files a SAR without a human approving it.
- **API key authentication**: every endpoint except `/health` requires an `X-API-Key`
  header matching the `API_KEY` env var. Verified: no key → 401, wrong key → 401,
  correct key → 200. `app/api/auth.py`
- **Persistent case storage**: real Postgres (`DATABASE_URL`), verified to survive
  restarts. In-memory fallback for local dev. `app/core/case_store.py`
- **Ticketing**: priority (auto-computed from risk score: LOW/MEDIUM/HIGH/CRITICAL),
  assignee, and threaded analyst comments on every case.
- **Case chat**: ask free-form questions about a specific case, answered from that
  case's actual data only (real Claude call with `ANTHROPIC_API_KEY`; without it,
  returns a clear "not configured" message rather than pretending to answer).
- **Analyst console** (`/`): a dashboard for the case queue, evidence trail,
  fraud-ring visualization, approve/reject actions, comments, and chat — single
  static file, no build step, served directly by the API. `app/static/index.html`
- **Real Claude SAR narration**, with a deterministic template fallback if
  `ANTHROPIC_API_KEY` isn't set. `app/agents/narration.py`
- **Langfuse tracing** on every agent node, no-op if not configured. `app/agents/tracing.py`
- **Admin endpoints** (`/admin/seed-graph`, `/admin/graph-status`) for on-demand,
  synchronous graph (re)seeding and direct inspection — no need to use the Neo4j
  console for routine checks.

## What's verified (actually run against real infra, not assumed)

- **Deployed live on Render**, running against a **real Neo4j Aura instance** and a
  **real Postgres database** — not just in-memory/SQLite fallbacks.
- Fraud ring detection confirmed correct against live Neo4j: `in_fraud_ring: true`,
  correct 5-account ring, each transfer edge appearing exactly once (no duplication).
- **Case persistence survives a real restart**: created a case, approved it, killed
  the process, restarted, fetched the same case — data fully intact.
- **Auth enforced live**: no/wrong `X-API-Key` → 401; correct key → 200; `/health`
  stays open (needed for platform health checks).
- Full HITL flow live: `/investigate` → `PENDING_APPROVAL` → `/approve` → `ESCALATE_SAR`
  → `/sar/{id}` → real narrative. Reject path verified too: downgrades to
  `MANUAL_REVIEW`, SAR generation correctly blocked (400).
- Ticketing endpoints (`PATCH /case/{id}`, `POST /case/{id}/comment`) verified live.
- Dashboard confirmed rendering correctly in a real browser against the live API
  (queue, evidence trail, priority/assignee editing, approve button).

## What's honestly NOT verified / not built

- **Docker image build is untested** — no Docker available in the environment this
  was built in. The Dockerfile is written correctly against the real dependency
  list, but hasn't been through an actual `docker build`. Render deploys directly
  from source (`--source .`-style), so this hasn't blocked deployment, but verify
  it yourself before using the Dockerfile elsewhere.
- **Chat has not been tested with a real `ANTHROPIC_API_KEY`** — the fallback path
  (no key set) is verified; the real-Claude path is written the same way as the
  already-verified SAR narration code, but hasn't been exercised live yet.
- **Langfuse tracing has not been tested with real credentials** — same situation:
  correctly implemented no-op fallback verified, real-tracing path unexercised.
- **No Kafka streaming** — transactions are a static synthetic batch, not a live feed.
- **AUC of 1.0 on synthetic data is a red flag, not a win** — the synthetic fraud is
  trivially separable. Re-validate on real or harder synthetic data before trusting this.
- **Single shared API key, not per-analyst auth** — fine for one person, wrong for a
  team. Anyone with the key can also wipe/reseed the graph via `/admin/seed-graph`.
  Add OAuth2/JWT + RBAC and lock down `/admin/*` separately before opening this up
  to multiple users.
- **Multi-worker caveat**: `WEB_CONCURRENCY`/`--workers > 1` was never tested. The
  graph-store contextvar pattern (`set_graph_store`) is per-process; multiple workers
  would each need their own graph connection, which should work but is unverified.
  Current Render deployment runs with `--workers 1`.
- **No dashboard button to start a new investigation yet** — new cases still need to
  go through `/investigate` via Swagger or curl; the dashboard is read/manage-only
  for now.

## Run locally

```bash
pip install -r requirements.txt
python3 -m uvicorn app.api.main:app --reload --port 8001
```

Open `http://localhost:8001/` for the dashboard, or `http://localhost:8001/docs` for
the API directly.

## Full flow, end to end

```bash
# 1. Investigate a known fraud-ring transaction (TXRING0000-0004 are always injected)
curl -X POST localhost:8001/investigate -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"transaction_id":"TXRING0000"}'
# -> status: "PENDING_APPROVAL", case_id in the response, priority auto-computed

# 2. Analyst approves (or rejects) the SAR escalation
curl -X POST localhost:8001/case/<case_id>/approve -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"decision":"approved"}'

# 3. Generate the SAR narrative
curl -X POST localhost:8001/sar/<case_id> -H "X-API-Key: <key>"

# 4. Ticketing: assign, prioritize, comment
curl -X PATCH localhost:8001/case/<case_id> -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"assignee":"analyst1","priority":"CRITICAL"}'
curl -X POST localhost:8001/case/<case_id>/comment -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"author":"analyst1","text":"Reviewing ring evidence."}'

# 5. Ask about the case
curl -X POST localhost:8001/case/<case_id>/chat -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"message":"Why was this escalated?"}'
```

## Deploy

Currently deployed on **Render**, building directly from source (Dockerfile-based).
Env vars used in production: `API_KEY`, `DATABASE_URL` (Postgres), `NEO4J_URI` /
`NEO4J_USER` / `NEO4J_PASSWORD` (Aura). `ANTHROPIC_API_KEY` and `LANGFUSE_*` are
supported but not yet set.

Copy `.env.example` to `.env` for the full list of what's configurable — everything
is optional; the app degrades gracefully for anything left blank.

## Recommended next steps, in order

1. **Add `ANTHROPIC_API_KEY`** — turns on real Claude chat and SAR narration; both
   are coded and pattern-matched against already-verified code, just need the key.
2. **Add a "new investigation" button to the dashboard** — right now it's Swagger-only.
3. **Re-validate the ML model on real transaction data** — the current AUC is
   inflated by trivially-separable synthetic fraud.
4. **Add per-analyst auth (OAuth2/JWT + RBAC)** before more than one person uses this,
   and lock down `/admin/*` behind a separate, more privileged credential.
5. **Swap synthetic batch data for a real feed** (Kafka or even a CSV export first).

## Project structure

```
app/
  core/synthetic_data.py        # dev/test data generator
  core/case_store.py            # Postgres-backed case persistence
  ml/fraud_model.py             # XGBoost detection model
  graph/graph_store.py          # Neo4j + in-memory fraud ring detection (batched writes)
  agents/investigation_graph.py # LangGraph multi-agent pipeline + HITL gate
  agents/narration.py           # Claude LLM narration + case chat (template/fallback)
  agents/tracing.py             # Langfuse tracing (no-op fallback)
  api/auth.py                   # API key authentication
  api/main.py                   # FastAPI service (investigation, ticketing, admin)
  static/index.html             # Analyst console dashboard
Dockerfile
.env.example
requirements.txt
```
