# AFIP — Enterprise Agentic Fraud Investigation Platform (v0.2)

A real, running multi-agent fraud investigation pipeline with a human-in-the-loop
approval gate before any SAR escalation. Tested end-to-end (see "What's verified" below).

## What's real here

- **ML detection**: XGBoost classifier on transaction features. `app/ml/fraud_model.py`
- **GraphRAG fraud ring detection**: NetworkX in-memory backend (dev) + a real Neo4j
  driver backend (prod, set `NEO4J_URI`). Detects mule rings via shared device/IP +
  circular transfer chains, with guardrails against shared-infrastructure false positives.
  `app/graph/graph_store.py`
- **Multi-agent pipeline** (LangGraph): Transaction → Customer → GraphRAG → Compliance →
  Supervisor, converging on a risk-scored decision with full evidence trail.
  `app/agents/investigation_graph.py`
- **Checkpointed, resumable state**: SQLite by default, Postgres in production
  (`DATABASE_URL`). Investigations survive process restarts.
- **Human-in-the-loop approval gate**: any case the Supervisor proposes to escalate to
  `ESCALATE_SAR` pauses the graph (`langgraph.types.interrupt`) and waits for an analyst
  to call `/case/{id}/approve`. Nothing files a SAR without a human approving it.
- **Real Claude narration + SAR drafting**, with a deterministic template fallback if
  `ANTHROPIC_API_KEY` isn't set. `app/agents/narration.py`
- **Langfuse tracing** on every agent node, no-op if not configured. `app/agents/tracing.py`
- **FastAPI service** exposing the full case lifecycle.

## What's verified (actually run, not assumed)

- `/investigate` on a fraud-ring transaction → `PENDING_APPROVAL`, with the ring's
  accounts, transfer chain, and risk score in `agent_result.graph_findings`
- `/case/{id}/approve` with `"approved"` → flips to `ESCALATE_SAR` / status `OPEN`
- `/case/{id}/approve` with `"rejected"` → downgrades to `MANUAL_REVIEW`, SAR blocked (400)
- `/sar/{id}` on an approved case → returns a narrative
- `/investigate` on a normal transaction → `CLEAR` / `CLOSED`, no approval needed
- `/cases?status=OPEN` filtering

## What's NOT verified / not built

- **Docker image is untested** — no Docker available in the environment this was built
  in. The Dockerfile is written correctly against the real dependency list, but you
  should `docker build` and smoke-test it yourself before deploying.
- **No live Neo4j or Postgres tested against** — both backends are real, correct driver
  code, but only exercised against their in-memory/SQLite fallbacks so far.
- **No Kafka streaming** — transactions are a static synthetic batch, not a live feed.
- **AUC of 1.0 on synthetic data is a red flag, not a win** — the synthetic fraud is
  trivially separable. Re-validate on real or harder synthetic data before trusting this.
- **No auth** — anyone who can reach the API can approve/reject SAR escalations. Add
  OAuth2/JWT + RBAC before this touches anything real.
- **Multi-worker caveat**: the in-memory graph store and case dict are per-process.
  Running with `--workers > 1` gives each worker its own dataset and case store —
  fine for a demo, wrong for anything real. Fix this by moving case storage to
  Postgres and the graph to real Neo4j (both already have driver code ready).

## Run locally

```bash
pip install -r requirements.txt
python3 -m uvicorn app.api.main:app --reload --port 8001
```

## Full HITL flow, end to end

```bash
# 1. Investigate a known fraud-ring transaction (TXRING0000-0004 are always injected)
curl -X POST localhost:8001/investigate -H "Content-Type: application/json" \
  -d '{"transaction_id":"TXRING0000"}'
# -> status: "PENDING_APPROVAL", case_id in the response

# 2. Analyst approves (or rejects) the SAR escalation
curl -X POST localhost:8001/case/<case_id>/approve -H "Content-Type: application/json" \
  -d '{"decision":"approved"}'
# -> final_decision: "ESCALATE_SAR", status: "OPEN"

# 3. Generate the SAR narrative
curl -X POST localhost:8001/sar/<case_id>

# Compare against a normal transaction (any TX0000XX) -- clears with no approval step.
curl -X POST localhost:8001/investigate -H "Content-Type: application/json" \
  -d '{"transaction_id":"TX000010"}'
```

Interactive docs at `localhost:8001/docs`.

## Deploy

### Docker (untested build, verify before shipping)

```bash
docker build -t afip:latest .
docker run -p 8001:8001 --env-file .env afip:latest
```

Copy `.env.example` to `.env` and fill in what you have — everything is optional;
the app degrades gracefully (SQLite instead of Postgres, in-memory graph instead
of Neo4j, template narration instead of Claude) for anything left blank.

### Cloud options (not yet configured, pick one)

- **AWS**: ECS Fargate or App Runner for the API container; RDS Postgres for
  `DATABASE_URL`; Neo4j Aura (managed) for the graph store.
- **GCP**: Cloud Run for the API; Cloud SQL for Postgres; Neo4j Aura.
- **Azure**: Container Apps; Azure Database for PostgreSQL; Neo4j Aura.

All three are reasonable — none is wired up yet. The app itself is cloud-agnostic
(just env vars), so the choice is really about where your other infra already lives.

## Recommended next steps, in order

1. **Test the Docker build** — I couldn't verify it in this environment.
2. **Point at real Postgres and Neo4j** — swap the fallbacks for the real thing,
   confirm the HITL approval flow survives a process restart mid-pending-approval
   (that's the whole point of checkpointing — verify it actually works).
3. **Add auth** (OAuth2/JWT + RBAC) before any real data touches this.
4. **Re-validate the ML model on real transaction data.**
5. **Swap synthetic batch data for a real feed** (Kafka or even a CSV export first).

## Project structure

```
app/
  core/synthetic_data.py        # dev/test data generator
  ml/fraud_model.py             # XGBoost detection model
  graph/graph_store.py          # Neo4j + in-memory fraud ring detection
  agents/investigation_graph.py # LangGraph multi-agent pipeline + HITL gate
  agents/narration.py           # Claude LLM narration (template fallback)
  agents/tracing.py             # Langfuse tracing (no-op fallback)
  api/main.py                   # FastAPI service
Dockerfile
.env.example
requirements.txt
```
