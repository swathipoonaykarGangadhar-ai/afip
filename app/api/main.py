"""
AFIP API — v0.2, wired to the checkpointed multi-agent graph.

Endpoints:
  POST /investigate           -> run the agent pipeline on a transaction_id
  POST /case/{case_id}/approve -> analyst approves/rejects a pending SAR escalation
  GET  /case/{case_id}        -> retrieve a case (including pending-approval state)
  POST /sar/{case_id}         -> generate a SAR narrative for an escalated+approved case
  GET  /cases                 -> list cases, optionally filtered by status
  GET  /health                -> liveness

On startup, loads synthetic data into memory + the graph store, and
trains/loads the fraud model. Swap `DATA` for real Kafka/Snowflake/
Postgres sources when moving past the prototype stage — the agent
graph itself doesn't care where the data came from.
"""
import uuid
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.synthetic_data import generate_dataset
from app.graph.graph_store import InMemoryGraphStore
from app.agents.investigation_graph import build_investigation_graph, set_graph_store
from app.agents.narration import generate_sar_narrative
from app.ml import fraud_model
from langgraph.types import Command

app = FastAPI(title="Enterprise Agentic Fraud Investigation Platform", version="0.2.0")

STATE = {"cases": {}}


@app.on_event("startup")
def startup():
    ds = generate_dataset(n_customers=300, n_transactions=4000)
    if not os.path.exists(fraud_model.MODEL_PATH):
        fraud_model.train(ds["transactions"])

    store = InMemoryGraphStore()
    store.load_data(ds["customers"], ds["accounts"], ds["transactions"])
    set_graph_store(store)  # context-scoped; see investigation_graph.py for why

    STATE["transactions_by_id"] = {t["transaction_id"]: t for t in ds["transactions"]}
    STATE["customers_by_id"] = {c["customer_id"]: c for c in ds["customers"]}
    STATE["accounts_by_id"] = {a["account_id"]: a for a in ds["accounts"]}
    STATE["graph_store"] = store
    STATE["model"] = fraud_model.load_model()
    STATE["agent_graph"] = build_investigation_graph()
    STATE["ready"] = True


class InvestigateRequest(BaseModel):
    transaction_id: str


class ApprovalRequest(BaseModel):
    decision: str  # "approved" | "rejected"


@app.get("/health")
def health():
    return {"status": "ok", "ready": STATE.get("ready", False)}


def _extract_findings(result: dict) -> dict:
    return {
        "transaction_findings": result.get("transaction_findings"),
        "customer_findings": result.get("customer_findings"),
        "graph_findings": result.get("graph_findings"),
        "compliance_findings": result.get("compliance_findings"),
    }


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    txn = STATE["transactions_by_id"].get(req.transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail=f"Unknown transaction_id {req.transaction_id}")

    customer = STATE["customers_by_id"].get(txn["customer_id"], {})
    velocity = sum(1 for t in STATE["transactions_by_id"].values() if t["account_id"] == txn["account_id"])

    case_id = f"CASE-{uuid.uuid4().hex[:10]}"
    # thread_id ties this case to its checkpointed graph state, so a later
    # /approve call can resume the exact same paused run.
    config = {"configurable": {"thread_id": case_id}}

    set_graph_store(STATE["graph_store"])  # ensure context is set on this request's worker
    result = STATE["agent_graph"].invoke({
        "transaction": txn,
        "customer": customer,
        "account_velocity": velocity,
    }, config=config)

    pending_approval = "__interrupt__" in result
    decision = result.get("final_decision") or result.get("proposed_decision")

    case = {
        "case_id": case_id,
        "alert_id": req.transaction_id,
        "transaction": txn,
        "customer": customer,
        "agent_result": _extract_findings(result),
        "final_decision": decision if not pending_approval else "PENDING_APPROVAL",
        "final_risk_score": result.get("final_risk_score"),
        "explanation": result.get("explanation"),
        "status": "PENDING_APPROVAL" if pending_approval else ("OPEN" if decision != "CLEAR" else "CLOSED"),
        "created_date": datetime.utcnow().isoformat(),
    }
    STATE["cases"][case_id] = case
    return case


@app.post("/case/{case_id}/approve")
def approve_case(case_id: str, req: ApprovalRequest):
    case = STATE["cases"].get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    if case["status"] != "PENDING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} is not pending approval (status: {case['status']}).",
        )
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    config = {"configurable": {"thread_id": case_id}}
    set_graph_store(STATE["graph_store"])
    result = STATE["agent_graph"].invoke(Command(resume=req.decision), config=config)

    case["final_decision"] = result["final_decision"]
    case["status"] = "OPEN" if result["final_decision"] != "CLEAR" else "CLOSED"
    case["human_approval"] = req.decision
    return case


@app.get("/case/{case_id}")
def get_case(case_id: str):
    case = STATE["cases"].get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    return case


@app.post("/sar/{case_id}")
def generate_sar(case_id: str):
    case = STATE["cases"].get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    if case["final_decision"] != "ESCALATE_SAR":
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} decision is '{case['final_decision']}', not ESCALATE_SAR — SAR not warranted.",
        )

    narrative = generate_sar_narrative(case)
    return {"case_id": case_id, "sar_narrative": narrative, "generated_date": datetime.utcnow().isoformat()}


@app.get("/cases")
def list_cases(status: str | None = None):
    cases = list(STATE["cases"].values())
    if status:
        cases = [c for c in cases if c["status"] == status]
    return {"count": len(cases), "cases": cases}

