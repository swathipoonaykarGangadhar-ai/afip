"""
AFIP API — v0.3: persistent case storage + API key auth.

Endpoints (all require X-API-Key header if API_KEY env var is set;
/health is always open):
  POST /investigate            -> run the agent pipeline on a transaction_id
  POST /case/{case_id}/approve -> analyst approves/rejects a pending SAR escalation
  GET  /case/{case_id}         -> retrieve a case (including pending-approval state)
  POST /sar/{case_id}          -> generate a SAR narrative for an escalated+approved case
  GET  /cases                  -> list cases, optionally filtered by status
  GET  /health                 -> liveness (no auth required)

On startup, loads synthetic data into memory + the graph store, and
trains/loads the fraud model. Swap `DATA` for real Kafka/Snowflake
sources when moving past the prototype stage — the agent graph itself
doesn't care where the data came from.
"""
import uuid
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("afip")

from app.core.synthetic_data import generate_dataset
from app.core import case_store
from app.graph.graph_store import get_graph_store, Neo4jGraphStore
from app.agents.investigation_graph import build_investigation_graph, set_graph_store
from app.agents.narration import generate_sar_narrative, chat_about_case
from app.ml import fraud_model
from app.api.auth import require_api_key, auth_enabled
from langgraph.types import Command

app = FastAPI(title="Enterprise Agentic Fraud Investigation Platform", version="0.3.0")

STATE = {}

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.on_event("startup")
def startup():
    if not auth_enabled():
        logger.warning(
            "API_KEY is not set -- this deployment has NO AUTHENTICATION. "
            "Anyone with the URL can approve/reject SAR escalations. "
            "Set API_KEY before using this with anything real."
        )
    if not case_store.is_persistent():
        logger.warning(
            "DATABASE_URL is not set -- cases are stored in-memory only "
            "and will be LOST on restart/redeploy/scale-to-zero."
        )
    if not os.environ.get("NEO4J_URI"):
        logger.warning(
            "NEO4J_URI is not set -- using an in-memory graph rebuilt from "
            "synthetic data on every startup. No real transaction network "
            "data, no persistence. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD "
            "to use a real graph database."
        )

    ds = generate_dataset(n_customers=300, n_transactions=4000)
    if not os.path.exists(fraud_model.MODEL_PATH):
        fraud_model.train(ds["transactions"])

    store = get_graph_store()
    if isinstance(store, Neo4jGraphStore):
        logger.info("Connected to Neo4j at %s", os.environ.get("NEO4J_URI"))
    if store.is_empty():
        logger.info("Graph store is empty -- seeding with synthetic data.")
        try:
            store.load_data(ds["customers"], ds["accounts"], ds["transactions"])
            logger.info("Graph seeding completed successfully.")
        except Exception:
            # On Render's free tier the container can be torn down for
            # inactivity moments after a cold start, which can interrupt
            # this seeding mid-flight. Don't crash the whole app over it --
            # log it and let POST /admin/seed-graph (below) be the
            # reliable, on-demand way to (re)seed once the service is
            # confirmed live and staying up.
            logger.exception("Graph seeding failed/was interrupted during startup. "
                              "Call POST /admin/seed-graph once the service is confirmed live.")
    else:
        logger.info("Graph store already has data -- skipping seed (using existing data).")
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


class CaseUpdateRequest(BaseModel):
    priority: str | None = None
    assignee: str | None = None
    status: str | None = None


class CommentRequest(BaseModel):
    author: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list | None = None  # [{"role": "user"|"assistant", "content": "..."}]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": STATE.get("ready", False),
        "auth_enabled": auth_enabled(),
        "persistent_storage": case_store.is_persistent(),
        "graph_backend": "neo4j" if isinstance(STATE.get("graph_store"), Neo4jGraphStore) else "in-memory",
    }


@app.post("/admin/seed-graph", dependencies=[Depends(require_api_key)])
def seed_graph(force: bool = False):
    """
    Explicitly (re)seed the graph store with synthetic data, synchronously,
    while the service is confirmed live. Use this instead of relying on
    automatic startup seeding -- on Render's free tier, the container can
    be torn down for inactivity moments after a cold start, which can
    interrupt startup-time seeding mid-write and leave the graph empty or
    partially loaded. Calling this endpoint directly avoids that race
    entirely: the HTTP response only returns after the write actually
    completes, so a 200 here is real proof the data is fully in place.

    force=false (default): only seeds if the graph is currently empty.
    force=true: wipes existing data first, then reseeds from scratch --
    use this to fix a partially-seeded or duplicate-laden graph.
    """
    store = STATE["graph_store"]
    if isinstance(store, Neo4jGraphStore) and force:
        with store.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph data (force=true).")

    if not force and not store.is_empty():
        return {
            "status": "skipped",
            "reason": "graph store already has data; pass ?force=true to wipe and reseed",
        }

    customers = list(STATE["customers_by_id"].values())
    accounts = list(STATE["accounts_by_id"].values())
    transactions = list(STATE["transactions_by_id"].values())
    store.load_data(customers, accounts, transactions)

    result = {
        "status": "seeded",
        "customers": len(customers),
        "accounts": len(accounts),
        "transactions": len(transactions),
    }
    if isinstance(store, Neo4jGraphStore):
        with store.driver.session() as session:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            )
            result["node_counts"] = {r["label"]: r["c"] for r in counts}
    logger.info("Graph seeding via /admin/seed-graph completed: %s", result)
    return result


@app.get("/admin/graph-status", dependencies=[Depends(require_api_key)])
def graph_status():
    """Direct visibility into what's actually in the graph store right
    now -- node counts by label, and whether the known fraud ring
    (TXRING0000-4) is currently detectable. Use this instead of the
    Neo4j Aura console to verify state without leaving the API."""
    store = STATE["graph_store"]
    result = {"backend": "neo4j" if isinstance(store, Neo4jGraphStore) else "in-memory"}
    if isinstance(store, Neo4jGraphStore):
        with store.driver.session() as session:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            )
            result["node_counts"] = {r["label"]: r["c"] for r in counts}
            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY rel"
            )
            result["relationship_counts"] = {r["rel"]: r["c"] for r in rel_counts}
    rings = store.find_fraud_rings()
    result["rings_detected"] = len(rings)
    result["rings"] = rings
    return result


def _extract_findings(result: dict) -> dict:
    return {
        "transaction_findings": result.get("transaction_findings"),
        "customer_findings": result.get("customer_findings"),
        "graph_findings": result.get("graph_findings"),
        "compliance_findings": result.get("compliance_findings"),
    }


@app.post("/investigate", dependencies=[Depends(require_api_key)])
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

    priority = "CRITICAL" if (result.get("final_risk_score") or 0) >= 0.9 else \
               "HIGH" if (result.get("final_risk_score") or 0) >= 0.7 else \
               "MEDIUM" if (result.get("final_risk_score") or 0) >= 0.4 else "LOW"

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
        "priority": priority,
        "assignee": None,
        "comments": [],
    }
    case_store.save_case(case)
    return case


@app.post("/case/{case_id}/approve", dependencies=[Depends(require_api_key)])
def approve_case(case_id: str, req: ApprovalRequest):
    case = case_store.get_case(case_id)
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
    case_store.save_case(case)
    return case


@app.get("/case/{case_id}", dependencies=[Depends(require_api_key)])
def get_case(case_id: str):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    return case


@app.patch("/case/{case_id}", dependencies=[Depends(require_api_key)])
def update_case(case_id: str, req: CaseUpdateRequest):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    if req.priority is not None:
        if req.priority not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise HTTPException(status_code=400, detail="priority must be LOW, MEDIUM, HIGH, or CRITICAL")
        case["priority"] = req.priority
    if req.assignee is not None:
        case["assignee"] = req.assignee
    if req.status is not None:
        if req.status not in ("OPEN", "CLOSED", "PENDING_APPROVAL"):
            raise HTTPException(status_code=400, detail="status must be OPEN, CLOSED, or PENDING_APPROVAL")
        case["status"] = req.status
    case_store.save_case(case)
    return case


@app.post("/case/{case_id}/comment", dependencies=[Depends(require_api_key)])
def add_comment(case_id: str, req: CommentRequest):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    comment = {
        "author": req.author,
        "text": req.text,
        "timestamp": datetime.utcnow().isoformat(),
    }
    case.setdefault("comments", []).append(comment)
    case_store.save_case(case)
    return case


@app.post("/case/{case_id}/chat", dependencies=[Depends(require_api_key)])
def chat(case_id: str, req: ChatRequest):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    reply = chat_about_case(case, req.message, history=req.history)
    return {"case_id": case_id, "reply": reply}


@app.post("/sar/{case_id}", dependencies=[Depends(require_api_key)])
def generate_sar(case_id: str):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    if case["final_decision"] != "ESCALATE_SAR":
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} decision is '{case['final_decision']}', not ESCALATE_SAR — SAR not warranted.",
        )

    narrative = generate_sar_narrative(case)
    return {"case_id": case_id, "sar_narrative": narrative, "generated_date": datetime.utcnow().isoformat()}


@app.get("/cases", dependencies=[Depends(require_api_key)])
def list_cases(status: str | None = None):
    cases = case_store.list_cases(status=status)
    return {"count": len(cases), "cases": cases}
