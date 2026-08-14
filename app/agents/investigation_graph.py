"""
Agentic Investigation Supervisor + specialist agents, built on LangGraph.

v0.2 additions over the first slice:
  - Checkpointing (SqliteSaver by default, PostgresSaver if DATABASE_URL
    is set) so investigations survive restarts and support time-travel
    debugging/audit -- required for anything compliance-adjacent.
  - Human-in-the-loop approval gate: any case that would auto-escalate
    to SAR filing pauses via LangGraph's interrupt() and waits for an
    analyst to approve or reject before the decision is finalized.
    Nothing files a SAR without a human in the loop.
  - Real Langfuse tracing on every agent node (no-op if not configured).
  - Real Claude-generated narratives via app/agents/narration.py
    (template fallback if no API key is set).
"""
import os
import contextvars
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from app.ml.fraud_model import load_model, score_transaction
from app.graph.graph_store import InMemoryGraphStore
from app.agents.narration import narrate
from app.agents.tracing import traced_agent, flush as flush_traces

# The graph store is a live connection object and can't go through
# LangGraph's checkpointed `state` (gets serialized to sqlite/postgres
# between steps) or reliably through `config` (observed to be dropped
# once a checkpointer + interrupt/resume is in play with this LangGraph
# version). A contextvar, set once per investigation before invoke() and
# read inside the node, is the standard Python pattern for exactly this
# kind of request-scoped external resource and sidesteps both issues.
_current_graph_store: contextvars.ContextVar = contextvars.ContextVar(
    "current_graph_store", default=None
)


def set_graph_store(store) -> None:
    _current_graph_store.set(store)


class InvestigationState(TypedDict, total=False):
    transaction: dict
    customer: dict
    account_velocity: int
    # NOTE: graph_store is deliberately NOT part of state -- state gets
    # checkpointed/serialized (sqlite/postgres) and a live graph connection
    # object isn't serializable. Pass it via config["configurable"] instead
    # (see graphrag_agent below and the __main__ example).

    ml_risk_score: float
    customer_findings: dict
    transaction_findings: dict
    graph_findings: dict
    compliance_findings: dict

    proposed_decision: str
    human_approval: Optional[str]  # "approved" | "rejected" | None (pending)

    final_decision: str
    final_risk_score: float
    explanation: str
    case_id: str


# ---------- Agent nodes ----------

@traced_agent("transaction_agent")
def transaction_agent(state: InvestigationState) -> InvestigationState:
    model = load_model()
    txn = state["transaction"]
    velocity = state.get("account_velocity", 1)
    risk = score_transaction(model, txn, account_velocity=velocity)

    flags = []
    if txn.get("amount", 0) > 2000:
        flags.append("high_amount")
    if velocity > 5:
        flags.append("high_velocity")
    if txn.get("merchant") == "WIRE_TRANSFER":
        flags.append("wire_transfer")

    return {
        "ml_risk_score": risk,
        "transaction_findings": {"risk_score": risk, "flags": flags},
    }


@traced_agent("customer_agent")
def customer_agent(state: InvestigationState) -> InvestigationState:
    customer = state.get("customer", {})
    flags = []
    if customer.get("kyc_status") == "PENDING":
        flags.append("kyc_pending")
    if customer.get("risk_level") == "HIGH":
        flags.append("high_risk_customer")
    if customer.get("country") in ("NG", "RU"):
        flags.append("high_risk_jurisdiction")

    return {"customer_findings": {"flags": flags, "customer_id": customer.get("customer_id")}}


@traced_agent("graphrag_agent")
def graphrag_agent(state: InvestigationState) -> InvestigationState:
    store: InMemoryGraphStore = _current_graph_store.get()
    txn = state["transaction"]
    findings = {"in_fraud_ring": False, "ring": None}

    if store is not None:
        rings = store.find_fraud_rings()
        for ring in rings:
            if txn.get("account_id") in ring.get("ring_participants", ring.get("accounts", [])):
                findings = {"in_fraud_ring": True, "ring": ring}
                break

    return {"graph_findings": findings}


@traced_agent("compliance_agent")
def compliance_agent(state: InvestigationState) -> InvestigationState:
    customer = state.get("customer", {})
    txn = state["transaction"]
    flags = []

    if txn.get("amount", 0) >= 10000:
        flags.append("ctr_threshold_triggered")
    if customer.get("country") in ("NG", "RU"):
        flags.append("sanctions_screening_required")
    if customer.get("kyc_status") == "PENDING":
        flags.append("kyc_incomplete_block")

    return {"compliance_findings": {"flags": flags}}


@traced_agent("supervisor")
def supervisor_agent(state: InvestigationState) -> InvestigationState:
    ml_score = state.get("ml_risk_score", 0.0)
    graph_findings = state.get("graph_findings", {})
    customer_findings = state.get("customer_findings", {})
    compliance_findings = state.get("compliance_findings", {})
    txn_findings = state.get("transaction_findings", {})

    reasons = []
    score = ml_score

    if graph_findings.get("in_fraud_ring"):
        score = max(score, graph_findings["ring"]["risk_score"])
        reasons.append("account is part of a detected fraud ring (shared device/IP + transfer chain)")

    if "high_risk_jurisdiction" in customer_findings.get("flags", []):
        score = min(1.0, score + 0.1)
        reasons.append("customer in high-risk jurisdiction")

    if "kyc_pending" in customer_findings.get("flags", []):
        score = min(1.0, score + 0.05)
        reasons.append("KYC verification incomplete")

    reasons.extend(f"transaction flag: {f}" for f in txn_findings.get("flags", []))

    if score >= 0.8 or graph_findings.get("in_fraud_ring"):
        decision = "ESCALATE_SAR"
    elif score >= 0.5:
        decision = "MANUAL_REVIEW"
    else:
        decision = "CLEAR"

    if compliance_findings.get("flags"):
        reasons.extend(f"compliance flag: {f}" for f in compliance_findings["flags"])
        if decision == "CLEAR":
            decision = "MANUAL_REVIEW"

    summary = {
        "transaction_id": state["transaction"].get("transaction_id"),
        "decision": decision,
        "risk_score": score,
        "reasons": reasons,
    }

    return {
        "proposed_decision": decision,
        "final_risk_score": score,
        "explanation": narrate(summary),
    }


def human_approval_gate(state: InvestigationState) -> InvestigationState:
    """
    Pauses the graph for a human analyst to approve or reject an
    auto-proposed SAR escalation. Requires a checkpointer to resume
    correctly -- see build_investigation_graph().

    Resume with: app.invoke(Command(resume="approved"), config=...)
                 app.invoke(Command(resume="rejected"), config=...)
    """
    decision = interrupt({
        "type": "sar_approval_required",
        "transaction_id": state["transaction"].get("transaction_id"),
        "proposed_decision": state["proposed_decision"],
        "risk_score": state.get("final_risk_score"),
        "explanation": state.get("explanation"),
    })
    approved = decision == "approved"
    return {
        "human_approval": decision,
        "final_decision": "ESCALATE_SAR" if approved else "MANUAL_REVIEW",
    }


def finalize_no_approval_needed(state: InvestigationState) -> InvestigationState:
    return {"final_decision": state["proposed_decision"]}


def route_after_supervisor(state: InvestigationState) -> str:
    if state.get("proposed_decision") == "ESCALATE_SAR":
        return "human_approval_gate"
    return "finalize"


def build_investigation_graph():
    graph = StateGraph(InvestigationState)

    graph.add_node("transaction_agent", transaction_agent)
    graph.add_node("customer_agent", customer_agent)
    graph.add_node("graphrag_agent", graphrag_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("human_approval_gate", human_approval_gate)
    graph.add_node("finalize", finalize_no_approval_needed)

    graph.set_entry_point("transaction_agent")
    graph.add_edge("transaction_agent", "customer_agent")
    graph.add_edge("customer_agent", "graphrag_agent")
    graph.add_edge("graphrag_agent", "compliance_agent")
    graph.add_edge("compliance_agent", "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "human_approval_gate": "human_approval_gate",
        "finalize": "finalize",
    })
    graph.add_edge("human_approval_gate", END)
    graph.add_edge("finalize", END)

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


def get_checkpointer():
    """
    PostgresSaver if DATABASE_URL is set (production), otherwise a local
    SQLite file (dev/prototype). Either way, checkpointing is real --
    investigations survive process restarts and support time-travel
    debugging via the same API.
    """
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        from langgraph.checkpoint.postgres import PostgresSaver
        saver_cm = PostgresSaver.from_conn_string(db_url)
        saver = saver_cm.__enter__()
        saver.setup()
        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "afip_checkpoints.sqlite")
    # Keep the connection open explicitly rather than via the contextmanager
    # helper -- the cm form gets closed by GC of the unreferenced generator.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.core.synthetic_data import generate_dataset

    ds = generate_dataset()
    store = InMemoryGraphStore()
    store.load_data(ds["customers"], ds["accounts"], ds["transactions"])

    customers_by_id = {c["customer_id"]: c for c in ds["customers"]}
    app = build_investigation_graph()

    ring_txn = next(t for t in ds["transactions"] if t["transaction_id"] == "TXRING0000")
    normal_txn = ds["transactions"][0]

    set_graph_store(store)

    # --- Normal transaction: no interrupt expected ---
    config = {"configurable": {"thread_id": "demo-normal-1"}}
    result = app.invoke({
        "transaction": normal_txn,
        "customer": customers_by_id.get(normal_txn["customer_id"], {}),
        "account_velocity": 3,
    }, config=config)
    print(f"\n=== NORMAL TXN ({normal_txn['transaction_id']}) ===")
    print(f"Decision: {result.get('final_decision')} | Risk: {result.get('final_risk_score', 0):.2f}")

    # --- Ring transaction: expect interrupt, then resume with approval ---
    config = {"configurable": {"thread_id": "demo-ring-1"}}
    result = app.invoke({
        "transaction": ring_txn,
        "customer": customers_by_id.get(ring_txn["customer_id"], {}),
        "account_velocity": 3,
    }, config=config)

    print(f"\n=== RING TXN ({ring_txn['transaction_id']}) — first pass ===")
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print(f"PAUSED for human approval: {interrupt_payload}")

        # Simulate an analyst approving the SAR escalation
        result2 = app.invoke(Command(resume="approved"), config=config)
        print(f"After analyst approval -> Decision: {result2.get('final_decision')}")
    else:
        print(f"Decision: {result.get('final_decision')} (no interrupt triggered — check routing)")

    flush_traces()
