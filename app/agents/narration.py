"""
LLM-generated investigation narratives via the Anthropic API.

Falls back to a deterministic template if ANTHROPIC_API_KEY isn't set,
so the rest of the system (and local dev/tests) never breaks just
because a key is missing. In production, set ANTHROPIC_API_KEY and
this transparently switches to real Claude-generated analyst prose.
"""
import os

MODEL = "claude-sonnet-4-6"


def _template_narrative(summary: dict) -> str:
    reasons = summary.get("reasons", [])
    reason_text = "; ".join(reasons) if reasons else "no significant risk indicators"
    return (
        f"Investigation of transaction {summary.get('transaction_id')} "
        f"concluded with decision '{summary.get('decision')}' "
        f"(risk score {summary.get('risk_score', 0):.2f}). "
        f"Key factors: {reason_text}."
    )


def narrate(summary: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_narrative(summary)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        reasons = summary.get("reasons", [])
        prompt = (
            "You are a fraud investigation analyst writing a concise, factual case "
            "note for a compliance file. Do not embellish or speculate beyond the "
            "evidence given. Write 2-4 sentences, plain professional tone.\n\n"
            f"Transaction ID: {summary.get('transaction_id')}\n"
            f"Decision: {summary.get('decision')}\n"
            f"Risk score: {summary.get('risk_score', 0):.2f}\n"
            f"Evidence found by investigation agents:\n"
            + "\n".join(f"- {r}" for r in reasons)
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "".join(text_blocks) if text_blocks else _template_narrative(summary)

    except Exception as e:
        # Never let a narration failure break the investigation pipeline --
        # degrade to the template and let the caller/logs know why.
        import logging
        logging.getLogger(__name__).warning(f"Claude narration failed, using template: {e}")
        return _template_narrative(summary)


def generate_sar_narrative(case: dict) -> str:
    """LLM-generated SAR narrative, with the same template fallback pattern."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    txn = case["transaction"]
    customer = case["customer"]

    fallback = (
        f"Suspicious Activity Report\n"
        f"Case: {case['case_id']}\n"
        f"Subject: {customer.get('name', 'Unknown')} ({customer.get('customer_id')})\n"
        f"Transaction: {txn['transaction_id']}, amount ${txn['amount']:.2f}, "
        f"merchant/type: {txn['merchant']}, timestamp: {txn['timestamp']}\n"
        f"Risk score: {case['final_risk_score']:.2f}\n"
        f"Narrative: {case['explanation']}\n"
        f"Recommended action: File SAR with FinCEN within statutory deadline; "
        f"restrict account pending compliance review.\n"
    )

    if not api_key:
        return fallback

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Draft a formal Suspicious Activity Report (SAR) narrative section "
            "for a US financial institution's BSA/AML compliance file. Use precise, "
            "factual, regulator-appropriate language. Include: what was suspicious, "
            "what evidence supports it, and recommended next steps. 4-6 sentences.\n\n"
            f"Case ID: {case['case_id']}\n"
            f"Subject: {customer.get('name', 'Unknown')} ({customer.get('customer_id')})\n"
            f"Transaction: {txn['transaction_id']}, amount ${txn['amount']:.2f}, "
            f"type: {txn['merchant']}, timestamp: {txn['timestamp']}\n"
            f"Risk score: {case['final_risk_score']:.2f}\n"
            f"Investigation findings: {case['explanation']}\n"
        )
        response = client.messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        narrative = "".join(text_blocks) if text_blocks else None
        if narrative:
            return f"Suspicious Activity Report\nCase: {case['case_id']}\n\n{narrative}"
        return fallback
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Claude SAR generation failed, using template: {e}")
        return fallback
