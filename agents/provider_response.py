from typing import Dict, Optional

VALID = {"PAY", "REJECT", "DENY", "ADJUST"}

def _list_to_english(items):
    items = [str(i) for i in (items or []) if str(i).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def provider_response_agent(reco: Dict, decision: Optional[str] = None) -> Dict:
    decision = (decision or reco.get("decision") or reco.get("outcome") or "").upper()
    if decision not in VALID:
        parsed = {
            "status": "error",
            "reason": "missing_or_invalid_decision",
            "message": "Recommendation did not include a valid decision (PAY/REJECT/DENY/ADJUST).",
            "decision": decision or None,
            "reco": reco,
        }
        trace = {
            # BUSINESS-FRIENDLY PROMPT
            "system_prompt": (
                "Draft a professional, easy‑to‑understand message for the provider explaining the "
                "decision. Include contract references, matched service/CPT, and guidance for "
                "future submissions."
            ),
            "user_input": reco,
            "model_output": parsed.get("message"),
            "parsed": parsed,
        }
        return {**parsed, "trace": trace}

    plan_name = reco.get("plan_name") or "the member's health plan"
    plan_id = reco.get("plan_id") or "the member’s plan"
    product = reco.get("product_type") or "PPO"
    episode_code = reco.get("episode_code") or "the bundled episode"
    contract_id = reco.get("contract_id") or "the applicable provider agreement"

    ev = (reco.get("evidence") or "").strip()
    terms = _list_to_english(reco.get("matched_terms", []))
    short_reason = (reco.get("reason") or "").strip()

    if decision == "PAY":
        draft = (
            "Approved for payment. The billed service is not part of the bundled coverage "
            f"defined under {plan_name} {product} (Plan {plan_id}). "
        )
        if short_reason:
            draft += f"{short_reason} "
        if ev:
            draft += f"Supporting note: {ev}"
        next_action = "Queue for payment and notify provider of approval."

    elif decision in {"REJECT", "DENY"}:
        draft = (
            f"This claim has been denied because the billed service is included in {episode_code} "
            f"under {plan_name} {product} (Plan {plan_id}) and provider agreement {contract_id}. "
            "Per contract terms, services that are part of the episode should not be billed separately. "
            "Please bill this service under the episode claim or submit a corrected claim."
        )
        if terms:
            draft += f" Matched bundle terms/CPTs: {terms}."
        if ev:
            draft += f" Supporting note: {ev}"
        next_action = (
            "Return claim to provider with denial explanation and contract reference. "
            "Advise resubmission under the bundled episode (or corrected billing)."
        )

    else:  # ADJUST
        draft = (
            f"This service falls under {episode_code} for {plan_name} (Plan {plan_id}). "
            "The claim will be adjusted per contract terms and allowed amounts for the bundled episode."
        )
        if terms:
            draft += f" Relevant matched terms/CPTs: {terms}."
        if ev:
            draft += f" Supporting note: {ev}"
        next_action = "Apply contract adjustment and notify provider of revised allowed amount."

    parsed = {"status": "ok", "decision": decision, "draft": draft, "next_action": next_action}
    trace = {
        # BUSINESS-FRIENDLY PROMPT
        "system_prompt": (
            "Draft a professional, easy‑to‑understand message for the provider explaining the "
            "decision. Include contract references, matched service/CPT, and guidance for "
            "future submissions."
        ),
        "user_input": reco,
        "model_output": draft,
        "parsed": parsed,
    }
    return {**parsed, "trace": trace}
