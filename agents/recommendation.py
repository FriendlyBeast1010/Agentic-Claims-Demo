from typing import Dict, Any

def recommendation_agent(claim: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    if match.get("status") == "skipped":
        parsed = {
            "decision": "PAY",
            "reason": "No applicable bundled episode for this plan/claim.",
            "evidence": match.get("evidence"),
            "matched_terms": [],
            "contract_id": match.get("contract_id"),
            "plan_id": match.get("plan_id"),
            "plan_name": match.get("plan_name"),
            "product_type": match.get("product_type"),
            "episode_code": match.get("episode_code"),
            "trace": {"bundled_match": match},
            "confidence": 0.9,
        }
    elif match.get("is_in_bundle"):
        parsed = {
            "decision": "REJECT",
            "reason": "Service included in the episode bundle; bill under the episode.",
            "evidence": match.get("evidence"),
            "matched_terms": match.get("matched_terms", []),
            "contract_id": match.get("contract_id"),
            "plan_id": match.get("plan_id"),
            "plan_name": match.get("plan_name"),
            "product_type": match.get("product_type"),
            "episode_code": match.get("episode_code"),
            "trace": {"bundled_match": match},
            "confidence": 0.9,
        }
    else:
        parsed = {
            "decision": "PAY",
            "reason": "Service not included in covered bundle terms.",
            "evidence": match.get("evidence"),
            "matched_terms": match.get("matched_terms", []),
            "contract_id": match.get("contract_id"),
            "plan_id": match.get("plan_id"),
            "plan_name": match.get("plan_name"),
            "product_type": match.get("product_type"),
            "episode_code": match.get("episode_code"),
            "trace": {"bundled_match": match},
            "confidence": 0.9,
        }

    trace = {
        # BUSINESS-FRIENDLY PROMPT
        "system_prompt": (
            "Based on the bundled match result, decide whether to PAY, REJECT, or DENY the claim. "
            "Provide a clear reason, citing contract terms and supporting evidence, so the decision "
            "can be reviewed by providers and internal audit teams."
        ),
        "user_input": {"claim": claim, "bundled_match": match},
        "model_output": "Policy decision; no model output.",
        "parsed": parsed,
    }
    return {**parsed, "trace": trace}
