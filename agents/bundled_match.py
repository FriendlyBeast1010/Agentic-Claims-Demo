from typing import Dict, List, Tuple, Union

# Synonyms to improve string matching when service labels vary
SERVICE_SYNONYMS = {
    "physical therapy": {"pt", "pt rehab", "therapy", "rehab", "physiotherapy", "post-op rehab"},
    "influenza vaccine": {"flu shot", "flu vaccine", "influenza immunization"},
}

def _tokens(s: str) -> List[str]:
    return [t for t in (s or "").lower().replace("_", " ").split() if t]

def _service_in_bundle(service: str, covered: List[str]) -> Tuple[bool, List[str]]:
    """
    Loose string/alias match between a claim's service label and the contract's covered services.
    """
    s = (service or "").strip().lower()
    if not s:
        return False, []
    aliases = {s}
    for base, syns in SERVICE_SYNONYMS.items():
        if s == base or s in syns:
            aliases.update({base, *syns})
    covered_norm = [str(c or "").strip().lower() for c in covered]
    matched = []
    for c in covered_norm:
        for a in aliases:
            if a in c or c in a or (set(_tokens(a)) & set(_tokens(c))):
                matched.append(c)
                break
    return (len(matched) > 0, matched)

def _evaluate_against_contract(pt_claim: Dict, contract: Dict) -> Dict:
    """
    Evaluate a single contract (nested: plan/provider_agreement/episode) against the claim.
    Returns parsed result + an agent trace (deterministic, no LLM).
    """
    # --- pull nested fields ---
    plan = contract.get("plan", {})
    prov = contract.get("provider_agreement", {})
    ep   = contract.get("episode", {})

    plan_name    = plan.get("payer", "")
    plan_id      = plan.get("plan_id", "")
    product_type = plan.get("product_type", "")
    contract_id  = prov.get("provider_contract_id", "UNKNOWN")
    episode_code = ep.get("episode_code", "")

    covered_services = ep.get("covered_services", []) or []
    covered_cpts     = [str(c) for c in (ep.get("covered_cpts", []) or [])]

    # --- derive claim fields ---
    service = (pt_claim.get("service") or "").strip().lower()
    cpt     = str(pt_claim.get("cpt") or "")

    # --- matching logic ---
    cpt_in = bool(cpt and cpt in covered_cpts)
    svc_in, svc_matched = _service_in_bundle(service, covered_services)

    is_in = cpt_in or svc_in
    matched_terms = (svc_matched or []) + ([cpt] if cpt_in else [])

    # --- parsed result used downstream ---
    parsed = {
        "status": "ok",
        "plan_name": plan_name,
        "plan_id": plan_id,
        "product_type": product_type,
        "contract_id": contract_id,
        "episode_code": episode_code,
        "service": service,
        "cpt": cpt,
        "is_in_bundle": is_in,
        "matched_terms": matched_terms,
        "evidence": f"Matched by CPT or service under {episode_code}.",
        "debug": {
            "covered_services": covered_services,
            "covered_cpts": covered_cpts,
            "svc_match": svc_in,
            "cpt_match": cpt_in,
            "svc_matched_terms": svc_matched,
        },
    }

    # --- agent trace (for the UI) ---
    trace = {
        # BUSINESS-FRIENDLY PROMPT
        "system_prompt": (
            "Compare the submitted claim’s service and CPT against the bundled payment contract. "
            "Determine if this claim’s service is already covered under an active episode "
            "(e.g., post‑op physical therapy after a knee surgery) and capture matching evidence."
        ),
        "user_input": {
            "claim": {"service": service, "cpt": cpt},
            "contract": {
                "plan": {"payer": plan_name, "plan_id": plan_id, "product_type": product_type},
                "provider_agreement": {"provider_contract_id": contract_id},
                "episode": {
                    "episode_code": episode_code,
                    "covered_services": covered_services,
                    "covered_cpts": covered_cpts,
                },
            },
        },
        "model_output": "Deterministic match; no model output.",
        "parsed": parsed,
    }

    return {**parsed, "trace": trace}

def bundled_match_agent(pt_claim: Dict, contracts: Union[Dict, List[Dict]]) -> Dict:
    """
    Evaluate one or many contracts and return the best match.
    Adds a trace even when there are no contracts.
    """
    # Accept dict of named contracts or list/single flat contract
    if isinstance(contracts, dict) and "contract_id" in contracts:
        return _evaluate_against_contract(pt_claim, contracts)

    if isinstance(contracts, dict):
        results = [_evaluate_against_contract(pt_claim, c) for c in contracts.values()]
    else:
        results = [_evaluate_against_contract(pt_claim, c) for c in (contracts or [])]

    if not results:
        parsed = {
            "status": "error",
            "reason": "no_contracts",
            "is_in_bundle": False,
            "contract_id": "UNKNOWN",
            "matched_terms": [],
            "evidence": "No contracts supplied.",
        }
        trace = {
            "system_prompt": (
                "Compare the submitted claim’s service and CPT against the bundled payment contract. "
                "Determine if the claim is part of an active episode and capture matching evidence."
            ),
            "user_input": {"claim": pt_claim, "contracts": "∅"},
            "model_output": "No contracts available to evaluate.",
            "parsed": parsed,
        }
        return {**parsed, "trace": trace}

    # Prefer in-bundle; tie-break by # of matched terms
    results.sort(key=lambda r: (r.get("is_in_bundle", False), len(r.get("matched_terms", []))), reverse=True)
    return results[0]
