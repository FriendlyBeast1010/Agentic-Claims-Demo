# agents/claim_intake.py
from typing import Any, Dict, List
from langsmith.run_helpers import traceable  # <-- tracing decorator

# CPT → service label aligned to contracts
CPT_TO_SERVICE = {
    "97110": "physical therapy",
    "97112": "physical therapy",
    "90686": "influenza vaccine",
}

def _first_line_cpt(lines: List[Dict[str, Any]]) -> Any:
    if isinstance(lines, list) and lines:
        return lines[0].get("cpt")
    return None

def _canon_service(claim: Dict[str, Any]) -> str:
    """
    Decide on a canonical service label for downstream agents.
    Prefer explicit 'service'/'description'/'procedure', else map from CPT.
    """
    svc = (claim.get("service") or claim.get("description") or claim.get("procedure") or "").strip()
    if svc:
        return svc

    cpt = claim.get("cpt") or claim.get("cpt_code") or _first_line_cpt(claim.get("lines") or [])
    if not cpt:
        return ""
    return CPT_TO_SERVICE.get(str(cpt), "")

def _coerce_lines(lines: Any) -> List[Dict[str, Any]]:
    """Ensure lines is a list of dicts; coerce CPTs to strings when present."""
    if not isinstance(lines, list):
        return []
    out: List[Dict[str, Any]] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        ln = dict(ln)  # shallow copy
        if "cpt" in ln and ln["cpt"] is not None:
            ln["cpt"] = str(ln["cpt"])
        out.append(ln)
    return out

def normalize_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a consistent, downstream-safe shape.
    - diagnosis is always a list
    - lines is always a list[dict]
    - cpt is stringified (top-level) if present/derivable
    - service is derived when missing
    """
    lines = _coerce_lines(claim.get("lines"))
    diagnosis = claim.get("diagnosis") or claim.get("dx") or []
    if isinstance(diagnosis, str):
        diagnosis = [diagnosis]

    top_cpt = claim.get("cpt") or claim.get("cpt_code") or _first_line_cpt(lines)
    if top_cpt is not None and top_cpt != "":
        top_cpt = str(top_cpt)

    normalized = {
        # canonical keys used by downstream agents
        "claim_id": claim.get("claim_id"),
        "member_id": claim.get("member_id"),
        "provider_id": claim.get("provider_id"),
        "dos": claim.get("dos"),
        "diagnosis": diagnosis,
        "lines": lines,
        "cpt": top_cpt,
        "service": _canon_service({"lines": lines, **claim}),
        # keep originals (canonical keys above take precedence)
        **claim,
    }
    return normalized

@traceable(name="Claim Intake Agent")  # <-- adds a LangSmith span
def claim_intake_agent(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic intake (no LLM): normalize the claim and return a trace.
    """
    normalized = normalize_claim(claim)

    note = (
        f"Normalized claim {normalized.get('claim_id')} with CPT {normalized.get('cpt')} "
        f"as service '{normalized.get('service') or 'UNKNOWN'}'."
    )

    trace = {
        # Business-friendly description shown in the UI
        "system_prompt": (
            "Read and normalize the incoming claim. Extract and standardize key fields like "
            "member ID, service description, CPT, provider ID, and date of service. Ensure "
            "service names align with internal contract terminology for later matching."
        ),
        "user_input": claim,
        "model_output": "Rule-based normalization; no model output.",
        "parsed": normalized,
    }

    return {
        "status": "ok",
        "normalized_claim": normalized,
        "notes": [note],
        "trace": trace,
    }
