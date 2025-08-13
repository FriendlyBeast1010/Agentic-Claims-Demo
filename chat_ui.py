import json, io, zipfile, os
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st
from dotenv import load_dotenv

# LangSmith tracing (pipeline-level)
from langsmith.run_helpers import traceable

# === Agents ===
from agents.claim_intake import claim_intake_agent
from agents.bundled_match import bundled_match_agent
from agents.recommendation import recommendation_agent
from agents.provider_response import provider_response_agent

def main() -> None:
    """Claims Copilot UI (wrapped so app.py can call it cleanly)."""
    # === App setup ===
    load_dotenv()  # load .env if present
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # Must be called once and before other Streamlit calls
    st.set_page_config(page_title="Claims Copilot", layout="wide")

    BASE_DIR = Path(__file__).parent.resolve()
    CONTRACTS = json.loads((BASE_DIR / "data" / "contracts.json").read_text(encoding="utf-8"))

    st.title("Claims Copilot")
    st.caption("Upload claim(s) and chat. I’ll normalize, match, recommend, and draft a provider response.")

    # --- UI tweaks: wrap code blocks and make textareas pleasant to read ---
    st.markdown(
        """
    <style>
    div[data-testid="stCodeBlock"] pre { white-space: pre-wrap !important; word-break: break-word !important; }
    textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------------------
    # Business-friendly default prompts (shown if an agent doesn't provide its own prompt)
    # --------------------------------------------------------------------------------------
    DEFAULT_AGENT_PROMPTS: Dict[str, str] = {
        "intake": (
            "Read and normalize the incoming claim. Extract and standardize key fields like "
            "member ID, service description, CPT, provider ID, and date of service. Ensure "
            "service names align with internal contract terminology for later matching."
        ),
        "bundled_match": (
            "Compare the submitted claim’s service and CPT against the bundled payment contract. "
            "Determine if this claim’s service is already covered under an active episode "
            "(e.g., post‑op physical therapy after a knee surgery) and capture matching evidence."
        ),
        "recommendation": (
            "Based on the bundled match result, decide whether to PAY, REJECT, or DENY the claim. "
            "Provide a clear reason, citing contract terms and supporting evidence, so the decision "
            "can be reviewed by providers and internal audit teams."
        ),
        "provider_response": (
            "Draft a professional, easy‑to‑understand message for the provider explaining the decision. "
            "Include contract references, matched service/CPT, and guidance for future submissions."
        ),
    }

    # --------------------------------------------------------------------------------------
    # Helper to render agent traces (Intake, Bundled Match, Recommendation, Provider Response)
    # NOTE: key_prefix makes widget keys unique per claim.
    # --------------------------------------------------------------------------------------
    def _safe_show_json(value):
        """Show dict/list as JSON; otherwise show as code so we never throw JSON parse errors."""
        if isinstance(value, (dict, list)):
            st.json(value)
        else:
            st.code(str(value), language="json")

    def render_agent_traces(res: Dict[str, Any], *, key_prefix: str = "") -> None:
        """Render a four-tab view that shows system prompt, user input, model output, and parsed JSON."""
        if not isinstance(res, dict):
            return

        with st.expander("Agent Traces", expanded=False):
            tabs = st.tabs(["Intake", "Bundled Match", "Recommendation", "Provider Response"])

            def show_trace(slot, title: str, trace_dict: Dict[str, Any] | None, agent_key: str) -> None:
                trace_dict = trace_dict or {}
                with slot:
                    st.markdown(f"**{title}**")
                    col1, col2 = st.columns(2)

                    # SYSTEM PROMPT (wrapped textarea; falls back to business-friendly default)
                    with col1:
                        st.caption("System prompt")
                        raw_prompt = str(trace_dict.get("system_prompt") or "").strip()
                        prompt_text = raw_prompt if raw_prompt and not raw_prompt.upper().startswith("N/A") \
                            else DEFAULT_AGENT_PROMPTS.get(agent_key, "No prompt available")
                        st.text_area(
                            "System prompt",
                            value=prompt_text,
                            key=f"{key_prefix}{agent_key}_sys_prompt",
                            height=150,
                            label_visibility="collapsed",
                        )

                        st.caption("User input")
                        _safe_show_json(trace_dict.get("user_input", "N/A"))

                    # MODEL OUTPUT (wrapped textarea)
                    with col2:
                        st.caption("Model output")
                        mo = trace_dict.get("model_output", "N/A")
                        st.text_area(
                            "Model output",
                            value=str(mo) if not isinstance(mo, (dict, list)) else json.dumps(mo, indent=2),
                            key=f"{key_prefix}{agent_key}_model_output",
                            height=150,
                            label_visibility="collapsed",
                        )

                        with st.expander("Parsed JSON", expanded=False):
                            _safe_show_json(trace_dict.get("parsed", {}))

            show_trace(tabs[0], "Claim Intake", res.get("intake", {}).get("trace", {}), agent_key="intake")
            show_trace(tabs[1], "Bundled Match", res.get("match", {}).get("trace", {}), agent_key="bundled_match")
            show_trace(tabs[2], "Recommendation", res.get("reco", {}).get("trace", {}), agent_key="recommendation")
            show_trace(tabs[3], "Provider Response", res.get("response", {}).get("trace", {}), agent_key="provider_response")

    # --------------------------------------------------------------------------------------
    # Pipeline + helpers (TRACED)
    # --------------------------------------------------------------------------------------
    @traceable(name="Claims Pipeline")
    def run_pipeline(claim: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents and return a dict with each step's output."""
        intake = claim_intake_agent(claim)
        pt_claim = intake.get("normalized_claim", claim)

        match = bundled_match_agent(pt_claim, CONTRACTS)
        reco = recommendation_agent(pt_claim, match)

        decision = (reco.get("decision") or reco.get("outcome") or "_").upper()
        response = provider_response_agent({**reco, "decision": decision}, decision=decision)

        return {
            "intake": intake,
            "match": match,
            "reco": {**reco, "decision": decision},
            "response": response,
        }

    def extract_claims(files) -> List[Dict[str, Any]]:
        """Accept .json (single or dataset-style) or .zip of JSONs; return a flat list of claim dicts."""
        claims: List[Dict[str, Any]] = []
        for f in files or []:
            name = f.name.lower()
            if name.endswith(".json"):
                data = json.loads(f.getvalue().decode("utf-8"))
                if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
                    claims.extend(list(data.values()))
                else:
                    claims.append(data)
            elif name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                    for nm in z.namelist():
                        if nm.lower().endswith(".json"):
                            data = json.loads(z.read(nm).decode("utf-8"))
                            if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
                                claims.extend(list(data.values()))
                            else:
                                claims.append(data)
        return claims

    # --------------------------------------------------------------------------------------
    # Sidebar upload
    # --------------------------------------------------------------------------------------
    st.sidebar.header("Upload claims")
    uploaded = st.sidebar.file_uploader(
        "Drop .json (single claim) or .zip (many JSON files)",
        type=["json", "zip"],
        accept_multiple_files=True,
    )

    # --------------------------------------------------------------------------------------
    # Main processing
    # --------------------------------------------------------------------------------------
    if uploaded:
        claims = extract_claims(uploaded)
        if claims:
            with st.chat_message("user"):
                st.write(f"Uploaded {len(claims)} claim file(s). Please process.")
        else:
            with st.chat_message("user"):
                st.write("Uploaded file(s), but no JSON claims found.")

        for i, c in enumerate(claims, 1):
            res = run_pipeline(c)
            st.session_state[f"res_{i}"] = res  # keep each separately, avoids confusion

            intake, match, reco, resp = res["intake"], res["match"], res["reco"], res["response"]

            with st.chat_message("assistant"):
                st.markdown(f"**Claim {i} Decision:** `{reco['decision']}` — {reco.get('reason', '')}")

                cols = st.columns(3)
                cols[0].metric("Service", intake["normalized_claim"].get("service", "_"))
                cols[1].metric("CPT", str(intake["normalized_claim"].get("cpt", "_")))
                cols[2].metric("Contract", match.get("contract_id", "_"))

                st.text_area(
                    label="Provider message",
                    value=resp.get("draft", ""),
                    height=160,
                    key=f"provider_msg_{i}",
                    label_visibility="collapsed",
                    disabled=True,
                    help="This box wraps long lines; copy text as needed.",
                )

                with st.expander("Details / Debug", expanded=False):
                    st.json({
                        "normalized_claim": intake["normalized_claim"],
                        "bundled_match": match,
                        "recommendation": reco,
                    })

                # >>> UNIQUE keys per claim <<<
                render_agent_traces(res, key_prefix=f"c{i}_")

    # Optional: show the most recent only (disabled to avoid duplicate keys)
    # if "res_1" in st.session_state:
    #     render_agent_traces(st.session_state["res_1"], key_prefix="latest_")


# Allow running directly for local dev, too
if __name__ == "__main__":
    main()
