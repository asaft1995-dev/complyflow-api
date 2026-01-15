import os
import uuid
from typing import Optional, Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest


# -------------------------
# Config (ENV)
# -------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "asafturjeman")
LOCATION = os.getenv("LOCATION", "us-central1")
AGENT_ID = os.getenv("AGENT_ID", "28b9247b-716c-4fac-9d86-22791f29f510")

# Optional: force the conversation to start/continue from a specific playbook
# Must be the FULL resource name:
# projects/<PROJECT>/locations/<LOC>/agents/<AGENT_ID>/playbooks/<PLAYBOOK_ID>
CURRENT_PLAYBOOK = os.getenv("CURRENT_PLAYBOOK")  # optional

LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "he")  # "he" or "en"
DIALOGFLOW_API_BASE = os.getenv("DIALOGFLOW_API_BASE", "https://dialogflow.googleapis.com/v3")

AGENT_PATH = f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ID}"


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="ComplyFlow Gateway (Cloud Run → Conversational Agent)")


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    # Keep the UI stable: these are extracted if available, otherwise None/empty
    business_profile: Dict[str, Optional[str]]
    compliance_checklist: List[str]
    # Optional: include raw response for debugging (set DEBUG_RAW=1)
    raw_agent_response: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "agent": AGENT_PATH}


def _get_access_token() -> str:
    """
    Uses Cloud Run's service account (Application Default Credentials)
    to obtain an OAuth2 access token for calling Dialogflow API.
    """
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GoogleAuthRequest())
    if not creds.token:
        raise RuntimeError("Failed to obtain access token from ADC")
    return creds.token


def _extract_reply_text(df_response: Dict[str, Any]) -> str:
    """
    Extracts the best-effort text response from detectIntent response.
    """
    # detectIntent response has queryResult.responseMessages[]
    query_result = df_response.get("queryResult", {})
    msgs = query_result.get("responseMessages", []) or []

    texts: List[str] = []
    for m in msgs:
        # Typical: { "text": { "text": ["..."] } }
        t = (m.get("text") or {}).get("text")
        if isinstance(t, list):
            texts.extend([x for x in t if isinstance(x, str)])
        elif isinstance(t, str):
            texts.append(t)

    # fallback
    if texts:
        return "\n".join(texts).strip()
    return "קיבלתי. איך תרצה להמשיך?"


def _extract_session_params(df_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Session parameters often appear under queryResult.parameters
    (format: Struct). We best-effort extract known keys.
    """
    query_result = df_response.get("queryResult", {})
    params = query_result.get("parameters") or {}
    # Some responses wrap parameters differently; keep best-effort.
    if not isinstance(params, dict):
        return {}
    return params


def _normalize_for_ui(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map session parameters to the UI schema you want.
    Adjust keys to match what your playbooks actually write.
    """
    # These keys are placeholders — change them to match your playbook parameter names
    business_stage = params.get("business_stage")
    license_status = params.get("license_status")
    business_tags = params.get("business_tags")

    # Checklist might come as an array parameter from your playbook
    checklist = params.get("compliance_checklist") or []

    # Normalize checklist to list[str]
    if isinstance(checklist, str):
        checklist_list = [checklist]
    elif isinstance(checklist, list):
        checklist_list = [str(x) for x in checklist]
    else:
        checklist_list = []

    return {
        "business_profile": {
            "business_stage": str(business_stage) if business_stage is not None else None,
            "license_status": str(license_status) if license_status is not None else None,
            "business_tags": str(business_tags) if business_tags is not None else None,
        },
        "compliance_checklist": checklist_list
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = (req.session_id or str(uuid.uuid4()))[:36]  # DF recommends <= 36 chars :contentReference[oaicite:1]{index=1}
    session_path = f"{AGENT_PATH}/sessions/{session_id}"
    url = f"{DIALOGFLOW_API_BASE}/{session_path}:detectIntent"

    token = _get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    body: Dict[str, Any] = {
        "queryInput": {
            "languageCode": LANGUAGE_CODE,
            "text": {"text": req.message}
        }
    }

    # If you want to force start/continue the session from a specific playbook:
    # QueryParameters.currentPlaybook supports full playbook resource name :contentReference[oaicite:2]{index=2}
    if CURRENT_PLAYBOOK:
        body["queryParams"] = {"currentPlaybook": CURRENT_PLAYBOOK}

    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    df_response = resp.json()

    reply_text = _extract_reply_text(df_response)
    params = _extract_session_params(df_response)
    ui = _normalize_for_ui(params)

    debug_raw = os.getenv("DEBUG_RAW") == "1"
    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        business_profile=ui["business_profile"],
        compliance_checklist=ui["compliance_checklist"],
        raw_agent_response=df_response if debug_raw else None
    )
