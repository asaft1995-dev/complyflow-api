import os
import uuid
from typing import Optional, Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest


# ============================================================
# Config (ENV with safe defaults for YOUR current project)
# ============================================================

# Defaulting to your valid project/agent to avoid "invalid project" issues
PROJECT_ID = os.getenv("PROJECT_ID", "yuvalfrank").strip()
LOCATION = os.getenv("LOCATION", "us-central1").strip()
AGENT_ID = os.getenv("AGENT_ID", "b470ba36-2b1d-4f48-a334-3bcdc5162445").strip()

LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "he").strip()
DIALOGFLOW_API_BASE = os.getenv("DIALOGFLOW_API_BASE", "https://dialogflow.googleapis.com/v3").strip()

# Optional: force start/continue from a specific playbook (full resource name)
CURRENT_PLAYBOOK = os.getenv("CURRENT_PLAYBOOK")
CURRENT_PLAYBOOK = CURRENT_PLAYBOOK.strip() if CURRENT_PLAYBOOK else None

# Vision Tool (from your OpenAPI)
VISION_TOOL_URL = os.getenv(
    "VISION_TOOL_URL",
    "https://business-vision-analyzer-550357153823.us-central1.run.app/"
).strip()

# CORS (set to your GitHub Pages origin; use * only for quick testing)
# Example:
# CORS_ALLOW_ORIGINS="https://asaft1995.github.io"
CORS_ALLOW_ORIGINS_RAW = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
CORS_ALLOW_ORIGINS = ["*"] if CORS_ALLOW_ORIGINS_RAW == "*" else [
    o.strip() for o in CORS_ALLOW_ORIGINS_RAW.split(",") if o.strip()
]

AGENT_PATH = f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ID}"


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="ComplyFlow Gateway (Cloud Run → Dialogflow CX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    # base64 ONLY (no "data:image/...;base64," prefix)
    image_data: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    business_profile: Dict[str, Optional[str]]
    compliance_checklist: List[str]
    vision_debug: Optional[Dict[str, Any]] = None
    raw_agent_response: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_path": AGENT_PATH,
        "vision_tool_url": VISION_TOOL_URL,
        "language_code": LANGUAGE_CODE,
        "current_playbook": CURRENT_PLAYBOOK,
        "cors_allow_origins": CORS_ALLOW_ORIGINS,
    }


def _get_access_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GoogleAuthRequest())
    if not creds.token:
        raise RuntimeError("Failed to obtain access token from ADC")
    return creds.token


def _extract_reply_text(df_response: Dict[str, Any]) -> str:
    query_result = df_response.get("queryResult", {}) or {}
    msgs = query_result.get("responseMessages", []) or []

    chunks: List[str] = []
    for m in msgs:
        txt = (m.get("text") or {}).get("text")
        if isinstance(txt, list):
            chunks.extend([t for t in txt if isinstance(t, str)])
        elif isinstance(txt, str):
            chunks.append(txt)

    return "\n".join(chunks).strip() if chunks else "קיבלתי. איך תרצה להמשיך?"


def _extract_session_params(df_response: Dict[str, Any]) -> Dict[str, Any]:
    query_result = df_response.get("queryResult", {}) or {}
    params = query_result.get("parameters") or {}
    return params if isinstance(params, dict) else {}


def _normalize_for_ui(params: Dict[str, Any]) -> Dict[str, Any]:
    # Adjust these keys to match your playbook parameter names
    business_stage = params.get("business_stage")
    license_status = params.get("license_status")
    business_tags = params.get("business_tags")

    checklist = params.get("compliance_checklist") or []

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
        "compliance_checklist": checklist_list,
    }


def _call_vision_tool(image_b64: str) -> Dict[str, Any]:
    try:
        r = requests.post(
            VISION_TOOL_URL,
            headers={"Content-Type": "application/json"},
            json={"image_data": image_b64},
            timeout=30,
        )
        if r.status_code >= 400:
            return {"success": False, "error": f"Vision HTTP {r.status_code}: {r.text[:500]}"}

        data = r.json()
        if not isinstance(data, dict) or "vision_data" not in data:
            return {"success": False, "error": "Vision payload missing vision_data"}

        return data
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = (req.session_id or str(uuid.uuid4()))[:36]
    session_path = f"{AGENT_PATH}/sessions/{session_id}"
    url = f"{DIALOGFLOW_API_BASE}/{session_path}:detectIntent"

    token = _get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    # 1) Vision outside Dialogflow (only if image uploaded)
    vision_result: Optional[Dict[str, Any]] = None
    if req.image_data:
        vision_result = _call_vision_tool(req.image_data)

    # 2) Dialogflow detectIntent request
    body: Dict[str, Any] = {
        "queryInput": {
            "languageCode": LANGUAGE_CODE,
            "text": {"text": req.message},
        }
    }

    query_params: Dict[str, Any] = {}
    if CURRENT_PLAYBOOK:
        query_params["currentPlaybook"] = CURRENT_PLAYBOOK

    # Pass small vision_data (NOT the base64) as session parameters
    if vision_result and vision_result.get("success") and isinstance(vision_result.get("vision_data"), dict):
        query_params.setdefault("parameters", {})
        query_params["parameters"]["vision_data"] = vision_result["vision_data"]
        query_params["parameters"]["has_image"] = True

    if query_params:
        body["queryParams"] = query_params

    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    df_response = resp.json()

    reply_text = _extract_reply_text(df_response)
    params = _extract_session_params(df_response)
    ui = _normalize_for_ui(params)

    debug_raw = os.getenv("DEBUG_RAW") == "1"
    debug_vision = os.getenv("DEBUG_VISION") == "1"

    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        business_profile=ui["business_profile"],
        compliance_checklist=ui["compliance_checklist"],
        vision_debug=vision_result if debug_vision else None,
        raw_agent_response=df_response if debug_raw else None,
    )
