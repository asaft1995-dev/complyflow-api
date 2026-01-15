from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid

app = FastAPI(title="ComplyFlow API")

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    business_profile: Dict[str, Optional[str]]
    compliance_checklist: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_msg = req.message.lower()

    # Minimal demo logic for course
    business_stage = "unknown"
    if "לפתוח" in user_msg or "חדש" in user_msg:
        business_stage = "planned"
    elif "פעיל" in user_msg or "קיים" in user_msg:
        business_stage = "operating"

    if business_stage == "planned":
        reply = "הבנתי. מה סוג העסק שאתה מתכנן לפתוח? (מסעדה / מאפייה / קייטרינג וכו׳)"
    elif business_stage == "operating":
        reply = "הבנתי. מה סטטוס הרישוי של העסק? (אין / זמני / מלא / לא ידוע)"
    else:
        reply = "כדי להתחיל: האם העסק מתוכנן לפתיחה או כבר פעיל?"

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        business_profile={
            "business_stage": business_stage,
            "license_status": "unknown",
            "business_tags": None
        },
        compliance_checklist=[]
    )
