import os
import json
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

router = APIRouter(prefix="/cheating", tags=["Cheating Detection Service"])


# --- Schemas ---
class CheatingReportResponse(BaseModel):
    is_suspicious: bool
    confidence_score: float
    violations: List[str]
    timeline_events: List[Dict[str, Any]]
    summary: str


# --- Helpers ---
def analyze_logs_for_cheating(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    You are a proctoring AI. Analyze the following activity logs for suspicious behavior during an exam.
    
    Logs:
    {json.dumps(logs, indent=2)}
    
    Provide your output in valid JSON format with the following structure:
    {{
        "is_suspicious": true/false,
        "confidence_score": 0.0 to 1.0,
        "violations": ["Multiple faces detected", "Tab switching", "Unauthorized device"],
        "timeline_events": [
            {{
                "timestamp": "HH:MM:SS",
                "event": "Description",
                "risk_level": "high/medium/low"
            }}
        ],
        "summary": "Cohesive summary of the findings."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# --- Endpoints ---
@router.post("/analyze-logs", response_model=CheatingReportResponse)
async def analyze_exam_logs(logs: List[Dict[str, Any]]):
    result = analyze_logs_for_cheating(logs)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cheating_detection",
        "llm": f"groq/{GROQ_MODEL}",
    }


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8006)
