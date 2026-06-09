import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Configuration — local Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvaqwen")

router = APIRouter(prefix="/personalization", tags=["Study Plan Personalization"])


# --- Schemas ---
class StudyPlanRequest(BaseModel):
    student_id: str
    learning_style: str  # e.g., visual, auditory, kinesthetic
    available_hours_per_day: float
    subjects_to_focus: List[str]
    current_mood: Optional[str] = "motivated"


class StudySession(BaseModel):
    time_block: str
    subject: str
    activity: str
    resource_type: str


class StudyPlanResponse(BaseModel):
    student_id: str
    daily_schedule: List[StudySession]
    weekly_goals: List[str]
    motivation_quote: str


# --- Helpers ---
def ollama_personalization(request: StudyPlanRequest) -> Dict[str, Any]:
    prompt = f"""
    You are a personalized learning assistant. Create a daily study plan for a student.

    Student ID: {request.student_id}
    Learning Style: {request.learning_style}
    Hours available: {request.available_hours_per_day}
    Subjects: {", ".join(request.subjects_to_focus)}
    Mood: {request.current_mood}

    Provide your output in valid JSON format with the following structure:
    {{
        "student_id": "{request.student_id}",
        "daily_schedule": [
            {{
                "time_block": "08:00 - 09:00",
                "subject": "Math",
                "activity": "Solve problems using visual diagrams",
                "resource_type": "Video/Diagram"
            }}
        ],
        "weekly_goals": ["Goal 1", "Goal 2"],
        "motivation_quote": "A short inspiring quote"
    }}
    IMPORTANT: Respond with ONLY valid JSON. Start with {{ and end with }}
    """

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])
    except Exception as e:
        return {"error": str(e)}


# --- Endpoints ---
@router.post("/generate", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):
    result = ollama_personalization(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {"status": "ok", "service": "personalization", "llm": f"ollama/{OLLAMA_MODEL}"}


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8005)
