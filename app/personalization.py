import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
def groq_personalization(request: StudyPlanRequest) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    client = Groq(api_key=GROQ_API_KEY)

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
@router.post("/generate", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):
    result = groq_personalization(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {"status": "ok", "service": "personalization", "llm": f"groq/{GROQ_MODEL}"}


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8005)
