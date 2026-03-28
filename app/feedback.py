import os
import json
import re
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from contextlib import asynccontextmanager

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

router = APIRouter(prefix="/feedback", tags=["Exam Feedback & Grading"])


# --- Lifespan for Unified App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization logic (e.g., loading models if needed)
    print("[FEEDBACK] Initializing lifespan...")
    yield
    print("[FEEDBACK] Cleaning up lifespan...")


# --- Schemas ---
class FeedbackRequest(BaseModel):
    subject: str
    student_answer: str
    marking_scheme: str
    extra_context: Optional[str] = None


class FeedbackResponse(BaseModel):
    score: float
    feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]
    suggested_resources: List[str]


# --- Helpers ---
def groq_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    You are an expert academic evaluator. Analyze the student's answer based on the marking scheme and provide detailed feedback.
    
    Subject: {request.subject}
    Marking Scheme: {request.marking_scheme}
    Student Answer: {request.student_answer}
    Extra Context: {request.extra_context or "N/A"}
    
    Provide your output in valid JSON format with the following structure:
    {{
        "score": (a numeric value out of 100),
        "feedback": "a cohesive paragraph of constructive criticism",
        "strengths": ["list of strengths"],
        "areas_for_improvement": ["list of areas"],
        "suggested_resources": ["links or names of topics to study"]
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
@router.post("/analyze", response_model=FeedbackResponse)
async def analyze_feedback(request: FeedbackRequest):
    result = groq_feedback(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {"status": "ok", "service": "feedback", "llm": f"groq/{GROQ_MODEL}"}


if __name__ == "__main__":
    import uvicorn

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8001)
