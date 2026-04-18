import os
import json
import re
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
from contextlib import asynccontextmanager

# Configuration — local Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvaqwen")

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
def ollama_feedback(request: FeedbackRequest) -> Dict[str, Any]:
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
@router.post("/analyze", response_model=FeedbackResponse)
async def analyze_feedback(request: FeedbackRequest):
    result = ollama_feedback(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {"status": "ok", "service": "feedback", "llm": f"ollama/{OLLAMA_MODEL}"}


if __name__ == "__main__":
    import uvicorn

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8001)
