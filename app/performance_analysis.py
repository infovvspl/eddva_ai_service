import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

router = APIRouter(prefix="/performance", tags=["Student Performance Analysis"])


# --- Schemas ---
class StudentScores(BaseModel):
    subject: str
    scores: List[float]
    test_dates: List[str]


class PerformanceRequest(BaseModel):
    student_id: str
    subjects_data: List[StudentScores]


class PerformanceResponse(BaseModel):
    overall_summary: str
    subject_analysis: List[Dict[str, Any]]
    improvement_plan: List[str]
    predicted_grade: str


# --- Helpers ---
def groq_performance_analysis(request: PerformanceRequest) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    client = Groq(api_key=GROQ_API_KEY)

    data_summary = []
    for s in request.subjects_data:
        data_summary.append(f"{s.subject}: {s.scores} on {s.test_dates}")

    prompt = f"""
    You are an educational data analyst. Analyze the following student performance data and provide insights.
    
    Student ID: {request.student_id}
    Data:
    {" | ".join(data_summary)}
    
    Provide your output in valid JSON format with the following structure:
    {{
        "overall_summary": "general overview of performance",
        "subject_analysis": [
            {{
                "subject": "name",
                "trend": "improving/declining/stable",
                "key_issue": "why?",
                "avg_score": 85.0
            }}
        ],
        "improvement_plan": ["action 1", "action 2"],
        "predicted_grade": "A/B/C/etc"
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
@router.post("/analyze", response_model=PerformanceResponse)
async def analyze_performance(request: PerformanceRequest):
    result = groq_performance_analysis(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "performance_analysis",
        "llm": f"groq/{GROQ_MODEL}",
    }


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8003)
