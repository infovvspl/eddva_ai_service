import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

router = APIRouter(prefix="/career", tags=["Career Roadmap Builder"])


# --- Schemas ---
class CareerPlanRequest(BaseModel):
    interests: List[str]
    current_skills: List[str]
    goal: str
    timeline_months: int = 12


class RoadmapStep(BaseModel):
    milestone: str
    description: str
    recommended_resources: List[str]
    duration: str


class CareerPlanResponse(BaseModel):
    career_path: str
    summary: str
    roadmap: List[RoadmapStep]
    market_demand: str
    essential_skills: List[str]


# --- Helpers ---
def groq_career_plan(request: CareerPlanRequest) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    You are a career advisor. Create a detailed roadmap for a student.
    
    Interests: {", ".join(request.interests)}
    Current Skills: {", ".join(request.current_skills)}
    Goal: {request.goal}
    Timeline: {request.timeline_months} months
    
    Provide your output in valid JSON format with the following structure:
    {{
        "career_path": "short descriptive title",
        "summary": "overview of the path",
        "roadmap": [
            {{
                "milestone": "step name",
                "description": "what to learn",
                "recommended_resources": ["link or book"],
                "duration": "1-2 months"
            }}
        ],
        "market_demand": "brief outlook",
        "essential_skills": ["skill 1", "skill 2"]
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
@router.post("/generate", response_model=CareerPlanResponse)
async def generate_career_plan(request: CareerPlanRequest):
    result = groq_career_plan(request)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/health")
async def health():
    return {"status": "ok", "service": "career_roadmap", "llm": f"groq/{GROQ_MODEL}"}


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8002)
