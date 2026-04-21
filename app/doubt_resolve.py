from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import os
import httpx
import json

router = APIRouter(prefix="/doubt", tags=["Doubt Resolution"], redirect_slashes=False)


class StudentContext(BaseModel):
    source: Optional[str] = None
    sourceRefId: Optional[str] = None
    doubtId: Optional[str] = None
    batchId: Optional[str] = None


class ResolveDoubtRequest(BaseModel):
    questionText: str
    topicId: Optional[str] = None
    mode: str = "detailed"  # 'short' | 'detailed'
    studentContext: Optional[Any] = None


class ResolveDoubtResponse(BaseModel):
    explanation: str
    conceptLinks: list[str] = []


def _build_prompt(req: ResolveDoubtRequest) -> str:
    length = "concise (2-3 sentences)" if req.mode == "short" else "detailed and step-by-step"
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    return (
        f"You are an expert tutor. A student has asked the following question.{topic_hint}\n\n"
        f"Question: {req.questionText}\n\n"
        f"Provide a {length} explanation. Also list 2-3 key concept names as a JSON array in 'key_concepts'.\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"explanation": "...", "key_concepts": ["concept1", "concept2"]}}'
    )


async def _call_groq(prompt: str, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq error: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _call_gemini(prompt: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Gemini error: {resp.text}")
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


@router.post("/resolve", response_model=ResolveDoubtResponse)
async def resolve_doubt(req: ResolveDoubtRequest):
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key and not gemini_key:
        raise HTTPException(status_code=500, detail="No AI API key configured")

    prompt = _build_prompt(req)

    try:
        if groq_key:
            result = await _call_groq(prompt, groq_key)
        else:
            result = await _call_gemini(prompt, gemini_key)

        explanation = result.get("explanation", "")
        concepts = result.get("key_concepts", result.get("conceptLinks", []))

        if not explanation:
            raise HTTPException(status_code=500, detail="AI returned empty explanation")

        return {"explanation": explanation, "conceptLinks": concepts}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
