from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import os
import httpx
import json
from io import BytesIO
from ai_services.core.groq_keys import get_rotated_groq_keys
from PIL import Image
import easyocr
import numpy as np

router = APIRouter(prefix="/doubt", tags=["Doubt Resolution"], redirect_slashes=False)


class StudentContext(BaseModel):
    source: Optional[str] = None
    sourceRefId: Optional[str] = None
    doubtId: Optional[str] = None
    batchId: Optional[str] = None


class ResolveDoubtRequest(BaseModel):
    questionText: str
    questionImageUrl: Optional[str] = None
    topicId: Optional[str] = None
    mode: str = "detailed"  # 'short' | 'detailed'
    studentContext: Optional[Any] = None


class ResolveDoubtResponse(BaseModel):
    explanation: str
    conceptLinks: list[str] = []

class OcrImageRequest(BaseModel):
    imageUrl: str


class OcrImageResponse(BaseModel):
    text: str


def _build_prompt(req: ResolveDoubtRequest) -> str:
    length = "concise (2-3 sentences)" if req.mode == "short" else "detailed and step-by-step"
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    if not question_text:
        question_text = "No typed question provided."
    image_hint = (
        f"\nImage URL (for reference): {req.questionImageUrl}\n"
        if req.questionImageUrl
        else ""
    )
    return (
        f"You are an expert tutor. A student has asked the following question.{topic_hint}\n\n"
        f"Question text: {question_text}\n"
        f"{image_hint}\n"
        f"If this came from a screenshot or book photo, rely on extracted text and solve the exact asked problem.\n"
        f"For mathematical/derivation questions, use equation-first formatting with minimal prose.\n"
        f"Show symbolic steps clearly and end with a final result line.\n"
        f"Provide a {length} explanation. Also list 2-3 key concept names as a JSON array in 'key_concepts'.\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"explanation": "...", "key_concepts": ["concept1", "concept2"]}}'
    )


_ocr_reader: Optional[easyocr.Reader] = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en", "hi"], gpu=False)
    return _ocr_reader


def _prepare_ocr_variants(img_rgb: np.ndarray) -> list[np.ndarray]:
    # Multiple variants improve OCR across screenshots and camera-captured book pages.
    gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    bw = np.where(gray > 165, 255, 0).astype(np.uint8)
    p5, p95 = np.percentile(gray, (5, 95))
    if p95 > p5:
        stretch = np.clip((gray - p5) * (255.0 / (p95 - p5)), 0, 255).astype(np.uint8)
    else:
        stretch = gray
    return [img_rgb, gray, stretch, bw]


async def _extract_text_from_image_url(image_url: str) -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(image_url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    arr = np.array(img)
    reader = _get_ocr_reader()
    best = ""
    for v in _prepare_ocr_variants(arr):
        try:
            parts = reader.readtext(v, detail=0, paragraph=True)
            text = " ".join([str(x).strip() for x in parts if str(x).strip()]).strip()
            if len(text) > len(best):
                best = text
        except Exception:
            continue
    return best


@router.post("/ocr-image", response_model=OcrImageResponse)
async def ocr_image(req: OcrImageRequest):
    try:
        text = await _extract_text_from_image_url(req.imageUrl)
        return {"text": text or ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


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
    groq_keys = get_rotated_groq_keys()
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_keys and not gemini_key:
        raise HTTPException(status_code=500, detail="No AI API key configured")

    ocr_text = ""
    if req.questionImageUrl:
        try:
            ocr_text = await _extract_text_from_image_url(req.questionImageUrl)
        except Exception:
            ocr_text = ""

    req_for_prompt = req
    if ocr_text:
        merged_text = (req.questionText or "").strip()
        merged_text = (merged_text + "\n\nExtracted from image:\n" + ocr_text).strip()
        req_for_prompt = ResolveDoubtRequest(
            questionText=merged_text,
            questionImageUrl=req.questionImageUrl,
            topicId=req.topicId,
            mode=req.mode,
            studentContext=req.studentContext,
        )

    prompt = _build_prompt(req_for_prompt)

    try:
        if groq_keys:
            last_groq_error = None
            result = None
            for i, groq_key in enumerate(groq_keys):
                try:
                    result = await _call_groq(prompt, groq_key)
                    break
                except HTTPException as e:
                    last_groq_error = e
                    text = str(e.detail).lower()
                    key_like_error = any(x in text for x in ("rate limit", "quota", "too many requests", "invalid api key", "authentication"))
                    if key_like_error and i < len(groq_keys) - 1:
                        continue
                    raise
            if result is None and last_groq_error:
                raise last_groq_error
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
