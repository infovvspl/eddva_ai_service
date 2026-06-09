from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx

router = APIRouter(prefix="/translate", tags=["Translation"], redirect_slashes=False)

# Sarvam AI language codes
LANGUAGE_CODE_MAP = {
    "hi": "hi-IN",
    "en": "en-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "od": "od-IN",
}


class TranslateRequest(BaseModel):
    text: str
    targetLanguage: str  # e.g. 'hi', 'en', 'ta'


class TranslateResponse(BaseModel):
    translatedText: str


SARVAM_CHUNK_SIZE = 900  # Sarvam max is ~1000 chars


async def _translate_chunk(client: httpx.AsyncClient, text: str, source_code: str, target_code: str, api_key: str) -> str:
    resp = await client.post(
        "https://api.sarvam.ai/translate",
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        json={
            "input": text,
            "source_language_code": source_code,
            "target_language_code": target_code,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": False,
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Sarvam error: {resp.text}")
    data = resp.json()
    return data.get("translated_text", "")


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks at sentence boundaries where possible."""
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for sentence in text.replace("\n", " \n ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 2 <= size:
            current += ("" if not current else ". ") + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence[:size]
    if current:
        chunks.append(current)
    return chunks


@router.post("", response_model=TranslateResponse)
@router.post("/", response_model=TranslateResponse, include_in_schema=False)
async def translate_text(req: TranslateRequest):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured")

    target_code = LANGUAGE_CODE_MAP.get(req.targetLanguage, f"{req.targetLanguage}-IN")
    source_code = "en-IN" if req.targetLanguage != "en" else "hi-IN"

    try:
        chunks = _chunk_text(req.text, SARVAM_CHUNK_SIZE)
        translated_parts = []
        async with httpx.AsyncClient(timeout=60) as client:
            for chunk in chunks:
                part = await _translate_chunk(client, chunk, source_code, target_code, api_key)
                translated_parts.append(part)

        translated = " ".join(translated_parts).strip()
        if not translated:
            raise HTTPException(status_code=500, detail="Empty translation response")
        return {"translatedText": translated}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
