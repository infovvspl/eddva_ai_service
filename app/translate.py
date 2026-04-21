from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx

router = APIRouter(prefix="/translate", tags=["Translation"])

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


@router.post("/", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured")

    target_code = LANGUAGE_CODE_MAP.get(req.targetLanguage, f"{req.targetLanguage}-IN")
    # Detect source — assume English if target is not English, else Hindi
    source_code = "en-IN" if req.targetLanguage != "en" else "hi-IN"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.sarvam.ai/translate",
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "input": req.text,
                    "source_language_code": source_code,
                    "target_language_code": target_code,
                    "speaker_gender": "Female",
                    "mode": "formal",
                    "model": "mayura:v1",
                    "enable_preprocessing": False,
                },
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=resp.text)

        data = resp.json()
        translated = data.get("translated_text", "")
        if not translated:
            raise HTTPException(status_code=500, detail="Empty translation response")
        return {"translatedText": translated}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
