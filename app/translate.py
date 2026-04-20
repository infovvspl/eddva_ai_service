from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai

router = APIRouter(prefix="/translate", tags=["Translation"])

LANGUAGE_MAP = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


class TranslateRequest(BaseModel):
    text: str
    targetLanguage: str


class TranslateResponse(BaseModel):
    translatedText: str


@router.post("/", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    lang_name = LANGUAGE_MAP.get(req.targetLanguage, req.targetLanguage)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Translate the following text to {lang_name}. "
            f"Return only the translated text, no explanations.\n\n"
            f"{req.text}"
        )

        response = model.generate_content(prompt)
        translated = response.text.strip()
        return {"translatedText": translated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
