"""
sarvam_client.py

Synchronous Sarvam AI translation helper for use in Django views.

Sarvam AI — Translate API
  Endpoint:  POST https://api.sarvam.ai/translate
  Auth:      api-subscription-key header
  Model:     mayura:v1
  Max chars: ~1000 per request (we chunk at 900)

Supports all 11 Sarvam-supported Indian languages:
  hi, en, bn, te, mr, ta, gu, kn, ml, pa, od
"""

import logging
import os
import time

import requests as _requests

logger = logging.getLogger("ai_services.sarvam")

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"
SARVAM_MODEL    = "mayura:v1"
SARVAM_CHUNK    = 900          # Sarvam hard limit is ~1 000 chars; use 900 for safety
SARVAM_STT_ENDPOINT = os.getenv("SARVAM_STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "transcribe")
SARVAM_CHAT_ENDPOINT = os.getenv("SARVAM_CHAT_ENDPOINT", "https://api.sarvam.ai/v1/chat/completions")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")

# ISO 639-1 short code → Sarvam BCP-47 code
LANGUAGE_CODE_MAP: dict[str, str] = {
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
    "odia": "od-IN",
    "or": "od-IN",
    "od-in": "od-IN",
    "or-in": "od-IN",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_sarvam_code(lang: str) -> str:
    """Convert a short language code (e.g. 'hi') to a Sarvam BCP-47 code."""
    cleaned = str(lang or "").strip().lower()
    if not cleaned:
        return "unknown"
    if cleaned == "auto":
        return "unknown"
    if "-" in cleaned and cleaned.endswith("-in"):
        return LANGUAGE_CODE_MAP.get(cleaned, f"{cleaned.split('-', 1)[0]}-IN")
    return LANGUAGE_CODE_MAP.get(cleaned, f"{cleaned}-IN")


def _chunk_text(text: str, size: int = SARVAM_CHUNK) -> list[str]:
    """
    Split long text into chunks ≤ size chars, breaking at sentence boundaries
    ('. ') where possible so translation context is preserved.
    """
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in text.replace("\n", " \n ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 2 <= size:
            current += ("" if not current else ". ") + sentence
        else:
            if current:
                chunks.append(current)
            # If a single sentence is longer than size, hard-cut it
            current = sentence[:size]
    if current:
        chunks.append(current)
    return chunks or [text[:size]]


def _translate_chunk(
    text: str,
    source_code: str,
    target_code: str,
    api_key: str,
    timeout: int = 30,
) -> str:
    """
    Translate one chunk via the Sarvam API.
    Raises RuntimeError on HTTP / parsing errors.
    """
    resp = _requests.post(
        SARVAM_ENDPOINT,
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "input": text,
            "source_language_code": source_code,
            "target_language_code": target_code,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": SARVAM_MODEL,
            "enable_preprocessing": False,
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Sarvam API error {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    translated = data.get("translated_text", "")
    if not translated:
        raise RuntimeError("Sarvam returned an empty translated_text field")
    return translated


# ── Public API ────────────────────────────────────────────────────────────────

def translate(
    text: str,
    target_language: str,
    source_language: str = "auto",
) -> str:
    """
    Translate `text` into `target_language` using Sarvam AI.

    Args:
        text:             The text to translate.
        target_language:  Short code, e.g. 'hi', 'en', 'ta'.
        source_language:  Short code or 'auto'.  When 'auto', the source is
                          inferred as 'en' when translating to an Indian
                          language, and 'hi' when translating to English.

    Returns:
        Translated string.

    Raises:
        RuntimeError: if SARVAM_API_KEY is missing or the API call fails.
    """
    api_key = SARVAM_API_KEY
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set — add it to .env to enable Sarvam translation"
        )

    target_code = _to_sarvam_code(target_language)

    # Auto-detect source: translating TO English → source is probably Hindi;
    # translating to any Indian language → source is English.
    if source_language == "auto" or not source_language:
        source_language = "hi" if target_language == "en" else "en"
    source_code = _to_sarvam_code(source_language)

    if source_code == target_code:
        logger.info("sarvam_client: source == target (%s), skipping API call", source_code)
        return text

    logger.info(
        "Sarvam translate | %s → %s | %d chars",
        source_code, target_code, len(text),
    )

    chunks = _chunk_text(text)
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        try:
            part = _translate_chunk(chunk, source_code, target_code, api_key)
            parts.append(part)
            logger.debug("Sarvam chunk %d/%d OK (%d chars)", i + 1, len(chunks), len(part))
        except RuntimeError as exc:
            logger.error("Sarvam chunk %d/%d FAILED: %s", i + 1, len(chunks), exc)
            raise

    return " ".join(parts).strip()


def chat_complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    timeout: int = 90,
) -> dict:
    """
    Generate text with Sarvam's OpenAI-compatible chat completion API.

    Sarvam is used for Indian-language generation where Groq quality is weak.
    Returns a small dict shaped similarly to the local LLM client:
    {"content": str, "model": str, "usage": dict}
    """
    api_key = os.getenv("SARVAM_API_KEY", SARVAM_API_KEY)
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set -- add it to .env to enable Sarvam chat completion"
        )

    selected_model = (model or os.getenv("SARVAM_CHAT_MODEL", SARVAM_CHAT_MODEL) or "sarvam-105b").strip()
    payload: dict = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": str(system_prompt or "").strip()},
            {"role": "user", "content": str(user_prompt or "").strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {
        "api-subscription-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = _requests.post(
                SARVAM_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) * 2
                logger.warning("Sarvam chat busy/rate-limited (%s); retrying in %.1fs", resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Sarvam chat API error {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            choices = data.get("choices") or []
            message = (choices[0].get("message") if choices and isinstance(choices[0], dict) else {}) or {}
            raw_content = message.get("content")
            if isinstance(raw_content, list):
                content = "\n".join(
                    str(part.get("text") or part.get("content") or "")
                    for part in raw_content
                    if isinstance(part, dict)
                ).strip()
            else:
                content = str(raw_content or "").strip()
            if not content and choices and isinstance(choices[0], dict):
                content = str(choices[0].get("text") or "").strip()
            if not content:
                finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
                logger.warning(
                    "Sarvam chat returned empty content | model=%s | finish_reason=%s | usage=%s | body=%s",
                    data.get("model") or selected_model,
                    finish_reason,
                    data.get("usage"),
                    resp.text[:800],
                )
                if payload.pop("reasoning_effort", None) and attempt < 2:
                    logger.info("Retrying Sarvam chat without reasoning_effort after empty content")
                    time.sleep(1.0)
                    continue
                if finish_reason == "length" and attempt < 2:
                    payload["max_tokens"] = 4096
                    logger.info("Retrying Sarvam chat with larger max_tokens=%s after empty length finish", payload["max_tokens"])
                    time.sleep(1.0)
                    continue
                raise RuntimeError("Sarvam chat returned an empty message.content field")
            logger.info(
                "Sarvam chat OK | model=%s | chars=%d | total_tokens=%s",
                data.get("model") or selected_model,
                len(content),
                (data.get("usage") or {}).get("total_tokens"),
            )
            return {
                "content": content,
                "model": data.get("model") or selected_model,
                "usage": data.get("usage") or {},
            }
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep((2 ** attempt) * 1.5)
                continue
            break

    raise RuntimeError(f"Sarvam chat failed: {last_error}") from last_error


def transcribe_file(
    audio_path: str,
    language: str = "od",
    timeout: int = 60,
) -> str:
    """
    Transcribe a short audio chunk with Sarvam Speech-to-Text.

    The caller should pass short chunks (Sarvam REST is intended for quick
    responses under ~30 seconds). Use language='od' for Odia, which maps to
    Sarvam's BCP-47 code 'od-IN'.
    """
    api_key = os.getenv("SARVAM_API_KEY", SARVAM_API_KEY)
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set -- add it to .env to enable Sarvam speech-to-text"
        )
    if not audio_path or not os.path.exists(audio_path):
        raise RuntimeError(f"Audio chunk does not exist: {audio_path}")

    language_code = _to_sarvam_code(language)
    form_data = {
        "model": SARVAM_STT_MODEL,
        "mode": SARVAM_STT_MODE,
    }
    if language_code and language_code != "unknown":
        form_data["language_code"] = language_code

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with open(audio_path, "rb") as f:
                resp = _requests.post(
                    SARVAM_STT_ENDPOINT,
                    headers={"api-subscription-key": api_key},
                    files={"file": (os.path.basename(audio_path), f, "audio/mpeg")},
                    data=form_data,
                    timeout=timeout,
                )
            if resp.status_code in (429, 503) and attempt < 2:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) * 2
                logger.warning("Sarvam STT rate-limited/service busy; retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Sarvam STT API error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            transcript = str(data.get("transcript") or "").strip()
            if not transcript:
                raise RuntimeError("Sarvam STT returned an empty transcript field")
            logger.debug(
                "Sarvam STT chunk OK | lang=%s | detected=%s | chars=%d",
                language_code,
                data.get("language_code"),
                len(transcript),
            )
            return transcript
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep((2 ** attempt) * 1.5)
                continue
            break

    raise RuntimeError(f"Sarvam STT failed: {last_error}") from last_error
