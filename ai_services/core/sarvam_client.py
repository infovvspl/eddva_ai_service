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

import requests as _requests

logger = logging.getLogger("ai_services.sarvam")

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"
SARVAM_MODEL    = "mayura:v1"
SARVAM_CHUNK    = 900          # Sarvam hard limit is ~1 000 chars; use 900 for safety

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
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_sarvam_code(lang: str) -> str:
    """Convert a short language code (e.g. 'hi') to a Sarvam BCP-47 code."""
    return LANGUAGE_CODE_MAP.get(lang.lower(), f"{lang}-IN")


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
