"""
Gemini client — used only where a large context window is required.

Groq remains the default for this service. Grounded generation is the exception:
a chapter of a textbook is several thousand tokens, and Groq's on-demand tier
rejects any request whose prompt plus max_tokens exceeds 12,000 for the key
(see _MAX_TOKENS in views/ppt.py). Gemini's context comfortably fits a whole
chapter, so source-grounded calls go here and everything else stays on Groq.

The import is local to the call, matching core/note_images.py: google-genai is
optional in some environments, and importing at module load would break the
whole service where it is absent.
"""
import json
import logging
import os
import time

logger = logging.getLogger("ai_services.gemini")

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiUnavailable(RuntimeError):
    """Raised when the SDK or API key is missing, so callers can fall back."""


def is_available() -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def complete_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> dict:
    """Return {content: dict, model, latency_ms} for a JSON-mode Gemini call.

    Raises GeminiUnavailable when unusable so the caller can fall back to Groq
    rather than failing the request outright.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set")
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    started = time.time()
    client = genai.Client(api_key=api_key)
    result = client.models.generate_content(
        model=model,
        # Gemini has no separate system role here; the instructions are prepended
        # and the source material follows, which keeps the ordering the model
        # sees identical to the Groq path.
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    latency_ms = int((time.time() - started) * 1000)

    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    try:
        content = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", exc)
        raise RuntimeError("Gemini returned malformed JSON") from exc

    usage = getattr(result, "usage_metadata", None)
    return {
        "content": content,
        "model": model,
        "latency_ms": latency_ms,
        "tokens_input": getattr(usage, "prompt_token_count", 0) or 0,
        "tokens_output": getattr(usage, "candidates_token_count", 0) or 0,
    }
