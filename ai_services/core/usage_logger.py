import httpx
import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_NESTJS_BASE_URL = os.getenv('NESTJS_INTERNAL_URL', '')
_INTERNAL_API_KEY = os.getenv('INTERNAL_API_KEY', '')

MODEL_COSTS = {
    'llama-3.1-8b-instant':           {'input': 0.05,  'output': 0.08},
    'llama-3.3-70b-versatile':        {'input': 0.59,  'output': 0.79},
    'qwen/qwen3-32b':                  {'input': 0.29,  'output': 0.59},
    'openai/gpt-oss-120b':             {'input': 0.15,  'output': 0.60},
    'openai/gpt-oss-20b':              {'input': 0.075, 'output': 0.30},
    'meta-llama/llama-4-scout-17b-16e-instruct':     {'input': 0.11,  'output': 0.34},
    'meta-llama/llama-4-maverick-17b-128e-instruct': {'input': 0.20,  'output': 0.60},
    'llama-3.2-11b-vision-preview':    {'input': 0.18,  'output': 0.18},
    'llama-3.2-90b-vision-preview':    {'input': 0.90,  'output': 0.90},
    'qwen/qwen3.6-27b':                {'input': 0.11,  'output': 0.34},
    'gemini-2.5-flash':                {'input': 0.075, 'output': 0.30},
    'mayura:v1':                       {'input': 0.10,  'output': 0.10},
    'whisper-large-v3-turbo':          {'input': 0.04,  'output': 0.0},
    'faster-whisper-local':            {'input': 0.0,   'output': 0.0},
    'easyocr-local':                   {'input': 0.0,   'output': 0.0},
    'sarvam-stt':                      {'input': 0.04,  'output': 0.0},
}

def calculate_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    rates = MODEL_COSTS.get(model, MODEL_COSTS.get(model.split('/')[-1], None))
    if not rates:
        return 0.0
    return round(
        (tokens_input / 1_000_000) * rates['input'] +
        (tokens_output / 1_000_000) * rates['output'],
        6
    )


def log_ai_usage_sync(
    institute_id: str,
    institute_type: str,
    feature_id: str,
    feature_category: str,
    model_used: str,
    tokens_input: int = 0,
    tokens_output: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error_message: str = None,
    user_id: str = None,
):
    cost = calculate_cost(model_used, tokens_input, tokens_output)
    payload = {
        "instituteId": institute_id,
        "instituteType": institute_type,
        "featureId": feature_id,
        "featureCategory": feature_category,
        "modelUsed": model_used,
        "tokensInput": tokens_input,
        "tokensOutput": tokens_output,
        "estimatedCost": cost,
        "latencyMs": latency_ms,
        "success": success,
        "errorMessage": error_message,
        "userId": user_id,
    }
    # Read at call time so gunicorn worker always picks up the deployed .env values
    nestjs_url = os.getenv('NESTJS_INTERNAL_URL', '') or _NESTJS_BASE_URL
    api_key = os.getenv('INTERNAL_API_KEY', '') or _INTERNAL_API_KEY
    if not nestjs_url:
        logger.error("AI usage log skipped: NESTJS_INTERNAL_URL is not set (feature=%s)", feature_id)
        return
    url = f"{nestjs_url}/api/v1/internal/ai-usage/log"
    headers = {"X-Internal-Key": api_key}
    last_err = None
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
            resp.raise_for_status()
            logger.debug("AI usage logged: feature=%s model=%s cost=$%.6f", feature_id, model_used, cost)
            return
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s
    logger.error("AI usage logging failed after 3 attempts — url=%s feature=%s err=%s", url, feature_id, last_err)


def log_usage(
    institute_id: str,
    institute_type: str,
    feature_id: str,
    feature_category: str,
    model_used: str,
    tokens_input: int = 0,
    tokens_output: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error_message: str = None,
    user_id: str = None,
):
    """Fire-and-forget — never blocks the AI response."""
    # Count the spend against the tenant's daily budget.
    #
    # This is the ONE place tokens are booked. Every generating endpoint already
    # calls log_usage, so putting the accounting here means the budget reflects
    # all traffic rather than only the calls that happen to run through
    # ai_call() — previously the majority of spend (grounded content, vision
    # OCR, transcription) was invisible to the cap it was supposed to obey.
    #
    # Done on the caller's thread, before the reporting thread starts: the next
    # request's budget check must see this call, and a daemon thread gives no
    # such ordering. It is a Redis INCR, so the cost is microseconds, and any
    # failure is swallowed — accounting must never break generation.
    try:
        from ai_services.core.rate_limiter import get_shared_limiter
        total = (tokens_input or 0) + (tokens_output or 0)
        if total > 0:
            get_shared_limiter().record_usage(institute_id or "default", total)
    except Exception as exc:
        logger.warning("Could not record usage against the daily budget: %s", exc)

    t = threading.Thread(
        target=log_ai_usage_sync,
        kwargs=dict(
            institute_id=institute_id or '',
            institute_type=institute_type or 'school',
            feature_id=feature_id,
            feature_category=feature_category,
            model_used=model_used or 'unknown',
            tokens_input=tokens_input or 0,
            tokens_output=tokens_output or 0,
            latency_ms=latency_ms or 0,
            success=success,
            error_message=error_message,
            user_id=user_id,
        ),
        daemon=True,
    )
    t.start()
