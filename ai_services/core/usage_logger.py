import httpx
import json
import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_NESTJS_BASE_URL = os.getenv('NESTJS_INTERNAL_URL', '')
_INTERNAL_API_KEY = os.getenv('INTERNAL_API_KEY', '')


def _first_present(*values):
    """First value that is actually present, else None.

    Treats None, '' and whitespace-only as absent. Callers in views/bridge.py
    derive user_id from the request body with an `or ''` tail, so an empty string
    is the normal "not supplied" case and must not be mistaken for a real value —
    downstream it would only be normalised to NULL anyway.
    """
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None

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

# Together list prices per 1M tokens, read from Together's own /v1/models on
# 2026-09-28. Keys are the provider model ids, lowercased for lookup. Prices
# change: AI_TOGETHER_PRICING (JSON: {"<model id>": {"input": x, "output": y}})
# overrides or extends this without a code change. A Together model that is in
# neither table stays unpriced (None) rather than being recorded as free.
TOGETHER_MODEL_COSTS = {
    'openai/gpt-oss-120b':                 {'input': 0.15, 'output': 0.60},
    'qwen/qwen3.8-flash':                  {'input': 0.09, 'output': 0.282},
    'zai-org/glm-5.3-flash':               {'input': 0.15, 'output': 0.50},
    'deepseek-ai/deepseek-v4-flash-0731':  {'input': 0.14, 'output': 0.28},
    'qwen/qwen3.7-max':                    {'input': 1.50, 'output': 4.50},
}


def _together_rates(provider_model_id: str):
    """Rates for a Together model id, or None when it has no configured price."""
    key = (provider_model_id or "").strip().lower()
    if not key:
        return None
    raw = os.getenv("AI_TOGETHER_PRICING")
    if raw:
        try:
            overrides = json.loads(raw)
            hit = {str(k).lower(): v for k, v in overrides.items()}.get(key)
            if isinstance(hit, dict) and "input" in hit and "output" in hit:
                return {"input": float(hit["input"]), "output": float(hit["output"])}
        except (ValueError, TypeError):
            logger.warning("AI_TOGETHER_PRICING is not valid JSON; using built-in Together rates")
    return TOGETHER_MODEL_COSTS.get(key)


def _is_unpriced_router_provider(model) -> bool:
    """A "<provider>:<model id>" id from the model router for a provider with no
    configured price (e.g. "together:..."). Reporting no estimate is honest;
    reporting $0 is not. NestJS stores a missing estimatedCost as NULL.

    Ids like "mayura:v1" are unaffected: only a known router provider prefix counts.
    """
    if not model or ":" not in str(model):
        return False
    try:
        from ai_services.core.routing.registry import KNOWN_PROVIDERS, UNQUALIFIED_TELEMETRY_PROVIDERS
    except Exception:
        return False
    prefix = str(model).split(":", 1)[0]
    return prefix in KNOWN_PROVIDERS and prefix not in UNQUALIFIED_TELEMETRY_PROVIDERS


def calculate_cost(model: str, tokens_input: int, tokens_output: int) -> "float | None":
    # Routed Together ids ("together:zai-org/GLM-5.3-Flash") carry their own price
    # table: the bare model id means something different on Together than it does
    # on Groq (gpt-oss-120b is billed by each of them separately).
    if str(model or "").startswith("together:"):
        rates = _together_rates(str(model).split(":", 1)[1])
        if not rates:
            return None
        return round(
            (tokens_input / 1_000_000) * rates['input'] +
            (tokens_output / 1_000_000) * rates['output'],
            6
        )
    if _is_unpriced_router_provider(model):
        return None
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
    user_role: str = None,
    request_id: str = None,
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
        "userRole": user_role,
        "requestId": request_id,
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
            logger.debug("AI usage logged: feature=%s model=%s cost=%s", feature_id, model_used, cost)
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
    user_role: str = None,
    request_id: str = None,
):
    """Fire-and-forget — never blocks the AI response."""
    # P1-6: the authenticated request context is the SOURCE OF TRUTH for attribution.
    #
    # TenantAuthMiddleware stamps it from the ai-bridge's X-User-Id / X-User-Role /
    # X-Request-Id headers, which the bridge derives from the verified JWT. What a
    # caller passes as user_id comes from the request *body*
    # (`data.get('userId') ... or ''`), which is empty for worker/background calls
    # and client-supplied — therefore spoofable — when present. So the context wins
    # whenever it has a value, and the caller's value is used only as a fallback,
    # which preserves behaviour for callers running without an authenticated context.
    #
    # Empty strings count as absent. Previously `user_id=''` was not None, so the
    # fallback was skipped entirely and the empty string became NULL downstream —
    # every lecture/STT event lost its user attribution that way.
    try:
        from ai_services.core import request_context
        user_id = _first_present(request_context.get("user_id"), user_id)
        user_role = _first_present(request_context.get("user_role"), user_role)
        request_id = _first_present(request_context.get("request_id"), request_id)
    except Exception:
        # Attribution must never break generation; fall through with what we have.
        pass

    # Never ship an empty string to the usage webhook — normalise to None so the
    # payload says "unknown" rather than "". Also covers the path where the import
    # above failed.
    user_id = _first_present(user_id)
    user_role = _first_present(user_role)
    request_id = _first_present(request_id)
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
            user_role=user_role,
            request_id=request_id,
        ),
        daemon=True,
    )
    t.start()
