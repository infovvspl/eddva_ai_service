"""
Provider-attempt telemetry (P0-5).

llm_client / gemini_client rotate keys on a 429 or 5xx and then usually succeed on
the next key. Today that recovery is invisible: only the final success/failure is
logged, so "no rate limiting" and "constant rate limiting that rotation absorbed"
look identical. This module records each such attempt to NestJS so dashboards can
tell them apart.

Fire-and-forget on a daemon thread — telemetry must never add latency to, or break,
a generation call. Never sends a raw API key; only a short sha256 prefix.
"""
import hashlib
import logging
import os
import threading

import httpx

logger = logging.getLogger("ai_services.provider_events")

VALID_EVENT_TYPES = {"429", "5xx", "timeout", "provider_error", "retry", "failover"}


def key_fingerprint(api_key: str) -> str:
    """Short, non-reversible tag so we can tell keys apart in telemetry without
    ever storing or transmitting the key itself."""
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _post(payload: dict):
    nestjs_url = os.getenv("NESTJS_INTERNAL_URL", "")
    api_key = os.getenv("INTERNAL_API_KEY", "")
    if not nestjs_url:
        return  # not configured (e.g. local dev) — skip silently
    url = f"{nestjs_url}/api/v1/internal/ai-usage/provider-event"
    try:
        httpx.post(url, json=payload, headers={"X-Internal-Key": api_key}, timeout=5.0)
    except Exception as e:
        # Best-effort: a telemetry failure must never surface to the caller.
        logger.debug("provider-event post failed: %s", e)


def emit(
    *,
    event_type: str,
    provider: str = None,
    model: str = None,
    status_code: int = None,
    retry_after_ms: int = None,
    attempt_number: int = None,
    key_hash: str = None,
    feature: str = None,
    institute_id: str = None,
    request_id: str = None,
):
    """Record one provider attempt outcome. Reads institute/request/feature from the
    request context when not supplied, so call sites deep in rotation stay simple."""
    if event_type not in VALID_EVENT_TYPES:
        return
    try:
        from ai_services.core import request_context
        institute_id = institute_id or request_context.get("institute_id")
        request_id = request_id or request_context.get("request_id")
    except Exception:
        pass

    payload = {
        "eventType": event_type,
        "provider": provider,
        "model": model,
        "statusCode": status_code,
        "retryAfterMs": retry_after_ms,
        "attemptNumber": attempt_number,
        "keyHash": key_hash,
        "feature": feature,
        "instituteId": institute_id,
        "requestId": request_id,
    }
    threading.Thread(target=_post, args=(payload,), daemon=True).start()
