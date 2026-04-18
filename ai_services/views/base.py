"""
Shared utilities for all AI service views.

Full tenant-aware pipeline:
  1. Extract tenant from request (set by TenantAuthMiddleware)
  2. Check feature is enabled for this tenant
  3. Acquire concurrency slot (noisy-neighbor protection)
  4. Check tenant-scoped cache → return on hit
  5. Check tenant-specific daily token budget → 429 on hard cap
  6. Call LLM with right-sized model
  7. Store in tenant-scoped cache
  8. Record token usage (Redis + DB UsageLog)
  9. Release concurrency slot
  10. Return response
"""

import logging
from rest_framework.response import Response

from ai_services.core.llm_client import LLMClient
from ai_services.core.cache import ResponseCache
from ai_services.core.rate_limiter import UsageLimiter
from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template

logger = logging.getLogger("ai_services.views")

# Shared singletons — initialized once, used by all views
_llm = LLMClient()
_cache = ResponseCache()
_limiter = UsageLimiter()


def get_llm() -> LLMClient:
    return _llm


def get_cache() -> ResponseCache:
    return _cache


def get_limiter() -> UsageLimiter:
    return _limiter


def _log_usage_to_db(institute, institute_id: str, feature: str, result: dict, cache_hit: bool):
    """Persist usage to DB for billing — fire-and-forget."""
    try:
        from ai_services.models import UsageLog
        UsageLog.objects.create(
            institute=institute,
            institute_id_str=institute_id,
            feature=feature,
            model_used=result.get("model", "cached"),
            prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
            total_tokens=result.get("usage", {}).get("total_tokens", 0),
            latency_ms=result.get("latency_ms", 0),
            cache_hit=cache_hit,
        )
    except Exception as e:
        logger.error("Failed to log usage to DB: %s", e)


def ai_call_text(
    request,
    feature: str,
    user_prompt: str,
    wrap_fn,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    skip_cache: bool = False,
) -> Response:
    """
    Like ai_call() but calls LLM with json_mode=False (plain text).
    wrap_fn(text: str) -> dict  maps the raw text into the endpoint's expected JSON structure.
    """
    institute = getattr(request, "institute", None)
    institute_id = getattr(request, "institute_id", "default")

    if institute and not institute.is_feature_enabled(feature):
        return Response(
            {"error": f"Feature '{feature}' is not enabled for your plan", "plan": institute.plan},
            status=403,
        )

    max_concurrent = institute.max_concurrent_requests if institute else 10
    if not _limiter.acquire_concurrency_slot(institute_id, max_concurrent, timeout=30.0):
        return Response({"error": "Too many concurrent requests. Please retry shortly."}, status=429)

    try:
        if not skip_cache:
            cached = _cache.get(institute_id, feature, user_prompt)
            if cached is not None:
                _log_usage_to_db(institute, institute_id, feature, {}, cache_hit=True)
                return Response({**cached, "_meta": {"source": "cache", "model": "cached", "institute": institute_id}})

        soft_cap = institute.daily_soft_cap if institute else None
        hard_cap = institute.daily_hard_cap if institute else None
        is_allowed, is_warning, _ = _limiter.check_budget(institute_id, soft_cap, hard_cap)
        if not is_allowed:
            return Response({"error": "Daily token budget exceeded for your institute"}, status=429)

        template = get_template(feature)
        model = get_model_for_task(feature)
        try:
            result = _llm.complete(
                system_prompt=template.system,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=False,
                institute_id=institute_id,
            )
        except RuntimeError as e:
            logger.error("LLM text call failed for %s (institute=%s): %s", feature, institute_id, e)
            return Response({"error": str(e)}, status=502)

        text = result["content"] if isinstance(result["content"], str) else str(result["content"])
        response_data = wrap_fn(text)

        if not skip_cache:
            _cache.set(institute_id, feature, user_prompt, response_data)
        _limiter.record_usage(institute_id, result["usage"]["total_tokens"])
        _log_usage_to_db(institute, institute_id, feature, result, cache_hit=False)

        meta = {
            "_meta": {
                "source": "llm",
                "model": result["model"],
                "latency_ms": round(result["latency_ms"]),
                "tokens": 0,
                "institute": institute_id,
            }
        }
        if is_warning:
            meta["_meta"]["usage_warning"] = "Approaching daily token limit"

        return Response({**response_data, **meta})
    finally:
        _limiter.release_concurrency_slot(institute_id)


def ai_call(
    request,
    feature: str,
    user_prompt: str,
    temperature: float = 0.7,
    skip_cache: bool = False,
    max_tokens: int = 4096,
) -> Response:
    """
    Tenant-aware LLM call pipeline.

    Args:
        request: DRF request (must have request.institute and request.institute_id
                 set by TenantAuthMiddleware)
        feature: Feature name (e.g., "feedback_analyze", "test_generate")
        user_prompt: The user-facing prompt (system prompt is auto-selected)
        temperature: LLM temperature
        skip_cache: Force skip cache (e.g., for cheating detection)
    """
    # Extract tenant context (set by middleware)
    institute = getattr(request, "institute", None)
    institute_id = getattr(request, "institute_id", "default")

    # 1. Check feature is enabled for this tenant
    if institute and not institute.is_feature_enabled(feature):
        return Response(
            {"error": f"Feature '{feature}' is not enabled for your plan", "plan": institute.plan},
            status=403,
        )

    # 2. Acquire concurrency slot (noisy-neighbor protection)
    max_concurrent = institute.max_concurrent_requests if institute else 10
    if not _limiter.acquire_concurrency_slot(institute_id, max_concurrent, timeout=30.0):
        return Response(
            {"error": "Too many concurrent requests. Please retry shortly."},
            status=429,
        )

    try:
        return _do_ai_call(institute, institute_id, feature, user_prompt, temperature, skip_cache, max_tokens)
    finally:
        # Always release the concurrency slot
        _limiter.release_concurrency_slot(institute_id)


def _do_ai_call(institute, institute_id, feature, user_prompt, temperature, skip_cache, max_tokens=4096) -> Response:
    """Inner pipeline after concurrency slot is acquired."""

    # 3. Tenant-scoped cache lookup
    if not skip_cache:
        cached = _cache.get(institute_id, feature, user_prompt)
        if cached is not None:
            # Log cache hit for billing
            _log_usage_to_db(institute, institute_id, feature, {}, cache_hit=True)
            return Response({
                **cached,
                "_meta": {"source": "cache", "model": "cached", "institute": institute_id},
            })

    # 4. Check tenant-specific daily token budget
    soft_cap = institute.daily_soft_cap if institute else None
    hard_cap = institute.daily_hard_cap if institute else None
    is_allowed, is_warning, current_usage = _limiter.check_budget(institute_id, soft_cap, hard_cap)

    if not is_allowed:
        return Response(
            {
                "error": "Daily token budget exceeded for your institute",
                "usage": _limiter.get_usage_summary(institute_id, soft_cap, hard_cap),
                "upgrade": "Contact support to increase your plan limits",
            },
            status=429,
        )

    # 5. LLM call with right-sized model
    template = get_template(feature)
    model = get_model_for_task(feature)

    try:
        result = _llm.complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("LLM call failed for %s (institute=%s): %s", feature, institute_id, e)
        return Response({"error": str(e)}, status=502)

    # 6. Tenant-scoped cache store
    if not skip_cache and isinstance(result["content"], dict):
        _cache.set(institute_id, feature, user_prompt, result["content"])

    # 7. Record usage (Redis for real-time + DB for billing)
    _limiter.record_usage(institute_id, result["usage"]["total_tokens"])
    _log_usage_to_db(institute, institute_id, feature, result, cache_hit=False)

    # 8. Build response
    response_data = result["content"] if isinstance(result["content"], dict) else {"raw": result["content"]}
    meta = {
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "tokens": result["usage"]["total_tokens"],
            "institute": institute_id,
        }
    }
    if is_warning:
        meta["_meta"]["usage_warning"] = "Approaching daily token limit"
        meta["_meta"]["usage"] = _limiter.get_usage_summary(institute_id, soft_cap, hard_cap)

    return Response({**response_data, **meta})
