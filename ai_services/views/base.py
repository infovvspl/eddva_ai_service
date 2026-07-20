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
  8. Record token usage (Redis + DB UsageLog) — DB write is async (BUG-8 fix)
  9. Release concurrency slot
  10. Return response
"""

import logging
import threading
import os
from rest_framework.response import Response

from ai_services.core.llm_client import LLMClient
from ai_services.core.cache import ResponseCache
from ai_services.core.rate_limiter import UsageLimiter
from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.boards import board_instruction
from ai_services.core.usage_logger import log_usage as _log_usage_nestjs

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


def _do_log_usage_to_db(institute, institute_id: str, feature: str, result: dict, cache_hit: bool, vertical: str):
    """Inner function — runs in a daemon thread, never blocks the request path."""
    try:
        from ai_services.models import UsageLog
        UsageLog.objects.create(
            institute=institute,
            institute_id_str=institute_id,
            feature=feature,
            vertical=vertical,
            model_used=result.get("model", "cached"),
            prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
            total_tokens=result.get("usage", {}).get("total_tokens", 0),
            latency_ms=result.get("latency_ms", 0),
            cache_hit=cache_hit,
        )
        
        # Also sync to NestJS for real-time dashboard billing
        if not cache_hit:
            _log_usage_nestjs(
                institute_id=institute_id,
                institute_type=vertical or 'school',
                feature_id=feature,
                feature_category='general',
                model_used=result.get('model', 'unknown'),
                tokens_input=result.get('tokens_input', 0) or result.get('usage', {}).get('prompt_tokens', 0),
                tokens_output=result.get('tokens_output', 0) or result.get('usage', {}).get('completion_tokens', 0),
                latency_ms=int(result.get('latency_ms', 0)),
                success=True,
            )
    except Exception as e:
        logger.error("Failed to log usage to DB: %s", e)


def _log_usage_to_db(institute, institute_id: str, feature: str, result: dict, cache_hit: bool, vertical: str = "base"):
    """
    FIX BUG-8: Persist usage to DB asynchronously on a daemon thread.
    The LLM response is returned to the client immediately — this write
    never adds latency to the request/response cycle.
    """
    t = threading.Thread(
        target=_do_log_usage_to_db,
        args=(institute, institute_id, feature, result, cache_hit, vertical),
        daemon=True,
    )
    t.start()



def _cache_scope(vertical: str, board: str) -> str:
    """
    Cache namespace. School answers are board-specific (a CBSE answer must never be
    served to an ICSE school), so the board joins the key for the school vertical.
    Coaching keys keep their existing shape, so coaching cache stays warm.
    """
    if (vertical or "").lower() == "school" and board:
        return f"{vertical}-{board}"
    return vertical


def _apply_board(system_prompt: str, vertical: str, board: str) -> str:
    """
    Prepend board framing (CBSE / ICSE / State) to a SCHOOL system prompt.

    The school prompts are derived at import time and cannot know the board, which
    is per-request — so the board block is layered on here. Applied to the school
    vertical only: coaching targets JEE/NEET, where the school board is not the
    governing syllabus, and adding it there would change coaching behaviour.
    """
    if (vertical or "").lower() != "school":
        return system_prompt
    return board_instruction(board) + system_prompt


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
    vertical = getattr(request, "vertical", "base")
    board = getattr(request, "board", "")

    # Vertical-level gate first: some features are meaningless for a vertical
    # (e.g. resume_analyze / interview_prep for Classes 1-10). Return a clear 403
    # rather than generating nonsense for a 10-year-old.
    profile = getattr(request, "profile", None)
    if profile is not None and not profile.allows_feature(feature):
        return Response(
            {"error": f"Feature '{feature}' is not available for the '{vertical}' vertical"},
            status=403,
        )

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
            cached = _cache.get(institute_id, feature, user_prompt, _cache_scope(vertical, board))
            if cached is not None:
                _log_usage_to_db(institute, institute_id, feature, {}, cache_hit=True, vertical=vertical)
                return Response({**cached, "_meta": {"source": "cache", "model": "cached", "institute": institute_id, "vertical": vertical}})

        soft_cap = institute.daily_soft_cap if institute else None
        hard_cap = institute.daily_hard_cap if institute else None
        is_allowed, is_warning, _ = _limiter.check_budget(institute_id, soft_cap, hard_cap)
        if not is_allowed:
            return Response({"error": "Daily token budget exceeded for your institute"}, status=429)

        template = get_template(feature, vertical)
        model = get_model_for_task(feature, vertical)
        system_prompt = _apply_board(template.system, vertical, board)
        try:
            result = _llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=False,
                institute_id=institute_id,
            )
        except RuntimeError as e:
            logger.error("LLM text call failed for %s (institute=%s, vertical=%s): %s", feature, institute_id, vertical, e)
            return Response({"error": str(e)}, status=502)

        text = result["content"] if isinstance(result["content"], str) else str(result["content"])
        response_data = wrap_fn(text)

        if not skip_cache:
            _cache.set(institute_id, feature, user_prompt, response_data, _cache_scope(vertical, board))
        _limiter.record_usage(institute_id, result["usage"]["total_tokens"])
        _log_usage_to_db(institute, institute_id, feature, result, cache_hit=False, vertical=vertical)

        meta = {
            "_meta": {
                "source": "llm",
                "model": result["model"],
                "latency_ms": round(result["latency_ms"]),
                "tokens": result["usage"]["total_tokens"],
                "institute": institute_id,
                "vertical": vertical,
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
    vertical = getattr(request, "vertical", "base")
    board = getattr(request, "board", "")

    # 1. Check feature is enabled for this tenant
    # Vertical-level gate first: some features are meaningless for a vertical
    # (e.g. resume_analyze / interview_prep for Classes 1-10). Return a clear 403
    # rather than generating nonsense for a 10-year-old.
    profile = getattr(request, "profile", None)
    if profile is not None and not profile.allows_feature(feature):
        return Response(
            {"error": f"Feature '{feature}' is not available for the '{vertical}' vertical"},
            status=403,
        )

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
        return _do_ai_call(institute, institute_id, feature, user_prompt, temperature, skip_cache, max_tokens, vertical, board)
    finally:
        # Always release the concurrency slot
        _limiter.release_concurrency_slot(institute_id)


def _do_ai_call(institute, institute_id, feature, user_prompt, temperature, skip_cache, max_tokens=4096, vertical="base", board="") -> Response:
    """Inner pipeline after concurrency slot is acquired."""

    # 3. Tenant- + vertical-scoped cache lookup
    if not skip_cache:
        cached = _cache.get(institute_id, feature, user_prompt, _cache_scope(vertical, board))
        if cached is not None:
            # Log cache hit for billing
            _log_usage_to_db(institute, institute_id, feature, {}, cache_hit=True, vertical=vertical)
            return Response({
                **cached,
                "_meta": {"source": "cache", "model": "cached", "institute": institute_id, "vertical": vertical},
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

    # 5. LLM call with right-sized model (vertical-aware prompt + model)
    template = get_template(feature, vertical)
    model = get_model_for_task(feature, vertical)
    system_prompt = _apply_board(template.system, vertical, board)

    try:
        result = _llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("LLM call failed for %s (institute=%s, vertical=%s): %s", feature, institute_id, vertical, e)
        return Response({"error": str(e)}, status=502)

    # 6. Tenant- + vertical-scoped cache store
    if not skip_cache and isinstance(result["content"], dict):
        _cache.set(institute_id, feature, user_prompt, result["content"], _cache_scope(vertical, board))

    # 7. Record usage (Redis for real-time + DB for billing)
    _limiter.record_usage(institute_id, result["usage"]["total_tokens"])
    _log_usage_to_db(institute, institute_id, feature, result, cache_hit=False, vertical=vertical)

    # 8. Build response
    response_data = result["content"] if isinstance(result["content"], dict) else {"raw": result["content"]}
    meta = {
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "tokens": result["usage"]["total_tokens"],
            "institute": institute_id,
            "vertical": vertical,
        }
    }
    if is_warning:
        meta["_meta"]["usage_warning"] = "Approaching daily token limit"
        meta["_meta"]["usage"] = _limiter.get_usage_summary(institute_id, soft_cap, hard_cap)

    return Response({**response_data, **meta})
