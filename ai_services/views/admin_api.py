"""
Tenant admin API endpoints — usage dashboard, cache management.
These are tenant-scoped: each institute can only see its own data.
"""

from django.db.models import Sum, Count, Avg, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.models import UsageLog
from .base import get_limiter, get_cache


@api_view(["GET"])
def usage_dashboard(request):
    """
    GET /admin-api/usage/
    Returns token usage summary for the authenticated institute.
    Query params: ?days=7 (default 7)
    """
    institute = getattr(request, "institute", None)
    institute_id = getattr(request, "institute_id", "default")
    # ISSUE-7 FIX: Clamp days to 1–90 to prevent full table scans.
    # Malformed values fall back to the default of 7.
    try:
        days = max(1, min(int(request.query_params.get("days", 7)), 90))
    except (ValueError, TypeError):
        days = 7
    since = timezone.now() - timedelta(days=days)

    # Real-time usage from rate limiter
    limiter = get_limiter()
    soft_cap = institute.daily_soft_cap if institute else None
    hard_cap = institute.daily_hard_cap if institute else None
    today_summary = limiter.get_usage_summary(institute_id, soft_cap, hard_cap)

    # Historical from DB
    logs = UsageLog.objects.filter(
        institute_id_str=institute_id,
        created_at__gte=since,
    )

    daily_breakdown = (
        logs.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(
            total_tokens=Sum("total_tokens"),
            total_calls=Count("id"),
            cache_hits=Count("id", filter=Q(cache_hit=True)),
            avg_latency=Avg("latency_ms"),
        )
        .order_by("-day")
    )

    feature_breakdown = (
        logs.values("feature")
        .annotate(
            total_tokens=Sum("total_tokens"),
            total_calls=Count("id"),
            cache_hits=Count("id", filter=Q(cache_hit=True)),
        )
        .order_by("-total_tokens")
    )

    vertical_breakdown = (
        logs.values("vertical")
        .annotate(
            total_tokens=Sum("total_tokens"),
            total_calls=Count("id"),
            cache_hits=Count("id", filter=Q(cache_hit=True)),
        )
        .order_by("-total_tokens")
    )

    return Response({
        "institute": institute_id,
        "plan": institute.plan if institute else "unknown",
        "vertical": getattr(institute, "vertical", None) if institute else None,
        "today": today_summary,
        "period_days": days,
        "daily_breakdown": list(daily_breakdown),
        "feature_breakdown": list(feature_breakdown),
        "vertical_breakdown": list(vertical_breakdown),
        "total_tokens": logs.aggregate(total=Sum("total_tokens"))["total"] or 0,
        "total_calls": logs.count(),
    })


@api_view(["POST"])
def flush_cache(request):
    """
    POST /admin-api/cache/flush/
    Flush all cached responses for the authenticated institute.
    """
    institute_id = getattr(request, "institute_id", "default")
    cache = get_cache()
    cache.flush_tenant(institute_id)
    return Response({"message": f"Cache flushed for institute: {institute_id}"})


@api_view(["GET"])
def institute_info(request):
    """
    GET /admin-api/info/
    Returns the current institute's config (for debugging).
    """
    institute = getattr(request, "institute", None)
    if not institute:
        return Response({"error": "No institute context"}, status=401)

    return Response({
        "id": str(institute.id),
        "name": institute.name,
        "slug": institute.slug,
        "plan": institute.plan,
        "vertical": institute.vertical,
        "is_service_account": institute.is_service_account,
        "daily_soft_cap": institute.daily_soft_cap,
        "daily_hard_cap": institute.daily_hard_cap,
        "max_concurrent_requests": institute.max_concurrent_requests,
        "features_enabled": institute.features_enabled,
    })


@api_view(["GET"])
def cache_stats(request):
    """
    GET /admin-api/cache/stats/?days=7

    Returns Redis cache performance stats for this institute:
      - Hit rate (% of requests served from cache without calling LLM)
      - Tokens saved (total tokens NOT sent to Groq/Gemini)
      - Cost saved in USD and INR
      - Daily breakdown for charting

    This is your AI cost optimization dashboard.
    A 60%+ hit rate means 60% of AI calls are FREE.
    """
    institute_id = getattr(request, "institute_id", "default")
    try:
        days = max(1, min(int(request.query_params.get("days", 7)), 90))
    except (ValueError, TypeError):
        days = 7

    cache = get_cache()
    stats = cache.get_stats(institute_id, days=days)
    return Response(stats)


@api_view(["GET"])
def cache_health(request):
    """
    GET /admin-api/cache/health/

    Quick health check for ops/monitoring:
      - Is Redis connected?
      - How many entries in memory cache?
      - Redis server info (version, memory usage)

    Use this to verify Redis is working in production.
    """
    from ai_services.core.cache import get_redis

    r = get_redis()
    if not r:
        return Response({
            "redis_connected": False,
            "warning": "Redis is NOT connected. Caching is per-worker only. Set REDIS_URL.",
            "memory_entries": get_cache()._memory.size(),
        }, status=503)

    try:
        info = r.info("memory")
        return Response({
            "redis_connected": True,
            "redis_version": r.info("server").get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "maxmemory_human": info.get("maxmemory_human", "unlimited"),
            "memory_policy": r.info("memory").get("maxmemory_policy", "noeviction"),
            "memory_entries": get_cache()._memory.size(),
            "status": "ok",
        })
    except Exception as e:
        return Response({"redis_connected": True, "error": str(e)}, status=500)

