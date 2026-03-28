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
    days = int(request.query_params.get("days", 7))
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

    return Response({
        "institute": institute_id,
        "plan": institute.plan if institute else "unknown",
        "today": today_summary,
        "period_days": days,
        "daily_breakdown": list(daily_breakdown),
        "feature_breakdown": list(feature_breakdown),
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
        "daily_soft_cap": institute.daily_soft_cap,
        "daily_hard_cap": institute.daily_hard_cap,
        "max_concurrent_requests": institute.max_concurrent_requests,
        "features_enabled": institute.features_enabled,
    })
