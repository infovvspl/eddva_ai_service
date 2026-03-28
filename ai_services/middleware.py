"""
Tenant authentication middleware — compatible with both:
  1. Direct API calls: X-API-Key header
  2. NestJS ai-bridge calls: Authorization: Bearer <api_key> + X-Tenant-ID header

NestJS backend flow:
  - TenantMiddleware resolves tenant from subdomain → sets req.tenantId (UUID)
  - ai-bridge.service.ts calls Django with:
      Authorization: Bearer <AI_API_KEY>
      X-Tenant-ID: <tenantId UUID>

This middleware:
  1. Extracts API key from X-API-Key OR Authorization: Bearer
  2. Extracts tenant from X-Tenant-ID header (NestJS) or from API key lookup
  3. Attaches request.institute and request.institute_id
  4. Rejects invalid/missing auth (401) or inactive institutes (403)

Exempt paths: health checks, root, admin, static files.
"""

import time
import logging
import threading
from django.http import JsonResponse

logger = logging.getLogger("ai_services.middleware")

# In-memory cache for Institute lookups — avoids DB hit on every request
_institute_cache = {}
_tenant_id_cache = {}  # Cache by tenant external ID (UUID from NestJS)
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes


EXEMPT_PATHS = {
    "/",
    "/admin",
    "/favicon.ico",
}
EXEMPT_SUFFIXES = ("/health/",)


def _get_cached_institute_by_api_key(api_key: str):
    """Look up institute by API key with 5-minute cache."""
    now = time.time()

    with _cache_lock:
        entry = _institute_cache.get(api_key)
        if entry and entry["expires_at"] > now:
            return entry["institute"]

    from ai_services.models import Institute
    institute = Institute.get_by_api_key(api_key)

    with _cache_lock:
        _institute_cache[api_key] = {
            "institute": institute,
            "expires_at": now + _CACHE_TTL,
        }

    return institute


def _get_cached_institute_by_tenant_id(tenant_id: str):
    """Look up institute by external tenant ID (UUID from NestJS) with cache."""
    now = time.time()

    with _cache_lock:
        entry = _tenant_id_cache.get(tenant_id)
        if entry and entry["expires_at"] > now:
            return entry["institute"]

    from ai_services.models import Institute
    try:
        institute = Institute.objects.filter(
            external_tenant_id=tenant_id, is_active=True
        ).first()
    except Exception:
        institute = None

    with _cache_lock:
        _tenant_id_cache[tenant_id] = {
            "institute": institute,
            "expires_at": now + _CACHE_TTL,
        }

    return institute


def invalidate_institute_cache(api_key: str = None):
    """Call this when institute config changes (e.g., from admin save)."""
    with _cache_lock:
        if api_key:
            _institute_cache.pop(api_key, None)
        else:
            _institute_cache.clear()
            _tenant_id_cache.clear()


def _extract_api_key(request) -> str:
    """
    Extract API key from request in priority order:
    1. X-API-Key header (direct calls)
    2. Authorization: Bearer <key> (NestJS ai-bridge calls)
    3. ?api_key= query param (browser testing)
    """
    # 1. X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    # 2. Authorization: Bearer <key>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 3. Query param fallback
    return request.GET.get("api_key", "")


class TenantAuthMiddleware:
    """
    Django middleware that enforces tenant authentication.

    Supports two flows:
      Flow A (Direct): X-API-Key → lookup Institute by api_key
      Flow B (NestJS): Bearer token + X-Tenant-ID → validate token, resolve tenant by external ID

    Attaches to request:
      - request.institute: Institute model instance (or None for exempt paths)
      - request.institute_id: str slug (or "anonymous")
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip auth for exempt paths
        if self._is_exempt(request.path):
            request.institute = None
            request.institute_id = "anonymous"
            return self.get_response(request)

        # Extract API key (from X-API-Key or Bearer token)
        api_key = _extract_api_key(request)

        if not api_key:
            return JsonResponse(
                {
                    "error": "Missing authentication",
                    "hint": "Include X-API-Key header or Authorization: Bearer <key>",
                },
                status=401,
            )

        # Validate the API key
        institute = _get_cached_institute_by_api_key(api_key)

        if institute is None:
            logger.warning("Invalid API key attempted: %s...", api_key[:8])
            return JsonResponse({"error": "Invalid API key"}, status=401)

        if not institute.is_active:
            return JsonResponse(
                {"error": "Institute account is deactivated"},
                status=403,
            )

        # Check for X-Tenant-ID (from NestJS) — allows the master API key to
        # operate on behalf of a specific tenant
        nest_tenant_id = request.headers.get("X-Tenant-ID")
        if nest_tenant_id:
            # The API key is the master service key; resolve the actual tenant
            tenant_institute = _get_cached_institute_by_tenant_id(nest_tenant_id)
            if tenant_institute:
                institute = tenant_institute

        # Attach tenant context to request
        request.institute = institute
        request.institute_id = str(institute.slug)

        response = self.get_response(request)

        # Debugging headers
        response["X-Institute"] = institute.slug
        response["X-Plan"] = institute.plan

        return response

    def _is_exempt(self, path: str) -> bool:
        if path in EXEMPT_PATHS or path == "":
            return True
        for suffix in EXEMPT_SUFFIXES:
            if path.endswith(suffix):
                return True
        if path.startswith("/admin/") or path == "/admin":
            return True
        return False
