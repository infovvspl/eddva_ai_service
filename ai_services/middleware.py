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
  2. Validates the key against the Institute table
  3. For service-account keys only: resolves tenant from X-Tenant-ID header
     (FIX BUG-9: regular institute keys CANNOT switch tenant via X-Tenant-ID)
  4. Attaches request.institute and request.institute_id
  5. Rejects invalid/missing auth (401) or inactive institutes (403)

Exempt paths: health checks, root, admin, static files.
"""

import time
import logging
import threading
from django.http import JsonResponse

from ai_services.core.verticals import (
    normalize_vertical,
    get_profile,
    env_default_vertical,
)
from ai_services.core.boards import (
    normalize_board,
    get_board,
    env_default_board,
)

logger = logging.getLogger("ai_services.middleware")

# ---------------------------------------------------------------------------
# In-memory cache for Institute lookups — avoids DB hit on every request.
# FIX BUG-4: Both caches are invalidated together when a specific tenant
# changes, preventing stale data for NestJS-routed (X-Tenant-ID) requests.
# ---------------------------------------------------------------------------
_institute_cache: dict = {}       # api_key  → {institute, expires_at}
_tenant_id_cache: dict = {}       # ext UUID → {institute, expires_at}
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes


EXEMPT_PATHS = {
    "/",
    "/admin",
    "/favicon.ico",
}
EXEMPT_SUFFIXES = ("/health/",)
EXEMPT_PREFIXES = (
    "/generated-note-images/",
)


def _get_cached_institute_by_api_key(api_key: str):
    """Look up institute by API key with 5-minute TTL cache."""
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
    """Look up institute by external tenant ID (UUID from NestJS) with TTL cache."""
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


def invalidate_institute_cache(api_key: str = None, external_tenant_id: str = None):
    """
    Invalidate cached institute data when config changes (e.g. admin save).

    FIX BUG-4: Previously, calling invalidate_institute_cache(api_key=<key>)
    only cleared _institute_cache, leaving _tenant_id_cache stale.
    Now both caches are cleared for the affected tenant.

    Args:
        api_key: If set, evicts only the entry for this API key.
        external_tenant_id: If set, evicts only the entry for this NestJS UUID.
        (Both can be passed together — both will be evicted.)
        If neither is passed → full flush of both caches.
    """
    with _cache_lock:
        if api_key or external_tenant_id:
            if api_key:
                _institute_cache.pop(api_key, None)
            if external_tenant_id:
                _tenant_id_cache.pop(external_tenant_id, None)
        else:
            # Full flush
            _institute_cache.clear()
            _tenant_id_cache.clear()


def _extract_vertical(request) -> str:
    """
    Per-request vertical override, if the caller supplied one explicitly.
    Reads (in order) the X-Vertical header then the ?vertical= query param.
    Returns "" when none is supplied (so the tenant/env default is used).

    Note: we deliberately do NOT read the JSON body here — consuming the body
    stream in middleware breaks DRF parsing downstream. Per-request selection
    travels via the X-Vertical header (set by the NestJS ai-bridge).
    """
    return (request.headers.get("X-Vertical") or request.GET.get("vertical") or "").strip()


def _resolve_vertical(request, institute) -> str:
    """
    Resolve the effective vertical with precedence:
        explicit per-request  >  institute default  >  DEFAULT_VERTICAL env  >  hard default
    Always returns a valid, registered vertical key (never raises).
    """
    explicit = _extract_vertical(request)
    if explicit:
        return normalize_vertical(explicit)
    institute_vertical = getattr(institute, "vertical", None) if institute else None
    if institute_vertical:
        return normalize_vertical(institute_vertical)
    return env_default_vertical()


def _resolve_board(request, institute) -> str:
    """
    Resolve the education board with the same precedence as the vertical:
        X-Board header  >  Institute.board  >  DEFAULT_BOARD env  >  "cbse"

    The backend reads the real value from its `institutes.board` column and sends it
    as X-Board, so the header is the normal path; Institute.board is the fallback.
    Always returns a valid board key (unknown/typo values normalise to the default).
    """
    explicit = (request.headers.get("X-Board") or request.GET.get("board") or "").strip()
    if explicit:
        return normalize_board(explicit)
    institute_board = getattr(institute, "board", None) if institute else None
    if institute_board:
        return normalize_board(institute_board)
    return env_default_board()


def _extract_api_key(request) -> str:
    """
    Extract API key from request in priority order:
    1. X-API-Key header (direct calls)
    2. Authorization: Bearer <key> (NestJS ai-bridge calls)
    ?api_key= query param is NOT supported — it leaks keys into server logs.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return ""


class TenantAuthMiddleware:
    """
    Django middleware that enforces tenant authentication.

    Supports two flows:
      Flow A (Direct): X-API-Key → lookup Institute by api_key
      Flow B (NestJS): Bearer token + X-Tenant-ID → validate token, resolve tenant by external ID
                       FIX BUG-9: Flow B only works if the authenticated key has is_service_account=True.

    Attaches to request:
      - request.institute: Institute model instance (or None for exempt paths)
      - request.institute_id: str NestJS UUID (X-Tenant-ID for service-account calls, else external_tenant_id → slug)
      - request.vertical: resolved vertical key
      - request.profile: VerticalProfile instance
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip auth for exempt paths
        if self._is_exempt(request.path):
            request.institute = None
            request.institute_id = "anonymous"
            request.vertical = _resolve_vertical(request, None)
            request.profile = get_profile(request.vertical)
            request.board = _resolve_board(request, None)
            request.board_profile = get_board(request.board)
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

        # FIX BUG-9: X-Tenant-ID impersonation is only allowed for service-account keys.
        # Capture is_service_account NOW, before institute may be reassigned to the
        # school tenant object (which has is_service_account=False).
        api_key_is_service_account = institute.is_service_account

        nest_tenant_id = request.headers.get("X-Tenant-ID")
        if nest_tenant_id:
            if not api_key_is_service_account:
                # Suspicious: a non-service-account key is trying to impersonate a tenant.
                # Log it for security auditing and reject the override (still serve the
                # authenticated institute, do NOT switch context).
                logger.warning(
                    "X-Tenant-ID override rejected: API key %s... is not a service account "
                    "(institute=%s). Possible impersonation attempt.",
                    api_key[:8],
                    institute.slug,
                )
                # Fall through — serve the authenticated institute without switching context.
            else:
                # Authenticated service-account key: resolve the real tenant.
                tenant_institute = _get_cached_institute_by_tenant_id(nest_tenant_id)
                if tenant_institute:
                    institute = tenant_institute
                else:
                    logger.warning(
                        "Service account %s... supplied unknown X-Tenant-ID: %s",
                        api_key[:8],
                        nest_tenant_id,
                    )

        # Attach tenant context to request
        request.institute = institute
        # X-Tenant-ID is always the NestJS school UUID when sent by the ai-bridge.
        # Use it directly for usage attribution (ai_usage_daily institute_id) so
        # analytics queries by school UUID match what's stored.
        # The service-account check above handles DATA-SERVING context only;
        # usage attribution is safe to key by X-Tenant-ID regardless of key type.
        nest_tid = request.headers.get("X-Tenant-ID", "")
        if nest_tid:
            request.institute_id = nest_tid
        else:
            request.institute_id = str(institute.external_tenant_id or institute.slug)
        request.vertical = _resolve_vertical(request, institute)
        request.profile = get_profile(request.vertical)
        request.board = _resolve_board(request, institute)
        request.board_profile = get_board(request.board)

        response = self.get_response(request)

        # Debugging headers (safe — these are internal slugs, not secrets)
        response["X-Institute"] = institute.slug
        response["X-Plan"] = institute.plan
        response["X-Vertical"] = request.vertical
        response["X-Board"] = request.board

        return response

    def _is_exempt(self, path: str) -> bool:
        if path in EXEMPT_PATHS or path == "":
            return True
        for suffix in EXEMPT_SUFFIXES:
            if path.endswith(suffix):
                return True
        for prefix in EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return True
        if path.startswith("/admin/") or path == "/admin":
            return True
        return False
