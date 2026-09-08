"""
Per-request attribution context (thread-local).

The usage/provider-event loggers are called from many places that do not have the
DRF request in hand (deep inside llm_client rotation, background chunk workers,
etc.). Threading user_id / request_id through every one of those call sites would
be a large, risky change. Instead TenantAuthMiddleware stamps the authenticated
identity here once per request, and the loggers read it as a fallback.

Sync gunicorn workers handle one request per thread at a time, so a thread-local
is correct: set at request start, cleared in a finally at request end. Values are
identity/correlation only — never secrets.
"""
import threading

_ctx = threading.local()


def set_context(*, request_id=None, user_id=None, user_role=None,
                institute_id=None, vertical=None):
    _ctx.request_id = request_id
    _ctx.user_id = user_id
    _ctx.user_role = user_role
    _ctx.institute_id = institute_id
    _ctx.vertical = vertical


def get(name, default=None):
    return getattr(_ctx, name, default)


def clear():
    for name in ("request_id", "user_id", "user_role", "institute_id", "vertical"):
        if hasattr(_ctx, name):
            delattr(_ctx, name)
