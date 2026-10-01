"""
EDVA AI model / provider router.

    from ai_services.core import routing
    result = routing.get_router().execute(routing.AIRequest(...))

LLMClient.complete() is the integration point: every existing Groq text call
already goes through it, so the router sits underneath AiBridgeService ->
Django view -> LLMClient without a second execution path. See router.py for the
compatibility guarantees, config.py for configuration, and
docs/ai-router-together-local-testing.md for local Together testing.
"""
from __future__ import annotations

import threading
from typing import Optional

from ai_services.core.routing.config import FeatureRoute, RoutePolicy, RouterConfig, load_config
from ai_services.core.routing.errors import (
    FallbackExhaustedError,
    NonRetryableProviderError,
    NoRouteError,
    ProviderConfigError,
    ProviderError,
    RetryableProviderError,
    RoutingError,
)
from ai_services.core.routing.registry import resolve_model_ref
from ai_services.core.routing.router import (
    AIRequest,
    Attempt,
    Candidate,
    ModelRouter,
    RoutePlan,
    telemetry_model_from_error,
    telemetry_model_from_results,
    telemetry_model_id,
)

__all__ = [
    "AIRequest", "Attempt", "Candidate", "FallbackExhaustedError", "FeatureRoute", "ModelRouter",
    "NoRouteError", "NonRetryableProviderError", "ProviderConfigError", "ProviderError",
    "RetryableProviderError", "RoutePlan", "RoutePolicy", "RouterConfig", "RoutingError",
    "get_router", "load_config", "reset_router", "resolve_model_ref", "router_enabled",
    "telemetry_model_from_error", "telemetry_model_from_results", "telemetry_model_id",
]

_lock = threading.Lock()
_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Process-wide router, built once from the environment."""
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = ModelRouter(load_config())
    return _router


def reset_router(router: Optional[ModelRouter] = None) -> None:
    """Replace (or clear, to reload from env on next use) the process router."""
    global _router
    with _lock:
        _router = router


def router_enabled() -> bool:
    return get_router().config.enabled
