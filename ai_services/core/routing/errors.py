"""
Error taxonomy for the model router.

The one question every provider failure must answer is: *would asking a
different provider plausibly succeed?* That decides whether the router is
allowed to fall back.

    retryable      provider 429, provider 5xx, timeout, network failure,
                   a provider's own retry budget exhausted on those classes
    non-retryable  malformed request, request too large, invalid input,
                   authentication / configuration failure, deterministic
                   structured-output failure, empty response

Falling back on a non-retryable error does not help and hides the real
problem: the same malformed prompt is malformed on every provider.

One exception: a failure of *this provider's account* (rejected or revoked
key 401/403, out of credit 402, model id unknown to this account 404). The
request itself is fine, and another provider has its own account, so the
router may serve it there - a teacher's deck must not fail because one
vendor's key or balance did. The router logs that at ERROR level every time,
so the broken account still surfaces; it is never retried on the same
provider. See ``ACCOUNT_KINDS`` and ``ModelRouter.execute``.

Every class subclasses RuntimeError on purpose. All existing call sites wrap
LLMClient.complete() in ``except RuntimeError``; a new exception hierarchy that
escaped those handlers would turn today's graceful 502s into unhandled 500s.
"""
from __future__ import annotations

from typing import Optional


class RoutingError(RuntimeError):
    """Base for everything raised by the router or a provider adapter."""


class NoRouteError(RoutingError):
    """No candidate model satisfies the request (unknown capability, or every
    candidate fails a hard requirement such as vision)."""


class ProviderError(RoutingError):
    """A single provider attempt failed."""

    retryable: bool = False
    kind: str = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        kind: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        # What the provider itself asked us to wait, from its Retry-After header.
        # Guessing a backoff when the provider has told us the number is how a
        # rate limit turns into a longer rate limit.
        self.retry_after_ms = retry_after_ms
        if kind:
            self.kind = kind


class RetryableProviderError(ProviderError):
    """Transient: another provider may succeed."""

    retryable = True
    kind = "transient"


class NonRetryableProviderError(ProviderError):
    """Deterministic: another provider would fail the same way, or the failure
    is ours (bad request, bad output contract)."""

    retryable = False
    kind = "deterministic"


class ProviderConfigError(NonRetryableProviderError):
    """Missing/invalid credentials or model configuration. Never falls back:
    a misconfiguration must surface, not be papered over by another vendor."""

    kind = "config"


class FallbackExhaustedError(RetryableProviderError):
    """Primary and every eligible fallback failed with retryable errors."""

    kind = "exhausted"

    def __init__(self, message: str, *, attempts: list, **kw):
        super().__init__(message, **kw)
        self.attempts = attempts


RETRYABLE_KINDS = frozenset({"rate_limited", "server_error", "timeout", "network", "exhausted", "transient"})
NON_RETRYABLE_KINDS = frozenset({
    "bad_request", "request_too_large", "structured_output", "auth", "billing", "config",
    "validation", "empty_response", "deterministic",
})
# Non-retryable on the same provider, but a different provider (its own
# account) may serve the request. See the module docstring.
ACCOUNT_KINDS = frozenset({"auth", "billing", "config"})


def is_account_failure(exc: BaseException) -> bool:
    """True when ``exc`` is a provider-account failure (see ``ACCOUNT_KINDS``)."""
    return isinstance(exc, ProviderError) and not exc.retryable and exc.kind in ACCOUNT_KINDS


def classify_status(status_code: Optional[int]) -> tuple[bool, str]:
    """Map an HTTP status to (retryable, kind).

    404 is treated as configuration: for an inference API it almost always means
    the model id does not exist for this account, which no retry will fix.
    """
    if status_code is None:
        return True, "network"
    if status_code == 429:
        return True, "rate_limited"
    if status_code == 408:
        return True, "timeout"
    if status_code >= 500:
        return True, "server_error"
    if status_code in (401, 403):
        return False, "auth"
    if status_code == 402:
        return False, "billing"   # out of credit / payment required
    if status_code == 404:
        return False, "config"
    if status_code == 413:
        return False, "request_too_large"
    return False, "bad_request"


def error_for_status(
    status_code: int, message: str, *, provider: str, model: Optional[str],
    retry_after_ms: Optional[int] = None,
) -> ProviderError:
    retryable, kind = classify_status(status_code)
    if retryable:
        cls = RetryableProviderError
    elif kind in ACCOUNT_KINDS:
        cls = ProviderConfigError
    else:
        cls = NonRetryableProviderError
    return cls(message, provider=provider, model=model, status_code=status_code, kind=kind,
               retry_after_ms=retry_after_ms)
