import os
import threading
import time


_LOCK = threading.Lock()
_NEXT_INDEX = 0
_DISABLED_KEYS: set[str] = set()

# ── Short-lived cooldown for transiently failing keys ─────────────────────────
# A key that just returned 429 (its quota is spent) or 503 (its project has no
# capacity right now) will almost certainly do so again seconds later. Without
# memory, every request rediscovers the same dead keys serially — and a 503 is
# not cheap: Google attempts the generation before giving up, so it costs about
# as long as a successful call (measured: 14-27s per key, turning a ~13s deck
# into 57s).
#
# These are NOT _DISABLED_KEYS: that set is permanent and reserved for keys the
# API rejected outright. Quota and capacity both come back, so the key is only
# stepped over until its cooldown expires.
_COOLING_KEYS: "dict[str, float]" = {}

# Measured, not guessed: the same two keys 503'd at 14:43, 14:49 and 16:14, and
# one 503 took 40s to come back. These failures are persistent rather than
# momentary, and each rediscovery costs most of a request — so the window has to
# outlive the gap between a teacher's generations, not just a burst.
_COOLDOWN_S = {
    "429": float(os.getenv("GEMINI_COOLDOWN_429_S", "300")),   # quota window
    "503": float(os.getenv("GEMINI_COOLDOWN_503_S", "600")),   # capacity starvation
}

# Redis key prefix. The cooldown MUST be shared: gunicorn runs 3 workers here, so
# a process-local set means each worker pays to rediscover the same dead keys.
_COOL_PREFIX = "gemini:cooling:"


def _cool_redis():
    """Shared Redis, or None. Never raises — this is an optimisation, not a gate."""
    try:
        from ai_services.core.cache import get_redis
        return get_redis()
    except Exception:
        return None


# Consecutive-failure counter, so a key that keeps failing is retried less and
# less often instead of costing a full rediscovery every cooldown expiry.
_FAIL_PREFIX = "gemini:failstreak:"
_COOLDOWN_MAX_S = float(os.getenv("GEMINI_COOLDOWN_MAX_S", "3600"))


def mark_gemini_key_cooling(api_key: str, kind: str) -> None:
    """Step over this key after a 429/503, across every worker.

    The window doubles per consecutive failure (10min, 20min, 40min, capped at
    an hour for 503). A key broken all afternoon then costs one probe an hour
    instead of one every ten minutes, while a key that fails once and recovers
    is barely penalised — its streak resets on the next success.

    Self-healing and best-effort: entries expire on their own, and Redis being
    unavailable degrades to process-local memory rather than failing the call.
    """
    base = _COOLDOWN_S.get(kind)
    if not base:
        return

    fp = _fingerprint(api_key)
    r = _cool_redis()
    if r is not None:
        try:
            # Track the streak for longer than the cooldown itself, or the
            # counter would expire alongside it and never escalate.
            streak = r.incr(f"{_FAIL_PREFIX}{fp}")
            r.expire(f"{_FAIL_PREFIX}{fp}", int(_COOLDOWN_MAX_S * 2))
            seconds = min(base * (2 ** (max(1, int(streak)) - 1)), _COOLDOWN_MAX_S)
            r.setex(f"{_COOL_PREFIX}{fp}", int(seconds), kind)
            return
        except Exception:
            pass
    with _LOCK:
        _COOLING_KEYS[api_key] = time.time() + base


def mark_gemini_key_healthy(api_key: str) -> None:
    """Clear a key's failure streak after it succeeds.

    Without this the streak only ever grows, so a key that recovers would keep
    inheriting an hour-long penalty from a bad afternoon.
    """
    r = _cool_redis()
    if r is None:
        return
    try:
        r.delete(f"{_FAIL_PREFIX}{_fingerprint(api_key)}")
    except Exception:
        pass


def _fingerprint(api_key: str) -> str:
    """Stable short id for a key. Never store or log the key itself."""
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def _cooling_now(keys: "list[str]") -> "set[str]":
    """Keys currently cooling, from Redis when available plus local fallback."""
    cooling = set()
    r = _cool_redis()
    if r is not None:
        try:
            fps = {_fingerprint(k): k for k in keys}
            vals = r.mget([f"{_COOL_PREFIX}{fp}" for fp in fps])
            cooling.update(
                key for (fp, key), v in zip(fps.items(), vals) if v is not None
            )
        except Exception:
            pass

    now = time.time()
    with _LOCK:
        for k, until in list(_COOLING_KEYS.items()):
            if until <= now:
                del _COOLING_KEYS[k]
        cooling.update(_COOLING_KEYS)
    return cooling


def _live_keys(keys: "list[str]") -> "list[str]":
    """Keys worth trying now: not permanently disabled, not cooling.

    Fails OPEN. If every key is cooling, the cooldowns are ignored rather than
    returning nothing — a slow attempt beats refusing to generate at all.
    """
    with _LOCK:
        disabled = set(_DISABLED_KEYS)
    cooling = _cooling_now(keys)

    usable = [k for k in keys if k not in disabled and k not in cooling]
    if usable:
        return usable
    return [k for k in keys if k not in disabled]

# ── Per-key model availability ────────────────────────────────────────────────
# Google withdraws a model "for new users" rather than for everyone: a key whose
# Google Cloud project predates the withdrawal keeps calling gemini-2.5-flash,
# while a key issued today gets 404 "no longer available to new users" for the
# same model. So the usable model is a property of the *key*, not of the
# deployment — and adding a fresh key to widen quota silently adds a key that
# fails every request it is handed.
#
# Measured against the live API across the whole pool: gemini-flash-latest is
# served by every key, old project and new, and returns valid JSON under the
# same response_mime_type/thinking config the callers already pass. It is
# therefore the common fallback. gemini-3.5-flash was rejected as a fallback —
# it produced malformed JSON on one of the new keys.
_MODEL_FALLBACK_DEFAULT = "gemini-flash-latest,gemini-3-flash-preview"

# Models known to be withdrawn for new projects. Anything not listed is assumed
# universally available and gets the same fallback chain if it ever 404s.
_FALLBACK_CHAIN = tuple(
    m.strip()
    for m in os.getenv("GEMINI_MODEL_FALLBACKS", _MODEL_FALLBACK_DEFAULT).split(",")
    if m.strip()
)

# (api_key, model) pairs the API has told us this project cannot use. Cached for
# the process so a key costs at most one wasted round-trip per model, ever,
# instead of one on every request.
_BLOCKED_MODELS: set[tuple[str, str]] = set()

# Callers pass thinking_budget=0 to stop Gemini's internal reasoning eating the
# output budget (it was truncating question papers mid-question). Gemini 3-era
# models reject a zero budget outright with 400 INVALID_ARGUMENT — measured:
# gemini-flash-latest refuses it in every combination, gemini-3-flash-preview
# accepts it. Since gemini-flash-latest is a moving alias, which model refuses
# is not something to hard-code; it is discovered on the first 400 and cached
# here for the process.
_ZERO_THINKING_REJECTED: set[str] = set()


def get_gemini_api_keys() -> list[str]:
    keys: list[str] = []

    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        keys.append(single)

    csv_keys = os.getenv("GEMINI_API_KEYS", "").strip()
    if csv_keys:
        keys.extend(k.strip() for k in csv_keys.split(",") if k.strip())

    for i in range(1, 21):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if key:
            keys.append(key)

    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def get_rotated_gemini_keys() -> list[tuple[int, str]]:
    global _NEXT_INDEX
    keys = get_gemini_api_keys()
    if not keys:
        return []

    # Try keys that are not known-bad first; a key cooling off after a 429/503
    # is stepped over rather than re-probed on every request.
    usable = _live_keys(keys)
    if not usable:
        return []

    with _LOCK:
        start = _NEXT_INDEX % len(usable)
        _NEXT_INDEX = (_NEXT_INDEX + 1) % len(usable)

    rotated = usable[start:] + usable[:start]
    # Key numbers stay 1-based over the FULL configured list so log lines and
    # provider events keep naming the same key across requests.
    return [(keys.index(key) + 1, key) for key in rotated]


def resolve_gemini_model(api_key: str, model: str) -> str:
    """Return the model this particular key can actually serve.

    Callers keep asking for the model they want; this swaps in a fallback only
    for the keys whose project cannot serve it. A key from an older project is
    unaffected and keeps using the requested model, so behaviour for the
    existing pool is unchanged.
    """
    if not api_key or not model:
        return model
    with _LOCK:
        if (api_key, model) not in _BLOCKED_MODELS:
            return model
        for candidate in _FALLBACK_CHAIN:
            if (api_key, candidate) not in _BLOCKED_MODELS:
                return candidate
    # Every known model is blocked for this key. Return the original so the
    # caller's own error handling reports a real API error rather than us
    # inventing one.
    return model


def mark_gemini_model_unavailable(api_key: str, model: str) -> str | None:
    """Record that this key's project cannot use this model.

    Returns the next model to try on the *same* key, or None when the chain is
    exhausted. Recording is what makes the fallback free from the second call
    onwards — the 404 is paid once per key/model, not once per request.
    """
    if not api_key or not model:
        return None
    with _LOCK:
        _BLOCKED_MODELS.add((api_key, model))
        for candidate in _FALLBACK_CHAIN:
            if (api_key, candidate) not in _BLOCKED_MODELS:
                return candidate
    return None


def model_rejects_zero_thinking(model: str) -> bool:
    with _LOCK:
        return model in _ZERO_THINKING_REJECTED


def mark_zero_thinking_rejected(model: str) -> None:
    if not model:
        return
    with _LOCK:
        _ZERO_THINKING_REJECTED.add(model)


def is_gemini_invalid_argument_error(message: str) -> bool:
    """A 400 the request shape caused, as opposed to a bad key or model."""
    text = str(message or "").lower()
    return "invalid_argument" in text or (
        "400" in text and "invalid argument" in text
    )


def is_gemini_model_unavailable_error(message: str) -> bool:
    """The key is valid but its project has no access to the requested model.

    Distinct from a bad key: the key must stay in rotation and be retried on a
    model it can serve, not be disabled.
    """
    text = str(message or "").lower()
    return (
        "no longer available" in text
        or "is not found for api version" in text
        or ("404" in text and "model" in text)
    )


def mark_gemini_key_disabled(key: str) -> None:
    if not key:
        return
    with _LOCK:
        _DISABLED_KEYS.add(key)


def gemini_key_count() -> int:
    return len(get_gemini_api_keys())


def has_gemini_api_key() -> bool:
    return bool(get_gemini_api_keys())


def is_gemini_permanent_key_error(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        token in text
        for token in (
            "api key not valid",
            "api_key_invalid",
            "invalid api key",
            "permission denied",
            "unauthorized",
            "unauthenticated",
            "forbidden",
            "invalid_argument",
            "invalid argument",
        )
    )


def is_gemini_retryable_error(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        token in text
        for token in (
            "429",
            "503",
            "resource_exhausted",
            "unavailable",
            "high demand",
            "rate",
            "quota",
            "timeout",
            "temporarily",
        )
    )
