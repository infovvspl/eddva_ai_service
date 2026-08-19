import os
import threading


_LOCK = threading.Lock()
_NEXT_INDEX = 0
_DISABLED_KEYS: set[str] = set()

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

    with _LOCK:
        start = _NEXT_INDEX % len(keys)
        _NEXT_INDEX = (_NEXT_INDEX + 1) % len(keys)
        disabled = set(_DISABLED_KEYS)

    rotated = keys[start:] + keys[:start]
    indexed = []
    for key in rotated:
        if key not in disabled:
            indexed.append((keys.index(key) + 1, key))
    return indexed


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
