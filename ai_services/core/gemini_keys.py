import os
import threading


_LOCK = threading.Lock()
_NEXT_INDEX = 0
_DISABLED_KEYS: set[str] = set()


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
            "forbidden",
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
