import os
import threading
from typing import List

_lock = threading.Lock()
_rr_index = 0


def _normalize_keys(keys: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for k in keys:
        v = (k or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def get_groq_api_keys() -> List[str]:
    keys: List[str] = []

    # Supports:
    # - GROQ_API_KEY
    # - GROQ_API_KEYS (comma-separated)
    # - GROQ_API_KEY_1..N
    single = os.getenv("GROQ_API_KEY", "")
    if single:
        keys.append(single)

    csv_keys = os.getenv("GROQ_API_KEYS", "")
    if csv_keys:
        keys.extend([x.strip() for x in csv_keys.split(",") if x.strip()])

    i = 1
    while True:
        v = os.getenv(f"GROQ_API_KEY_{i}", "")
        if not v:
            break
        keys.append(v)
        i += 1

    return _normalize_keys(keys)


def get_rotated_groq_keys() -> List[str]:
    keys = get_groq_api_keys()
    if not keys:
        return []
    global _rr_index
    with _lock:
        start = _rr_index % len(keys)
        _rr_index = (_rr_index + 1) % len(keys)
    return keys[start:] + keys[:start]


def is_key_exhausted_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403, 429):
        return True
    return any(
        token in msg
        for token in (
            "rate limit",
            "quota",
            "exceeded",
            "insufficient_quota",
            "too many requests",
            "authentication",
            "invalid api key",
            "api key",
        )
    )

