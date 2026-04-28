"""
LLM client -- Groq API (llama-3.3-70b-versatile).

Keeps the same .complete() interface as the previous Ollama client so every
view continues to work without changes.
"""

import json
import logging
import os
import time
import threading
from typing import Optional

from ai_services.core.groq_keys import (
    get_groq_api_keys,
    get_rotated_groq_keys,
    is_key_exhausted_error,
)

logger = logging.getLogger("ai_services.llm")
_KEY_STATE_LOCK = threading.Lock()
_DISABLED_GROQ_KEYS: set[str] = set()

# Round-robin cursor: each complete() call starts from a different key so sequential
# chunk calls (notes generation) spread load evenly instead of hammering key[0] each time.
_KEY_CURSOR = 0
_KEY_CURSOR_LOCK = threading.Lock()


def _next_key_offset() -> int:
    global _KEY_CURSOR
    with _KEY_CURSOR_LOCK:
        idx = _KEY_CURSOR
        _KEY_CURSOR += 1
        return idx

# -- Groq config (multi-key pool for rate-limit rotation) ----------------------
_GROQ_KEYS_RAW = [
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
    os.getenv("GROQ_API_KEY_8", ""),
    os.getenv("GROQ_API_KEY_9", ""),
    os.getenv("GROQ_API_KEY_10", ""),
    os.getenv("GROQ_API_KEY_11", ""),
    os.getenv("GROQ_API_KEY_12", ""),
    os.getenv("GROQ_API_KEY_13", ""),
]
GROQ_API_KEYS: list[str] = [k for k in _GROQ_KEYS_RAW if k]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # backward compat
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Models that can be requested by name -- anything else falls back to GROQ_MODEL
_GROQ_ALLOWED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "quiz",
}
_GROQ_MODEL_ALIAS = {
    # llama-3.3-70b-versatile: best quiz quality on Groq free tier.
    # 3,500-char chunks ≈ 3,000 tokens total — fits within the 6,000 TPM limit.
    "quiz": "llama-3.3-70b-versatile",
}


def _resolve_model(model: str) -> str:
    if model in _GROQ_MODEL_ALIAS:
        return _GROQ_MODEL_ALIAS[model]
    if model in _GROQ_ALLOWED_MODELS:
        return model
    return GROQ_MODEL


# -- Ollama config (kept for future use) ---------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://213.192.2.90:40077")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvav2")

_ANTI_HALLUCINATION_PREFIX = (
    "You are EDVA AI, an expert Indian education assistant "
    "for JEE, NEET, and CBSE (Class 10-12).\n"
    "Only answer what is asked. Stay on topic. "
    "Use correct scientific facts only.\n\n"
)

_JSON_MODE_SUFFIX = "\n\nRespond with ONLY a JSON object. No markdown. No code fences. No explanation."

# For tutor/teacher-style replies: JSON is required, but string values may include Markdown and formulas.
_JSON_MODE_TUTOR_SUFFIX = (
    "\n\nRespond with ONLY one valid JSON object. "
    "In string fields (especially \"response\"), you may use Markdown, **bold**, and normal math text for equations."
)


def _extract_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def get_llm() -> "LLMClient":
    return LLMClient()


def _get_groq_client():
    """Startup validation: confirm at least one key is configured."""
    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY configured -- set at least one in .env")
    from groq import Groq
    return Groq(api_key=GROQ_API_KEYS[0])


def check_groq_keys() -> dict:
    """
    Health-check every configured Groq key with a minimal LLM call (max_tokens=5).
    Permanently disables invalid/restricted keys so they are never used in production.
    Returns a summary dict and logs a table.  Designed to run in a background thread
    at startup so it never blocks Django boot.
    """
    from groq import Groq

    keys = GROQ_API_KEYS
    if not keys:
        logger.critical("GROQ HEALTH CHECK: No keys configured — set GROQ_API_KEY in .env")
        return {"total": 0, "ok": 0, "rate_limited": 0, "dead": 0}

    logger.info("GROQ HEALTH CHECK: testing %d key(s) ...", len(keys))

    ok_count = 0
    rate_limited_count = 0
    dead_count = 0
    error_count = 0

    for i, key in enumerate(keys):
        key_num = i + 1
        key_hint = f"{key[:8]}…{key[-4:]}" if len(key) > 12 else key
        try:
            from groq import RateLimitError as _RLE, AuthenticationError as _AE
            client = Groq(api_key=key)
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=5,
                temperature=0,
            )
            ok_count += 1
            logger.info("  [%2d/%d] %s  →  OK", key_num, len(keys), key_hint)

        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            is_rate = (
                getattr(exc, "status_code", None) == 429
                or "rate limit" in msg_lower
                or "too many requests" in msg_lower
            )
            is_dead = (
                getattr(exc, "status_code", None) in (401, 403)
                or "invalid api key" in msg_lower
                or "invalid_api_key" in msg_lower
                or "organization has been restricted" in msg_lower
                or "organization_restricted" in msg_lower
            )

            if is_dead:
                dead_count += 1
                with _KEY_STATE_LOCK:
                    _DISABLED_GROQ_KEYS.add(key)
                logger.error(
                    "  [%2d/%d] %s  →  DEAD (disabled) — %s",
                    key_num, len(keys), key_hint, msg[:120],
                )
            elif is_rate:
                rate_limited_count += 1
                logger.warning(
                    "  [%2d/%d] %s  →  RATE LIMITED (will auto-recover)",
                    key_num, len(keys), key_hint,
                )
            else:
                error_count += 1
                logger.warning(
                    "  [%2d/%d] %s  →  ERROR — %s",
                    key_num, len(keys), key_hint, msg[:120],
                )

    usable = ok_count + rate_limited_count
    logger.info(
        "GROQ HEALTH CHECK DONE: %d total | %d OK | %d rate-limited | %d dead | %d error",
        len(keys), ok_count, rate_limited_count, dead_count, error_count,
    )
    if usable == 0:
        logger.critical(
            "GROQ: NO usable keys! All %d keys are dead/errored. Check .env immediately.",
            len(keys),
        )
    elif dead_count:
        logger.warning("GROQ: %d dead key(s) permanently disabled — remove them from .env", dead_count)

    return {
        "total": len(keys),
        "ok": ok_count,
        "rate_limited": rate_limited_count,
        "dead": dead_count,
        "error": error_count,
        "usable": usable,
    }


class LLMClient:
    """
    Single entry-point for all LLM calls, backed by Groq with multi-key rotation.

    complete() returns:
        {
            "content":    <dict (json_mode=True) | str (json_mode=False)>,
            "usage":      {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "model":      str,
            "latency_ms": float,
        }
    """

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = True,
        institute_id: Optional[str] = None,
        json_mode_suffix: Optional[str] = None,
    ) -> dict:
        from groq import Groq, RateLimitError as GroqRateLimitError

        if not GROQ_API_KEYS:
            raise RuntimeError("No GROQ_API_KEY configured -- set at least one in .env")

        effective_system = _ANTI_HALLUCINATION_PREFIX + system_prompt
        if json_mode:
            effective_system += (json_mode_suffix if json_mode_suffix is not None else _JSON_MODE_SUFFIX)

        effective_model = _resolve_model(model)

        kwargs = dict(
            model=effective_model,
            messages=[
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[str] = None

        def _parse_retry_after(msg: str, default: float = 65.0) -> float:
            """Parse 'Please try again in 1m46.5s' → seconds.
            Capped at 75s: Groq TPM windows are 60s, so 75s is always enough to reset.
            Never wait longer — a longer Groq-suggested time means TPD (daily) exhaustion
            on that specific key, but other keys will be available in round 2."""
            import re as _re
            m = _re.search(r"(\d+)m(\d+\.?\d*)s", msg or "")
            if m:
                return min(int(m.group(1)) * 60 + float(m.group(2)) + 2, 75.0)
            m = _re.search(r"(\d+\.?\d*)s", msg or "")
            if m:
                return min(float(m.group(1)) + 2, 75.0)
            return default

        def _is_permanently_bad_key_error(msg: str) -> bool:
            m = (msg or "").lower()
            return any(
                token in m
                for token in (
                    "invalid api key",
                    "invalid_api_key",
                    "organization has been restricted",
                    "organization_restricted",
                )
            )

        def _active_keys() -> list[str]:
            with _KEY_STATE_LOCK:
                keys = [k for k in GROQ_API_KEYS if k and k not in _DISABLED_GROQ_KEYS]
            return keys

        # Round-robin starting key: each invocation starts from a different key so that
        # sequential chunk calls (notes generation) distribute load evenly across the key pool
        # instead of every call starting at key[0] and triggering needless rate-limit retries.
        start_offset = _next_key_offset()

        # Three rounds across all keys: instant rotation on any error, sleep between rounds.
        # Round 1: normal attempt. Round 2 (after 65s): TPM windows guaranteed reset.
        # Round 3 (after another 65s): final attempt before giving up.
        for round_num in range(3):
            keys_this_round = _active_keys()
            if not keys_this_round:
                raise RuntimeError("No active GROQ keys left. Check invalid/restricted keys in .env")

            n = len(keys_this_round)
            offset = start_offset % n
            ordered_keys = keys_this_round[offset:] + keys_this_round[:offset]

            for key_idx, api_key in enumerate(ordered_keys):
                actual_key_num = (offset + key_idx) % n + 1  # human-readable key number for logs
                try:
                    client = Groq(api_key=api_key)
                    start = time.perf_counter()
                    resp = client.chat.completions.create(**kwargs)
                    latency_ms = (time.perf_counter() - start) * 1000

                    raw: str = resp.choices[0].message.content or ""
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    }

                    if not json_mode:
                        logger.info(
                            "LLM (text) | key=%d/%d model=%s latency=%.0fms",
                            actual_key_num, n, effective_model, latency_ms,
                        )
                        return {
                            "content": raw,
                            "usage": usage,
                            "model": effective_model,
                            "latency_ms": latency_ms,
                        }

                    try:
                        content = json.loads(_extract_json(raw))
                    except json.JSONDecodeError:
                        logger.warning("JSON parse failure on key %d -- retrying next key", actual_key_num)
                        last_error = "JSON parse failure"
                        continue

                    logger.info(
                        "LLM (json) | key=%d/%d model=%s latency=%.0fms",
                        actual_key_num, n, effective_model, latency_ms,
                    )
                    return {
                        "content": content,
                        "usage": usage,
                        "model": effective_model,
                        "latency_ms": latency_ms,
                    }

                except GroqRateLimitError as exc:
                    last_error = str(exc)
                    logger.warning(
                        "LLM key %d/%d rate-limited -- rotating to next key",
                        actual_key_num, n,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    if _is_permanently_bad_key_error(last_error):
                        with _KEY_STATE_LOCK:
                            _DISABLED_GROQ_KEYS.add(api_key)
                        logger.error(
                            "LLM key %d/%d permanently disabled (%s)",
                            actual_key_num, n, last_error,
                        )
                        continue
                    logger.error(
                        "LLM key %d/%d error (%s) -- rotating to next key",
                        actual_key_num, n, last_error,
                    )

            # All keys failed this round — wait before next round.
            # 75s cap guarantees any TPM window (60s) is fully reset before we retry.
            if round_num < 2:
                wait_s = _parse_retry_after(last_error or "", default=65.0)
                logger.warning(
                    "All %d LLM keys failed (round %d) -- waiting %.0fs before retry",
                    n, round_num + 1, wait_s,
                )
                time.sleep(wait_s)

        raise RuntimeError(
            f"LLM call failed after 3 rounds across all {len(GROQ_API_KEYS)} keys "
            f"(check dead/exhausted keys in .env): {last_error}"
        )

    # ── Internal: single-key call (no rotation) ───────────────────────────────

    def _groq_call_single_key(self, *, api_key: str, kwargs: dict) -> dict:
        """
        Make exactly ONE Groq call using the given api_key.
        Returns the same structure as complete().
        Raises on any error so the caller can decide what to do.
        """
        from groq import Groq
        client = Groq(api_key=api_key)
        start = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        raw: str = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        content = raw
        if kwargs.get("response_format", {}).get("type") == "json_object":
            try:
                content = json.loads(_extract_json(raw))
            except json.JSONDecodeError:
                raise RuntimeError(f"JSON parse failure from key")

        return {
            "content": content,
            "usage": usage,
            "model": kwargs["model"],
            "latency_ms": latency_ms,
        }

    # ── Parallel multi-chunk dispatch ─────────────────────────────────────────

    def parallel_complete_many(
        self,
        *,
        tasks: list[dict],
        model: str,
        temperature: float = 0.3,
        json_mode: bool = False,
        institute_id: Optional[str] = None,
    ) -> list[Optional[dict]]:
        """
        Fire every task in parallel, each pinned to a DIFFERENT Groq API key.

        tasks: list of dicts, each with:
            - system_prompt: str
            - user_prompt:   str
            - max_tokens:    int  (optional, default 1024)

        Returns a list of results in the same order as tasks.
        Failed tasks return None (caller should handle gracefully).

        With 13 keys and 8 chunks, all 8 LLM calls fire simultaneously —
        each uses a fresh key's full TPM budget, so total time ≈ 1 call time.
        """
        import concurrent.futures

        effective_model = _resolve_model(model)

        def _active_keys() -> list[str]:
            with _KEY_STATE_LOCK:
                return [k for k in GROQ_API_KEYS if k and k not in _DISABLED_GROQ_KEYS]

        active_keys = _active_keys()
        if not active_keys:
            raise RuntimeError("No active GROQ keys for parallel dispatch")

        n_tasks = len(tasks)
        results: list[Optional[dict]] = [None] * n_tasks

        def _run_task(task_idx: int, task: dict, api_key: str) -> tuple[int, Optional[dict]]:
            sys_prompt = _ANTI_HALLUCINATION_PREFIX + task["system_prompt"]
            kwargs = dict(
                model=effective_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": task["user_prompt"]},
                ],
                temperature=temperature,
                max_tokens=task.get("max_tokens", 1024),
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # Try the assigned key first
            try:
                result = self._groq_call_single_key(api_key=api_key, kwargs=kwargs)
                logger.info(
                    "parallel chunk %d/%d | key=%d/%d model=%s latency=%.0fms",
                    task_idx + 1, n_tasks,
                    active_keys.index(api_key) + 1, len(active_keys),
                    effective_model, result["latency_ms"],
                )
                return task_idx, result
            except Exception as primary_err:
                logger.warning(
                    "parallel chunk %d/%d | assigned key failed (%s) — falling back to rotation",
                    task_idx + 1, n_tasks, str(primary_err)[:120],
                )

            # Fall back: let standard complete() rotate across remaining keys
            try:
                result = self.complete(
                    system_prompt=task["system_prompt"],
                    user_prompt=task["user_prompt"],
                    model=model,
                    temperature=temperature,
                    max_tokens=task.get("max_tokens", 1024),
                    json_mode=json_mode,
                    institute_id=institute_id,
                )
                return task_idx, result
            except Exception as fallback_err:
                logger.error(
                    "parallel chunk %d/%d | fallback also failed: %s",
                    task_idx + 1, n_tasks, str(fallback_err)[:200],
                )
                return task_idx, None

        # Assign each task to a distinct key, cycling if tasks > keys
        assigned_keys = [active_keys[i % len(active_keys)] for i in range(n_tasks)]

        max_workers = min(n_tasks, len(active_keys))
        logger.info(
            "parallel_complete_many | %d chunks × %d keys | model=%s | workers=%d",
            n_tasks, len(active_keys), effective_model, max_workers,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_run_task, i, task, assigned_keys[i])
                for i, task in enumerate(tasks)
            ]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        successes = sum(1 for r in results if r is not None)
        logger.info(
            "parallel_complete_many done | %d/%d chunks succeeded",
            successes, n_tasks,
        )
        return results
