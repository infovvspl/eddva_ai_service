"""
LLM client -- Groq API (openai/gpt-oss-120b).

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
#
# get_groq_api_keys() is the single source of truth. It reads ALL supported forms:
#   GROQ_API_KEY  |  GROQ_API_KEYS (comma-separated)  |  GROQ_API_KEY_1..N
#
# This used to be followed by a second, hardcoded GROQ_API_KEYS = [...] built only
# from GROQ_API_KEY and GROQ_API_KEY_1..20, which SHADOWED the line above and
# silently discarded any keys supplied via the comma-separated GROQ_API_KEYS.
# That is exactly how production ended up with "No GROQ_API_KEY configured" while
# holding 20 valid keys in the environment. Do not reintroduce it.
GROQ_API_KEYS: list[str] = get_groq_api_keys()
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # backward compat
# Groq decommissioned llama-3.1-8b-instant + llama-3.3-70b-versatile on
# 2026-08-16. Default is now GPT-OSS (Groq's recommended replacement).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Models that can be requested by name -- anything else falls back to GROQ_MODEL.
# The old llama-3.x names are intentionally dropped so any lingering request
# for them resolves to the current default instead of a decommissioned model.
_GROQ_ALLOWED_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "gemma2-9b-it",
    "quiz",
}
_GROQ_MODEL_ALIAS = {
    # Legacy aliases remapped onto current models.
    "quiz": "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "qwen/qwen3-32b": "qwen/qwen3-32b",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "math": "qwen/qwen3-32b",
    "reasoning": "openai/gpt-oss-120b",
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
    "Use correct scientific facts only.\n"
    "CRITICAL: Do NOT include internal reasoning, chain-of-thought, or <think> tags. "
    "DO NOT THINK. START YOUR RESPONSE DIRECTLY WITH '{'.\n\n"
)

_JSON_MODE_SUFFIX = "\n\nRespond with ONLY a JSON object. No markdown. No code fences. No explanation."

# For tutor/teacher-style replies: JSON is required, but string values may include Markdown and formulas.
_JSON_MODE_TUTOR_SUFFIX = (
    "\n\nRespond with ONLY one valid JSON object. "
    "In string fields (especially \"response\"), you may use Markdown, **bold**, and normal math text for equations."
)


def _extract_json(raw: str) -> str:
    import re
    # 1. Aggressively remove <think> blocks (including unclosed ones)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Strip markdown fences
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    
    # 3. Find the first '{' and the last '}' to isolate the JSON object
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
        
    return stripped

def strip_think_tags(text: str) -> str:
    import re
    # Remove closed blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed blocks
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
                model="openai/gpt-oss-20b",
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
        max_tokens: int = 3500,
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

        def _parse_retry_after(msg: str, default: float = 15.0) -> float:
            """Parse 'Please try again in Xs' from Groq rate-limit error.
            Cap at 20s — if Groq says wait longer it means TPD exhaustion on that key,
            so we should move to the next key quickly rather than blocking the request."""
            import re as _re
            m = _re.search(r"(\d+)m(\d+\.?\d*)s", msg or "")
            if m:
                return min(int(m.group(1)) * 60 + float(m.group(2)) + 1, 20.0)
            m = _re.search(r"(\d+\.?\d*)s", msg or "")
            if m:
                return min(float(m.group(1)) + 1, 20.0)
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

        def _is_request_too_large_error(msg: str) -> bool:
            # 413 "Request too large ... reduce your message size" is a hard
            # per-request token cap tied to the model/tier, not a per-key quota --
            # every key hits the identical limit against the identical request, so
            # rotating keys or waiting for the next round can never succeed.
            m = (msg or "").lower()
            return any(token in m for token in ("413", "request too large", "reduce your message size"))

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
                        raw = strip_think_tags(raw)
                        logger.info(
                            "LLM (text) | key=%d/%d model=%s latency=%.0fms",
                            actual_key_num, n, effective_model, latency_ms,
                        )
                        return {
                            "content": raw,
                            "usage": usage,
                            "model": effective_model,
                            "latency_ms": latency_ms,
                            "tokens_input": usage.get("prompt_tokens", 0),
                            "tokens_output": usage.get("completion_tokens", 0),
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
                        "tokens_input": usage.get("prompt_tokens", 0),
                        "tokens_output": usage.get("completion_tokens", 0),
                    }

                except GroqRateLimitError as exc:
                    last_error = str(exc)
                    logger.warning(
                        "LLM key %d/%d rate-limited -- rotating to next key",
                        actual_key_num, n,
                    )
                    # P0-5: record the 429 that rotation is about to absorb, so a
                    # request that ultimately succeeds still leaves evidence of the
                    # rate-limit pressure it hit.
                    try:
                        from ai_services.core import provider_events as _pev
                        _pev.emit(
                            event_type="429", provider="groq", model=effective_model,
                            status_code=429, attempt_number=actual_key_num,
                            retry_after_ms=int(_parse_retry_after(last_error, default=0.0) * 1000) or None,
                            key_hash=_pev.key_fingerprint(api_key),
                        )
                    except Exception:
                        pass
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
                    if _is_request_too_large_error(last_error):
                        # Fail fast instead of burning the remaining keys and up to
                        # 2 more 5s-spaced rounds (~30+ calls) on a request that is
                        # guaranteed to be rejected identically every time -- the
                        # caller needs to reduce max_tokens/prompt size, not retry.
                        logger.error(
                            "LLM key %d/%d request too large for model %s -- not retrying (%s)",
                            actual_key_num, n, effective_model, last_error,
                        )
                        raise RuntimeError(f"LLM request too large for model {effective_model}: {last_error}") from exc
                    logger.error(
                        "LLM key %d/%d error (%s) -- rotating to next key",
                        actual_key_num, n, last_error,
                    )
                    # P0-5: a non-429 provider error we're rotating past.
                    try:
                        from ai_services.core import provider_events as _pev
                        _sc = getattr(exc, "status_code", None)
                        _pev.emit(
                            event_type="5xx" if (_sc and _sc >= 500) else "provider_error",
                            provider="groq", model=effective_model, status_code=_sc,
                            attempt_number=actual_key_num,
                            key_hash=_pev.key_fingerprint(api_key),
                        )
                    except Exception:
                        pass

            # All keys failed this round — short wait then try again.
            # Cap at 5s to avoid blocking a gunicorn worker for too long.
            if round_num < 2:
                wait_s = min(_parse_retry_after(last_error or "", default=5.0), 5.0)
                logger.warning(
                    "All %d LLM keys failed (round %d) -- waiting %.0fs before retry",
                    n, round_num + 1, wait_s,
                )
                time.sleep(wait_s)

        raise RuntimeError(
            f"LLM call failed after 3 rounds across all {len(GROQ_API_KEYS)} keys "
            f"(check dead/exhausted keys in .env): {last_error}"
        )

    def parallel_complete_many(
        self,
        tasks: list[dict],
        model: str = None,
        temperature: float = 0.3,
        json_mode: bool = False,
        institute_id: str = "default",
    ) -> list[dict]:
        """
        Execute multiple LLM completion tasks in parallel.
        Each task is a dict: {"system_prompt": str, "user_prompt": str, "max_tokens": int}
        Uses different keys for each task to maximize parallel TPM capacity.
        """
        from concurrent.futures import ThreadPoolExecutor

        n_tasks = len(tasks)
        if n_tasks == 0:
            return []

        def _worker(task_idx):
            task = tasks[task_idx]
            return self.complete(
                system_prompt=task["system_prompt"],
                user_prompt=task["user_prompt"],
                model=model,
                temperature=temperature,
                max_tokens=task.get("max_tokens", 3500),
                json_mode=json_mode,
                institute_id=institute_id,
            )

        with ThreadPoolExecutor(max_workers=n_tasks) as executor:
            results = list(executor.map(_worker, range(n_tasks)))

        return results
