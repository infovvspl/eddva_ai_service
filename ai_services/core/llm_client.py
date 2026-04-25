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
    get_rotated_groq_keys,
    is_key_exhausted_error,
)

logger = logging.getLogger("ai_services.llm")
_KEY_STATE_LOCK = threading.Lock()
_DISABLED_GROQ_KEYS: set[str] = set()

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
]
GROQ_API_KEYS: list[str] = [k for k in _GROQ_KEYS_RAW if k]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # backward compat
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Models that can be requested by name -- anything else falls back to GROQ_MODEL
_GROQ_ALLOWED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "quiz",
}
_GROQ_MODEL_ALIAS = {
    "quiz": "llama-3.1-8b-instant",
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
    ) -> dict:
        from groq import Groq, RateLimitError as GroqRateLimitError

        if not GROQ_API_KEYS:
            raise RuntimeError("No GROQ_API_KEY configured -- set at least one in .env")

        effective_system = _ANTI_HALLUCINATION_PREFIX + system_prompt
        if json_mode:
            effective_system += _JSON_MODE_SUFFIX

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

        # Two rounds across all keys: instant rotation on any error, sleep only if all fail
        for round_num in range(2):
            keys_this_round = _active_keys()
            if not keys_this_round:
                raise RuntimeError("No active GROQ keys left. Check invalid/restricted keys in .env")

            for key_idx, api_key in enumerate(keys_this_round):
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
                            key_idx + 1, len(keys_this_round), effective_model, latency_ms,
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
                        logger.warning("JSON parse failure on key %d -- retrying next key", key_idx + 1)
                        last_error = "JSON parse failure"
                        continue

                    logger.info(
                        "LLM (json) | key=%d/%d model=%s latency=%.0fms",
                        key_idx + 1, len(keys_this_round), effective_model, latency_ms,
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
                        key_idx + 1, len(keys_this_round),
                    )
                except Exception as exc:
                    last_error = str(exc)
                    if _is_permanently_bad_key_error(last_error):
                        with _KEY_STATE_LOCK:
                            _DISABLED_GROQ_KEYS.add(api_key)
                        logger.error(
                            "LLM key %d/%d permanently disabled (%s)",
                            key_idx + 1, len(keys_this_round), last_error,
                        )
                        continue
                    logger.error(
                        "LLM key %d/%d error (%s) -- rotating to next key",
                        key_idx + 1, len(keys_this_round), last_error,
                    )

            # All keys failed this round -- wait 10s before round 2
            if round_num == 0:
                logger.warning(
                    "All %d LLM keys failed (round 1) -- waiting 10s before retry",
                    len(keys_this_round),
                )
                time.sleep(10)

        raise RuntimeError(
            f"LLM call failed after exhausting all {len(GROQ_API_KEYS)} keys: {last_error}"
        )
