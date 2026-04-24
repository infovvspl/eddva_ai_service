"""
LLM client — Groq API (llama-3.3-70b-versatile).

Keeps the same .complete() interface as the previous Ollama client so every
view continues to work without changes.  Ollama config env-vars are preserved
but unused — they will be wired back once edvav2 is fine-tuned.
"""

import json
import logging
import os
import time
from typing import Optional

from ai_services.core.groq_keys import (
    get_rotated_groq_keys,
    is_key_exhausted_error,
)

logger = logging.getLogger("ai_services.llm")

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Models that can be requested by name — anything else falls back to GROQ_MODEL
_GROQ_ALLOWED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "quiz",  # alias → 8b for low-token quiz calls
}
_GROQ_MODEL_ALIAS = {
    "quiz": "llama-3.1-8b-instant",
}

def _resolve_model(model: str) -> str:
    """Return the actual Groq model ID to use."""
    if model in _GROQ_MODEL_ALIAS:
        return _GROQ_MODEL_ALIAS[model]
    if model in _GROQ_ALLOWED_MODELS:
        return model
    return GROQ_MODEL

# ── Ollama config (kept for future use) ──────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://213.192.2.90:40077")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvav2")

# Prepended to every system prompt.
_ANTI_HALLUCINATION_PREFIX = (
    "You are EDVA AI, an expert Indian education assistant "
    "for JEE, NEET, and CBSE (Class 10-12).\n"
    "Only answer what is asked. Stay on topic. "
    "Use correct scientific facts only.\n\n"
)

_JSON_MODE_SUFFIX = (
    "\n\nRespond with ONLY a JSON object. No markdown. No code fences. No explanation."
)

# ── Per-key Groq client cache ─────────────────────────────────────────────────
_groq_clients = {}


def _get_groq_client(api_key: str):
    if api_key not in _groq_clients:
        from groq import Groq
        _groq_clients[api_key] = Groq(api_key=api_key)
        logger.info("Groq client ready → model=%s", GROQ_MODEL)
    return _groq_clients[api_key]


def _extract_json(raw: str) -> str:
    """Strip markdown code fences if the model wraps JSON in ```json ... ```."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


class LLMClient:
    """
    Single entry-point for all LLM calls.  Now backed by Groq.

    complete() returns:
        {
            "content":    <dict (json_mode=True) | str (json_mode=False)>,
            "usage":      {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "model":      str,
            "latency_ms": float,
        }
    """

    MAX_RETRIES   = 3
    RETRY_BACKOFF = (2, 5, 10)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt:   str,
        model:         str,           # accepted but overridden by GROQ_MODEL
        temperature:   float = 0.7,
        max_tokens:    int   = 4096,
        json_mode:     bool  = True,
        institute_id:  Optional[str] = None,
    ) -> dict:

        effective_system = _ANTI_HALLUCINATION_PREFIX + system_prompt
        if json_mode:
            effective_system += _JSON_MODE_SUFFIX

        last_error: Optional[str] = None

        effective_model = _resolve_model(model)
        for attempt in range(self.MAX_RETRIES):
            key_candidates = get_rotated_groq_keys()
            if not key_candidates:
                raise RuntimeError("No GROQ API key configured. Set GROQ_API_KEY or GROQ_API_KEY_1..N")
            try:
                for key_idx, key in enumerate(key_candidates):
                    client = _get_groq_client(key)
                    kwargs = dict(
                        model=effective_model,
                        messages=[
                            {"role": "system", "content": effective_system},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                    start = time.perf_counter()
                    try:
                        resp = client.chat.completions.create(**kwargs)
                    except Exception as key_err:
                        if is_key_exhausted_error(key_err) and key_idx < len(key_candidates) - 1:
                            last_error = str(key_err)
                            logger.warning(
                                "Groq key exhausted/limited (attempt %d key %d/%d) — rotating key",
                                attempt + 1, key_idx + 1, len(key_candidates),
                            )
                            continue
                        raise

                    latency_ms = (time.perf_counter() - start) * 1000

                    raw: str = resp.choices[0].message.content or ""
                    usage = {
                        "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                        "total_tokens":      resp.usage.total_tokens      if resp.usage else 0,
                    }

                    # ── Plain-text mode ───────────────────────────────────────────
                    if not json_mode:
                        logger.info(
                            "LLM (text) | model=%s latency=%.0fms institute=%s",
                            effective_model, latency_ms, institute_id or "global",
                        )
                        return {
                            "content":    raw,
                            "usage":      usage,
                            "model":      effective_model,
                            "latency_ms": latency_ms,
                        }

                    # ── JSON mode ─────────────────────────────────────────────────
                    try:
                        content = json.loads(_extract_json(raw))
                    except json.JSONDecodeError:
                        if attempt < self.MAX_RETRIES - 1:
                            last_error = "JSON parse failure"
                            logger.warning("JSON parse failure attempt %d — retrying", attempt + 1)
                            break
                        logger.warning("JSON parse failed on all attempts — returning raw fallback")
                        content = {"raw": raw, "parse_error": True}

                    logger.info(
                        "LLM (json) | model=%s latency=%.0fms institute=%s",
                        effective_model, latency_ms, institute_id or "global",
                    )
                    return {
                        "content":    content,
                        "usage":      usage,
                        "model":      effective_model,
                        "latency_ms": latency_ms,
                    }

            except Exception as e:
                last_error = str(e)
                logger.error(
                    "Groq error attempt %d/%d: %s",
                    attempt + 1, self.MAX_RETRIES, last_error,
                )

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_BACKOFF[attempt])

        raise RuntimeError(
            f"LLM call failed after {self.MAX_RETRIES} retries: {last_error}"
        )
