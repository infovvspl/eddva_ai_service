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

logger = logging.getLogger("ai_services.llm")

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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

# ── Singleton Groq client ─────────────────────────────────────────────────────
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client ready → model=%s", GROQ_MODEL)
    return _groq_client


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

        client = _get_groq_client()
        last_error: Optional[str] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                kwargs = dict(
                    model=GROQ_MODEL,
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
                resp  = client.chat.completions.create(**kwargs)
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
                        GROQ_MODEL, latency_ms, institute_id or "global",
                    )
                    return {
                        "content":    raw,
                        "usage":      usage,
                        "model":      GROQ_MODEL,
                        "latency_ms": latency_ms,
                    }

                # ── JSON mode ─────────────────────────────────────────────────
                try:
                    content = json.loads(_extract_json(raw))
                except json.JSONDecodeError:
                    if attempt < self.MAX_RETRIES - 1:
                        last_error = "JSON parse failure"
                        logger.warning("JSON parse failure attempt %d — retrying", attempt + 1)
                        continue
                    logger.warning("JSON parse failed on all attempts — returning raw fallback")
                    content = {"raw": raw, "parse_error": True}

                logger.info(
                    "LLM (json) | model=%s latency=%.0fms institute=%s",
                    GROQ_MODEL, latency_ms, institute_id or "global",
                )
                return {
                    "content":    content,
                    "usage":      usage,
                    "model":      GROQ_MODEL,
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
