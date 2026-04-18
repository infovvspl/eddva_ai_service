"""
LLM client — remote Ollama endpoint on RunPod GPU.

One httpx.Client per process (singleton). All views share it.
Tuned for a remote GPU server: longer connect timeout, no CPU thread
hints, larger context window, exponential retry backoff for network
transients.
"""

import json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("ai_services.llm")

# ── Config — read once at import time ─────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://213.192.2.90:40077")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvav2")

# Prepended to every system prompt.
# JSON-specific instructions are added separately only when json_mode=True.
_ANTI_HALLUCINATION_PREFIX = (
    "You are EDVA AI, an expert Indian education assistant "
    "for JEE, NEET, and CBSE (Class 10-12).\n"
    "Only answer what is asked. Stay on topic. "
    "Use correct scientific facts only.\n\n"
)

_JSON_MODE_SUFFIX = (
    "\n\nRespond with ONLY a JSON object. No markdown. No code fences. No explanation."
)

# ── Singleton httpx client ─────────────────────────────────────────────────────
_http_client: Optional[httpx.Client] = None


def _get_http_client() -> httpx.Client:
    """
    Thread-safe lazy singleton.
    Tuned for RunPod:
      - connect=10s  (remote server, allow for cold-start)
      - read=300s    (GPU inference can take a while on large prompts)
      - pool limits  (prevent thundering-herd on a single GPU box)
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            base_url=OLLAMA_URL,
            timeout=httpx.Timeout(
                connect=10.0,   # network round-trip to RunPod
                read=300.0,     # generous — GPU inference
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30,
            ),
        )
        logger.info(
            "Ollama client ready → %s  model=%s", OLLAMA_URL, OLLAMA_MODEL
        )
    return _http_client


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
    Single entry-point for all LLM calls.

    complete() returns:
        {
            "content":    <dict (json_mode=True) | str (json_mode=False)>,
            "usage":      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model":      str,
            "latency_ms": float,
        }
    """

    MAX_RETRIES  = 3
    RETRY_BACKOFF = (2, 5, 10)  # seconds — longer gaps suit network retries

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt:   str,
        model:         str,           # accepted but always overridden by OLLAMA_MODEL
        temperature:   float = 0.7,
        max_tokens:    int   = 4096,
        json_mode:     bool  = True,
        institute_id:  Optional[str] = None,
    ) -> dict:

        effective_system = _ANTI_HALLUCINATION_PREFIX + system_prompt
        if json_mode:
            effective_system += _JSON_MODE_SUFFIX

        payload = {
            "model":    OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx":     4096,   # GPU has VRAM — use a decent context window
                # num_thread omitted: RunPod runs on GPU, not CPU threads
            },
        }

        client     = _get_http_client()
        last_error: Optional[str] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                start      = time.perf_counter()
                resp       = client.post("/api/chat", json=payload)
                latency_ms = (time.perf_counter() - start) * 1000

                resp.raise_for_status()
                raw: str = resp.json()["message"]["content"]
                usage    = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                # ── Plain-text mode ───────────────────────────────────────────
                if not json_mode:
                    logger.info(
                        "LLM (text) | model=%s latency=%.0fms institute=%s",
                        OLLAMA_MODEL, latency_ms, institute_id or "global",
                    )
                    return {
                        "content":    raw,
                        "usage":      usage,
                        "model":      OLLAMA_MODEL,
                        "latency_ms": latency_ms,
                    }

                # ── JSON mode ─────────────────────────────────────────────────
                try:
                    content = json.loads(_extract_json(raw))
                except json.JSONDecodeError:
                    if attempt < self.MAX_RETRIES - 1:
                        last_error = "JSON parse failure"
                        logger.warning(
                            "JSON parse failure attempt %d — retrying", attempt + 1
                        )
                        continue
                    # All retries exhausted — graceful fallback, never crash
                    logger.warning("JSON parse failed on all attempts — returning raw fallback")
                    content = {"raw": raw, "parse_error": True}

                logger.info(
                    "LLM (json) | model=%s latency=%.0fms institute=%s",
                    OLLAMA_MODEL, latency_ms, institute_id or "global",
                )
                return {
                    "content":    content,
                    "usage":      usage,
                    "model":      OLLAMA_MODEL,
                    "latency_ms": latency_ms,
                }

            except httpx.ConnectError as e:
                last_error = f"ConnectError: {e}"
                logger.error(
                    "Cannot reach RunPod Ollama (%s) attempt %d/%d",
                    OLLAMA_URL, attempt + 1, self.MAX_RETRIES,
                )
            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.error(
                    "Timeout from RunPod Ollama attempt %d/%d",
                    attempt + 1, self.MAX_RETRIES,
                )
            except Exception as e:
                last_error = str(e)
                logger.error(
                    "LLM error attempt %d/%d: %s",
                    attempt + 1, self.MAX_RETRIES, last_error,
                )

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_BACKOFF[attempt])

        raise RuntimeError(
            f"LLM call failed after {self.MAX_RETRIES} retries: {last_error}"
        )
