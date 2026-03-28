"""
Singleton LLM client with connection pooling and retry logic.
Creates ONE Groq client per process — never per-request.
"""

import os
import json
import time
import logging
import threading
from typing import Optional
from groq import Groq

logger = logging.getLogger("ai_services.llm")

_client_lock = threading.Lock()
_client_instance: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Thread-safe singleton Groq client."""
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError("GROQ_API_KEY environment variable is not set")
                _client_instance = Groq(api_key=api_key)
                logger.info("Groq client initialized (singleton)")
    return _client_instance


class LLMClient:
    """
    Unified LLM gateway. All views call this instead of Groq directly.
    Handles: model selection, retries, token tracking, JSON parsing.
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF = (1, 2, 4)  # seconds

    def __init__(self):
        self._client = get_groq_client()

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
        """
        Single entry point for all LLM calls.

        Returns:
            {
                "content": <parsed dict if json_mode else str>,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "model": str,
                "latency_ms": float,
            }
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                raw = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                content = json.loads(raw) if json_mode else raw

                logger.info(
                    "LLM call | model=%s tokens=%d latency=%.0fms institute=%s",
                    model,
                    usage["total_tokens"],
                    latency_ms,
                    institute_id or "global",
                )

                return {
                    "content": content,
                    "usage": usage,
                    "model": model,
                    "latency_ms": latency_ms,
                }

            except json.JSONDecodeError:
                # LLM returned non-JSON — return raw on last attempt
                if attempt == self.MAX_RETRIES - 1:
                    return {
                        "content": raw if json_mode else raw,
                        "usage": usage,
                        "model": model,
                        "latency_ms": latency_ms,
                    }
                last_error = "JSON parse failure, retrying"
                logger.warning("JSON parse failure on attempt %d", attempt + 1)

            except Exception as e:
                last_error = str(e)
                logger.error(
                    "LLM error attempt %d/%d: %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    last_error,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF[attempt])

        raise RuntimeError(f"LLM call failed after {self.MAX_RETRIES} retries: {last_error}")
