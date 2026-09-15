"""
Provider adapters: the only code that knows how to talk to a vendor.

Each adapter takes a resolved ModelSpec + ProviderCall, returns the result dict
LLMClient has always returned::

    {"content", "usage", "model", "latency_ms", "tokens_input", "tokens_output"}

and raises only routing.errors types, classified retryable / non-retryable.

    GroqAdapter      delegates to LLMClient._complete_groq — the existing
                     multi-key rotation, 413 / json_validate fail-fast and
                     provider-event telemetry, unchanged.
    GeminiAdapter    delegates to gemini_client.complete_text / complete_json —
                     the existing key rotation, cooldowns and model fallback.
    TogetherAdapter  OpenAI-compatible chat completions over httpx.
                     Exactly ONE HTTP request per call: retry and fallback
                     belong to the router, and a retry layer here would multiply
                     with it. Models the provider only serves as a stream are
                     requested as one streaming request and assembled here.

Speech (Whisper, Sarvam) and image generation (FLUX) are deliberately not
adapters. They are not chat completions and do not belong in this router.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from ai_services.core.routing.errors import (
    NonRetryableProviderError,
    ProviderConfigError,
    ProviderError,
    RetryableProviderError,
    error_for_status,
)
from ai_services.core.routing.registry import JSON_CAPABLE, STRUCTURED_NATIVE, ModelSpec

logger = logging.getLogger("ai_services.routing")

# Together's OpenAI-compatible endpoint. Overridable; no model ids are implied.
DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"

# Fields copied from the provider's model listing when it reports them. Nothing
# is filled in when a field is absent.
_DISCOVERY_FIELDS = ("id", "type", "display_name", "organization", "context_length")


@dataclass(frozen=True)
class ProviderCall:
    system_prompt: str
    user_prompt: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 3500
    json_mode: bool = True
    json_mode_suffix: Optional[str] = None
    institute_id: Optional[str] = None
    timeout_s: Optional[float] = None
    # Apply LLMClient's historical system-prompt shaping (anti-hallucination
    # prefix + JSON suffix). Groq always applies it inside _complete_groq; the
    # other adapters apply it when set, so a fallback provider receives exactly
    # the instructions the primary did. Grounded call sites turn it off: their
    # prompts reached Gemini unshaped and must reach any provider unshaped.
    legacy_prompt_shaping: bool = True


_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-~+/=]+")


def scrub(text, *secrets: Optional[str]) -> str:
    """Strip credentials from text that may be logged or raised."""
    out = "" if text is None else str(text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, "[redacted]")
    return _BEARER_RE.sub("Bearer [redacted]", out)


def _shaped_system_prompt(call: ProviderCall) -> str:
    if not call.legacy_prompt_shaping:
        return call.system_prompt
    from ai_services.core.llm_client import build_effective_system_prompt

    return build_effective_system_prompt(call.system_prompt, call.json_mode, call.json_mode_suffix)


def _usage(tokens_in: int, tokens_out: int) -> dict:
    return {
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


class GroqAdapter:
    name = "groq"

    def is_configured(self) -> bool:
        from ai_services.core import llm_client as lc

        return bool(lc.GROQ_API_KEYS)

    def complete(self, spec: ModelSpec, call: ProviderCall) -> dict:
        from ai_services.core import llm_client as lc

        return lc.LLMClient()._complete_groq(
            system_prompt=call.system_prompt,
            user_prompt=call.user_prompt,
            model=call.model_id,
            temperature=call.temperature,
            max_tokens=call.max_tokens,
            json_mode=call.json_mode,
            institute_id=call.institute_id,
            json_mode_suffix=call.json_mode_suffix,
            legacy_prompt_shaping=call.legacy_prompt_shaping,
        )


def classify_gemini_error(exc: Exception, model: Optional[str]) -> ProviderError:
    """gemini_client raises plain RuntimeErrors; map their known forms."""
    msg = str(exc)
    low = msg.lower()
    if "malformed json" in low:
        return NonRetryableProviderError(
            f"Gemini returned malformed JSON (model {model})",
            provider="gemini", model=model, kind="structured_output",
        )
    if "empty response" in low:
        return NonRetryableProviderError(
            f"Gemini returned an empty response (model {model})",
            provider="gemini", model=model, kind="empty_response",
        )
    if "exhausted model/config retries" in low:
        return ProviderConfigError(
            f"Gemini has no usable model/config for {model}", provider="gemini", model=model,
        )
    if "gemini key(s) failed" in low:
        try:
            from ai_services.core.gemini_keys import is_gemini_permanent_key_error

            if is_gemini_permanent_key_error(msg):
                return ProviderConfigError(
                    "Every Gemini key was rejected", provider="gemini", model=model, kind="auth",
                )
        except Exception:
            pass
        return RetryableProviderError(
            f"Gemini key rotation exhausted: {scrub(msg)[:300]}",
            provider="gemini", model=model, kind="exhausted",
        )
    # Unknown form: do not fall back. Masking an unrecognised failure behind a
    # second vendor would hide a bug.
    return NonRetryableProviderError(
        f"Gemini error: {scrub(msg)[:300]}", provider="gemini", model=model, kind="deterministic",
    )


class GeminiAdapter:
    name = "gemini"

    def is_configured(self) -> bool:
        try:
            from ai_services.core.gemini_keys import has_gemini_api_key

            return bool(has_gemini_api_key())
        except Exception:
            return False

    def complete(self, spec: ModelSpec, call: ProviderCall) -> dict:
        from ai_services.core import gemini_client as gc

        fn = gc.complete_json if call.json_mode else gc.complete_text
        try:
            result = fn(
                system_prompt=_shaped_system_prompt(call),
                user_prompt=call.user_prompt,
                model=call.model_id,
                temperature=call.temperature,
                max_output_tokens=call.max_tokens,
            )
        except gc.GeminiUnavailable as exc:  # subclass of RuntimeError: must come first
            raise ProviderConfigError(
                f"Gemini unavailable: {scrub(exc)}", provider="gemini", model=call.model_id,
            ) from None
        except ProviderError:
            raise
        except RuntimeError as exc:
            raise classify_gemini_error(exc, call.model_id) from exc
        out = dict(result)
        out.setdefault(
            "usage", _usage(int(out.get("tokens_input") or 0), int(out.get("tokens_output") or 0))
        )
        return out


def _retry_after_ms(resp) -> Optional[int]:
    raw = resp.headers.get("retry-after") if getattr(resp, "headers", None) else None
    try:
        return int(float(raw) * 1000) if raw else None
    except (TypeError, ValueError):
        return None


class TogetherAdapter:
    name = "together"

    @staticmethod
    def _api_key() -> str:
        return (os.getenv(TOGETHER_API_KEY_ENV) or "").strip()

    @staticmethod
    def base_url() -> str:
        return ((os.getenv("TOGETHER_BASE_URL") or "").strip() or DEFAULT_TOGETHER_BASE_URL).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self._api_key())

    def _event(self, event_type: str, model: Optional[str], status: Optional[int], key: str,
               retry_after_ms: Optional[int] = None) -> None:
        try:
            from ai_services.core import provider_events

            provider_events.emit(
                event_type=event_type, provider="together", model=model, status_code=status,
                retry_after_ms=retry_after_ms, key_hash=provider_events.key_fingerprint(key),
            )
        except Exception:
            pass  # telemetry must never break a call

    def _headers(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _raise_http_error(self, status: int, text: str, model: str, key: str,
                          retry_after_ms: Optional[int], streaming_hint: str) -> None:
        evt = "429" if status == 429 else ("5xx" if status >= 500 else "provider_error")
        self._event(evt, model, status, key, retry_after_ms)
        text = text or ""
        hint = streaming_hint if ("streaming_required" in text or "supports streaming" in text.lower()) else ""
        raise error_for_status(
            status, f"Together {status} for model {model}: {scrub(text[:300], key)}{hint}",
            provider="together", model=model,
        )

    def _post_completion(self, body: dict, key: str, model: str, timeout: float, streaming_hint: str):
        """One non-streaming request. Returns (status, raw_text, usage_or_None, reported_model)."""
        import httpx

        try:
            resp = httpx.post(
                f"{self.base_url()}/chat/completions", json=body, headers=self._headers(key), timeout=timeout,
            )
        except httpx.TimeoutException:
            self._event("timeout", model, None, key)
            raise RetryableProviderError(
                f"Together request timed out after {timeout:.0f}s (model {model})",
                provider="together", model=model, kind="timeout",
            ) from None
        except httpx.RequestError as exc:
            self._event("provider_error", model, None, key)
            raise RetryableProviderError(
                f"Together network error ({exc.__class__.__name__}) for model {model}",
                provider="together", model=model, kind="network",
            ) from None

        if resp.status_code >= 400:
            self._raise_http_error(resp.status_code, resp.text, model, key, _retry_after_ms(resp), streaming_hint)

        try:
            data = resp.json()
            raw = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            self._event("provider_error", model, resp.status_code, key)
            raise RetryableProviderError(
                f"Together returned an unreadable response envelope (model {model})",
                provider="together", model=model, status_code=resp.status_code, kind="server_error",
            ) from None
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        reported = data.get("model") if isinstance(data.get("model"), str) else None
        return resp.status_code, raw, usage, reported

    def _stream_completion(self, body: dict, key: str, model: str, timeout: float, streaming_hint: str):
        """One streaming request, assembled into a complete answer.

        Used only for models configured as streaming-only. Still exactly one HTTP
        request, and the timeout bounds the WHOLE stream, not just the gap
        between chunks — a slow trickle must not hold a worker indefinitely.
        """
        import httpx

        body = dict(body, stream=True)
        deadline = time.monotonic() + timeout
        parts: list[str] = []
        usage = None
        reported = None
        status = None
        try:
            with httpx.stream(
                "POST", f"{self.base_url()}/chat/completions", json=body, headers=self._headers(key), timeout=timeout,
            ) as resp:
                status = resp.status_code
                if status >= 400:
                    text = resp.read().decode("utf-8", "replace")
                    self._raise_http_error(status, text, model, key, _retry_after_ms(resp), streaming_hint)
                for line in resp.iter_lines():
                    if time.monotonic() > deadline:
                        self._event("timeout", model, None, key)
                        raise RetryableProviderError(
                            f"Together stream exceeded {timeout:.0f}s (model {model})",
                            provider="together", model=model, kind="timeout",
                        )
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue  # blank separators, SSE comments / keep-alives
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        self._event("provider_error", model, status, key)
                        raise RetryableProviderError(
                            f"Together returned an unreadable stream chunk (model {model})",
                            provider="together", model=model, status_code=status, kind="server_error",
                        ) from None
                    if isinstance(chunk.get("error"), dict):
                        # An error inside an accepted stream is not a capacity
                        # signal we can classify; never mask it behind a fallback.
                        self._event("provider_error", model, status, key)
                        raise NonRetryableProviderError(
                            f"Together stream error for model {model}: "
                            f"{scrub(json.dumps(chunk['error'])[:300], key)}",
                            provider="together", model=model, status_code=status, kind="deterministic",
                        )
                    if reported is None and isinstance(chunk.get("model"), str):
                        reported = chunk["model"]
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = (choice or {}).get("delta") or {}
                        if isinstance(delta.get("content"), str):
                            parts.append(delta["content"])
        except httpx.TimeoutException:
            self._event("timeout", model, None, key)
            raise RetryableProviderError(
                f"Together request timed out after {timeout:.0f}s (model {model})",
                provider="together", model=model, kind="timeout",
            ) from None
        except httpx.RequestError as exc:
            self._event("provider_error", model, None, key)
            raise RetryableProviderError(
                f"Together network error ({exc.__class__.__name__}) for model {model}",
                provider="together", model=model, kind="network",
            ) from None
        return status, "".join(parts), usage, reported

    def complete(self, spec: ModelSpec, call: ProviderCall) -> dict:
        from ai_services.core.llm_client import _extract_json, strip_think_tags

        key = self._api_key()
        model = call.model_id
        if not key:
            raise ProviderConfigError(
                f"Together is not configured ({TOGETHER_API_KEY_ENV} is unset)", provider="together", model=model,
            )
        if not model:
            hint = f" — set {spec.model_env}" if spec.model_env else ""
            raise ProviderConfigError(
                f"No Together model id configured for {spec.id}{hint}", provider="together", model=None,
            )
        # Never downgrade a JSON requirement. "prompted" is an explicit,
        # operator-selected compatibility path; "unknown" and "none" refuse
        # before any tokens are spent.
        if call.json_mode and spec.structured_output not in JSON_CAPABLE:
            hint = f"; verify the model, then set {spec.model_env}_STRUCTURED_OUTPUT=native|prompted" \
                if spec.model_env else ""
            raise NonRetryableProviderError(
                f"{spec.id} structured-output support is {spec.structured_output.upper()}: refusing a JSON "
                f"request rather than downgrading it{hint}",
                provider="together", model=model, kind="structured_output",
            )

        timeout = float(call.timeout_s or os.getenv("TOGETHER_TIMEOUT_S") or 60)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _shaped_system_prompt(call)},
                {"role": "user", "content": call.user_prompt},
            ],
            "temperature": call.temperature,
            "max_tokens": call.max_tokens,
        }
        if call.json_mode and spec.structured_output == STRUCTURED_NATIVE:
            body["response_format"] = {"type": "json_object"}

        streamed = spec.streaming is True
        streaming_hint = (
            f" — this model only accepts streaming requests; set {spec.model_env}_STREAMING=true"
            if spec.model_env else ""
        )
        started = time.perf_counter()
        if streamed:
            status_code, raw, usage, reported = self._stream_completion(body, key, model, timeout, streaming_hint)
        else:
            status_code, raw, usage, reported = self._post_completion(body, key, model, timeout, streaming_hint)
        latency_ms = (time.perf_counter() - started) * 1000

        tokens_reported = bool(usage) and ("prompt_tokens" in usage or "completion_tokens" in usage)
        tokens_in = int((usage or {}).get("prompt_tokens") or 0)
        tokens_out = int((usage or {}).get("completion_tokens") or 0)
        if not tokens_reported:
            # The usage row needs integers. Record 0, but say so: the row is not
            # provider-reported usage and must not be read as "free".
            logger.warning(
                "Together model %s returned no token usage; recorded as 0 with tokens_reported=false", model,
            )

        if call.json_mode:
            try:
                content = json.loads(_extract_json(raw))
            except json.JSONDecodeError:
                # Deterministic: same prompt, same model -> same malformed output.
                raise NonRetryableProviderError(
                    f"Together model {model} did not return valid JSON",
                    provider="together", model=model, status_code=status_code, kind="structured_output",
                ) from None
        else:
            content = strip_think_tags(raw)
            if not content:
                raise NonRetryableProviderError(
                    f"Together model {model} returned an empty response",
                    provider="together", model=model, status_code=status_code, kind="empty_response",
                )

        logger.info(
            "LLM (%s) | provider=together model=%s reported_model=%s stream=%s latency=%.0fms tokens=%s",
            "json" if call.json_mode else "text", model, reported or "-", streamed, latency_ms,
            f"{tokens_in}+{tokens_out}" if tokens_reported else "not-reported",
        )
        return {
            "content": content,
            "usage": _usage(tokens_in, tokens_out),
            "model": model,
            # What the provider says actually served the request (may carry a
            # version suffix the configured id does not).
            "provider_reported_model": reported,
            "latency_ms": latency_ms,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_reported": tokens_reported,
            "streamed": streamed,
        }

    def list_model_records(self) -> list[dict]:
        """The models this account can call, as the provider reports them. The
        only sanctioned way to obtain Together ids — they are never guessed."""
        import httpx

        key = self._api_key()
        if not key:
            raise ProviderConfigError(
                f"Together is not configured ({TOGETHER_API_KEY_ENV} is unset)", provider="together",
            )
        try:
            resp = httpx.get(f"{self.base_url()}/models", headers=self._headers(key), timeout=30.0)
        except httpx.RequestError as exc:
            raise RetryableProviderError(
                f"Together network error ({exc.__class__.__name__}) listing models",
                provider="together", kind="network",
            ) from None
        if resp.status_code >= 400:
            raise error_for_status(
                resp.status_code,
                f"Together {resp.status_code} listing models: {scrub(resp.text[:200], key)}",
                provider="together", model=None,
            )
        try:
            data = resp.json()
        except ValueError:
            raise RetryableProviderError(
                "Together returned an unreadable model listing", provider="together", kind="server_error",
            ) from None
        items = data if isinstance(data, list) else (data.get("data") or [])
        records = [
            {k: item[k] for k in _DISCOVERY_FIELDS if k in item}
            for item in items if isinstance(item, dict) and item.get("id")
        ]
        return sorted(records, key=lambda r: str(r["id"]))

    def list_models(self) -> list[str]:
        return [str(r["id"]) for r in self.list_model_records()]


def default_adapters() -> dict:
    return {"groq": GroqAdapter(), "gemini": GeminiAdapter(), "together": TogetherAdapter()}
