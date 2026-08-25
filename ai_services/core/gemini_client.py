"""
Gemini client — used only where a large context window is required.

Groq remains the default for this service. Grounded generation is the exception:
a chapter of a textbook is several thousand tokens, and Groq's on-demand tier
rejects any request whose prompt plus max_tokens exceeds 12,000 for the key
(see _MAX_TOKENS in views/ppt.py). Gemini's context comfortably fits a whole
chapter, so source-grounded calls go here and everything else stays on Groq.

The import is local to the call, matching core/note_images.py: google-genai is
optional in some environments, and importing at module load would break the
whole service where it is absent.
"""
import json
import logging
import os
import re
import time

from ai_services.core.gemini_keys import (
    _FALLBACK_CHAIN,
    get_gemini_api_keys,
    get_rotated_gemini_keys,
    is_gemini_invalid_argument_error,
    is_gemini_model_unavailable_error,
    mark_gemini_key_disabled,
    mark_gemini_model_unavailable,
    mark_zero_thinking_rejected,
    model_rejects_zero_thinking,
    resolve_gemini_model,
)

logger = logging.getLogger("ai_services.gemini")

DEFAULT_MODEL = "gemini-2.5-flash"

# Pause before trying the next key after a rate limit. Short, because rotating to
# a different key is the real remedy; the wait only helps when the limit is per
# project rather than per key.
_RETRY_BACKOFF_S = (0.5, 2.0, 5.0)


class GeminiUnavailable(RuntimeError):
    """Raised when the SDK or API key is missing, so callers can fall back."""


# A page citation emitted just outside the closing quote:
#     "Nonmetals make up most of the Earth's crust." [3],
# The model is following the instruction to cite its source but attaches the
# marker after the string rather than inside it, which is invalid JSON. Seen
# repeatedly on grounded generation, and the consequence was silent: the parse
# failed, the caller fell back to ungrounded output, and a teacher got a deck of
# general knowledge with no citations and no indication anything had gone wrong.
_STRAY_CITATION = re.compile(r'"\s*(\[p?\.?\s*\d+(?:\s*,\s*\d+)*\])\s*(?=[,\]\}])')


def _repair_json(text: str) -> str:
    """Fix the malformations Gemini produces on grounded output.

    Only touches sequences that are already invalid JSON, so well-formed output
    passes through untouched.
    """
    return _STRAY_CITATION.sub(lambda m: ' ' + m.group(1) + '"', text)


def is_available() -> bool:
    if not get_gemini_api_keys():
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _looks_rate_limited(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in ("429", "resource_exhausted", "rate limit", "quota"))


def _looks_key_rejected(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in ("api key not valid", "api_key_invalid", "permission_denied"))


def _looks_model_unavailable(exc: Exception) -> bool:
    """The key is valid but its project cannot use this model.

    gemini-2.5-flash is grandfathered: a project created after it was withdrawn
    gets 404 "no longer available to new users", while an older project keeps
    working. Such a key used to be retired here, which kept the pool honest but
    threw away real quota — every key bought after the withdrawal date was dead
    on arrival. It is now retried on a model its project *can* serve instead
    (see gemini_keys.resolve_gemini_model), so a new key adds capacity rather
    than noise.
    """
    return is_gemini_model_unavailable_error(str(exc))


def _without_zero_thinking(config):
    """Drop a zero thinking budget from a config, or return None if there isn't one.

    Returning None lets the caller tell "nothing to change" apart from "changed",
    so a 400 that had nothing to do with thinking is reported rather than retried.
    """
    thinking = getattr(config, "thinking_config", None)
    if thinking is None or getattr(thinking, "thinking_budget", None) != 0:
        return None
    try:
        return config.model_copy(update={"thinking_config": None})
    except Exception:
        try:  # older pydantic
            return config.copy(update={"thinking_config": None})
        except Exception:
            return None


def gemini_generate(api_key: str, *, model: str, contents, config):
    """One generate_content call on one key, on a model and config that key accepts.

    Two things vary by key rather than by deployment, and both are invisible
    until the API rejects the call:

    * the model — gemini-2.5-flash is withdrawn for projects created after the
      cutoff, so a key bought today 404s on it while an older key succeeds;
    * the thinking budget — the Gemini 3-era models those keys fall back to
      reject thinking_budget=0 with 400 INVALID_ARGUMENT.

    Both are learned on first refusal and cached, so a key pays each lesson once
    per process rather than once per request. Errors that are not about the
    model or the request shape propagate untouched for the caller to classify.
    """
    try:
        from google import genai
    except Exception as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    want = resolve_gemini_model(api_key, model)
    cfg = config
    if model_rejects_zero_thinking(want):
        cfg = _without_zero_thinking(config) or config

    # Each lesson costs at most one retry, and there are two kinds of lesson per
    # model in the chain, so this is bounded and cannot spin.
    for _ in range(2 * (len(_FALLBACK_CHAIN) + 1)):
        try:
            # Bind the client to a local: as a bare temporary it can be collected
            # while the request is in flight, closing its transport underneath
            # the call ("Cannot send a request, as the client has been closed").
            client = genai.Client(api_key=api_key)
            return client.models.generate_content(
                model=want, contents=contents, config=cfg
            )
        except Exception as exc:
            if is_gemini_model_unavailable_error(str(exc)):
                nxt = mark_gemini_model_unavailable(api_key, want)
                if not nxt:
                    raise
                logger.warning(
                    "Gemini: %s is withdrawn for this key's project, retrying on %s",
                    want, nxt,
                )
                want = nxt
                cfg = config
                if model_rejects_zero_thinking(want):
                    cfg = _without_zero_thinking(config) or config
                continue
            if is_gemini_invalid_argument_error(str(exc)):
                relaxed = _without_zero_thinking(cfg)
                if relaxed is None:
                    raise  # a 400 about something else — do not mask it
                mark_zero_thinking_rejected(want)
                logger.warning(
                    "Gemini: %s rejects thinking_budget=0, retrying without it", want
                )
                cfg = relaxed
                continue
            raise
    raise RuntimeError(f"Gemini: exhausted model/config retries for {model}")


def generate_with_rotation(*, contents, config, model: str = DEFAULT_MODEL, what: str = "request"):
    """Run one generate_content call, moving to another key rather than failing.

    Bulk work — indexing a school's whole library — is exactly what exhausts a
    single key's quota, and the keys to spread that over are already configured.
    A key the API rejects outright is disabled for the process rather than
    retried on every subsequent call.

    Returns the raw SDK result so the caller can inspect finish_reason; a
    truncated response is a successful call with an incomplete answer, which
    only the caller knows how to interpret.
    """
    keys = get_rotated_gemini_keys()
    if not keys:
        raise GeminiUnavailable("No Gemini API key is configured")
    try:
        from google import genai
    except Exception as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    last_exc: Exception | None = None
    for attempt, (key_no, key) in enumerate(keys):
        try:
            # gemini_generate settles the model and config this key can use; only
            # errors that are properties of the *key* reach here.
            return gemini_generate(key, model=model, contents=contents, config=config)
        except Exception as exc:
            last_exc = exc
            if _looks_model_unavailable(exc):
                # The key is fine, its project just has no model we can use.
                # Keep it in rotation and try the next key.
                logger.warning(
                    "Gemini key #%d has no usable model for %s; trying the next key",
                    key_no, what,
                )
                continue
            if _looks_key_rejected(exc):
                mark_gemini_key_disabled(key)
                logger.warning("Gemini key #%d rejected, disabling it: %s", key_no, exc)
                continue
            if _looks_rate_limited(exc):
                logger.warning(
                    "Gemini key #%d rate-limited on %s; trying the next key", key_no, what
                )
                time.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                continue
            # Anything else is a real failure — a bad prompt or an unreadable
            # file will fail identically on every key, so do not burn them all.
            raise

    raise RuntimeError(f"All {len(keys)} Gemini key(s) failed for {what}: {last_exc}")


def check_gemini_health() -> str:
    """Verify a configured Gemini key actually WORKS, not just that one exists.

    is_available() only checks that a key string is present and the SDK imports;
    a key can still be expired, project-restricted, or out of quota. This does
    one tiny generate so a broken Gemini surfaces at boot instead of as a
    teacher's grounded PPT/notes silently falling back to general knowledge.

    Returns one of: "ok", "no_keys", "key_rejected", "rate_limited",
    "model_unavailable", "error".
    """
    if not get_gemini_api_keys():
        return "no_keys"
    try:
        from google.genai import types
    except Exception:
        return "error"
    try:
        generate_with_rotation(
            contents=["ping"],
            config=types.GenerateContentConfig(max_output_tokens=8, temperature=0.0),
            what="health check",
        )
        return "ok"
    except Exception as exc:
        if _looks_key_rejected(exc):
            return "key_rejected"
        if _looks_rate_limited(exc):
            return "rate_limited"
        if _looks_model_unavailable(exc):
            return "model_unavailable"
        logger.warning("Gemini health check error: %s", exc)
        return "error"


def complete_text(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> dict:
    """Plain-text (Markdown) completion. Same contract as complete_json minus the
    JSON parsing, for the content types that return Markdown rather than a
    structured object."""
    try:
        from google.genai import types
    except Exception as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    started = time.time()
    result = generate_with_rotation(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # Gemini 2.5 charges its internal reasoning against max_output_tokens.
            # Grounded writing is faithful reproduction, not problem solving, and
            # the reasoning was eating enough budget to truncate a question paper
            # mid-question (finish_reason MAX_TOKENS with ~1.4k thinking tokens).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        what="text completion",
    )
    latency_ms = int((time.time() - started) * 1000)
    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")

    usage = getattr(result, "usage_metadata", None)
    return {
        "content": text,
        # The model actually used, which is not always the one asked for:
        # a key whose project cannot serve `model` is answered by a fallback,
        # and usage_logger prices per model, so reporting the request here
        # would bill the wrong rate.
        "model": getattr(result, "model_version", None) or model,
        "latency_ms": latency_ms,
        "tokens_input": getattr(usage, "prompt_token_count", 0) or 0,
        "tokens_output": getattr(usage, "candidates_token_count", 0) or 0,
    }


def complete_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> dict:
    """Return {content: dict, model, latency_ms} for a JSON-mode Gemini call.

    Raises GeminiUnavailable when unusable so the caller can fall back to Groq
    rather than failing the request outright.
    """
    try:
        from google.genai import types
    except Exception as exc:
        raise GeminiUnavailable(f"google-genai not installed: {exc}") from exc

    started = time.time()
    result = generate_with_rotation(
        model=model,
        # Gemini has no separate system role here; the instructions are prepended
        # and the source material follows, which keeps the ordering the model
        # sees identical to the Groq path.
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # See complete_text: reasoning tokens come out of the same budget.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        what="JSON completion",
    )
    latency_ms = int((time.time() - started) * 1000)

    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    try:
        content = json.loads(text)
    except json.JSONDecodeError as first_exc:
        # Retry once through the repair rather than discarding the response. The
        # alternative is falling back to ungrounded generation, which costs the
        # teacher the one thing they asked for — the book as the source.
        try:
            content = json.loads(_repair_json(text))
            logger.info("Gemini JSON repaired (was: %s)", first_exc)
        except json.JSONDecodeError:
            logger.error("Gemini returned invalid JSON: %s", first_exc)
            raise RuntimeError("Gemini returned malformed JSON") from first_exc

    usage = getattr(result, "usage_metadata", None)
    return {
        "content": content,
        # The model actually used, which is not always the one asked for:
        # a key whose project cannot serve `model` is answered by a fallback,
        # and usage_logger prices per model, so reporting the request here
        # would bill the wrong rate.
        "model": getattr(result, "model_version", None) or model,
        "latency_ms": latency_ms,
        "tokens_input": getattr(usage, "prompt_token_count", 0) or 0,
        "tokens_output": getattr(usage, "candidates_token_count", 0) or 0,
    }
