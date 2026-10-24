"""
Model route for chapter lesson plans (content types lesson_plan /
lesson_plan_extract / lesson_plan_section).

Policy (owner decision): Gemini is never called for lesson plans — not as the
primary, not as a fallback, not for a repair or a section rewrite. To make that
structural rather than a matter of configuration, lesson plans do NOT go
through the model router (whose grounded policy falls back to Gemini, and whose
legacy path pins Gemini at the grounded call site). They call exactly one
explicitly configured, registered, non-Gemini model through its provider adapter,
with no fallback at all. Anything missing, unknown or Gemini fails closed.

    LESSON_PLAN_AI_MODEL     required; a registry reference, e.g. together/glm-5.3-flash
                             (its provider model id comes from that model's env, e.g.
                             TOGETHER_MODEL_GLM53_FLASH)

Optional pilot cost guard (off unless LESSON_PLAN_COST_GUARD=true). When on, every
call is checked BEFORE it is made: the worst-case cost of the call (estimated input
tokens + the output cap, priced at the configured rates) plus everything already
spent must stay under the ceiling, and the attempt count under its cap. Each call is
recorded in a local JSON-lines ledger. Any missing guard setting fails closed.

    LESSON_PLAN_COST_GUARD=true
    LESSON_PLAN_LEDGER_PATH            file the ledger is appended to
    LESSON_PLAN_CEILING_INR            e.g. 450
    LESSON_PLAN_MAX_ATTEMPTS           e.g. 6 (all calls, including repairs)
    LESSON_PLAN_PRICE_INPUT_INR_PER_M  price per 1M input tokens, taxes included
    LESSON_PLAN_PRICE_OUTPUT_INR_PER_M price per 1M output tokens, taxes included
    LESSON_PLAN_OUTPUT_WORST_FACTOR    multiplier on max_tokens for the worst case (default 2:
                                       reasoning tokens may not be capped by max_tokens)
    LESSON_PLAN_MAX_ATTEMPTS_PER_CASE  optional; calls per case label (the bridge labels a call
                                       "<subject>|<chapter>" unless the request names an evalLabel)
    LESSON_PLAN_GUARD_REFUSE_REPAIRS   optional; true refuses the backend's repair call

The ledger file must already exist: the guard never creates one, so a mistyped path
or a lost file cannot silently restart the count at zero. Create it once per pilot
with init_ledger(path, note=...). Counting is per call_id and survives restarts
(the file is re-read before every call). One server process only: the lock below is
per process.

A call's cost is "actual" only when the provider reported token usage; otherwise it
is counted at its pre-call worst case and marked "worst_case_estimate". If the
provider reports a different model than the one configured, the call is recorded
as a mismatch and every later call is refused.

Secrets are never logged or written: the ledger holds counts, ids and costs only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ai_services.lesson_plan_route")

ALLOWED_PROVIDERS = frozenset({"together", "groq"})
# Conservative token estimate for the pre-call guard: ~4 chars/token is typical for
# English; Devanagari and similar scripts tokenise far denser, so assume 2.
CHARS_PER_TOKEN_ESTIMATE = 2.0

_LOCK = threading.Lock()


class LessonPlanRouteError(RuntimeError):
    """The lesson-plan route is missing, invalid or not allowed (fail closed)."""


class LessonPlanBudgetError(RuntimeError):
    """The pilot cost guard refused a call before it was made."""


@dataclass(frozen=True)
class Route:
    registry_id: str
    provider: str
    model_id: str
    # The registered spec as configured (streaming, structured output, recorded
    # request params such as <model env>_REASONING_EFFORT). Calling with a bare
    # spec dropped those: GLM then reasoned at its default effort, spent the whole
    # output budget thinking and returned nothing (pilot, 22 Oct 2026).
    spec: object = None


def _is_gemini(*values: Optional[str]) -> bool:
    return any("gemini" in str(v or "").lower() for v in values)


def resolve_route(env: Optional[dict] = None) -> Route:
    """The configured lesson-plan model, or LessonPlanRouteError."""
    env = os.environ if env is None else env
    ref = (env.get("LESSON_PLAN_AI_MODEL") or "").strip()
    if not ref:
        raise LessonPlanRouteError("LESSON_PLAN_AI_MODEL is not set; lesson plans need an explicitly configured model")
    if _is_gemini(ref):
        raise LessonPlanRouteError("Gemini is not allowed for lesson plans")
    from ai_services.core.routing.config import load_config
    from ai_services.core.routing.registry import resolve_model_ref

    spec = resolve_model_ref(load_config(env).models, ref)
    if spec is None:
        raise LessonPlanRouteError(f"LESSON_PLAN_AI_MODEL={ref!r} is not a registered model")
    if spec.provider not in ALLOWED_PROVIDERS or _is_gemini(spec.provider, spec.id, spec.provider_model_id):
        raise LessonPlanRouteError(f"Provider {spec.provider!r} is not allowed for lesson plans")
    if not spec.provider_model_id:
        raise LessonPlanRouteError(f"{spec.id} has no provider model id configured ({spec.model_env or 'model env'})")
    return Route(registry_id=spec.id, provider=spec.provider, model_id=spec.provider_model_id, spec=spec)


def _adapter(provider: str):
    from ai_services.core.routing.providers import default_adapters

    adapter = default_adapters().get(provider)
    if adapter is None or not adapter.is_configured():
        raise LessonPlanRouteError(f"Provider {provider!r} is not configured")
    return adapter


# ── pilot cost guard and ledger ──────────────────────────────────────────────

REPAIR_MARKER = "YOUR PREVIOUS REPLY WAS NOT USABLE"

def _guard_enabled(env) -> bool:
    return str(env.get("LESSON_PLAN_COST_GUARD", "")).strip().lower() in {"1", "true", "yes"}


def _guard_config(env) -> dict:
    keys = ["LESSON_PLAN_LEDGER_PATH", "LESSON_PLAN_CEILING_INR", "LESSON_PLAN_MAX_ATTEMPTS",
            "LESSON_PLAN_PRICE_INPUT_INR_PER_M", "LESSON_PLAN_PRICE_OUTPUT_INR_PER_M"]
    missing = [k for k in keys if not str(env.get(k, "")).strip()]
    if missing:
        raise LessonPlanBudgetError(f"Cost guard is on but not fully configured: {', '.join(missing)}")
    try:
        return {
            "ledger": env["LESSON_PLAN_LEDGER_PATH"],
            "ceiling": float(env["LESSON_PLAN_CEILING_INR"]),
            "max_attempts": int(env["LESSON_PLAN_MAX_ATTEMPTS"]),
            "price_in": float(env["LESSON_PLAN_PRICE_INPUT_INR_PER_M"]),
            "price_out": float(env["LESSON_PLAN_PRICE_OUTPUT_INR_PER_M"]),
            "worst_factor": float(env.get("LESSON_PLAN_OUTPUT_WORST_FACTOR") or 2),
            "max_per_case": int(env["LESSON_PLAN_MAX_ATTEMPTS_PER_CASE"])
            if str(env.get("LESSON_PLAN_MAX_ATTEMPTS_PER_CASE", "")).strip() else None,
        }
    except (TypeError, ValueError) as exc:
        raise LessonPlanBudgetError(f"Cost guard setting is not a number: {exc}") from exc


def init_ledger(path: str, *, note: str) -> None:
    """Create a new, empty pilot ledger (one header row). Refuses to touch an existing file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "init", "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note},
                            ensure_ascii=False) + "\n")


def read_ledger(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_guard_ledger(path: str) -> list:
    """The ledger for a guarded call: it must exist and be readable, or no call is made."""
    if not os.path.exists(path):
        raise LessonPlanBudgetError(
            f"Cost ledger not found at {path}; create it once with init_ledger() — the guard never starts a new count by itself")
    try:
        return read_ledger(path)
    except (OSError, ValueError) as exc:
        raise LessonPlanBudgetError(f"Cost ledger is unreadable ({type(exc).__name__}); no call made") from exc


def summarise(rows: list) -> dict:
    """Per call (by call_id, latest row wins): attempts made and money committed.

    A call's cost is its actual cost when the provider reported usage, otherwise its
    pre-call worst case (a started, failed or usage-less call still counts).
    `spent_inr` = confirmed + estimated; the two are also given apart, because an
    estimate is not money known to be spent.
    """
    calls: dict = {}
    for r in rows:
        if r.get("type") == "init":
            continue
        calls[r.get("call_id") or id(r)] = r
    confirmed = sum(r["cost_inr"] for r in calls.values() if r.get("cost_inr") is not None)
    estimated = sum(r.get("worst_case_inr", 0.0) for r in calls.values() if r.get("cost_inr") is None)
    by_label: dict = {}
    for r in calls.values():
        by_label[r.get("label") or ""] = by_label.get(r.get("label") or "", 0) + 1
    return {"attempts": len(calls), "spent_inr": confirmed + estimated, "confirmed_inr": confirmed,
            "estimated_inr": estimated, "by_label": by_label,
            "model_mismatch": any(r.get("model_mismatch") for r in calls.values()),
            "calls": list(calls.values())}


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_input_tokens(system_prompt: str, user_prompt: str) -> int:
    return int((len(system_prompt) + len(user_prompt)) / CHARS_PER_TOKEN_ESTIMATE) + 1


# ── the call ─────────────────────────────────────────────────────────────────

def complete(*, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float,
             institute_id: Optional[str], content_type: str, label: str = "",
             timeout_s: float = 200.0, env: Optional[dict] = None) -> dict:
    """One call to the configured lesson-plan model. No router, no fallback, no Gemini."""
    env = os.environ if env is None else env
    route = resolve_route(env)
    adapter = _adapter(route.provider)
    from ai_services.core.routing.providers import ProviderCall, REASONING_EFFORTS, scrub

    guard = _guard_config(env) if _guard_enabled(env) else None
    est_in = estimate_input_tokens(system_prompt, user_prompt)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": label, "content_type": content_type,
        "provider": route.provider, "model": route.model_id, "max_tokens": max_tokens,
        "estimated_input_tokens": est_in, "call_id": uuid.uuid4().hex,
    }
    with _LOCK:
        if guard:
            # Pilot control: a per-case attempt cap the backend cannot see. Its repair call
            # quotes the previous reply under this marker (lesson-plan.service.ts runGeneration).
            if (str(env.get("LESSON_PLAN_GUARD_REFUSE_REPAIRS", "")).strip().lower() in {"1", "true", "yes"}
                    and REPAIR_MARKER in user_prompt):
                logger.warning("Lesson plan repair call refused by the pilot guard (no provider call made)")
                raise LessonPlanBudgetError("Repair calls are disabled for this pilot; call refused")
            summary = summarise(_read_guard_ledger(guard["ledger"]))
            worst = (est_in * guard["price_in"] + max_tokens * guard["worst_factor"] * guard["price_out"]) / 1_000_000
            spent = summary["spent_inr"]
            if summary["model_mismatch"]:
                raise LessonPlanBudgetError("A previous call reported a different model than configured; pilot stopped")
            if summary["attempts"] >= guard["max_attempts"]:
                raise LessonPlanBudgetError(f"Attempt cap reached ({summary['attempts']}/{guard['max_attempts']}); no further calls")
            if guard["max_per_case"] is not None:
                if not label:
                    raise LessonPlanBudgetError("A per-case cap is set but this call has no case label; call refused")
                if summary["by_label"].get(label, 0) >= guard["max_per_case"]:
                    raise LessonPlanBudgetError(
                        f"Case cap reached for {label!r} ({summary['by_label'][label]}/{guard['max_per_case']}); call refused")
            if spent + worst > guard["ceiling"]:
                raise LessonPlanBudgetError(
                    f"Projected cost ₹{spent + worst:.2f} would exceed the ceiling ₹{guard['ceiling']:.2f}; call refused")
            row.update({"attempt": summary["attempts"] + 1, "worst_case_inr": round(worst, 4), "spent_before_inr": round(spent, 4)})
            # Reserve the worst case now, so a crash mid-call still counts against the ceiling.
            _append(guard["ledger"], {**row, "result": "started", "cost_inr": None, "cost_basis": "worst_case_estimate"})

    started = time.time()
    # A lesson plan is long structured output: keep reasoning short so the output
    # budget goes to the answer. LESSON_PLAN_REASONING_EFFORT overrides (none|low|medium|high).
    effort = (env.get("LESSON_PLAN_REASONING_EFFORT") or "low").strip().lower()
    try:
        result = adapter.complete(route.spec, ProviderCall(
            system_prompt=system_prompt, user_prompt=user_prompt, model_id=route.model_id,
            temperature=temperature, max_tokens=max_tokens, json_mode=False, institute_id=institute_id,
            timeout_s=timeout_s, legacy_prompt_shaping=False,
            reasoning_effort=effort if effort in REASONING_EFFORTS else None,
        ))
    except Exception as exc:
        kind = getattr(exc, "kind", None)
        detail = scrub(str(exc))[:300]
        if guard:
            with _LOCK:
                _append(guard["ledger"], {**row, "result": "error", "error": type(exc).__name__, "error_kind": kind,
                                          "error_detail": detail, "cost_basis": "worst_case_estimate",
                                          "latency_ms": int((time.time() - started) * 1000), "cost_inr": None})
        logger.warning("Lesson plan model call failed | provider=%s model=%s error=%s kind=%s detail=%s",
                       route.provider, route.model_id, type(exc).__name__, kind, detail)
        raise

    usage = result.get("usage") or {}
    tin = usage.get("prompt_tokens")
    tout = usage.get("completion_tokens")
    reported = isinstance(tin, int) and isinstance(tout, int) and (tin + tout) > 0
    # Only what the provider itself said; None when it did not say (never the configured id).
    reported_model = result.get("provider_reported_model")
    mismatch = bool(reported_model) and reported_model != route.model_id
    if mismatch:
        logger.error("Lesson plan provider reported model %s, configured %s", reported_model, route.model_id)
    if guard:
        cost = round((tin * guard["price_in"] + tout * guard["price_out"]) / 1_000_000, 6) if reported else None
        # The raw reply, kept locally for evaluation (model text only: no prompt, no secrets).
        raw_path = os.path.join(os.path.dirname(os.path.abspath(guard["ledger"])), "raw", f"{row['call_id']}.txt")
        try:
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as fh:
                fh.write(result.get("content") if isinstance(result.get("content"), str) else json.dumps(result.get("content")))
        except OSError:
            raw_path = None
        with _LOCK:
            _append(guard["ledger"], {**row, "result": "ok", "input_tokens": tin, "output_tokens": tout,
                                      "usage_reported": reported, "cost_inr": cost,
                                      "cost_basis": "actual" if reported else "worst_case_estimate",
                                      "hit_output_cap": isinstance(tout, int) and tout >= max_tokens,
                                      "latency_ms": int((time.time() - started) * 1000),
                                      "provider_reported_model": reported_model, "model_mismatch": mismatch,
                                      "raw_path": raw_path})
    logger.info("Lesson plan model call ok | provider=%s model=%s tokens_in=%s tokens_out=%s latency_ms=%d",
                route.provider, route.model_id, tin, tout, int((time.time() - started) * 1000))
    return {**result, "provider": route.provider, "model_id": route.model_id, "registry_id": route.registry_id}
