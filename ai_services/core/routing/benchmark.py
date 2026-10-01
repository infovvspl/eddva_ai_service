"""
Benchmark support: run a candidate model on a representative EDVA prompt
WITHOUT touching production routing.

A benchmark call goes straight to the candidate's adapter — no policy, no
fallback, no override, no tenant usage row. A candidate that cannot run (no
provider model id, no credentials, unverified JSON support) is reported
BLOCKED or FAIL with the reason, never skipped silently and never substituted.

This module measures. It does not rank. Quality has to be judged on the
outputs against EDVA's own rubric; latency and token counts alone must not be
read as "model X is better".
"""
from __future__ import annotations

import statistics
import time
from typing import Iterable, Optional

from ai_services.core.routing.errors import ProviderError
from ai_services.core.routing.providers import ProviderCall, scrub
from ai_services.core.routing.registry import primary_capability, resolve_model_ref

PASS, FAIL, BLOCKED, NOT_SUPPORTED = "PASS", "FAIL", "BLOCKED", "NOT_SUPPORTED"


def candidate_ids_for(router, selector: str) -> list[str]:
    """'all' | 'capability:<cap>' | comma-separated registry ids or provider:alias refs."""
    selector = (selector or "").strip()
    if selector == "all":
        return sorted(router.config.models)
    if selector.startswith("capability:"):
        cap = selector.split(":", 1)[1]
        policy = router.config.policies.get(cap)
        if policy is None:
            return []
        ordered = [policy.primary, *policy.fallbacks, *policy.candidates]
        return list(dict.fromkeys(ordered))
    ids = []
    for part in (x.strip() for x in selector.split(",") if x.strip()):
        spec = resolve_model_ref(router.config.models, part)
        ids.append(spec.id if spec is not None else part)  # unknown ids are reported BLOCKED
    return ids


def resolve_selector(router, provider: str, model: str) -> Optional[str]:
    """--provider together --model qwen3.8-flash (or a configured provider model id)."""
    spec = resolve_model_ref(router.config.models, model) or \
        resolve_model_ref(router.config.models, f"{provider}:{model}")
    if spec is None or spec.provider != provider:
        return None
    return spec.id


def benchmark_candidate(
    router,
    registry_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    repeat: int = 1,
    timeout_s: Optional[float] = 120.0,
    legacy_prompt_shaping: bool = True,
    include_output: bool = False,
    label: Optional[str] = None,
) -> list[dict]:
    base = {"registry_id": registry_id, "label": label}
    spec = router.config.models.get(registry_id)
    if spec is None:
        return [{**base, "status": BLOCKED, "reason": "unknown registry id"}]
    base.update(provider=spec.provider, model=spec.provider_model_id, capability=primary_capability(spec))
    if not spec.is_configured:
        hint = f"set {spec.model_env}" if spec.model_env else "no provider model id"
        return [{**base, "status": BLOCKED, "reason": f"model id not configured ({hint})"}]
    adapter = router.adapters.get(spec.provider)
    if adapter is None:
        return [{**base, "status": BLOCKED, "reason": f"no adapter for provider {spec.provider}"}]
    if not adapter.is_configured():
        return [{**base, "status": BLOCKED, "reason": f"{spec.provider} credentials not configured"}]

    records = []
    for run in range(1, max(1, repeat) + 1):
        call = ProviderCall(
            system_prompt=system_prompt, user_prompt=user_prompt, model_id=spec.provider_model_id,
            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            timeout_s=timeout_s, legacy_prompt_shaping=legacy_prompt_shaping,
        )
        t0 = time.perf_counter()
        rec = {**base, "run": run}
        try:
            result = adapter.complete(spec, call)
        except ProviderError as exc:
            rec.update(
                status=FAIL, kind=exc.kind, retryable=exc.retryable, status_code=exc.status_code,
                wall_ms=int((time.perf_counter() - t0) * 1000), error=scrub(exc)[:300],
            )
        except Exception as exc:  # report, never crash a sweep
            rec.update(
                status=FAIL, kind="unclassified", wall_ms=int((time.perf_counter() - t0) * 1000),
                error=scrub(f"{exc.__class__.__name__}: {exc}")[:300],
            )
        else:
            content = result.get("content")
            text = content if isinstance(content, str) else repr(content)
            usage = result.get("usage") or {}  # same normalisation the router applies
            rec.update(
                status=PASS,
                wall_ms=int((time.perf_counter() - t0) * 1000),
                provider_latency_ms=int(result.get("latency_ms") or 0),
                reported_model=result.get("model"),
                provider_reported_model=result.get("provider_reported_model"),
                tokens_input=int(result.get("tokens_input") or usage.get("prompt_tokens") or 0),
                tokens_output=int(result.get("tokens_output") or usage.get("completion_tokens") or 0),
                tokens_reported=bool(result.get("tokens_reported", True)),
                output_chars=len(text or ""),
                structured=isinstance(content, (dict, list)),
            )
            if include_output:
                rec["output"] = content
        records.append(rec)
    return records


def discover(router, provider: str) -> dict:
    """List the provider account's models and validate each configured registry
    id against that list. Never fills in ids — it only reports."""
    adapter = router.adapters.get(provider)
    if adapter is None or not hasattr(adapter, "list_model_records"):
        return {"status": NOT_SUPPORTED, "provider": provider}
    if not adapter.is_configured():
        return {"status": BLOCKED, "provider": provider, "reason": f"{provider} credentials not configured"}
    try:
        records = adapter.list_model_records()
    except ProviderError as exc:
        return {"status": FAIL, "provider": provider, "kind": exc.kind, "error": scrub(exc)[:300]}
    ids = {str(r["id"]) for r in records}
    validation = []
    for spec in sorted(router.config.models.values(), key=lambda s: s.id):
        if spec.provider != provider:
            continue
        validation.append({
            "registry_id": spec.id,
            "env": spec.model_env,
            "configured": spec.is_configured,
            "model": spec.provider_model_id,
            "present_in_account": (spec.provider_model_id in ids) if spec.is_configured else None,
        })
    return {"status": PASS, "provider": provider, "count": len(records), "models": records,
            "validation": validation}


def summarize(records: Iterable[dict]) -> list[dict]:
    """Per-candidate aggregates computed only from runs that actually happened."""
    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r["registry_id"], []).append(r)
    rows = []
    for rid, rs in by_id.items():
        passed = [r for r in rs if r["status"] == PASS]
        lat = [r["wall_ms"] for r in passed]
        reported = [r for r in passed if r.get("tokens_reported", True)]
        rows.append({
            "registry_id": rid,
            "provider": next((r.get("provider") for r in rs if r.get("provider")), None),
            "capability": next((r.get("capability") for r in rs if r.get("capability")), None),
            "runs": sum(1 for r in rs if r["status"] != BLOCKED),
            "pass": len(passed),
            "fail": sum(1 for r in rs if r["status"] == FAIL),
            "blocked": any(r["status"] == BLOCKED for r in rs),
            "blocked_reason": next((r.get("reason") for r in rs if r["status"] == BLOCKED), None),
            "median_wall_ms": int(statistics.median(lat)) if lat else None,
            "mean_tokens_output": int(statistics.mean(r["tokens_output"] for r in reported)) if reported else None,
        })
    return rows
