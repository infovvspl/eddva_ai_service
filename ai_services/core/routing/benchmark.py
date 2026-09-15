"""
Benchmark support: run a candidate model on a representative EDVA prompt
WITHOUT touching production routing.

A benchmark call goes straight to the candidate's adapter — no policy, no
fallback, no tenant usage row. A candidate that cannot run (no provider model
id, no credentials) is reported BLOCKED with the reason, never skipped
silently and never substituted.

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

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"


def candidate_ids_for(router, selector: str) -> list[str]:
    """'all' | 'capability:<cap>' | comma-separated registry ids."""
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
    return [x.strip() for x in selector.split(",") if x.strip()]


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
    base.update(provider=spec.provider, model=spec.provider_model_id)
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
            rec.update(
                status=PASS,
                wall_ms=int((time.perf_counter() - t0) * 1000),
                provider_latency_ms=int(result.get("latency_ms") or 0),
                reported_model=result.get("model"),
                tokens_input=int(result.get("tokens_input") or 0),
                tokens_output=int(result.get("tokens_output") or 0),
                output_chars=len(text or ""),
                structured=isinstance(content, (dict, list)),
            )
            if include_output:
                rec["output"] = content
        records.append(rec)
    return records


def summarize(records: Iterable[dict]) -> list[dict]:
    """Per-candidate aggregates computed only from runs that actually happened."""
    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r["registry_id"], []).append(r)
    rows = []
    for rid, rs in by_id.items():
        passed = [r for r in rs if r["status"] == PASS]
        lat = [r["wall_ms"] for r in passed]
        rows.append({
            "registry_id": rid,
            "runs": sum(1 for r in rs if r["status"] != BLOCKED),
            "pass": len(passed),
            "fail": sum(1 for r in rs if r["status"] == FAIL),
            "blocked": any(r["status"] == BLOCKED for r in rs),
            "blocked_reason": next((r.get("reason") for r in rs if r["status"] == BLOCKED), None),
            "median_wall_ms": int(statistics.median(lat)) if lat else None,
            "mean_tokens_output": int(statistics.mean(r["tokens_output"] for r in passed)) if passed else None,
        })
    return rows
