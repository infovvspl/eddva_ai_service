"""
Model router: feature -> capability -> policy -> primary/fallback -> execute -> telemetry.

The router decides *which* model runs and *whether* to fall back. It contains
no educational business logic: prompts, grounding, feature output parsing and
feature-specific degradation (e.g. "drop the textbook and retry ungrounded")
stay in the views that already implement them.

Compatibility guarantees (pinned by tests_model_router):

  * A call site that names a model ("pinned") gets exactly that model as
    primary, with the model string passed through unchanged. Every existing
    LLMClient.complete() call is pinned, so default routing executes the same
    Groq call as before the router existed.
  * Cross-provider fallback is OFF unless AI_ROUTER_FALLBACK_ENABLED=true AND
    the fallback's provider credentials AND provider model id are configured.
  * Fallback happens only on retryable errors (429 / 5xx / timeout / network /
    exhausted rotation). Deterministic, validation, structured-output and
    configuration errors raise immediately.
  * When only one attempt ran, the original exception object is re-raised, so
    callers see exactly what they saw before.
  * One logical request stays one usage row. The router annotates the result
    with provider/fallback metadata and never logs usage itself; individual
    attempts are observable as provider events (failover / 429 / 5xx / timeout)
    and log lines carrying request_id — never prompts, never credentials.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

from ai_services.core.routing.config import RoutePolicy, RouterConfig
from ai_services.core.routing.errors import (
    FallbackExhaustedError,
    NoRouteError,
    ProviderConfigError,
    ProviderError,
)
from ai_services.core.routing.providers import ProviderCall, default_adapters, scrub
from ai_services.core.routing.registry import (
    STRUCTURED_NATIVE,
    ModelSpec,
    find_by_provider_model,
    unmet_requirements,
)

logger = logging.getLogger("ai_services.routing")

# When a pinned model must be mapped to a capability, prefer the everyday
# classes; "premium" is last so nothing lands there by inference.
_CAPABILITY_PREFERENCE = ("reasoning", "lightweight", "content", "bulk_text", "grounded", "vision", "premium")


@dataclass
class AIRequest:
    system_prompt: str
    user_prompt: str
    feature: Optional[str] = None
    capability: Optional[str] = None
    # Provider model id named by a legacy call site. Pinned = used as primary.
    model: Optional[str] = None
    provider: str = "groq"
    # Metadata only: admission control is enforced upstream by NestJS.
    priority: Optional[str] = None
    language: Optional[str] = None
    requires_vision: bool = False
    requires_long_context: bool = False
    min_context_tokens: Optional[int] = None
    requires_grounding: bool = False
    # None = derive from json_mode.
    requires_structured_output: Optional[bool] = None
    temperature: float = 0.7
    max_tokens: int = 3500
    json_mode: bool = True
    json_mode_suffix: Optional[str] = None
    institute_id: Optional[str] = None
    legacy_prompt_shaping: bool = True

    @property
    def effective_structured_output(self) -> bool:
        if self.requires_structured_output is not None:
            return bool(self.requires_structured_output)
        return bool(self.json_mode)


@dataclass(frozen=True)
class Candidate:
    spec: ModelSpec
    model_id: str
    role: str     # primary | fallback
    source: str   # pinned | policy | feature_override


@dataclass(frozen=True)
class RoutePlan:
    capability: str
    policy: RoutePolicy
    candidates: tuple
    skipped: tuple  # ((registry_id, reason), ...)


@dataclass
class Attempt:
    registry_id: str
    provider: str
    model: str
    role: str
    outcome: str  # success | error
    kind: Optional[str] = None
    retryable: Optional[bool] = None
    status_code: Optional[int] = None
    latency_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "registry_id": self.registry_id,
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "outcome": self.outcome,
            "kind": self.kind,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
        }


def _ctx(name: str):
    try:
        from ai_services.core import request_context

        return request_context.get(name)
    except Exception:
        return None


class ModelRouter:
    def __init__(self, config: RouterConfig, adapters: Optional[dict] = None):
        self.config = config
        self.adapters = adapters if adapters is not None else default_adapters()

    # ── planning ─────────────────────────────────────────────────────────────
    def _canonical_model(self, provider: str, model: str) -> str:
        """The id that will actually execute. Groq call sites use aliases
        ("quiz", "reasoning") that LLMClient resolves; lookups must see the
        resolved id, while the original string is still what gets sent."""
        if provider == "groq":
            try:
                from ai_services.core.llm_client import _resolve_model

                return _resolve_model(model)
            except Exception:
                return model
        return model

    def _pinned_spec(self, req: AIRequest) -> ModelSpec:
        canonical = self._canonical_model(req.provider, req.model)
        spec = find_by_provider_model(dict(self.config.models), req.provider, canonical)
        if spec is not None:
            return spec
        # Not in the registry: still honoured exactly — the call site chose it.
        return ModelSpec(
            id=f"{req.provider}/{canonical}",
            provider=req.provider,
            provider_model_id=canonical,
            capabilities=frozenset(),
            structured_output=STRUCTURED_NATIVE,
            metadata_source="pinned",
        )

    def _resolve_capability(self, req: AIRequest, pinned: Optional[ModelSpec]) -> str:
        if req.capability:
            if req.capability not in self.config.policies:
                raise NoRouteError(f"Unknown AI capability {req.capability!r}")
            return req.capability
        if req.feature and req.feature in self.config.features:
            return self.config.features[req.feature].capability
        if pinned is not None:
            for cap in _CAPABILITY_PREFERENCE:
                if cap in pinned.capabilities and cap in self.config.policies:
                    return cap
        return "reasoning"

    def _require_configured(self, spec: ModelSpec, role: str) -> None:
        if not spec.is_configured:
            hint = f" (set {spec.model_env})" if spec.model_env else ""
            raise ProviderConfigError(
                f"{role} model {spec.id} has no provider model id configured{hint}",
                provider=spec.provider, model=None,
            )
        if spec.provider not in self.adapters:
            raise ProviderConfigError(
                f"{role} model {spec.id} uses provider {spec.provider!r} with no adapter",
                provider=spec.provider, model=spec.provider_model_id,
            )

    def plan(self, req: AIRequest) -> RoutePlan:
        pinned = self._pinned_spec(req) if req.model else None
        capability = self._resolve_capability(req, pinned)
        policy = self.config.policies[capability]
        feature_route = self.config.features.get(req.feature) if req.feature else None

        # Primary: explicit operator override > call-site pin > policy default.
        if feature_route is not None and feature_route.primary:
            spec = self.config.models[feature_route.primary]
            self._require_configured(spec, "primary")
            primary = Candidate(spec, spec.provider_model_id, "primary", "feature_override")
        elif pinned is not None:
            if pinned.provider not in self.adapters:
                raise ProviderConfigError(
                    f"pinned provider {pinned.provider!r} has no adapter", provider=pinned.provider, model=req.model,
                )
            # The original string, untouched: identical to the pre-router call.
            primary = Candidate(pinned, req.model, "primary", "pinned")
        else:
            spec = self.config.models[policy.primary]
            self._require_configured(spec, "primary")
            primary = Candidate(spec, spec.provider_model_id, "primary", "policy")

        # A pin is the call site's existing, working choice and is not second-
        # guessed. A model the router picked must actually meet the request.
        if primary.source != "pinned":
            unmet = unmet_requirements(primary.spec, req)
            if unmet:
                raise NoRouteError(
                    f"primary {primary.spec.id} for capability {capability!r} does not satisfy {unmet}"
                )

        candidates = [primary]
        skipped = []
        if feature_route is not None and feature_route.fallbacks is not None:
            fallback_ids, fb_source = feature_route.fallbacks, "feature_override"
        else:
            fallback_ids, fb_source = policy.fallbacks, "policy"

        seen = {(primary.spec.provider, self._canonical_model(primary.spec.provider, primary.model_id or ""))}
        for fid in fallback_ids:
            if not self.config.fallback_enabled:
                skipped.append((fid, "fallback_disabled"))
                continue
            spec = self.config.models.get(fid)
            if spec is None:
                skipped.append((fid, "unknown_model"))
                continue
            if not spec.is_configured:
                skipped.append((fid, "model_id_not_configured"))
                continue
            adapter = self.adapters.get(spec.provider)
            if adapter is None:
                skipped.append((fid, "no_adapter"))
                continue
            if not adapter.is_configured():
                skipped.append((fid, "provider_not_configured"))
                continue
            unmet = unmet_requirements(spec, req)
            if unmet:
                skipped.append((fid, "unmet:" + ",".join(unmet)))
                continue
            key = (spec.provider, spec.provider_model_id)
            if key in seen:
                skipped.append((fid, "duplicate"))
                continue
            seen.add(key)
            candidates.append(Candidate(spec, spec.provider_model_id, "fallback", fb_source))

        return RoutePlan(capability=capability, policy=policy, candidates=tuple(candidates), skipped=tuple(skipped))

    # ── execution ────────────────────────────────────────────────────────────
    def _log_attempt(self, req: AIRequest, plan: RoutePlan, a: Attempt) -> None:
        # Plain primary success is the overwhelmingly common case and the
        # provider already logs it; keep it at DEBUG. Anything the router had
        # to do — an error, a fallback — is INFO/WARNING so it is visible.
        level = logging.DEBUG if (a.outcome == "success" and a.role == "primary") else (
            logging.INFO if a.outcome == "success" else logging.WARNING
        )
        logger.log(
            level,
            "AI route attempt | request_id=%s institute=%s feature=%s capability=%s role=%s "
            "provider=%s model=%s outcome=%s kind=%s status=%s latency_ms=%d",
            _ctx("request_id") or "-", _ctx("institute_id") or req.institute_id or "-",
            req.feature or "-", plan.capability, a.role, a.provider, a.model, a.outcome,
            a.kind or "-", a.status_code if a.status_code is not None else "-", a.latency_ms,
        )

    def _emit_failover(self, req: AIRequest, prev: Attempt, attempt_number: int) -> None:
        try:
            from ai_services.core import provider_events

            provider_events.emit(
                event_type="failover", provider=prev.provider, model=prev.model,
                status_code=prev.status_code, attempt_number=attempt_number, feature=req.feature,
                institute_id=req.institute_id if req.institute_id not in (None, "default") else None,
            )
        except Exception:
            pass

    def execute(self, req: AIRequest) -> dict:
        plan = self.plan(req)
        started = time.monotonic()
        attempts: list[Attempt] = []
        last_exc: Optional[BaseException] = None

        for idx, cand in enumerate(plan.candidates):
            if idx > 0:
                elapsed = time.monotonic() - started
                if elapsed > plan.policy.fallback_deadline_s:
                    logger.warning(
                        "AI route fallback skipped | request_id=%s feature=%s capability=%s "
                        "elapsed_s=%.1f deadline_s=%.1f next=%s",
                        _ctx("request_id") or "-", req.feature or "-", plan.capability,
                        elapsed, plan.policy.fallback_deadline_s, cand.spec.id,
                    )
                    break
                self._emit_failover(req, attempts[-1], idx + 1)

            adapter = self.adapters[cand.spec.provider]
            call = ProviderCall(
                system_prompt=req.system_prompt,
                user_prompt=req.user_prompt,
                model_id=cand.model_id,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                json_mode=req.json_mode,
                json_mode_suffix=req.json_mode_suffix,
                institute_id=req.institute_id,
                timeout_s=plan.policy.timeout_s,
                legacy_prompt_shaping=req.legacy_prompt_shaping,
            )
            t0 = time.perf_counter()
            try:
                result = adapter.complete(cand.spec, call)
            except ProviderError as exc:
                a = Attempt(
                    cand.spec.id, cand.spec.provider, cand.model_id or "-", cand.role, "error",
                    kind=exc.kind, retryable=exc.retryable, status_code=exc.status_code,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
                attempts.append(a)
                self._log_attempt(req, plan, a)
                last_exc = exc
                if not exc.retryable:
                    raise
                continue
            except Exception as exc:
                # Unclassified (import error, programming error): never mask it
                # behind another provider.
                a = Attempt(
                    cand.spec.id, cand.spec.provider, cand.model_id or "-", cand.role, "error",
                    kind="unclassified", retryable=False,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
                attempts.append(a)
                self._log_attempt(req, plan, a)
                raise

            a = Attempt(
                cand.spec.id, cand.spec.provider, str(result.get("model") or cand.model_id), cand.role,
                "success", latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            attempts.append(a)
            self._log_attempt(req, plan, a)
            return self._finalize(result, cand, plan, attempts, started)

        if len(attempts) <= 1:
            raise last_exc  # the original exception, exactly as before the router
        summary = "; ".join(f"{a.registry_id} {a.kind}" for a in attempts)
        last = attempts[-1]
        raise FallbackExhaustedError(
            f"All routed providers failed for capability {plan.capability!r} ({summary}). "
            f"Last error: {scrub(last_exc)[:300]}",
            attempts=[x.as_dict() for x in attempts],
            provider=last.provider, model=last.model, status_code=last.status_code,
        ) from last_exc

    def _finalize(self, result: dict, cand: Candidate, plan: RoutePlan, attempts: list, started: float) -> dict:
        out = dict(result)
        if "usage" not in out:
            ti, to = int(out.get("tokens_input") or 0), int(out.get("tokens_output") or 0)
            out["usage"] = {"prompt_tokens": ti, "completion_tokens": to, "total_tokens": ti + to}
        out.setdefault("tokens_input", out["usage"].get("prompt_tokens", 0))
        out.setdefault("tokens_output", out["usage"].get("completion_tokens", 0))
        out["provider"] = cand.spec.provider
        out["registry_id"] = cand.spec.id
        out["capability"] = plan.capability
        out["fallback_used"] = cand.role == "fallback"
        out["route_attempts"] = [a.as_dict() for a in attempts]
        out["route_latency_ms"] = int((time.monotonic() - started) * 1000)
        return out

    def with_config(self, **changes) -> "ModelRouter":
        """A copy with config fields replaced — for benchmarks and tests."""
        return ModelRouter(replace(self.config, **changes), self.adapters)
