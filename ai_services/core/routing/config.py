"""
Router configuration: built-in defaults, then a JSON document, then targeted
environment overrides. Switching a model, provider, fallback or policy is a
configuration change, never a code change.

    AI_ROUTER_ENABLED            "false" bypasses the router entirely (kill switch)
    AI_ROUTER_FALLBACK_ENABLED   "true" allows cross-provider fallback (default OFF)
    AI_ROUTING_CONFIG            inline JSON document (see below)
    AI_ROUTING_CONFIG_FILE       path to the same JSON document
    AI_ROUTE_<CAPABILITY>_PRIMARY
    AI_ROUTE_<CAPABILITY>_FALLBACKS          comma-separated registry ids
    AI_ROUTE_<CAPABILITY>_TIMEOUT_S
    AI_ROUTE_<CAPABILITY>_FALLBACK_DEADLINE_S

  Together candidates (ids are NEVER built in):
    TOGETHER_API_KEY
    TOGETHER_MODEL_GPT_OSS_120B | _QWEN38_FLASH | _GLM53_FLASH | _DEEPSEEK_V4_FLASH | _QWEN37_MAX
    <model env>_STRUCTURED_OUTPUT   native | prompted | none | unknown
    <model env>_LONG_CONTEXT        true | false | unknown
    <model env>_MULTIMODAL          true | false | unknown
    <model env>_GROUNDING           true | false | unknown
    <model env>_CONTEXT_TOKENS      positive integer | unknown
    <model env>_STREAMING           true (model only accepts streaming) | false | unknown

  Local-development model override (server-side only, OFF by default):
    AI_MODEL_OVERRIDE_ENABLED       must be "true"
    AI_MODEL_OVERRIDE_<CAPABILITY>  e.g. together:qwen3.8-flash
    AI_MODEL_OVERRIDE_ALL           applies to every capability without its own override
  Honoured only when Django DEBUG is true. Refused (with an error) otherwise.

JSON document::

    {
      "models":   {"together/qwen3.8-flash": {"provider_model_id": "...", "long_context": true}},
      "policies": {"content": {"primary": "...", "fallbacks": ["..."], "timeout_s": 90}},
      "features": {"content_generate": {"capability": "content", "primary": "...", "fallbacks": []}}
    }

Loading never raises. An invalid override is dropped with a warning and the
default stays in force: a typo in routing config must not take the AI service
down at boot. Credentials are never read from this document — only from the
provider's own environment variables.

DEFAULTS REPRODUCE CURRENT PRODUCTION ROUTING. Every primary below is the model
the corresponding call site already uses; the intended candidates (Qwen, GLM,
DeepSeek, Gemini-as-content) are recorded as ``candidates`` for benchmarking and
are never executed by routing until an operator promotes them in config.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

from ai_services.core.routing.registry import (
    CAPABILITIES,
    KNOWN_PROVIDERS,
    STRUCTURED_VALUES,
    ModelSpec,
    default_registry,
    resolve_model_ref,
)

logger = logging.getLogger("ai_services.routing")

_TRUE = {"1", "true", "yes", "on"}
_SECRET_LIKE = ("key", "secret", "token", "password", "credential", "bearer")

OVERRIDE_ENABLE_ENV = "AI_MODEL_OVERRIDE_ENABLED"
OVERRIDE_PREFIX = "AI_MODEL_OVERRIDE_"
OVERRIDE_ALL = "__all__"


@dataclass(frozen=True)
class RoutePolicy:
    capability: str
    primary: str
    fallbacks: tuple = ()
    # Benchmark targets. Recorded so the intended strategy is visible in one
    # place; routing NEVER executes these.
    candidates: tuple = ()
    # Per-attempt timeout for adapters that accept one (Together). Groq and
    # Gemini keep their existing, separately tuned timeouts.
    timeout_s: Optional[float] = None
    # If the primary has already consumed this much wall clock, skip fallback:
    # the NestJS caller has most likely timed out and a second provider call
    # would be paid for and thrown away.
    fallback_deadline_s: float = 90.0
    description: str = ""


@dataclass(frozen=True)
class FeatureRoute:
    capability: str
    primary: Optional[str] = None
    fallbacks: Optional[tuple] = None


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool
    fallback_enabled: bool
    models: Mapping[str, ModelSpec]
    policies: Mapping[str, RoutePolicy]
    features: Mapping[str, FeatureRoute]
    warnings: tuple = field(default=())
    # capability (or OVERRIDE_ALL) -> registry id. Empty unless the override is
    # explicitly enabled AND Django DEBUG is true.
    overrides: Mapping[str, str] = field(default_factory=dict)
    debug: bool = False


def default_policies() -> dict[str, RoutePolicy]:
    """TOGETHER-FIRST. Every text capability names a Together model as primary.

    The previous production models are kept as fallbacks so cross-provider
    failover can be switched on later; fallback itself stays OFF by default.
    Where a Together primary is not configured, or cannot meet a request's
    requirements, the router uses the call site's existing model instead (see
    router.py, source=legacy) — so an environment without Together configuration
    keeps its current behaviour exactly.
    """
    return {
        "reasoning": RoutePolicy(
            capability="reasoning",
            primary="together/gpt-oss-120b",
            fallbacks=("groq/gpt-oss-120b",),
            timeout_s=60.0,
            description="Doubts, tutor, quiz, assessments, grading, teacher analysis.",
        ),
        "lightweight": RoutePolicy(
            capability="lightweight",
            primary="together/deepseek-v4-flash",
            fallbacks=("groq/gpt-oss-20b",),
            timeout_s=60.0,
            description="Classification, extraction, transcript cleanup, metadata.",
        ),
        "content": RoutePolicy(
            capability="content",
            primary="together/qwen3.8-flash",
            fallbacks=("groq/gpt-oss-120b",),
            candidates=("together/glm-5.3-flash", "gemini/gemini-2.5-flash"),
            timeout_s=120.0,
            fallback_deadline_s=120.0,
            description="Textbook/chapter content, lecture notes, ungrounded PPT.",
        ),
        "premium": RoutePolicy(
            capability="premium",
            primary="together/qwen3.7-max",
            fallbacks=("together/gpt-oss-120b",),
            timeout_s=120.0,
            description="Opt-in only. No feature routes here by default.",
        ),
        "bulk_text": RoutePolicy(
            capability="bulk_text",
            primary="together/deepseek-v4-flash",
            fallbacks=("groq/gpt-oss-20b",),
            timeout_s=60.0,
            description="High-volume background text.",
        ),
        "grounded": RoutePolicy(
            capability="grounded",
            # GLM-5.3 Flash rather than Qwen3.8 Flash: grounded generation is
            # faithful reproduction of supplied passages, and Gemini's grounded
            # calls disable thinking because reasoning tokens truncated question
            # papers. In local probes Qwen3.8 Flash spent ~1,250 output tokens
            # on a two-sentence answer; GLM-5.3 Flash spent ~90. Switchable with
            # AI_ROUTE_GROUNDED_PRIMARY.
            primary="together/glm-5.3-flash",
            fallbacks=("gemini/gemini-2.5-flash",),
            candidates=("together/qwen3.8-flash",),
            timeout_s=200.0,
            fallback_deadline_s=200.0,
            description="Textbook-grounded content and PPT. Retrieval, citations and the "
                        "ungrounded fallback stay in EDVA code.",
        ),
        "vision": RoutePolicy(
            capability="vision",
            primary="gemini/gemini-2.5-flash",
            fallbacks=(),
            timeout_s=60.0,
            description="Not routed: vision/OCR call sites remain specialized (see routing/catalog.py).",
        ),
    }


def default_features() -> dict[str, FeatureRoute]:
    """Feature -> capability, derived from the existing tier map so the router
    and model_tier can never disagree about what a feature is."""
    from ai_services.core.model_tier import FEATURE_TIER_MAP, ModelTier

    features = {
        name: FeatureRoute(capability="lightweight" if tier == ModelTier.FAST else "reasoning")
        for name, tier in FEATURE_TIER_MAP.items()
    }
    # Features that call LLMClient directly with a feature label the tier map
    # does not know. Capability only — primaries stay pinned by the call site.
    features.setdefault("doubt_resolver", FeatureRoute(capability="reasoning"))
    features.setdefault("content_generate", FeatureRoute(capability="content"))
    features.setdefault("ppt_generate", FeatureRoute(capability="content"))
    features.setdefault("ai_lecture_notes", FeatureRoute(capability="content"))
    return features


def _truthy(value: Optional[str], default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in _TRUE


def _detect_debug(env: Mapping[str, str]) -> bool:
    """Django's DEBUG when settings are loaded (the authoritative value), else
    the same DJANGO_DEBUG env var settings.py reads."""
    try:
        from django.conf import settings

        if settings.configured:
            return bool(settings.DEBUG)
    except Exception:
        pass
    return str(env.get("DJANGO_DEBUG", "false")).strip().lower() in ("true", "1", "yes")


def _load_json_document(env: Mapping[str, str], warnings: list) -> dict:
    raw = (env.get("AI_ROUTING_CONFIG") or "").strip()
    source = "AI_ROUTING_CONFIG"
    path = (env.get("AI_ROUTING_CONFIG_FILE") or "").strip()
    if not raw and path:
        source = f"AI_ROUTING_CONFIG_FILE={path}"
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            warnings.append(f"{source}: unreadable ({exc.__class__.__name__}); using defaults")
            return {}
    if not raw:
        return {}
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        warnings.append(f"{source}: invalid JSON ({exc.msg} at line {exc.lineno}); using defaults")
        return {}
    if not isinstance(doc, dict):
        warnings.append(f"{source}: top level must be an object; using defaults")
        return {}
    return doc


def _contains_secret_like_field(obj) -> Optional[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            # provider_model_id / model_env are legitimate; anything else that
            # looks like a credential field is refused.
            if kl not in ("provider_model_id", "model_env") and any(s in kl for s in _SECRET_LIKE):
                return str(k)
            found = _contains_secret_like_field(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _contains_secret_like_field(v)
            if found:
                return found
    return None


# Tri-state support flags: true / false / null (UNKNOWN).
_MODEL_FLAG_FIELDS = ("multimodal", "long_context", "supports_grounding", "streaming")
_MODEL_STR_FIELDS = ("quality_tier", "cost_tier", "notes", "model_env")


def _apply_model_overrides(models: dict, doc_models, warnings: list) -> None:
    if not isinstance(doc_models, dict):
        if doc_models is not None:
            warnings.append("models: must be an object; ignored")
        return
    for model_id, fields in doc_models.items():
        if not isinstance(fields, dict):
            warnings.append(f"models.{model_id}: must be an object; ignored")
            continue
        base = models.get(model_id)
        if base is None:
            provider = fields.get("provider")
            caps = fields.get("capabilities") or []
            if provider not in KNOWN_PROVIDERS:
                warnings.append(f"models.{model_id}: new model needs a known provider {sorted(KNOWN_PROVIDERS)}; ignored")
                continue
            bad_caps = set(caps) - CAPABILITIES
            if not caps or bad_caps:
                warnings.append(f"models.{model_id}: new model needs valid capabilities; ignored")
                continue
            base = ModelSpec(id=model_id, provider=provider, provider_model_id=None,
                             capabilities=frozenset(caps))
        updates = {}
        if "provider" in fields and fields["provider"] != base.provider:
            warnings.append(f"models.{model_id}: provider cannot be changed on an existing id; ignored field")
        if "provider_model_id" in fields:
            updates["provider_model_id"] = (str(fields["provider_model_id"]).strip() or None) \
                if fields["provider_model_id"] is not None else None
        if "capabilities" in fields:
            caps = set(fields["capabilities"] or [])
            if caps and not (caps - CAPABILITIES):
                updates["capabilities"] = frozenset(caps)
            else:
                warnings.append(f"models.{model_id}.capabilities: invalid; ignored field")
        for f in _MODEL_FLAG_FIELDS:
            if f in fields:
                if fields[f] is None or isinstance(fields[f], bool):
                    updates[f] = fields[f]
                else:
                    warnings.append(f"models.{model_id}.{f}: must be true, false or null (UNKNOWN); ignored field")
        if "context_tokens" in fields:
            v = fields["context_tokens"]
            if v is None or (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                updates["context_tokens"] = v
            else:
                warnings.append(f"models.{model_id}.context_tokens: must be a positive integer or null; ignored field")
        if "structured_output" in fields:
            if fields["structured_output"] in STRUCTURED_VALUES:
                updates["structured_output"] = fields["structured_output"]
            else:
                warnings.append(f"models.{model_id}.structured_output: must be one of {sorted(STRUCTURED_VALUES)}; ignored field")
        for f in _MODEL_STR_FIELDS:
            if f in fields and isinstance(fields[f], str):
                updates[f] = fields[f]
        if updates:
            updates["metadata_source"] = "config"
        models[model_id] = replace(base, **updates)


_META_ENV_SUFFIXES = {
    "STRUCTURED_OUTPUT": "structured_output",
    "LONG_CONTEXT": "long_context",
    "MULTIMODAL": "multimodal",
    "GROUNDING": "supports_grounding",
    "STREAMING": "streaming",
    "CONTEXT_TOKENS": "context_tokens",
}


def _apply_together_env_metadata(models: dict, env: Mapping[str, str], warnings: list) -> None:
    """<model env>_<FIELD> lets an operator record what they have verified about
    a Together model without writing a JSON document. Together only: Groq and
    Gemini metadata is repository-verified and not overridable this way."""
    for model_id, spec in list(models.items()):
        if spec.provider != "together" or not spec.model_env:
            continue
        updates = {}
        for suffix, fname in _META_ENV_SUFFIXES.items():
            name = f"{spec.model_env}_{suffix}"
            raw = (env.get(name) or "").strip()
            if not raw:
                continue
            low = raw.lower()
            if fname == "structured_output":
                if low in STRUCTURED_VALUES:
                    updates[fname] = low
                else:
                    warnings.append(f"{name}: must be one of {sorted(STRUCTURED_VALUES)}; ignored")
            elif fname == "context_tokens":
                if low == "unknown":
                    updates[fname] = None
                elif low.isdigit() and int(low) > 0:
                    updates[fname] = int(low)
                else:
                    warnings.append(f"{name}: must be a positive integer or 'unknown'; ignored")
            else:
                if low in ("true", "yes", "1"):
                    updates[fname] = True
                elif low in ("false", "no", "0"):
                    updates[fname] = False
                elif low == "unknown":
                    updates[fname] = None
                else:
                    warnings.append(f"{name}: must be true, false or unknown; ignored")
        if updates:
            models[model_id] = replace(spec, metadata_source="config", **updates)


def _apply_policy_fields(policy: RoutePolicy, fields: dict, models: dict, where: str, warnings: list) -> RoutePolicy:
    updates = {}
    if "primary" in fields:
        if fields["primary"] in models:
            updates["primary"] = fields["primary"]
        else:
            warnings.append(f"{where}.primary: unknown model {fields['primary']!r}; ignored field")
    if "fallbacks" in fields:
        fb = fields["fallbacks"]
        if isinstance(fb, str):
            fb = [x.strip() for x in fb.split(",") if x.strip()]
        if isinstance(fb, (list, tuple)):
            unknown = [x for x in fb if x not in models]
            if unknown:
                warnings.append(f"{where}.fallbacks: unknown models {unknown}; ignored field")
            else:
                updates["fallbacks"] = tuple(fb)
        else:
            warnings.append(f"{where}.fallbacks: must be a list; ignored field")
    for f in ("timeout_s", "fallback_deadline_s"):
        if f in fields:
            try:
                val = float(fields[f])
                if val <= 0:
                    raise ValueError
                updates[f] = val
            except (TypeError, ValueError):
                warnings.append(f"{where}.{f}: must be a positive number; ignored field")
    return replace(policy, **updates) if updates else policy


def _load_overrides(env: Mapping[str, str], models: dict, policies: dict, debug: bool, warnings: list) -> dict:
    """Local-development model override. Server-side configuration only: nothing
    here reads a request, so no client can select a provider or model."""
    requested = {}
    for cap in policies:
        name = f"{OVERRIDE_PREFIX}{cap.upper()}"
        if (env.get(name) or "").strip():
            requested[cap] = (name, env[name].strip())
    if (env.get(f"{OVERRIDE_PREFIX}ALL") or "").strip():
        requested[OVERRIDE_ALL] = (f"{OVERRIDE_PREFIX}ALL", env[f"{OVERRIDE_PREFIX}ALL"].strip())
    if not requested:
        return {}
    names = sorted(n for n, _ in requested.values())
    if not _truthy(env.get(OVERRIDE_ENABLE_ENV), False):
        warnings.append(f"model override variables {names} are set but {OVERRIDE_ENABLE_ENV} is not true; ignored")
        return {}
    if not debug:
        warnings.append(
            f"REFUSED model overrides {names}: local-development only and Django DEBUG is not true. "
            "Production routing is unchanged."
        )
        return {}
    active = {}
    for cap, (name, value) in requested.items():
        spec = resolve_model_ref(models, value)
        if spec is None:
            warnings.append(f"{name}={value!r}: not a registered model (use provider:alias, e.g. together:qwen3.8-flash); ignored")
            continue
        if not spec.is_configured:
            hint = f" — set {spec.model_env}" if spec.model_env else ""
            warnings.append(f"{name}: {spec.id} has no provider model id configured{hint}; ignored")
            continue
        active[cap] = spec.id
    return active


def load_config(env: Optional[Mapping[str, str]] = None, *, debug: Optional[bool] = None) -> RouterConfig:
    env = os.environ if env is None else env
    debug = _detect_debug(env) if debug is None else bool(debug)
    warnings: list[str] = []

    models = default_registry()
    # 1. Provider model ids from each model's own env var.
    for model_id, spec in list(models.items()):
        if spec.model_env:
            val = (env.get(spec.model_env) or "").strip()
            if val:
                models[model_id] = spec.with_model_id(val)
    _apply_together_env_metadata(models, env, warnings)

    policies = default_policies()
    features = default_features()

    # 2. JSON document.
    doc = _load_json_document(env, warnings)
    if doc:
        secret_field = _contains_secret_like_field(doc)
        if secret_field:
            warnings.append(
                f"routing config contains credential-like field {secret_field!r}; entire document "
                "ignored. Credentials belong in provider environment variables only."
            )
            doc = {}
    if doc:
        _apply_model_overrides(models, doc.get("models"), warnings)
        doc_policies = doc.get("policies") or {}
        if isinstance(doc_policies, dict):
            for cap, fields in doc_policies.items():
                if cap not in policies or not isinstance(fields, dict):
                    warnings.append(f"policies.{cap}: unknown capability or not an object; ignored")
                    continue
                policies[cap] = _apply_policy_fields(policies[cap], fields, models, f"policies.{cap}", warnings)
        doc_features = doc.get("features") or {}
        if isinstance(doc_features, dict):
            for name, fields in doc_features.items():
                if not isinstance(fields, dict):
                    warnings.append(f"features.{name}: must be an object; ignored")
                    continue
                cap = fields.get("capability") or (features[name].capability if name in features else None)
                if cap not in policies:
                    warnings.append(f"features.{name}.capability: unknown capability {cap!r}; ignored")
                    continue
                primary = fields.get("primary")
                if primary is not None and primary not in models:
                    warnings.append(f"features.{name}.primary: unknown model {primary!r}; ignored")
                    continue
                fb = fields.get("fallbacks")
                if fb is not None:
                    if not isinstance(fb, list) or any(x not in models for x in fb):
                        warnings.append(f"features.{name}.fallbacks: unknown models; ignored")
                        continue
                    fb = tuple(fb)
                features[name] = FeatureRoute(capability=cap, primary=primary, fallbacks=fb)

    # 3. Targeted env overrides (highest precedence among production settings).
    for cap in list(policies):
        prefix = f"AI_ROUTE_{cap.upper()}_"
        fields = {}
        if (env.get(prefix + "PRIMARY") or "").strip():
            fields["primary"] = env[prefix + "PRIMARY"].strip()
        if env.get(prefix + "FALLBACKS") is not None:
            fields["fallbacks"] = env[prefix + "FALLBACKS"]
        for f, suffix in (("timeout_s", "TIMEOUT_S"), ("fallback_deadline_s", "FALLBACK_DEADLINE_S")):
            if (env.get(prefix + suffix) or "").strip():
                fields[f] = env[prefix + suffix]
        if fields:
            policies[cap] = _apply_policy_fields(policies[cap], fields, models, prefix.rstrip("_"), warnings)

    # 4. Local-development model override (last, and gated).
    overrides = _load_overrides(env, models, policies, debug, warnings)

    cfg = RouterConfig(
        enabled=_truthy(env.get("AI_ROUTER_ENABLED"), True),
        fallback_enabled=_truthy(env.get("AI_ROUTER_FALLBACK_ENABLED"), False),
        models=models,
        policies=policies,
        features=features,
        warnings=tuple(warnings),
        overrides=overrides,
        debug=debug,
    )
    for w in cfg.warnings:
        logger.warning("AI routing config: %s", w)
    if overrides:
        logger.warning(
            "LOCAL MODEL OVERRIDE ACTIVE (development only, DEBUG=true): %s",
            ", ".join(f"{('ALL' if k == OVERRIDE_ALL else k)} -> {v}" for k, v in sorted(overrides.items())),
        )
    return cfg
