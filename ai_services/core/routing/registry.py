"""
Model registry: what each model *is*, independent of which feature uses it.

A ModelSpec separates three things the old code conflated in one string:

    MODEL       registry id, e.g. "together/qwen3.8-flash" (ours, stable)
    PROVIDER    who executes it: groq | together | gemini
    PROVIDER MODEL ID
                the exact id the provider's API expects

Provider model ids come from two places only:

  * ids already used by this repository and proven against the live provider
    (the Groq GPT-OSS ids, gemini-2.5-flash) — written below;
  * configuration, for every model this repository has never called
    (all Together candidates). Those default to ``None`` and the router
    treats them as unconfigured, so nothing is ever sent to a guessed id.

Technical support flags are tri-state: True = verified supported, False =
verified unsupported, None = UNKNOWN. UNKNOWN never satisfies a requirement, so
an unverified model is never handed vision, long-context or JSON work on the
strength of a guess. Operators promote a flag through configuration once they
have verified it (see config.py).

Metadata here is reference metadata for routing and benchmarking, not billing
truth: cost estimation stays in usage_logger.MODEL_COSTS, and the relative cost
tiers exist only so an operator can reason about a policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

KNOWN_PROVIDERS = frozenset({"groq", "together", "gemini"})

# Providers whose model ids are recorded bare in usage telemetry, exactly as they
# were before the router existed. Every other provider's ids are recorded as
# "<provider>:<model id>": the same model id (e.g. openai/gpt-oss-120b) can be
# served by more than one provider, and the usage row has no provider column.
UNQUALIFIED_TELEMETRY_PROVIDERS = frozenset({"groq", "gemini"})

# How a model satisfies a JSON-only output requirement.
#   native    provider honours response_format / a JSON mime type
#   prompted  JSON produced by instruction and extracted from text. An explicit
#             operator-selected compatibility path; never sent response_format
#   none      verified unable to produce structured output
#   unknown   not verified — JSON requests are refused, never downgraded
STRUCTURED_NATIVE = "native"
STRUCTURED_PROMPTED = "prompted"
STRUCTURED_NONE = "none"
STRUCTURED_UNKNOWN = "unknown"
STRUCTURED_VALUES = frozenset({STRUCTURED_NATIVE, STRUCTURED_PROMPTED, STRUCTURED_NONE, STRUCTURED_UNKNOWN})
JSON_CAPABLE = frozenset({STRUCTURED_NATIVE, STRUCTURED_PROMPTED})

CAPABILITIES = frozenset({
    "reasoning",     # doubts, tutor, hard Q&A, assessment reasoning
    "lightweight",   # classification, extraction, cleanup, metadata
    "content",       # long educational documents, notes, PPT content
    "premium",       # very hard reasoning — opt-in only, never a default
    "bulk_text",     # high-volume background summarisation / cleanup
    "grounded",      # generation that must stay inside supplied sources
    "vision",        # image understanding
})

# When a pinned model must be mapped to a capability, prefer the everyday
# classes; "premium" is last so nothing lands there by inference.
CAPABILITY_PREFERENCE = ("reasoning", "lightweight", "content", "bulk_text", "grounded", "vision", "premium")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    provider_model_id: Optional[str]
    capabilities: frozenset
    # Env var that supplies provider_model_id. For unverified models this is the
    # ONLY way an id gets set.
    model_env: Optional[str] = None
    multimodal: Optional[bool] = None
    long_context: Optional[bool] = None
    context_tokens: Optional[int] = None
    structured_output: str = STRUCTURED_UNKNOWN
    supports_grounding: Optional[bool] = None
    # True = the provider only accepts streaming requests for this model (the
    # adapter streams and assembles the full answer); False/None = a normal
    # request is used. Set from configuration after the provider refuses one.
    streaming: Optional[bool] = None
    quality_tier: str = "standard"   # light | standard | high | premium
    cost_tier: str = "medium"        # low | medium | high
    # Where the metadata came from: "repository" = exercised by this codebase in
    # production; "unverified" = candidate with UNKNOWN flags; "config" = set by
    # an operator; "pinned" = named by a call site and not in the registry.
    metadata_source: str = "unverified"
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def alias(self) -> str:
        """Short name after the provider, e.g. "qwen3.8-flash"."""
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    @property
    def is_configured(self) -> bool:
        return bool((self.provider_model_id or "").strip())

    def with_model_id(self, provider_model_id: Optional[str]) -> "ModelSpec":
        return replace(self, provider_model_id=(provider_model_id or None))


def support_label(value: Optional[bool]) -> str:
    return "UNKNOWN" if value is None else ("yes" if value else "no")


def primary_capability(spec: ModelSpec) -> Optional[str]:
    for cap in CAPABILITY_PREFERENCE:
        if cap in spec.capabilities:
            return cap
    return None


def _caps(*names: str) -> frozenset:
    unknown = set(names) - CAPABILITIES
    if unknown:  # programming error in the defaults below
        raise ValueError(f"unknown capabilities in registry defaults: {sorted(unknown)}")
    return frozenset(names)


# Groq's on-demand tier rejects any single request above ~12,000 tokens
# (see bridge.generate_topic_content: "Groq's on-demand tier rejects any request
# over 12,000 tokens outright"). That, not the model's native window, is the
# ceiling EDVA actually hits, so Groq models are not long_context here.
_GROQ_EFFECTIVE_REQUEST_TOKENS = 12_000

DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    # ── Groq — ids and behaviour proven in this repository ────────────────────
    ModelSpec(
        id="groq/gpt-oss-120b",
        provider="groq",
        provider_model_id="openai/gpt-oss-120b",
        capabilities=_caps("reasoning", "content"),
        multimodal=False,
        long_context=False,
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
        supports_grounding=False,
        quality_tier="high",
        cost_tier="medium",
        metadata_source="repository",
        notes="Current production default for doubts, tutor, quiz, content, assessments.",
    ),
    ModelSpec(
        id="groq/gpt-oss-20b",
        provider="groq",
        provider_model_id="openai/gpt-oss-20b",
        capabilities=_caps("lightweight", "bulk_text"),
        multimodal=False,
        long_context=False,
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
        supports_grounding=False,
        quality_tier="light",
        cost_tier="low",
        metadata_source="repository",
        notes="Current production FAST tier: subject detection, transcript cleanup, chunk notes.",
    ),
    ModelSpec(
        id="groq/qwen3-32b",
        provider="groq",
        provider_model_id="qwen/qwen3-32b",
        capabilities=_caps("reasoning"),
        multimodal=False,
        long_context=False,
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
        supports_grounding=False,
        quality_tier="high",
        cost_tier="high",
        metadata_source="repository",
        notes="Coaching math doubts (bridge._select_doubt_model).",
    ),
    # ── Gemini — kept available; replaceable, not removed ─────────────────────
    ModelSpec(
        id="gemini/gemini-2.5-flash",
        provider="gemini",
        provider_model_id="gemini-2.5-flash",
        model_env="GEMINI_TEXT_MODEL",
        capabilities=_caps("content", "grounded", "vision"),
        multimodal=True,
        long_context=True,
        structured_output=STRUCTURED_NATIVE,
        supports_grounding=True,
        quality_tier="high",
        cost_tier="medium",
        metadata_source="repository",
        notes="Grounded PPT / grounded content / vision / Odia. Carries 30k-token grounding prompts today.",
    ),
    # ── Together — candidates. NO model ids and NO assumed support flags. ─────
    # Ids: TOGETHER_MODEL_* only. Flags: UNKNOWN until verified and configured.
    ModelSpec(
        id="together/gpt-oss-120b",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_GPT_OSS_120B",
        capabilities=_caps("reasoning", "content"),
        quality_tier="high",
        cost_tier="medium",
        notes="Same model family as groq/gpt-oss-120b on a different provider. Candidate, not a default.",
    ),
    ModelSpec(
        id="together/qwen3.8-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_QWEN38_FLASH",
        capabilities=_caps("content"),
        quality_tier="high",
        cost_tier="medium",
        notes="Content candidate (textbook, chapters, notes, PPT, multilingual). Long context / multimodal "
              "UNKNOWN until verified. Not assumed better than Gemini.",
    ),
    ModelSpec(
        id="together/glm-5.3-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_GLM53_FLASH",
        capabilities=_caps("content"),
        quality_tier="standard",
        cost_tier="medium",
        notes="Content candidate (long-context / structured educational generation, if verified).",
    ),
    ModelSpec(
        id="together/deepseek-v4-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_DEEPSEEK_V4_FLASH",
        capabilities=_caps("bulk_text", "lightweight"),
        quality_tier="standard",
        cost_tier="low",
        notes="High-volume text candidate: summarisation, cleanup, extraction, classification.",
    ),
    ModelSpec(
        id="together/qwen3.7-max",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_QWEN37_MAX",
        capabilities=_caps("premium"),
        quality_tier="premium",
        cost_tier="high",
        notes="Premium opt-in only. Never a default route.",
    ),
)


def default_registry() -> dict[str, ModelSpec]:
    return {m.id: m for m in DEFAULT_MODELS}


def find_by_provider_model(
    models: dict, provider: str, provider_model_id: str
) -> Optional[ModelSpec]:
    for spec in models.values():
        if spec.provider == provider and spec.provider_model_id == provider_model_id:
            return spec
    return None


def resolve_model_ref(models, value: Optional[str]) -> Optional[ModelSpec]:
    """Resolve an operator reference to a registered model.

    Accepts "together:qwen3.8-flash" (provider:alias), "together/qwen3.8-flash"
    (registry id) or "together:<configured provider model id>". Anything that is
    not already in the registry resolves to None — a reference can select a
    registered model, never introduce an arbitrary one.
    """
    v = (value or "").strip()
    if not v:
        return None
    if v in models:
        return models[v]
    provider, sep, ref = v.partition(":")
    if not sep:
        return None
    provider, ref = provider.strip().lower(), ref.strip()
    for spec in models.values():
        if spec.provider != provider:
            continue
        if ref in (spec.alias, spec.id) or (spec.provider_model_id and ref == spec.provider_model_id):
            return spec
    return None


def unmet_requirements(spec: ModelSpec, request) -> list[str]:
    """Hard requirements a candidate fails. Empty list = eligible. UNKNOWN fails."""
    unmet = []
    if getattr(request, "requires_vision", False) and spec.multimodal is not True:
        unmet.append("vision")
    if getattr(request, "requires_long_context", False) and spec.long_context is not True:
        unmet.append("long_context")
    min_ctx = getattr(request, "min_context_tokens", None)
    if min_ctx and spec.long_context is not True and (spec.context_tokens is None or spec.context_tokens < min_ctx):
        unmet.append("context_tokens")
    if getattr(request, "requires_grounding", False) and spec.supports_grounding is not True:
        unmet.append("grounding")
    if getattr(request, "effective_structured_output", False) and spec.structured_output not in JSON_CAPABLE:
        unmet.append("structured_output")
    return unmet
