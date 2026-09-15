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

Metadata here is *reference* metadata for routing and benchmarking. It is not
billing truth: cost estimation stays in usage_logger.MODEL_COSTS, and relative
cost tiers below exist only so an operator can reason about a policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

KNOWN_PROVIDERS = frozenset({"groq", "together", "gemini"})

# How a model satisfies a JSON-only output requirement.
#   native    provider honours response_format / a JSON mime type
#   prompted  JSON produced by instruction and extracted from text — works,
#             but is less reliable, so it is never sent a response_format
#   none      must not be used for structured output
STRUCTURED_NATIVE = "native"
STRUCTURED_PROMPTED = "prompted"
STRUCTURED_NONE = "none"

CAPABILITIES = frozenset({
    "reasoning",     # doubts, tutor, hard Q&A, assessment reasoning
    "lightweight",   # classification, extraction, cleanup, metadata
    "content",       # long educational documents, notes, PPT content
    "premium",       # very hard reasoning — opt-in only, never a default
    "bulk_text",     # high-volume background summarisation / cleanup
    "grounded",      # generation that must stay inside supplied sources
    "vision",        # image understanding
})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    provider_model_id: Optional[str]
    capabilities: frozenset
    # Env var that supplies provider_model_id. For unverified models this is the
    # ONLY way an id gets set.
    model_env: Optional[str] = None
    multimodal: bool = False
    long_context: bool = False
    context_tokens: Optional[int] = None
    structured_output: str = STRUCTURED_PROMPTED
    supports_grounding: bool = False
    quality_tier: str = "standard"   # light | standard | high | premium
    cost_tier: str = "medium"        # low | medium | high
    # Where the capability flags above came from. "repository" = exercised by
    # this codebase in production; "unverified" = candidate, flags are
    # conservative defaults until confirmed by configuration or benchmark.
    metadata_source: str = "unverified"
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_configured(self) -> bool:
        return bool((self.provider_model_id or "").strip())

    def with_model_id(self, provider_model_id: Optional[str]) -> "ModelSpec":
        return replace(self, provider_model_id=(provider_model_id or None))


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
    # ── Groq — ids proven in this repository ──────────────────────────────────
    ModelSpec(
        id="groq/gpt-oss-120b",
        provider="groq",
        provider_model_id="openai/gpt-oss-120b",
        capabilities=_caps("reasoning", "content"),
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
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
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
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
        context_tokens=_GROQ_EFFECTIVE_REQUEST_TOKENS,
        structured_output=STRUCTURED_NATIVE,
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
    # ── Together — candidates. NO model ids: configuration-supplied only. ─────
    ModelSpec(
        id="together/gpt-oss-120b",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_GPT_OSS_120B",
        capabilities=_caps("reasoning", "content"),
        quality_tier="high",
        cost_tier="medium",
        notes="Intended fallback for groq/gpt-oss-120b (same model family, different provider).",
    ),
    ModelSpec(
        id="together/qwen3.8-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_QWEN3_8_FLASH",
        capabilities=_caps("content", "grounded"),
        quality_tier="high",
        cost_tier="medium",
        notes="Content candidate. Not assumed better than Gemini; must be benchmarked first.",
    ),
    ModelSpec(
        id="together/glm-5.3-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_GLM_5_3_FLASH",
        capabilities=_caps("content"),
        quality_tier="standard",
        cost_tier="medium",
        notes="Content fallback candidate.",
    ),
    ModelSpec(
        id="together/qwen3.7-max",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_QWEN3_7_MAX",
        capabilities=_caps("premium", "reasoning"),
        quality_tier="premium",
        cost_tier="high",
        notes="Premium opt-in only. Must never become a default route.",
    ),
    ModelSpec(
        id="together/deepseek-v4-flash",
        provider="together",
        provider_model_id=None,
        model_env="TOGETHER_MODEL_DEEPSEEK_V4_FLASH",
        capabilities=_caps("bulk_text", "lightweight"),
        quality_tier="standard",
        cost_tier="low",
        notes="High-volume text candidate. Not migrated to until benchmarked.",
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


def unmet_requirements(spec: ModelSpec, request) -> list[str]:
    """Hard requirements a candidate fails. Empty list = eligible."""
    unmet = []
    if getattr(request, "requires_vision", False) and not spec.multimodal:
        unmet.append("vision")
    if getattr(request, "requires_long_context", False) and not spec.long_context:
        unmet.append("long_context")
    min_ctx = getattr(request, "min_context_tokens", None)
    if min_ctx and not spec.long_context and (spec.context_tokens is None or spec.context_tokens < min_ctx):
        unmet.append("context_tokens")
    if getattr(request, "requires_grounding", False) and not spec.supports_grounding:
        unmet.append("grounding")
    if getattr(request, "effective_structured_output", False) and spec.structured_output == STRUCTURED_NONE:
        unmet.append("structured_output")
    return unmet
