"""
Model tiering — maps feature tiers to Groq model IDs.

LLMClient._resolve_model() validates the name against _GROQ_ALLOWED_MODELS
and falls back to GROQ_MODEL (openai/gpt-oss-120b) for unknowns.

Removed features (deleted from the platform):
  - performance_analysis   (POST /performance/analyze)
  - grade_subjective        (POST /grade/subjective)
  - engagement_detect       (POST /engage/detect)
  - cheating_analyze        (POST /cheating/analyze-logs/)
"""

from enum import Enum


class ModelTier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    POWER = "power"


# Groq model IDs recognised by LLMClient._resolve_model()
MODEL_MAP = {
    # Groq decommissioned llama-3.1-8b-instant + llama-3.3-70b-versatile on
    # 2026-08-16; migrated to the GPT-OSS models (Groq's recommended replacements).
    ModelTier.FAST:     "openai/gpt-oss-20b",     # low-latency, simple tasks
    ModelTier.BALANCED: "openai/gpt-oss-120b",    # general quality
    ModelTier.POWER:    "openai/gpt-oss-120b",    # complex generation (same model, max tokens)
}

# Which tier each feature uses
FEATURE_TIER_MAP = {
    # ── NestJS ai-bridge features ─────────────────────────────────────
    "doubt_resolve":      ModelTier.BALANCED,   # needs good reasoning
    "tutor_session":      ModelTier.BALANCED,   # conversational quality matters
    "tutor_continue":     ModelTier.BALANCED,   # must maintain context
    "content_recommend":  ModelTier.FAST,       # simple recommendation
    "stt_notes":          ModelTier.POWER,      # comprehensive note generation
    "feedback_generate":  ModelTier.BALANCED,   # motivational + analytical
    "notes_analyze":      ModelTier.BALANCED,   # content analysis
    "resume_analyze":     ModelTier.BALANCED,   # structured analysis
    "interview_prep":     ModelTier.BALANCED,   # question generation
    "plan_generate":      ModelTier.POWER,      # complex multi-day planning
    "syllabus_generate":  ModelTier.POWER,      # structured curriculum generation
    "quiz_generate":      ModelTier.POWER,      # careful reasoning on transcript

    # ── Legacy Django features ────────────────────────────────────────
    "health":             ModelTier.FAST,
    "content_suggest":    ModelTier.FAST,
    "notes_generate":     ModelTier.BALANCED,
    "study_plan":         ModelTier.BALANCED,
    "career_roadmap":     ModelTier.BALANCED,
    "career_guidance":    ModelTier.POWER,      # full academic + psychometric analysis
    "feedback_analyze":   ModelTier.POWER,
    "test_generate":      ModelTier.POWER,

    # ── QA / evaluation ───────────────────────────────────────────────
    "evaluate_batch":     ModelTier.POWER,   # accuracy fact-checking needs the strongest model
}


# ──────────────────────────────────────────────────────────────────────────────
#  Per-vertical model overrides.
#
#  The maps above are the canonical *base* (coaching → Groq models). A vertical
#  appears here only when it needs a different model. Within a vertical, use
#  "__all__" for a blanket override or a feature name for a per-feature one.
#
#  e.g. (Phase 3): "school": {"__all__": "deepseek-v4-pro"}
#  Empty by default → identical behaviour for every caller.
# ──────────────────────────────────────────────────────────────────────────────
VERTICAL_MODEL_OVERRIDES = {}  # type: dict[str, dict[str, str]]


def get_model_for_task(feature: str, vertical: str = "base") -> str:
    """
    Resolve the model ID for a feature, optionally specialized for a vertical.

    Resolution: per-feature vertical override → blanket vertical override
    → base tier model. Unknown verticals fall back to base.
    """
    overrides = VERTICAL_MODEL_OVERRIDES.get(vertical or "base", {})
    model = overrides.get(feature) or overrides.get("__all__")
    if model:
        return model
    tier = FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
    return MODEL_MAP[tier]


def get_tier_for_feature(feature: str) -> ModelTier:
    return FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
