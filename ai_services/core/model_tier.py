"""
Model tiering — maps feature tiers to Groq model IDs.

LLMClient._resolve_model() validates the name against _GROQ_ALLOWED_MODELS
and falls back to GROQ_MODEL (llama-3.3-70b-versatile) for unknowns.

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
    ModelTier.FAST:     "llama-3.1-8b-instant",       # low-latency, simple tasks
    ModelTier.BALANCED: "llama-3.3-70b-versatile",    # general quality
    ModelTier.POWER:    "llama-3.3-70b-versatile",    # complex generation (same model, max tokens)
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
    "feedback_analyze":   ModelTier.POWER,
    "test_generate":      ModelTier.POWER,
}


def get_model_for_task(feature: str) -> str:
    """Resolve the correct model ID for a given feature."""
    tier = FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
    return MODEL_MAP[tier]


def get_tier_for_feature(feature: str) -> ModelTier:
    return FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
