"""
Model tiering — all tiers now resolve to the local edvaqwen model via Ollama.

Tiers are kept so the architecture stays the same and the `get_model_for_task`
call sites in every view continue to work without any change.
The `model` value returned by each tier is passed to LLMClient.complete(),
where it is accepted for API compatibility but always overridden to edvaqwen.

Removed features (deleted from the platform):
  - performance_analysis   (POST /performance/analyze)
  - grade_subjective        (POST /grade/subjective)
  - engagement_detect       (POST /engage/detect)
  - cheating_analyze        (POST /cheating/analyze-logs/)
"""

import os
from enum import Enum


class ModelTier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    POWER = "power"


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvaqwen")

# All tiers map to the same local model
MODEL_MAP = {
    ModelTier.FAST:     OLLAMA_MODEL,
    ModelTier.BALANCED: OLLAMA_MODEL,
    ModelTier.POWER:    OLLAMA_MODEL,
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