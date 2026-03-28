"""
Model tiering — right-size LLM per feature.

Tier 1 (FAST):     llama-3.1-8b-instant    — simple tasks, engagement detection
Tier 2 (BALANCED): llama-3.3-70b-versatile  — tutoring, feedback, study plans
Tier 3 (POWER):    llama-3.3-70b-versatile  — grading, performance analysis, test generation

Saves 40-60% cost by not using 70B for trivial tasks.
"""

import os
from enum import Enum


class ModelTier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    POWER = "power"


# Configurable via env — override in production as needed
MODEL_MAP = {
    ModelTier.FAST: os.getenv("LLM_MODEL_FAST", "llama-3.1-8b-instant"),
    ModelTier.BALANCED: os.getenv("LLM_MODEL_BALANCED", "llama-3.3-70b-versatile"),
    ModelTier.POWER: os.getenv("LLM_MODEL_POWER", "llama-3.3-70b-versatile"),
}

# Which tier each feature uses
FEATURE_TIER_MAP = {
    # ── NestJS ai-bridge features ─────────────────────────────────────
    "doubt_resolve": ModelTier.BALANCED,        # needs good reasoning
    "tutor_session": ModelTier.BALANCED,         # conversational quality matters
    "tutor_continue": ModelTier.BALANCED,        # must maintain context
    "performance_analysis": ModelTier.POWER,     # complex analytics + rank prediction
    "grade_subjective": ModelTier.POWER,         # grading accuracy is critical
    "engagement_detect": ModelTier.FAST,         # simple signal analysis
    "content_recommend": ModelTier.FAST,         # simple recommendation
    "stt_notes": ModelTier.POWER,                # comprehensive note generation needs full capacity
    "feedback_generate": ModelTier.BALANCED,     # motivational + analytical
    "notes_analyze": ModelTier.BALANCED,         # content analysis
    "resume_analyze": ModelTier.BALANCED,        # structured analysis
    "interview_prep": ModelTier.BALANCED,        # question generation
    "plan_generate": ModelTier.POWER,            # complex multi-day planning
    "quiz_generate": ModelTier.POWER,            # needs careful reasoning about transcript content

    # ── Legacy Django features ────────────────────────────────────────
    "health": ModelTier.FAST,
    "content_suggest": ModelTier.FAST,
    "cheating_analyze": ModelTier.FAST,
    "notes_generate": ModelTier.BALANCED,
    "study_plan": ModelTier.BALANCED,
    "career_roadmap": ModelTier.BALANCED,
    "feedback_analyze": ModelTier.POWER,
    "test_generate": ModelTier.POWER,
    "performance_analyze": ModelTier.POWER,
}


def get_model_for_task(feature: str) -> str:
    """Resolve the correct model ID for a given feature."""
    tier = FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
    return MODEL_MAP[tier]


def get_tier_for_feature(feature: str) -> ModelTier:
    return FEATURE_TIER_MAP.get(feature, ModelTier.BALANCED)
