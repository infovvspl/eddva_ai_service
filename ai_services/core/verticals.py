"""
Vertical registry — the single source of truth for "what makes each product
vertical different".

A *vertical* (e.g. "coaching" for JEE/NEET, "school" for class 1-12) is a thin
configuration layer over the shared AI engine. Everything is shared by default;
a vertical only declares the handful of things that genuinely differ from the
canonical base (currently: prompt overrides, model overrides, enabled features,
academic framing).

Adding a future vertical = add one VerticalProfile here (+ any prompt/model
overrides in prompt_templates.py / model_tier.py). No engine code changes.
Re-segregating a vertical later = this profile + its overrides are self-contained.

Resolution precedence (handled in TenantAuthMiddleware):
    explicit per-request (X-Vertical header / ?vertical=)  >
    institute.vertical (tenant default)                    >
    DEFAULT_VERTICAL env                                   >
    "coaching" (hard default)

Unknown / missing values always fall back to DEFAULT_VERTICAL, so the service
never errors on an unexpected vertical — it degrades to canonical base behaviour.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# Canonical base vertical. The flat registries in prompt_templates.py and
# model_tier.py ARE the coaching/base content, so "coaching" needs no overrides.
DEFAULT_VERTICAL = "coaching"


@dataclass(frozen=True)
class VerticalProfile:
    """Immutable description of one product vertical."""

    key: str
    display_name: str
    # Default exam/answer style for this vertical ("competitive" vs "school").
    default_exam_mode: str = "competitive"
    # Boards / exams this vertical targets (informational + prompt context).
    boards: Tuple[str, ...] = ()
    # Inclusive grade range this vertical serves, e.g. (11, 12) or (1, 10).
    grade_range: Tuple[int, int] = ()
    # Allowlist of features for this vertical. None => all features enabled.
    # (Per-tenant toggles in Institute.features_enabled apply on top of this.)
    enabled_features: Optional[frozenset] = None
    # Denylist — features that make no sense for this vertical. Applied after the
    # allowlist. Usually the easier of the two to maintain: a vertical normally
    # wants everything EXCEPT a couple of features.
    disabled_features: frozenset = frozenset()
    # Free-form, vertical-specific knobs for future use (no engine coupling).
    extra: Dict[str, str] = field(default_factory=dict)

    def allows_feature(self, feature: str) -> bool:
        """Whether this vertical exposes a given feature at all."""
        if feature in self.disabled_features:
            return False
        if self.enabled_features is None:
            return True
        return feature in self.enabled_features


# ──────────────────────────────────────────────────────────────────────────────
#  Registry — seed with the two known verticals. Add more here over time.
# ──────────────────────────────────────────────────────────────────────────────
PROFILES: Dict[str, VerticalProfile] = {
    "coaching": VerticalProfile(
        key="coaching",
        display_name="Coaching (JEE / NEET / Competitive)",
        default_exam_mode="competitive",
        boards=("JEE", "NEET", "CBSE"),
        grade_range=(11, 12),
        enabled_features=None,  # all features
    ),
    "school": VerticalProfile(
        key="school",
        display_name="School (Class 1-12)",
        default_exam_mode="school",
        boards=("CBSE", "ICSE", "State Board"),
        grade_range=(1, 10),
        # All features EXCEPT the ones that are meaningless for Classes 1-12
        # (a 10-year-old has no résumé and is not prepping for college
        # interviews). Gating them off returns a clear 403 instead of generating
        # nonsense. School-specific prompts for everything else are derived in
        # ai_services/core/prompts/school.py.
        disabled_features=frozenset({"resume_analyze", "interview_prep"}),
    ),
}


def env_default_vertical() -> str:
    """Deployment-level default vertical (DEFAULT_VERTICAL env), validated."""
    return normalize_vertical(os.getenv("DEFAULT_VERTICAL"))


def normalize_vertical(value: Optional[str]) -> str:
    """
    Coerce any incoming value into a known vertical key.
    Falls back to DEFAULT_VERTICAL for empty/unknown values — never raises.
    """
    if not value:
        return DEFAULT_VERTICAL
    key = str(value).strip().lower()
    return key if key in PROFILES else DEFAULT_VERTICAL


def get_profile(vertical: Optional[str]) -> VerticalProfile:
    """Return the VerticalProfile for a vertical, normalized + guaranteed valid."""
    return PROFILES[normalize_vertical(vertical)]


def is_known_vertical(value: Optional[str]) -> bool:
    """True if value names a registered vertical (case-insensitive)."""
    return bool(value) and str(value).strip().lower() in PROFILES
