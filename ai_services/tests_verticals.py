"""
Regression tests for the vertical (multi-product) layer.

These lock in the Phase 1 guarantee: with no vertical supplied — or with the
default "coaching" vertical — the engine behaves byte-identically to the
pre-vertical codebase (canonical base). Verticals only diverge where an
explicit override is registered.
"""

from django.test import SimpleTestCase

from ai_services.core.verticals import (
    normalize_vertical, get_profile, DEFAULT_VERTICAL, PROFILES,
)
from ai_services.core.prompt_templates import get_template, TEMPLATES, VERTICAL_OVERRIDES
from ai_services.core.model_tier import get_model_for_task
from ai_services.core.cache import _make_cache_key


class VerticalResolutionTests(SimpleTestCase):
    def test_default_is_coaching(self):
        self.assertEqual(DEFAULT_VERTICAL, "coaching")

    def test_seeded_verticals(self):
        self.assertEqual(set(PROFILES), {"coaching", "school"})

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(normalize_vertical("martian"), DEFAULT_VERTICAL)

    def test_empty_falls_back_to_default(self):
        self.assertEqual(normalize_vertical(""), DEFAULT_VERTICAL)
        self.assertEqual(normalize_vertical(None), DEFAULT_VERTICAL)

    def test_case_insensitive(self):
        self.assertEqual(normalize_vertical("SCHOOL"), "school")

    def test_profile_lookup(self):
        self.assertEqual(get_profile("school").grade_range, (1, 10))
        self.assertEqual(get_profile("coaching").default_exam_mode, "competitive")


class PromptRegistryBackwardCompatTests(SimpleTestCase):
    def test_base_equals_coaching_when_no_override(self):
        base = get_template("doubt_resolve")
        coaching = get_template("doubt_resolve", "coaching")
        self.assertIs(base, coaching)

    def test_unspecified_vertical_falls_back_to_base(self):
        # No school override registered yet (Phase 1) → identical to base.
        if "doubt_resolve" not in VERTICAL_OVERRIDES:
            self.assertIs(get_template("doubt_resolve", "school"),
                          get_template("doubt_resolve"))

    def test_every_feature_resolves_for_every_vertical(self):
        for feature in TEMPLATES:
            for vertical in list(PROFILES) + ["base", "unknown"]:
                self.assertIsNotNone(get_template(feature, vertical))

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            get_template("does_not_exist")


class ModelSelectionBackwardCompatTests(SimpleTestCase):
    def test_base_equals_coaching(self):
        self.assertEqual(get_model_for_task("doubt_resolve"),
                         get_model_for_task("doubt_resolve", "coaching"))

    def test_vertical_without_override_uses_base(self):
        self.assertEqual(get_model_for_task("test_generate", "school"),
                         get_model_for_task("test_generate"))


class EvaluateFeatureTests(SimpleTestCase):
    def test_evaluate_registered(self):
        self.assertIn("evaluate_batch", TEMPLATES)

    def test_school_override_differs_from_base(self):
        base = get_template("evaluate_batch", "coaching")
        school = get_template("evaluate_batch", "school")
        self.assertIsNot(base, school)
        self.assertNotIn("JEE", school.system)
        self.assertNotIn("NEET", school.system)
        self.assertIn("Classes 1-10", school.system)

    def test_base_keeps_competitive_framing(self):
        self.assertIn("JEE", get_template("evaluate_batch", "coaching").system)


class CacheKeyTests(SimpleTestCase):
    def test_key_carries_vertical_segment(self):
        self.assertEqual(
            _make_cache_key("inst1", "doubt_resolve", "HASH"),
            "ai_svc:inst1:base:doubt_resolve:HASH",
        )

    def test_verticals_get_distinct_keys(self):
        k_base = _make_cache_key("inst1", "doubt_resolve", "HASH", "coaching")
        k_school = _make_cache_key("inst1", "doubt_resolve", "HASH", "school")
        self.assertNotEqual(k_base, k_school)
        self.assertIn(":school:", k_school)
