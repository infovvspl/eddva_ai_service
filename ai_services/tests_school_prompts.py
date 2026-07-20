"""
Locks in the school (Classes 1-10) vertical prompt layer.

Two things must hold for every school prompt:
  1. NO competitive-exam framing leaks through (JEE / NEET / IIT / AIIMS / UPSC).
  2. The base prompt's OUTPUT FORMAT / JSON SCHEMA is preserved byte-for-byte —
     the views parse against it, so a school variant that drifts from the schema
     would silently break response parsing.

Both are guaranteed by deriving school prompts from the base (prompts/school.py)
rather than hand-writing a second copy.
"""

import re

from django.test import SimpleTestCase

from ai_services.core.prompt_templates import (
    TEMPLATES, VERTICAL_OVERRIDES, get_template,
)
from ai_services.core.prompts.school import (
    SCHOOL_AUDIENCE, SCHOOL_DISABLED_FEATURES, SCHOOL_OVERRIDE_FEATURES, schoolify,
)
from ai_services.core.verticals import get_profile

COMPETITIVE = re.compile(r"(?i)\b(jee|neet|iit|aiims|upsc)\b")

# Prompts a school user can actually be served: no override AND not gated off.
# (A disabled feature's prompt never reaches school, so its competitive framing
# is harmless — e.g. interview_prep talks about IIT/AIIMS but 403s for school.)
SHAREABLE = [
    f for f in TEMPLATES
    if f not in SCHOOL_OVERRIDE_FEATURES and f not in SCHOOL_DISABLED_FEATURES
]


class SchoolPromptsHaveNoCompetitiveFramingTests(SimpleTestCase):
    def test_every_school_override_is_free_of_competitive_framing(self):
        for feature in SCHOOL_OVERRIDE_FEATURES:
            if feature not in TEMPLATES:
                continue
            system = get_template(feature, "school").system
            leaked = COMPETITIVE.findall(system)
            self.assertFalse(
                leaked,
                f"school prompt for '{feature}' leaks competitive framing: {set(leaked)}",
            )

    def test_school_prompts_carry_the_audience_block(self):
        for feature in SCHOOL_OVERRIDE_FEATURES:
            if feature not in TEMPLATES:
                continue
            self.assertIn(
                "Classes 1-10", get_template(feature, "school").system,
                f"school prompt for '{feature}' is missing the audience block",
            )

    def test_coaching_keeps_its_competitive_framing(self):
        # Regression guard: schoolify() must not have mutated the base prompts.
        self.assertIn("JEE", get_template("test_generate", "coaching").system)
        self.assertIn("JEE", get_template("tutor_session", "coaching").system)


class SchoolPromptsPreserveOutputSchemaTests(SimpleTestCase):
    """The derived prompt must keep the base's schema — parsing depends on it."""

    def test_json_schema_lines_survive_schoolify(self):
        # Every '"key":' line in the base schema must still be present in the
        # school variant. This is what makes deriving safe vs hand-writing.
        for feature in SCHOOL_OVERRIDE_FEATURES:
            if feature not in TEMPLATES:
                continue
            base = TEMPLATES[feature].system
            school = get_template(feature, "school").system
            for key in re.findall(r'"(\w+)"\s*:', base):
                self.assertIn(
                    f'"{key}"', school,
                    f"school prompt for '{feature}' dropped schema key '{key}'",
                )

    def test_user_template_is_identical_to_base(self):
        for feature in SCHOOL_OVERRIDE_FEATURES:
            if feature not in TEMPLATES:
                continue
            self.assertEqual(
                get_template(feature, "school").user_template,
                TEMPLATES[feature].user_template,
            )

    def test_school_prompt_is_base_plus_audience(self):
        # The derivation is exactly: audience block + neutralised base.
        f = "notes_analyze"
        self.assertTrue(get_template(f, "school").system.startswith(SCHOOL_AUDIENCE))


class SharedPromptsAreCleanTests(SimpleTestCase):
    def test_prompts_shared_with_school_have_no_competitive_framing(self):
        """A prompt with NO school override is served as-is to school users, so it
        must be vertical-neutral. If this fails, that feature needs an override."""
        for feature in SHAREABLE:
            if feature in ("doubt_resolve",):
                continue  # doubt is handled in bridge._build_solver_system_prompt
            system = TEMPLATES[feature].system
            leaked = COMPETITIVE.findall(system)
            self.assertFalse(
                leaked,
                f"'{feature}' has no school override but its base prompt contains "
                f"{set(leaked)} — add it to SCHOOL_OVERRIDE_FEATURES",
            )


class SchoolFeatureGatingTests(SimpleTestCase):
    def test_resume_and_interview_are_disabled_for_school(self):
        school = get_profile("school")
        self.assertFalse(school.allows_feature("resume_analyze"))
        self.assertFalse(school.allows_feature("interview_prep"))

    def test_school_still_allows_normal_features(self):
        school = get_profile("school")
        for f in ("tutor_session", "test_generate", "quiz_generate", "plan_generate"):
            self.assertTrue(school.allows_feature(f))

    def test_coaching_allows_everything(self):
        coaching = get_profile("coaching")
        for f in ("resume_analyze", "interview_prep", "tutor_session"):
            self.assertTrue(coaching.allows_feature(f))


class SolverFormulaKnowledgeBaseIsCoachingOnlyTests(SimpleTestCase):
    """
    The formula KB is built from JEE/NEET formula sheets (data/knowledge_base/).
    Grounding a Class 1-10 answer with IIT-JEE formulae would push it above grade
    level, so retrieval must be skipped for the school vertical.
    """

    def _solve_and_capture(self, vertical):
        """Run solve() far enough to see whether the KB was queried."""
        from unittest.mock import patch
        import asyncio

        calls = []

        class _FakeLLM:
            def complete(self, system_prompt, user_prompt, **kw):
                calls.append(system_prompt)
                # invalid python twice -> solve() gives up quickly, no sandbox run
                return {"content": "not python {", "model": "m", "latency_ms": 1,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        retrieved = []

        def _fake_retrieve(query, top_k=3):
            retrieved.append(query)
            return [{"text": "F = ma (JEE sheet)", "source": "IIT-JEE-Formula-PCM.pdf"}]

        import ai_services.solver.scientific_solver as ss
        with patch.object(ss.formula_retriever, "retrieve", _fake_retrieve):
            solver = ss.ScientificSolver.__new__(ss.ScientificSolver)
            solver.llm = _FakeLLM()
            asyncio.run(solver.solve("A ball is thrown upward", "detailed", vertical))
        return retrieved, calls

    def test_school_does_not_query_the_jee_formula_bank(self):
        retrieved, calls = self._solve_and_capture("school")
        self.assertEqual(retrieved, [], "school must NOT be grounded with JEE/NEET formulae")
        self.assertIn("Classes 1-10", calls[0])
        self.assertNotIn("VERIFIED FORMULAS FROM KNOWLEDGE BASE", calls[0])

    def test_coaching_does_query_the_formula_bank(self):
        retrieved, calls = self._solve_and_capture("coaching")
        self.assertEqual(len(retrieved), 1, "coaching should use the formula KB")
        self.assertIn("VERIFIED FORMULAS FROM KNOWLEDGE BASE", calls[0])


class DoubtModelRoutingByVerticalTests(SimpleTestCase):
    """School math routes to the cheaper GPT-OSS-120B instead of Qwen ($3/M out)."""

    def _model(self, subject, qtype, vertical):
        from ai_services.views.bridge import _select_doubt_model, GROQ_MODELS
        return _select_doubt_model(subject, qtype, vertical), GROQ_MODELS

    def test_school_math_avoids_the_expensive_qwen_model(self):
        model, G = self._model("math", "numerical", "school")
        self.assertEqual(model, G["reasoning"])          # gpt-oss-120b
        self.assertNotEqual(model, G["math"])            # NOT qwen

    def test_coaching_math_still_uses_qwen(self):
        model, G = self._model("math", "numerical", "coaching")
        self.assertEqual(model, G["math"])               # qwen (unchanged)

    def test_non_math_routing_is_unchanged_for_both(self):
        for v in ("school", "coaching"):
            phys_num, G = self._model("physics", "numerical", v)
            self.assertEqual(phys_num, G["reasoning"])
            theory, _ = self._model("physics", "conceptual", v)
            self.assertEqual(theory, G["general"])

    def test_gpt_oss_is_still_treated_as_a_reasoning_model_downstream(self):
        # School math now returns gpt-oss; the doubt view parses reasoning models
        # in text mode. Guard that gpt-oss stays in that set.
        from ai_services.views.bridge import GROQ_MODELS
        reasoning_set = {"openai/gpt-oss-120b", "qwen/qwen3-32b"}
        self.assertIn(GROQ_MODELS["reasoning"], reasoning_set)


class SchoolifyUnitTests(SimpleTestCase):
    def test_neutralises_common_phrases(self):
        out = schoolify("You are a tutor for JEE/NEET students preparing for competitive exams.")
        self.assertNotIn("JEE", out)
        self.assertNotIn("NEET", out)
        self.assertIn("Classes 1-10", out)

    def test_does_not_touch_the_schema(self):
        base = 'Return JSON:\n{\n  "score": <float>,\n  "feedback": "<text>"\n}'
        out = schoolify(base)
        self.assertIn('"score"', out)
        self.assertIn('"feedback"', out)
