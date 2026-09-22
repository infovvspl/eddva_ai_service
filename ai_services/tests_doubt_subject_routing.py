"""
Subject routing for the doubt solver.

The bug this prevents: a teacher pressed "Draft with AI" on a Class 10 civics
doubt ("difference between federalism and a unitary government") and the AI
replied *"I'm sorry, but I can only help with physics questions."*

Nothing was broken at the provider level. The doubt carried no subject_name, so
classification fell through to `_detect_subject_and_type_for_doubt`; every
keyword list there is STEM, so the question scored zero on all four; the LLM
classifier was only offered biology/chemistry/physics/math and its answer was
coerced to `physics`; and `_build_solver_system_prompt` then opened with
"Subject: PHYSICS", which the model obeyed by refusing.

Two independent rules are enforced here:

  1. a non-STEM question must not be routed to a STEM subject, and
  2. even when routing IS wrong, the prompt must never let the model refuse.

Rule 2 is the one that matters: classification is best-effort forever, so a
misroute has to degrade to a less-tailored answer, never to a non-answer.
"""

import io
import os
import re

from django.test import SimpleTestCase

from ai_services.views.bridge import (
    _DETECTOR_SUBJECT_ALIASES,
    _DETECTOR_VALID_SUBJECTS,
    _DOUBT_DETECTOR_SYSTEM,
    _SOLVER_SCOPE_RULE,
    _SUBJECT_DISPLAY,
    _build_solver_system_prompt,
    _detect_subject_by_keyword,
)

# The exact doubt from the report.
CIVICS_DOUBT = (
    "Can you help me understand the difference between federalism and unitary "
    "government with a simple real-world example from India?"
)


def _bridge_source() -> str:
    """
    Read bridge.py as text.

    resolve_doubt is wrapped by @api_view/@metered, so inspect.getsource() on the
    imported object returns DRF's wrapper, not the view body.
    """
    path = os.path.join(os.path.dirname(__file__), "views", "bridge.py")
    return io.open(path, encoding="utf-8").read()


def _function_body(name: str) -> str:
    src = _bridge_source()
    start = src.index(f"def {name}(")
    end = src.find("\ndef ", start + 1)
    return src[start:end if end != -1 else len(src)]


class KeywordDetectorScopeTests(SimpleTestCase):
    """Why the humanities path reaches the fallback at all."""

    def test_a_civics_question_scores_zero_on_every_keyword_list(self):
        # Not a defect in itself — it is the reason the fallback default matters.
        _, score, all_scores = _detect_subject_by_keyword(CIVICS_DOUBT)
        self.assertEqual(score, 0, f"unexpected keyword hits: {all_scores}")

    def test_a_physics_question_still_matches_confidently(self):
        subject, score, _ = _detect_subject_by_keyword(
            "A block of mass 2 kg slides down a frictionless incline; find its acceleration."
        )
        self.assertEqual(subject, "physics")
        self.assertGreaterEqual(score, 2, "physics routing must not regress")


class DetectorVocabularyTests(SimpleTestCase):
    """The classifier must be able to name the subjects the solver supports."""

    def test_non_stem_subjects_are_offered_to_the_classifier(self):
        for subject in ("social_science", "english", "hindi", "odia", "general"):
            self.assertIn(subject, _DOUBT_DETECTOR_SYSTEM)

    def test_every_offered_subject_is_accepted_back(self):
        # Offering a label the validator then rejects is exactly the old bug.
        for line in _DOUBT_DETECTOR_SYSTEM.splitlines():
            if line.startswith("- "):
                label = line[2:].split(":", 1)[0].strip()
                self.assertIn(label, _DETECTOR_VALID_SUBJECTS)

    def test_aliases_resolve_to_supported_subjects(self):
        for alias, target in _DETECTOR_SUBJECT_ALIASES.items():
            self.assertIn(
                target, _DETECTOR_VALID_SUBJECTS,
                f"alias {alias!r} maps to unsupported subject {target!r}",
            )

    def test_common_humanities_labels_are_aliased(self):
        for alias in ("civics", "political science", "history", "social science"):
            self.assertEqual(_DETECTOR_SUBJECT_ALIASES.get(alias), "social_science")

    def test_physics_is_no_longer_the_catch_all(self):
        # The two coercion sites in the detector must fall back to "general".
        src = _function_body("_detect_subject_and_type_for_doubt")
        self.assertNotIn('else "physics"', src)
        self.assertNotIn('"physics", "numerical"', src)
        self.assertIn('"general", "conceptual"', src)


class SolverPromptScopeTests(SimpleTestCase):
    """Rule 2: a wrong subject must not become a refusal."""

    def _prompt(self, subject, qtype="conceptual"):
        return _build_solver_system_prompt(subject, qtype, "detailed", "school", "CBSE")

    def test_general_subject_builds_a_usable_prompt(self):
        prompt = self._prompt("general")
        self.assertIn(_SUBJECT_DISPLAY["general"], prompt)
        self.assertNotIn("Subject: GENERAL.", prompt)

    def test_social_science_reads_naturally(self):
        self.assertIn("Subject: SOCIAL SCIENCE", self._prompt("social_science"))

    def test_unknown_subject_does_not_raise(self):
        # _SUBJECT_RULES has no entry for these; the builder must still work.
        for subject in ("english", "hindi", "odia", "computer", "general"):
            self.assertTrue(self._prompt(subject).strip())

    def test_stem_prompts_are_unchanged(self):
        # The tuned physics/chemistry/math prompts must not have been disturbed.
        self.assertIn("Subject: PHYSICS", self._prompt("physics", "numerical"))
        self.assertIn("PHYSICS RULES:", self._prompt("physics", "numerical"))
        self.assertIn("MATH RULES:", self._prompt("math", "derivation"))

    def test_scope_rule_forbids_refusing(self):
        self.assertIn("NEVER refuse", _SOLVER_SCOPE_RULE)
        self.assertIn("may be wrong", _SOLVER_SCOPE_RULE)

    def test_scope_rule_preserves_the_json_contract(self):
        # It is appended after the OUTPUT SCHEMA, so it must not invite prose.
        self.assertIn("JSON output schema above exactly", _SOLVER_SCOPE_RULE)

    def test_scope_rule_is_applied_to_every_solver_call(self):
        src = _function_body("resolve_doubt")
        self.assertIn("_build_solver_system_prompt(", src)
        for line in src.splitlines():
            if "solver_system = _build_solver_system_prompt(" in line:
                self.assertIn("_SOLVER_SCOPE_RULE", line)
                break
        else:  # pragma: no cover - guards against a silent rename
            self.fail("solver_system assignment not found in resolve_doubt")


class ProvidedSubjectMappingTests(SimpleTestCase):
    """
    When the doubt DOES carry a subject name, that mapping runs instead of the
    detector — and it matched the bare substring "science", so "Political
    Science" and "Computer Science" were both routed to biology.
    """

    def _map(self, provided_subject):
        """Mirror of the elif chain in resolve_doubt, read from the real source."""
        src = _function_body("resolve_doubt")
        start = src.index("provided_subject = ")
        block = src[start:src.index("# ── Step 2", start)]

        provided = provided_subject.strip().lower()
        # Each branch is `... for k in [...]): \n subject, qtype = "x", "y"`.
        pattern = re.compile(
            r"for k in \[([^\]]*)\]\s*\):\s*\n\s*subject, qtype = \"(\w+)\", \"(\w+)\"",
            re.S,
        )
        for keywords, subject, qtype in pattern.findall(block):
            for kw in re.findall(r'"([^"]+)"', keywords):
                if kw in provided:
                    return subject, qtype
        return None

    def test_political_science_is_social_science_not_biology(self):
        self.assertEqual(self._map("Political Science"), ("social_science", "conceptual"))

    def test_economics_is_social_science(self):
        self.assertEqual(self._map("Economics"), ("social_science", "conceptual"))

    def test_computer_science_is_not_biology(self):
        self.assertEqual(self._map("Computer Science"), ("computer", "conceptual"))

    def test_existing_mappings_are_unchanged(self):
        self.assertEqual(self._map("Physics"), ("physics", "numerical"))
        self.assertEqual(self._map("Chemistry"), ("chemistry", "conceptual"))
        self.assertEqual(self._map("Biology"), ("biology", "conceptual"))
        self.assertEqual(self._map("Mathematics"), ("math", "derivation"))
        self.assertEqual(self._map("English Literature"), ("english", "conceptual"))
        self.assertEqual(self._map("Social Science"), ("social_science", "conceptual"))
