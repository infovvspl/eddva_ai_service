"""A student's name must not reach a model provider, and must still reach the student."""
from django.test import TestCase

from ai_services.core.pii import (
    STUDENT_TOKEN,
    contains_token,
    needs_pseudonym,
    restore_name,
    to_prompt_name,
)


class PseudonymTests(TestCase):
    def test_real_name_is_replaced_in_the_prompt(self):
        self.assertEqual(to_prompt_name("Ananya Sharma"), STUDENT_TOKEN)

    def test_generic_placeholders_are_left_alone(self):
        """No identity to protect, so the prompt keeps its natural wording."""
        for value in ("Student", "student", "  STUDENT  ", "", None, "unknown", "N/A"):
            self.assertEqual(to_prompt_name(value), "Student", f"failed for {value!r}")
            self.assertFalse(needs_pseudonym(value))

    def test_a_real_name_is_recognised(self):
        self.assertTrue(needs_pseudonym("Ravi"))
        self.assertTrue(needs_pseudonym("Student Kumar"))  # not the bare placeholder


class RestoreTests(TestCase):
    def test_round_trip_through_a_string(self):
        out = restore_name(f"{STUDENT_TOKEN} should focus on Physics.", "Ananya")
        self.assertEqual(out, "Ananya should focus on Physics.")

    def test_restores_every_occurrence(self):
        text = f"{STUDENT_TOKEN} is strong in Maths. {STUDENT_TOKEN} should keep it up."
        self.assertEqual(restore_name(text, "Ravi").count("Ravi"), 2)

    def test_walks_nested_report_structure(self):
        """The career report is a nested object; the name can appear anywhere in it."""
        report = {
            "summary": f"{STUDENT_TOKEN} shows a strong analytical profile.",
            "sections": [
                {"title": "Strengths", "body": f"{STUDENT_TOKEN} scores well in Science."},
                {"title": "Next steps", "body": "Practise more numericals."},
            ],
            "meta": {"grade": 10, "confidence": 0.8},
        }
        out = restore_name(report, "Meera")
        self.assertIn("Meera", out["summary"])
        self.assertIn("Meera", out["sections"][0]["body"])
        self.assertEqual(out["sections"][1]["body"], "Practise more numericals.")
        self.assertEqual(out["meta"], {"grade": 10, "confidence": 0.8})
        self.assertFalse(contains_token(out))

    def test_tolerates_the_model_reformatting_the_sentinel(self):
        """Models sometimes drop an underscore or change case; debris in a
        student-facing report is worse than a slightly loose match."""
        for variant in ("__STUDENT__", "_STUDENT_", "__student__",
                        "__ STUDENT __", "___Student___"):
            self.assertEqual(
                restore_name(f"{variant} did well.", "Arjun"),
                "Arjun did well.",
                f"failed for {variant!r}",
            )

    def test_no_real_name_means_no_substitution(self):
        text = f"{STUDENT_TOKEN} did well."
        self.assertEqual(restore_name(text, "Student"), text)
        self.assertEqual(restore_name(text, ""), text)
        self.assertEqual(restore_name(text, None), text)

    def test_output_without_the_sentinel_is_untouched(self):
        """The model is free to write "the student" instead; nothing to restore."""
        text = "The student should consider engineering."
        self.assertEqual(restore_name(text, "Ananya"), text)

    def test_non_string_leaves_survive(self):
        report = {"scores": [1, 2.5, None, True], "ok": False}
        self.assertEqual(restore_name(report, "Ravi"), report)


class ExposureTests(TestCase):
    def test_the_prompt_value_never_contains_the_real_name(self):
        """The guarantee this module exists to provide."""
        for name in ("Ananya Sharma", "Ravi", "Zoya Khan", "A"):
            self.assertNotIn(name, to_prompt_name(name))

    def test_failure_mode_is_visible_not_wrong(self):
        """If restoration is skipped, the reader sees an obvious placeholder —
        never another child's name. That is why the sentinel is not name-like."""
        self.assertNotRegex(STUDENT_TOKEN, r"^[A-Z][a-z]+$")
        self.assertIn("_", STUDENT_TOKEN)


class CareerEndpointWiringTests(TestCase):
    """Guards the wiring itself, so a later edit cannot quietly reintroduce the
    real name. Reads the module source rather than inspect.getsource() on the
    view, because @api_view replaces the function with a DRF wrapper."""

    @staticmethod
    def _source():
        from ai_services.views import career
        return open(career.__file__, encoding="utf-8").read()

    def test_career_view_pseudonymises_and_restores(self):
        src = self._source()
        self.assertIn("prompt_name = to_prompt_name(student_name)", src)
        self.assertIn("student_name=prompt_name", src)
        self.assertIn("restore_name(report, student_name)", src)
        self.assertNotIn("student_name=student_name", src,
                         "the real name is being passed into the prompt")

    def test_student_name_is_not_used_as_a_usage_log_id(self):
        self.assertNotIn("data.get('studentName') or ''", self._source(),
                         "studentName is leaking into the usage log as user_id")
