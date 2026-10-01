"""A doubt may contain a false claim, and the answer must correct it, not dress it up.

Measured on 2026-09-30 against 18 false-premise questions from the CBSE Class 8
Science suite: 10 produced confident fabrications, including an invented
"BIS IS 1445:2009" calorific value, a "Bharat unit" of pressure attributed to the
NCERT textbook, a "Kutch Intensity Index" reading for the 2001 Bhuj earthquake,
and two answers that accepted a reversed premise ("rolling friction is greater
than sliding friction") and explained the opposite of the real physics.

Cause: _SOLVER_SCOPE_RULE, added so a misrouted subject never produced "I can
only help with physics", told the model that not answering is always failure.
It was applied to every doubt, so the model answered anything — including
questions whose premise was false — and invented the supporting detail.

The same run showed answers pitched years above the student: the class was sent
in studentContext but only the vision prompt ever used it.
"""
import io
import os

from django.test import SimpleTestCase

from ai_services.views.bridge import (
    _FALSE_PREMISE_RULE,
    _SOLVER_SCOPE_RULE,
    _class_level_rule,
)

SRC = io.open(os.path.join(os.path.dirname(__file__), "views", "bridge.py"), encoding="utf-8").read()


class FalsePremiseRuleTests(SimpleTestCase):
    def test_it_forbids_inventing_authority_and_numbers(self):
        rule = _FALSE_PREMISE_RULE.lower()
        for forbidden in ("standard number", "unit name", "official figure",
                          "percentage", "date", "organisation", "terminology"):
            self.assertIn(forbidden, rule, forbidden)
        self.assertIn("never invent", rule)

    def test_it_names_the_authorities_that_were_falsely_cited(self):
        rule = _FALSE_PREMISE_RULE.lower()
        for authority in ("ncert", "cbse", "textbook", "bis"):
            self.assertIn(authority, rule, authority)

    def test_it_covers_a_reversed_premise(self):
        rule = _FALSE_PREMISE_RULE.lower()
        self.assertIn("backwards", rule)
        self.assertIn("correct the direction", rule)
        self.assertIn("do not build an explanation on the reversed claim", rule)

    def test_it_tells_the_model_this_is_not_a_refusal(self):
        # Without this the new rule fights the scope rule and could bring back the
        # original bug: refusing an academic question outright.
        self.assertIn("NOT refusing", _FALSE_PREMISE_RULE)
        self.assertIn("Answer the corrected question fully", _FALSE_PREMISE_RULE)

    def test_the_scope_rule_no_longer_forbids_all_non_answers(self):
        # It must still stop SUBJECT refusals - that bug is not to be reintroduced.
        self.assertIn("NEVER say you can only help with one subject", _SOLVER_SCOPE_RULE)
        self.assertIn("subject refusal is a failed response", _SOLVER_SCOPE_RULE.lower())
        # ...but it must no longer say that any refusal whatsoever is failure.
        self.assertNotIn("NEVER refuse to answer an academic question. A refusal is a failed",
                         _SOLVER_SCOPE_RULE)
        self.assertIn("Correcting a wrong fact inside the question is not a refusal",
                      _SOLVER_SCOPE_RULE.replace("\n", " ").replace("  ", " "))


class PremiseCheckFieldTests(SimpleTestCase):
    """Prose alone did not work: re-running the 18 false-premise cases after the
    prose rule fixed only 1 of 10. The check is therefore a required schema field,
    which models comply with far more reliably than with prohibitions."""

    def _prompt(self, subject="biology", qtype="conceptual"):
        from ai_services.views.bridge import _build_solver_system_prompt

        return _build_solver_system_prompt(subject, qtype, "detailed", "school", "CBSE") + _FALSE_PREMISE_RULE

    def test_the_field_is_in_every_answer_schema(self):
        for qtype in ("conceptual", "numerical", "mcq"):
            self.assertIn('"premise_check"', self._prompt(qtype=qtype), qtype)

    def test_the_rule_defines_ok_and_the_alternative(self):
        rule = _FALSE_PREMISE_RULE
        self.assertIn('"premise_check"', rule)
        self.assertIn('exactly "ok"', rule)
        self.assertIn("FIRST line of the answer must state the correction", rule)

    def test_a_reported_problem_is_put_in_front_of_the_student(self):
        from ai_services.views.bridge import _surface_premise_warning

        brief = {"answer": "Urea was mandated in 1987.", "final_answer": "Urea",
                 "premise_check": "no such policy exists"}
        detailed = {"solution": "The policy...", "explanation": "The policy..."}
        self.assertTrue(_surface_premise_warning(brief, detailed, "q"))
        for text in (brief["answer"], brief["final_answer"],
                     detailed["solution"], detailed["explanation"]):
            self.assertTrue(text.startswith("**Note about the question:**"), text[:40])
        # the field itself must not leak into the rendered payload
        self.assertNotIn("premise_check", brief)
        self.assertNotIn("premise_check", detailed)

    def test_ok_leaves_the_answer_untouched(self):
        from ai_services.views.bridge import _surface_premise_warning

        for value in ("ok", "OK", "ok.", "none", "n/a", "", "   "):
            brief = {"answer": "Real answer.", "premise_check": value}
            detailed = {"solution": "Real solution."}
            self.assertFalse(_surface_premise_warning(brief, detailed, "q"), value)
            self.assertEqual(brief["answer"], "Real answer.")

    def test_a_missing_field_is_not_an_error(self):
        from ai_services.views.bridge import _surface_premise_warning

        brief, detailed = {"answer": "a"}, {"solution": "b"}
        self.assertFalse(_surface_premise_warning(brief, detailed, "q"))

    def test_the_note_is_not_stacked_twice(self):
        from ai_services.views.bridge import _surface_premise_warning

        brief = {"answer": "x", "premise_check": "bad premise"}
        detailed = {"solution": "y"}
        _surface_premise_warning(brief, detailed, "q")
        once = brief["answer"]
        brief["premise_check"] = "bad premise"
        _surface_premise_warning(brief, detailed, "q")
        self.assertEqual(brief["answer"], once)

    def test_resolve_doubt_calls_the_surfacer(self):
        self.assertIn("_surface_premise_warning(brief_obj, detailed_obj, question_text", SRC)


class ClassLevelRuleTests(SimpleTestCase):
    def test_the_class_is_named_in_the_instruction(self):
        rule = _class_level_rule("Class 8")
        self.assertIn("CLASS 8", rule)
        self.assertIn("The student is in Class 8", rule)

    def test_it_forbids_reaching_into_higher_classes(self):
        rule = _class_level_rule("Class 8").lower()
        self.assertIn("do not bring in material from higher classes", rule)
        self.assertIn("studied in higher classes", rule)

    def test_whitespace_in_the_class_name_is_tolerated(self):
        self.assertIn("CLASS 10", _class_level_rule("  Class 10  "))

    def test_it_is_applied_only_when_the_caller_supplies_a_class(self):
        body = SRC[SRC.find("def resolve_doubt"):]
        self.assertIn('_class_name = str(student_ctx.get("className")', body)
        self.assertIn("if _class_name:", body)
        self.assertIn("solver_system += _class_level_rule(_class_name)", body)
