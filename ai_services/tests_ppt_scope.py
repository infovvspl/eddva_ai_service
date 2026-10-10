"""Right first time: the plan is checked against the chapter before anything
is built, and a plan that strays is corrected once instead of being shipped.

A live Class 10 Respiration deck put "Krebs cycle, NADH, FADH2" on screen. The
gate noticed and logged it - after the pictures were bought - and shipped the
deck anyway, so the only fix left was the teacher generating it again.
"""
import copy
import time

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import scope
from ai_services.core.ppt_v2.planner import PlannerResult

KREBS = [{"title": "Aerobic Respiration", "blocks": [
    {"label": "Krebs Cycle", "text": "Pyruvate enters the Krebs cycle, making NADH."}]}]
CLEAN = [{"title": "Aerobic Respiration", "blocks": [
    {"label": "In mitochondria", "text": "Pyruvate is broken down to release energy."}]}]
BOOK = [{"page_no": 103, "content": "In mitochondria pyruvate is broken down using oxygen."}]
BOOK_11 = [{"page_no": 220, "content": "The Krebs cycle oxidises acetyl CoA in the matrix."}]


def plan(slides, grounded=False):
    return PlannerResult(title="T", slides=copy.deepcopy(slides), grounded=grounded)


class OffScopeTests(SimpleTestCase):
    def test_a_term_the_book_does_not_use_is_off_scope(self):
        self.assertEqual(scope.off_scope_terms(KREBS, passages=BOOK, class_name="Class 10"),
                         ["Krebs"])

    def test_a_term_the_book_does_use_is_in_scope_whatever_the_list_says(self):
        # Class 11 biology teaches the Krebs cycle, and its book says so.
        self.assertEqual(scope.off_scope_terms(KREBS, passages=BOOK_11, class_name="Class 11"), [])

    def test_without_a_book_the_list_applies_only_up_to_class_10(self):
        self.assertEqual(scope.off_scope_terms(KREBS, class_name="Class 10"), ["Krebs"])
        self.assertEqual(scope.off_scope_terms(KREBS, class_name="Class 11"), [])
        self.assertEqual(scope.off_scope_terms(KREBS, class_name=""), [])

    def test_a_clean_plan_has_nothing_to_correct(self):
        self.assertEqual(scope.off_scope_terms(CLEAN, passages=BOOK, class_name="Class 10"), [])

    def test_the_note_names_the_terms_and_the_reason(self):
        note = scope.revision_note(["Krebs"], class_name="Class 10", grounded=True)
        self.assertIn("Krebs", note)
        self.assertIn("SOURCE MATERIAL", note)


class RepairTests(SimpleTestCase):
    """Only the slides that strayed are rewritten - a whole re-plan cost 68s
    on a live deck to remove one word from one slide."""

    DECK = [CLEAN[0], KREBS[0], CLEAN[0]]

    def _run(self, rewrite=None, raises=False, started=None, deck=None):
        calls = []

        def revise(note, bad):
            calls.append((note, bad))
            if raises:
                raise RuntimeError("planner down")
            return rewrite if rewrite is not None else [CLEAN[0]] * len(bad)
        out = scope.replan_if_off_scope(
            plan(deck or self.DECK, grounded=True), revise=revise, passages=BOOK,
            class_name="Class 10", started=started or time.perf_counter())
        return out, calls

    def test_only_the_straying_slide_is_sent_and_replaced(self):
        out, calls = self._run()
        self.assertEqual(len(calls), 1)
        note, bad = calls[0]
        self.assertEqual(bad, [KREBS[0]])
        self.assertIn("Krebs", note)
        self.assertEqual(out.slides, [CLEAN[0], CLEAN[0], CLEAN[0]])

    def test_a_clean_plan_costs_no_call(self):
        out, calls = self._run(deck=[CLEAN[0]])
        self.assertEqual(calls, [])

    def test_a_rewrite_that_is_no_better_is_not_used(self):
        out, _ = self._run(rewrite=[KREBS[0]])
        self.assertEqual(out.slides, self.DECK)

    def test_a_rewrite_of_the_wrong_length_is_not_used(self):
        out, _ = self._run(rewrite=[CLEAN[0], CLEAN[0]])
        self.assertEqual(out.slides, self.DECK)

    def test_a_failed_rewrite_keeps_the_plan(self):
        out, _ = self._run(raises=True)
        self.assertEqual(out.slides, self.DECK)

    def test_no_rewrite_when_there_is_no_time_left(self):
        out, calls = self._run(started=time.perf_counter() - scope.REPLAN_WINDOW_S - 1)
        self.assertEqual(calls, [])
        self.assertEqual(out.slides, self.DECK)


class PipelineTests(SimpleTestCase):
    """generate_v2 end to end with stubbed providers: the corrected slide is the
    one that gets built, and only it was sent back."""

    def test_v2_builds_the_corrected_slide(self):
        from ai_services import tests_ppt_v2_service as svc

        straying = copy.deepcopy(svc.PHOTOSYNTHESIS)
        straying["slides"][0]["blocks"][0]["text"] = (
            "The Calvin cycle fixes carbon dioxide in the thylakoid stroma.")
        fixed = {"title": "", "slides": [copy.deepcopy(svc.PHOTOSYNTHESIS["slides"][0])]}

        class Sequence(svc._StubLLM):
            def __init__(self, payloads):
                super().__init__()
                self.payloads, self.prompts = payloads, []

            def complete(self, **kwargs):
                self.calls += 1
                self.prompts.append(kwargs["user_prompt"])
                return {"content": self.payloads[min(self.calls, len(self.payloads)) - 1],
                        "model": "stub-model"}

        llm = Sequence([straying, fixed])
        result = svc.run(llm=llm)
        self.assertEqual(llm.calls, 2)
        self.assertIn("REVISION", llm.prompts[1])
        self.assertIn("Rewrite ONLY these slides", llm.prompts[1])
        text = " ".join(b.text for s in result.slides for b in s.content_blocks)
        self.assertNotIn("Calvin", text)
        self.assertEqual(len(result.slides), 2)
