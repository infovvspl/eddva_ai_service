"""Fact check against the school's book, before anything is built.

A live Class 10 deck said aerobic respiration releases "three molecules of
carbon dioxide for each molecule of glucose"; the book says three per
pyruvate. The scope check could not catch it - every word was in the book.
"""
import copy
import json
import os
import time
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import factcheck

BOOK = [{"page_no": 101, "content": "The breakdown of pyruvate using oxygen takes place in "
                                    "the mitochondria. It gives three molecules of carbon dioxide."}]
SLIDES = [
    {"title": "Aerobic Respiration", "blocks": [
        {"label": "Where", "text": "Pyruvate is broken down in the mitochondria."},
        {"label": "Products", "text": "Each glucose molecule gives three molecules of carbon dioxide."},
    ]},
]


class FakeLLM:
    def __init__(self, reply, raises=False):
        self.reply, self.raises, self.calls = reply, raises, []

    def complete(self, **kw):
        self.calls.append(kw)
        if self.raises:
            raise RuntimeError("provider down")
        return {"content": self.reply, "model": "stub"}


def run(reply, slides=None, **kw):
    slides = copy.deepcopy(slides or SLIDES)
    llm = kw.pop("llm", None) or FakeLLM(reply)
    fixes = factcheck.check_facts(slides, passages=kw.pop("passages", BOOK), llm=llm,
                                  topic="Respiration", **kw)
    return slides, fixes, llm


WRONG = {"issues": [{"id": "S1.2", "problem": "three CO2 per pyruvate, not glucose",
                     "fix": "Each pyruvate molecule gives three molecules of carbon dioxide."}]}


class FactCheckTests(SimpleTestCase):
    def test_a_statement_that_contradicts_the_book_is_corrected(self):
        slides, fixes, llm = run(WRONG)
        self.assertEqual(slides[0]["blocks"][1]["text"],
                         "Each pyruvate molecule gives three molecules of carbon dioxide.")
        self.assertEqual(slides[0]["blocks"][0]["text"],
                         "Pyruvate is broken down in the mitochondria.")
        self.assertEqual([(f["slide"], f["id"]) for f in fixes], [(1, "S1.2")])

    def test_the_checker_sees_the_book_and_every_statement(self):
        _slides, _fixes, llm = run({"issues": []})
        prompt = llm.calls[0]["user_prompt"]
        self.assertIn("three molecules of carbon dioxide", prompt)
        self.assertIn("S1.1: Pyruvate is broken down in the mitochondria.", prompt)
        self.assertIn("S1.2:", prompt)

    def test_a_reply_as_a_json_string_is_read_too(self):
        slides, fixes, _ = run(json.dumps(WRONG))
        self.assertEqual(len(fixes), 1)

    def test_an_invented_id_changes_nothing(self):
        slides, fixes, _ = run({"issues": [{"id": "S9.9", "fix": "Something else entirely."}]})
        self.assertEqual(slides, SLIDES)
        self.assertEqual(fixes, [])

    def test_a_fix_that_is_an_essay_or_no_change_is_not_applied(self):
        essay = {"issues": [{"id": "S1.2", "fix": "word " * 200}]}
        same = {"issues": [{"id": "S1.2", "fix": SLIDES[0]["blocks"][1]["text"]}]}
        for reply in (essay, same):
            slides, fixes, _ = run(reply)
            self.assertEqual(slides, SLIDES)
            self.assertEqual(fixes, [])

    def test_a_failed_or_garbled_check_leaves_the_slides_as_planned(self):
        for llm in (FakeLLM(None, raises=True), FakeLLM("{not json")):
            with mock.patch.object(factcheck, "RETRY_PAUSE_S", 0):
                slides, fixes, _ = run(None, llm=llm)
            self.assertEqual(slides, SLIDES)
            self.assertEqual(fixes, [])

    def test_a_provider_blip_is_retried_once(self):
        class Blip(FakeLLM):
            def complete(self, **kw):
                self.calls.append(kw)
                if len(self.calls) == 1:
                    raise RuntimeError("Together 503 Service unavailable")
                return {"content": WRONG}
        with mock.patch.object(factcheck, "RETRY_PAUSE_S", 0):
            slides, fixes, llm = run(None, llm=Blip(None))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(fixes), 1)

    def test_a_check_that_could_not_run_never_claims_the_facts_match(self):
        # A live run met three 503s and still reported "Every fact matches".
        feed = []
        with mock.patch.object(factcheck, "RETRY_PAUSE_S", 0):
            run(None, llm=FakeLLM(None, raises=True), on_event=lambda *a: feed.append(a))
        self.assertEqual(feed[-1][2], "warn")
        self.assertIn("Couldn't complete the fact check", feed[-1][1])
        self.assertNotIn("matches", feed[-1][1])

    def test_without_a_book_there_is_nothing_to_check_and_no_call(self):
        slides, fixes, llm = run(WRONG, passages=None)
        self.assertEqual(llm.calls, [])
        self.assertEqual(slides, SLIDES)

    def test_no_check_is_started_when_the_deck_is_out_of_time(self):
        _s, _f, llm = run(WRONG, started=time.perf_counter() - factcheck.CHECK_WINDOW_S - 1)
        self.assertEqual(llm.calls, [])

    def test_it_can_be_switched_off(self):
        with mock.patch.dict(os.environ, {"PPT_FACT_CHECK": "false"}):
            _s, _f, llm = run(WRONG)
        self.assertEqual(llm.calls, [])

    def test_large_decks_are_checked_in_parallel_batches(self):
        deck = [copy.deepcopy(SLIDES[0]) for _ in range(12)]
        _s, _f, llm = run({"issues": []}, slides=deck)
        self.assertEqual(len(llm.calls), 3)          # 12 slides, 5 per batch

    def test_the_teacher_is_told_what_was_corrected(self):
        feed = []
        run(WRONG, on_event=lambda *a: feed.append(a))
        self.assertEqual(feed[0], ("facts", "Checking every fact against your textbook", "active"))
        self.assertEqual(feed[-1][2], "done")
        self.assertIn("Corrected 1 statement on slide 1", feed[-1][1])


class PipelineTests(SimpleTestCase):
    def test_v2_builds_the_corrected_statement(self):
        from ai_services import tests_ppt_v2_service as svc

        class Planner(svc._StubLLM):
            def complete(self, **kw):
                self.calls += 1
                if kw.get("feature") == "ppt_v2_factcheck":
                    first = svc.PHOTOSYNTHESIS["slides"][0]["blocks"][0]["text"]
                    return {"content": {"issues": [{
                        "id": "S1.1", "problem": "x",
                        "fix": "Leaves trap sunlight using the green pigment chlorophyll."}]},
                        "model": "stub"}
                return super().complete(**kw)

        result = svc.run(llm=Planner(), passages=[{"page_no": 1, "content": "Leaves trap sunlight."}])
        texts = [b.text for s in result.slides for b in s.content_blocks]
        self.assertIn("Leaves trap sunlight using the green pigment chlorophyll.", texts)
