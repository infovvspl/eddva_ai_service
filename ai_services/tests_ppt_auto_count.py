"""Automatic slide count: the chapter decides how many slides it needs.

A teacher should not have to guess the length of a deck. With slideCount
"auto" the outline reads the source and gives one slide per distinct idea,
inside a range sized to how much the book actually says.
"""
import re
import threading
from unittest import mock

from django.test import SimpleTestCase
from rest_framework.response import Response

from ai_services.core.ppt_v2 import planner
from ai_services.core.ppt_v2.planner import auto_slide_range, plan_deck


def book(words):
    return [{"page_no": 10 + i, "content": " ".join(["photosynthesis"] * 100)}
            for i in range(max(1, words // 100))]


class RangeTests(SimpleTestCase):
    def test_without_a_book_a_normal_lesson_size(self):
        self.assertEqual(auto_slide_range(None, "T", "C"), planner.AUTO_NO_BOOK_RANGE)

    def test_a_topic_is_narrower_than_its_chapter(self):
        self.assertEqual(auto_slide_range(book(3000), "T", "C", topic_only=True),
                         planner.AUTO_TOPIC_RANGE)

    def test_more_book_means_more_slides_within_limits(self):
        short = auto_slide_range(book(300), "photosynthesis", "C")
        long = auto_slide_range(book(4000), "photosynthesis", "C")
        self.assertLess(short[1], long[1])
        for lo, hi in (short, long):
            self.assertGreaterEqual(lo, planner.AUTO_MIN_SLIDES)
            self.assertLessEqual(hi, planner.AUTO_MAX_SLIDES)
            self.assertLess(lo, hi)


class Chooser:
    """Outlines `chosen` slides when allowed to choose; fills batches as asked."""

    def __init__(self, chosen):
        self.chosen, self.calls, self.lock = chosen, [], threading.Lock()

    def complete(self, **kw):
        with self.lock:
            self.calls.append(kw)
        if "You outline educational slide decks" in kw["system_prompt"]:
            return {"content": {"title": "Deck", "slides": [
                {"slideType": "concept", "title": "Idea %d" % (i + 1), "focus": "f"}
                for i in range(self.chosen)]}}
        n = int(re.search(r"Return exactly (\d+) slides", kw["user_prompt"]).group(1))
        first = re.search(r"writing slides (\d+)-", kw["user_prompt"])
        base = int(first.group(1)) if first else 1
        return {"content": {"title": "", "slides": [
            {"slideType": "concept", "title": "Slide %d" % (base + i),
             "blocks": [{"kind": "bullet", "label": "x", "text": "A fact."}],
             "visual": {"kind": "none"}} for i in range(n)]}}


class AutoPlanTests(SimpleTestCase):
    def test_the_outline_chooses_the_count_and_the_deck_has_exactly_that_many(self):
        llm = Chooser(chosen=8)
        result = plan_deck(topic="T", slide_count=8, llm=llm, auto_range=(6, 12))
        self.assertEqual(len(result.slides), 8)
        outline_prompt = llm.calls[0]
        self.assertIn("Decide how many slides this content needs, between 6 and 12",
                      outline_prompt["system_prompt"])
        self.assertIn("Return between 6 and 12 outline entries", outline_prompt["user_prompt"])

    def test_an_outline_past_the_ceiling_is_held_to_it(self):
        result = plan_deck(topic="T", slide_count=8, llm=Chooser(chosen=20), auto_range=(6, 12))
        self.assertEqual(len(result.slides), 12)

    def test_an_outline_under_the_floor_falls_back_to_the_target(self):
        llm = Chooser(chosen=2)
        result = plan_deck(topic="T", slide_count=8, llm=llm, auto_range=(6, 12))
        self.assertEqual(len(result.slides), 8)          # the one-call target

    def test_the_teacher_is_told_how_many_were_chosen(self):
        feed = []
        plan_deck(topic="T", slide_count=8, llm=Chooser(chosen=7), auto_range=(6, 12),
                  on_event=lambda *a: feed.append(a))
        outline = [t for k, t, s in feed if k == "outline" and s == "done"]
        self.assertTrue(outline[0].startswith("Chose 7 slides"))


class EndpointTests(SimpleTestCase):
    def _run(self, data):
        from ai_services.views import ppt
        seen = {}

        def fake_v2(**kw):
            seen.update(kw)
            return Response({"success": True, "data": {}})
        request = mock.Mock(data=data, headers={}, institute_id="i", vertical="school",
                            board="CBSE")
        with mock.patch.object(ppt, "ppt_version", return_value=ppt.V2), \
             mock.patch.object(ppt, "_try_generate_v2", side_effect=fake_v2):
            ppt._generate_presentation(request)
        return seen

    def test_auto_reaches_the_generator_as_a_range(self):
        seen = self._run({"topic": "Photosynthesis", "slideCount": "auto",
                          "chapterName": "Photosynthesis",
                          "sourcePassages": book(2000)})
        lo, hi = seen["auto_range"]
        self.assertLess(lo, hi)
        self.assertEqual(seen["slide_count"], (lo + hi) // 2)

    def test_a_topic_inside_a_chapter_gets_the_topic_range(self):
        seen = self._run({"topic": "Stomata", "slideCount": "auto", "topicName": "Stomata",
                          "chapterName": "Photosynthesis", "sourcePassages": book(2000)})
        self.assertEqual(seen["auto_range"], planner.AUTO_TOPIC_RANGE)

    def test_a_number_and_no_number_behave_exactly_as_before(self):
        self.assertIsNone(self._run({"topic": "T", "slideCount": 7})["auto_range"])
        self.assertEqual(self._run({"topic": "T", "slideCount": 7})["slide_count"], 7)
        self.assertEqual(self._run({"topic": "T"})["slide_count"], 5)
