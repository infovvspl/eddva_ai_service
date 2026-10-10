"""Deck scope: a deck asked for one topic stays on that topic, and says so.

A teacher reported decks for one topic coming out about another. A topic deck
is given its whole chapter as the source (the book is indexed by chapter) and
was told to follow the book's order, with the topic named only as "a lesson
on" it - nothing stopped it drifting into the chapter's other topics. These
tests pin the fix and the visibility around it:

  * a topic deck carries an explicit FOCUS rule in every planner call (the
    outline, the parallel batches, the one-call path and slide rewrites);
  * a chapter deck, or a topic named like its chapter, carries none;
  * the job records the scope it is generating for, from the start, and the
    activity feed opens with it, so the studio can show it while loading.

No network: the LLM is a stub that records the prompts it is given.
"""
from unittest import mock

from django.test import SimpleTestCase
from rest_framework.response import Response
from rest_framework.test import APIRequestFactory

from ai_services.core.ppt_v2 import jobs, planner
from ai_services.core.ppt_v2.planner import (
    PlannerError,
    build_outline_prompts,
    build_planner_prompts,
    focus_topic_for,
    plan_deck,
    revise_slides,
)

TOPIC_CTX = {"className": "Class 10", "subjectName": "Science",
             "chapterName": "Life Processes", "topicName": "Respiration"}
CHAPTER_CTX = {"className": "Class 10", "subjectName": "Science",
               "chapterName": "Life Processes", "topicName": ""}
BOOK = [{"page_no": 95, "content": "Nutrition: autotrophs make their own food."},
        {"page_no": 100, "content": "Respiration: glucose is broken down to release energy."},
        {"page_no": 105, "content": "Transportation: blood carries oxygen and food."}]


class Recorder:
    """An LLM that records every prompt and then fails, so nothing is planned."""

    def __init__(self):
        self.prompts = []

    def complete(self, system_prompt="", user_prompt="", **kw):
        self.prompts.append(user_prompt)
        raise RuntimeError("401 stop here")      # never retried


class FocusTopicTests(SimpleTestCase):
    def test_a_topic_inside_a_chapter_is_a_focus(self):
        self.assertEqual(focus_topic_for(TOPIC_CTX), "Respiration")

    def test_a_chapter_deck_has_no_focus(self):
        self.assertEqual(focus_topic_for(CHAPTER_CTX), "")
        self.assertEqual(focus_topic_for({}), "")
        self.assertEqual(focus_topic_for(None), "")

    def test_a_topic_named_like_its_chapter_is_the_chapter(self):
        self.assertEqual(focus_topic_for({"chapterName": "Light", "topicName": " light "}), "")


class PromptTests(SimpleTestCase):
    def test_the_planner_is_told_to_stay_inside_the_topic(self):
        _s, user = build_planner_prompts(topic="Respiration", slide_count=6,
                                         chapter="Life Processes", passages=BOOK,
                                         focus_topic="Respiration")
        self.assertIn('FOCUS: this deck covers ONLY the topic "Respiration"', user)
        self.assertIn('the chapter "Life Processes"', user)
        self.assertIn("do not make slides about the chapter's other topics", user)
        # Before the source, so the model reads the boundary first.
        self.assertLess(user.index("FOCUS:"), user.index("SOURCE MATERIAL"))

    def test_a_chapter_deck_prompt_is_unchanged(self):
        with_none = build_planner_prompts(topic="Life Processes", slide_count=6,
                                          chapter="Life Processes", passages=BOOK)
        with_empty = build_planner_prompts(topic="Life Processes", slide_count=6,
                                           chapter="Life Processes", passages=BOOK,
                                           focus_topic="")
        self.assertEqual(with_none, with_empty)
        self.assertNotIn("FOCUS:", with_none[1])

    def test_the_outline_carries_the_focus_too(self):
        for slide_range in (None, (5, 10)):
            _s, user = build_outline_prompts(topic="Respiration", slide_count=6,
                                             chapter="Life Processes", passages=BOOK,
                                             focus_topic="Respiration", slide_range=slide_range)
            self.assertIn("covers ONLY the topic \"Respiration\"", user, slide_range)

    def test_a_slide_rewrite_keeps_the_focus(self):
        llm = Recorder()
        with self.assertRaises(PlannerError):
            revise_slides(slides=[{"title": "x"}], note="fix it", topic="Respiration",
                          chapter="Life Processes", passages=BOOK, llm=llm,
                          focus_topic="Respiration")
        self.assertIn("covers ONLY the topic", llm.prompts[0])
        self.assertIn("focus_topic", planner.revise_args({"focus_topic": "Respiration"}))


class PlanDeckTests(SimpleTestCase):
    def _prompts(self, **kw):
        llm = Recorder()
        with self.assertRaises(PlannerError):
            plan_deck(topic="Respiration", chapter="Life Processes", passages=BOOK,
                      llm=llm, focus_topic="Respiration", **kw)
        return llm.prompts

    def test_every_planning_path_carries_the_focus(self):
        with mock.patch.object(planner, "parallel_enabled", return_value=True):
            parallel = self._prompts(slide_count=8)                 # outline call
            auto = self._prompts(slide_count=8, auto_range=(5, 10))
        with mock.patch.object(planner, "parallel_enabled", return_value=False):
            single = self._prompts(slide_count=3)
        for name, prompts in (("parallel", parallel), ("auto", auto), ("single", single)):
            self.assertTrue(prompts, name)
            self.assertTrue(all("covers ONLY the topic" in p for p in prompts), name)


class GeneratorWiringTests(SimpleTestCase):
    def test_v2_and_image_slides_plan_with_the_topic_focus(self):
        from ai_services.core.ppt_v2 import image_slides, service
        for ctx, expected in ((TOPIC_CTX, "Respiration"), (CHAPTER_CTX, "")):
            seen = {}

            def capture(**kw):
                seen.update(kw)
                raise PlannerError("llm_error", "stop")
            with mock.patch.object(service, "plan_deck", side_effect=capture):
                with self.assertRaises(Exception):
                    service.generate_v2(topic="T", slide_count=3, ctx=dict(ctx))
            self.assertEqual(seen.get("focus_topic"), expected, "v2 %s" % ctx)

            seen.clear()
            with mock.patch("ai_services.core.ppt_v2.planner.plan_deck", side_effect=capture):
                with self.assertRaises(PlannerError):
                    image_slides.generate_image_deck(topic="T", slide_count=3, ctx=dict(ctx),
                                                     render=lambda *a, **k: {})
            self.assertEqual(seen.get("focus_topic"), expected, "image %s" % ctx)


class ScopeReportTests(SimpleTestCase):
    def test_the_scope_says_what_kind_of_deck_it_is(self):
        from ai_services.views.ppt import _deck_scope
        self.assertEqual(_deck_scope(dict(TOPIC_CTX, topic="Respiration"))["kind"], "topic")
        self.assertEqual(_deck_scope(dict(CHAPTER_CTX, topic="Life Processes"))["kind"], "chapter")
        self.assertEqual(_deck_scope({"topic": "Volcanoes", "subjectName": "Geography"})["kind"],
                         "subject")
        self.assertEqual(_deck_scope({"topic": "Volcanoes"})["kind"], "free")
        # Malformed fields degrade to empty, never raise.
        self.assertEqual(_deck_scope({"topic": None, "chapterName": 7})["kind"], "free")

    def test_the_first_activity_line_names_the_deck(self):
        from ai_services.views.ppt import _deck_scope, _scope_line
        self.assertEqual(
            _scope_line(_deck_scope(dict(TOPIC_CTX, topic="Respiration"))),
            'Creating a presentation on the topic "Respiration" '
            "(Life Processes · Science · Class 10)")
        self.assertEqual(
            _scope_line(_deck_scope(dict(CHAPTER_CTX, topic="Life Processes"))),
            'Creating a presentation on the chapter "Life Processes" (Science · Class 10)')
        self.assertEqual(_scope_line(_deck_scope({"topic": "Volcanoes"})),
                         'Creating a presentation on "Volcanoes"')


class JobScopeTests(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.object(jobs, "_redis", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        jobs._memory.clear()
        jobs._memory_running.clear()
        p = mock.patch("ai_services.views.base.tenant_gate", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_every_status_carries_the_scope_from_the_first_poll(self):
        from ai_services.views import ppt
        import threading
        release = threading.Event()

        def slow(snapshot, progress=None):
            release.wait(2)
            return Response({"success": True, "data": {}})
        factory = APIRequestFactory()
        request = factory.post("/ppt/generate/start", dict(TOPIC_CTX, topic="Respiration"),
                               format="json")
        request.institute_id = "inst-1"
        with mock.patch.object(ppt, "_generate_presentation", side_effect=slow):
            job_id = ppt.generate_presentation_start(request).data["jobId"]
            poll = factory.get("/ppt/generate/status", {"job_id": job_id})
            poll.institute_id = "inst-1"
            status = ppt.generate_presentation_status(poll)
            release.set()
        self.assertEqual(status.data["status"], "running")
        self.assertEqual(status.data["scope"]["kind"], "topic")
        self.assertEqual(status.data["scope"]["topicName"], "Respiration")
        self.assertEqual(status.data["scope"]["chapterName"], "Life Processes")

    def test_the_feed_opens_with_the_scope(self):
        from ai_services.views import ppt
        progress = mock.Mock()
        request = mock.Mock(data=dict(TOPIC_CTX, topic="Respiration"), headers={},
                            institute_id="i", vertical="school", board="CBSE")
        with mock.patch.object(ppt, "ppt_version", return_value=ppt.V2), \
             mock.patch.object(ppt, "_try_generate_v2",
                               return_value=Response({"success": True, "data": {}})):
            ppt._generate_presentation(request, progress=progress)
        first = progress.event.call_args_list[0].args
        self.assertEqual(first[0], "scope")
        self.assertIn('the topic "Respiration"', first[1])
        self.assertEqual(first[2], "done")

    def test_no_scope_is_flagged_in_the_feed(self):
        from ai_services.views import ppt
        progress = mock.Mock()
        request = mock.Mock(data={"topic": "Volcanoes"}, headers={},
                            institute_id="i", vertical="school", board="CBSE")
        with mock.patch.object(ppt, "ppt_version", return_value=ppt.V2), \
             mock.patch.object(ppt, "_try_generate_v2",
                               return_value=Response({"success": True, "data": {}})):
            ppt._generate_presentation(request, progress=progress)
        self.assertEqual(progress.event.call_args_list[0].args[2], "warn")
