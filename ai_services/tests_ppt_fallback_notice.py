"""A V2 or image deck that falls back to the old generator says so.

A teacher asked for image slides; Together rate-limited the planner and timed
it out, and the deck silently came from the V1 generator - it looked like the
"old PPT" and nothing explained why. These tests pin the honesty around the
fallback: the job records why, the activity feed says it, and the finished
deck carries generation.fallback for the studio to show.

No network: the generators are stubs.
"""
from unittest import mock

from django.test import SimpleTestCase
from rest_framework.response import Response

from ai_services.core.ppt_v2 import jobs
from ai_services.core.ppt_v2.planner import PlannerError
from ai_services.core.ppt_v2.service import V2Error
from ai_services.views import ppt


def v1_deck():
    return Response({"success": True, "data": {"title": "Reflection", "slides": [{"title": "A"}]}})


class MarkFallbackTests(SimpleTestCase):
    def test_a_stand_in_v1_deck_is_tagged_with_what_was_asked_for(self):
        data = ppt._mark_fallback(v1_deck(), ppt.IMAGE, "llm_error").data["data"]
        self.assertEqual(data["generation"], {
            "version": "v1", "requested": "image", "fallback": True,
            "fallbackReason": "llm_error", "fallbackWhy": "the AI service was busy"})

    def test_a_deck_in_the_requested_style_is_left_alone(self):
        image = Response({"success": True, "data": {"slides": [], "generation": {"version": "image"}}})
        self.assertNotIn("fallback", ppt._mark_fallback(image, ppt.IMAGE, None).data["data"]["generation"])

    def test_v1_requested_or_errors_are_not_tagged(self):
        self.assertNotIn("generation", ppt._mark_fallback(v1_deck(), "v1", None).data["data"])
        error = Response({"error": "x"}, status=502)
        self.assertEqual(ppt._mark_fallback(error, ppt.IMAGE, "llm_error").data, {"error": "x"})

    def test_an_unknown_reason_still_explains_itself(self):
        data = ppt._mark_fallback(v1_deck(), ppt.V2, None).data["data"]
        self.assertEqual(data["generation"]["fallbackReason"], "unavailable")
        self.assertEqual(data["generation"]["fallbackWhy"], "of an unexpected problem")


class RecordReasonTests(SimpleTestCase):
    def _progress(self):
        p = mock.Mock(spec=["__call__", "event"])
        return p

    def test_the_image_path_records_why_it_gave_up(self):
        progress = self._progress()
        with mock.patch("ai_services.core.ppt_v2.image_slides.generate_image_deck",
                        side_effect=PlannerError("llm_error", "Together request timed out")):
            self.assertIsNone(ppt._try_generate_image_deck(
                topic="T", slide_count=3, language="English", ctx={}, board="CBSE", passages=[],
                institute_id="i", vertical="school", progress=progress))
        self.assertEqual(progress.fallback_reason, "llm_error")

    def test_the_v2_path_records_why_it_gave_up(self):
        progress = self._progress()
        with mock.patch("ai_services.core.ppt_v2.service.generate_v2",
                        side_effect=V2Error("llm_error", "timed out")), \
             mock.patch.object(ppt, "_v2_fallback_allowed", return_value=True):
            self.assertIsNone(ppt._try_generate_v2(
                topic="T", slide_count=3, language="English", ctx={}, board="CBSE", passages=[],
                institute_id="i", vertical="school", progress=progress))
        self.assertEqual(progress.fallback_reason, "llm_error")

    def test_no_progress_object_is_fine(self):
        ppt._record_fallback(None, "llm_error")         # the one-request route


class FeedAndJobTests(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.object(jobs, "_redis", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        jobs._memory.clear()
        jobs._memory_running.clear()

    def _generate(self, progress):
        request = mock.Mock(data={"topic": "Reflection of Light", "pptVersion": "image",
                                  "chapterName": "Light"},
                            headers={}, institute_id="i", vertical="school", board="CBSE")

        def failed_image(**kw):
            ppt._record_fallback(kw.get("progress"), "llm_error")
        with mock.patch.object(ppt, "ppt_version", return_value=ppt.IMAGE), \
             mock.patch.object(ppt, "_try_generate_image_deck", side_effect=failed_image), \
             mock.patch.object(ppt, "_generate_grounded", return_value=(None, "unavailable")), \
             mock.patch.object(ppt, "get_llm", return_value=mock.Mock(complete=mock.Mock(side_effect=RuntimeError("stop")))):
            try:
                ppt._generate_presentation(request, progress=progress)
            except Exception:
                pass
        return request

    def test_the_feed_says_the_chosen_style_could_not_be_made(self):
        progress = mock.Mock(spec=["__call__", "event"])
        request = self._generate(progress)
        lines = {c.args[0]: c.args[1:] for c in progress.event.call_args_list}
        self.assertIn("fallback", lines)
        text, state = lines["fallback"]
        self.assertEqual(state, "warn")
        self.assertIn("Image slides could not be made this time, because the AI service was busy", text)
        self.assertEqual(request.ppt_version, ppt.IMAGE)

    def test_the_job_result_carries_the_fallback(self):
        import time

        def generate(snapshot, progress=None):
            snapshot.ppt_version = ppt.IMAGE
            ppt._record_fallback(progress, "llm_error")
            return v1_deck()
        # The deck cache is not what this tests; without a Redis it would also
        # spend a connection timeout (about 1s on Windows) inside the job.
        with mock.patch.object(ppt, "_generate_presentation", side_effect=generate), \
             mock.patch("ai_services.views.base.tenant_gate", return_value=None), \
             mock.patch("ai_services.core.ppt_v2.deck_cache._client", return_value=None):
            from rest_framework.test import APIRequestFactory
            req = APIRequestFactory().post("/ppt/generate/start", {"topic": "T"}, format="json")
            req.institute_id = "inst-1"
            job_id = ppt.generate_presentation_start(req).data["jobId"]
            for _ in range(1000):
                job = jobs.get(job_id)
                if job and job.get("status") == "done":
                    break
                time.sleep(0.01)
        generation = job["result"]["data"]["generation"]
        self.assertTrue(generation["fallback"])
        self.assertEqual(generation["requested"], "image")


def unpainted_deck():
    """What generate_image_deck returns when the image provider refused every slide."""
    slides = [{"slideNumber": i + 1, "type": t, "title": "Slide %d" % (i + 1), "subtitle": "",
               "bullets": ["Fact %d." % (i + 1)], "speakerNotes": "Notes",
               "slideImageError": "Together image API error 401"}
              for i, t in enumerate(("title", "content", "summary"))]
    return {"title": "Respiration", "slides": slides,
            "generation": {"version": "image", "plannerModel": "groq/gpt-oss-120b", "imageModel": "q",
                           "imagesGenerated": 0, "failedSlides": [1, 2, 3], "grounded": True}}


class NothingPaintedTests(SimpleTestCase):
    """The image provider is down: the planned, checked slides are still the deck.

    Seen with Together returning 401: the slides were planned and fact-checked
    on the fallback provider, no slide could be painted, and the request then
    started over on the standard writer - which failed too ("Failed to
    generate slide content").
    """

    def _run(self, deck, passages=()):
        progress = mock.Mock(spec=["__call__", "event"])
        with mock.patch("ai_services.core.ppt_v2.image_slides.generate_image_deck", return_value=deck), \
             mock.patch.object(ppt, "_fetch_image_for_slide",
                               side_effect=lambda term, title, ctx=None: {"imageUrl": "u/" + title}):
            response = ppt._try_generate_image_deck(
                topic="Respiration", slide_count=3, language="English",
                ctx={"chapterName": "Life Processes", "topicName": "Respiration"}, board="CBSE",
                passages=list(passages), institute_id="i", vertical="school", progress=progress)
        return response, progress

    def test_the_planned_slides_come_back_as_standard_slides(self):
        response, progress = self._run(unpainted_deck(), passages=[{"text": "p"}])
        data = response.data["data"]
        self.assertEqual([s["title"] for s in data["slides"]], ["Slide 1", "Slide 2", "Slide 3"])
        self.assertEqual([s["type"] for s in data["slides"]], ["title", "content", "summary"])
        self.assertEqual(data["slides"][1]["bullets"], ["Fact 2."])
        self.assertEqual(data["slides"][1]["imageUrl"], "u/Slide 2")
        self.assertTrue(all("slideImageError" not in s for s in data["slides"]))
        self.assertEqual(data["generation"]["version"], "v1")
        self.assertTrue(data["generation"]["fallback"])
        self.assertEqual(data["generation"]["requested"], "image")
        self.assertEqual(data["generation"]["fallbackWhy"], "no slide could be painted")
        self.assertTrue(data["source"]["grounded"])
        self.assertEqual(progress.fallback_reason, "no_images")

    def test_a_picture_search_failure_still_returns_the_deck(self):
        with mock.patch.object(ppt, "_fetch_image_for_slide", side_effect=RuntimeError("serper down")):
            data = ppt._unpainted_as_standard(unpainted_deck(), {})
        self.assertEqual(len(data["slides"]), 3)

    def test_the_stand_in_is_never_kept_for_identical_requests(self):
        from ai_services.core.ppt_v2 import deck_cache
        response, _ = self._run(unpainted_deck())
        self.assertFalse(deck_cache.complete(response.data["data"], "image"))

    def test_no_usable_slides_falls_back_to_the_standard_writer(self):
        deck = unpainted_deck()
        deck["slides"] = []
        response, _ = self._run(deck)
        self.assertIsNone(response)


class StandardWriterRetryTests(SimpleTestCase):
    """The last step must not fail over one unreadable reply."""

    def _generate(self, *replies):
        llm =mock.Mock(complete=mock.Mock(side_effect=list(replies)))
        request = mock.Mock(data={"topic": "Respiration", "slideCount": 3}, headers={},
                            institute_id="i", vertical="school", board="CBSE")
        with mock.patch.object(ppt, "ppt_version", return_value="v1"), \
             mock.patch.object(ppt, "get_llm", return_value=llm), \
             mock.patch.object(ppt, "_decompose_ppt_coverage", return_value=None), \
             mock.patch.object(ppt, "_fetch_image_for_slide", return_value={}), \
             mock.patch.object(ppt, "_log"):
            return ppt._generate_presentation(request), llm

    @staticmethod
    def _ok():
        return {"content": {"title": "R", "slides": [{"type": "title", "title": "R", "bullets": []}]},
                "model": "m", "usage": {}}

    def test_a_reply_that_failed_the_json_check_is_written_once_more(self):
        from ai_services.core.routing.errors import NonRetryableProviderError
        bad = NonRetryableProviderError("json_validate_failed", provider="groq", status_code=400,
                                        kind="structured_output")
        response, llm = self._generate(bad, self._ok())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.complete.call_count, 2)
        self.assertEqual(llm.complete.call_args.kwargs["temperature"], 0.3)

    def test_an_unreadable_deck_is_written_once_more(self):
        response, llm = self._generate({"content": "not json at all", "model": "m", "usage": {}}, self._ok())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.complete.call_count, 2)

    def test_an_outage_is_not_retried_here(self):
        from ai_services.core.routing.errors import RetryableProviderError
        response, llm = self._generate(RetryableProviderError("429", provider="groq", kind="rate_limited"))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(llm.complete.call_count, 1)

    def test_two_bad_replies_fail_plainly(self):
        from ai_services.core.routing.errors import NonRetryableProviderError
        bad = NonRetryableProviderError("bad", provider="groq", kind="structured_output")
        response, llm = self._generate(bad, bad)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(llm.complete.call_count, 2)
