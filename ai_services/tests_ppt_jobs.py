"""Background deck generation: start now, poll, see the deck as it is built.

Held open as one request, a deck ran into every timeout on the way (the
production web worker kills a request at 120s, the backend stops at 240s) and
the teacher watched a bare spinner. These tests pin the job contract.
"""
import time
from unittest import mock

from django.test import SimpleTestCase
from rest_framework.response import Response
from rest_framework.test import APIRequestFactory

from ai_services.core.ppt_v2 import jobs
# Imported before any test patches planner.plan_deck: service binds plan_deck at
# import, and a first import inside a patch would keep the stub for every later
# test (the image-slide test below imports it through generate_image_deck).
from ai_services.core.ppt_v2 import service as _service  # noqa: F401


def wait_for(job_id, status=("done", "failed"), timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = jobs.get(job_id)
        if job and job.get("status") in status:
            return job
        time.sleep(0.02)
    raise AssertionError("job %s did not finish" % job_id)


class MemoryStore(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.object(jobs, "_redis", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        jobs._memory.clear()
        jobs._memory_running.clear()


class JobStoreTests(MemoryStore):
    def test_a_job_runs_in_the_background_and_keeps_its_result(self):
        def run(progress):
            progress("planning")
            progress("pictures", done=1, total=2, partial={"slides": [{"title": "A"}]})
            return Response({"success": True, "data": {"title": "Deck"}})
        job_id = jobs.start(run, institute_id="inst-1", meta={"sourceMode": "ebook"})
        job = wait_for(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"], {"success": True, "data": {"title": "Deck"}})
        self.assertEqual(job["partial"], {"slides": [{"title": "A"}]})
        self.assertEqual(job["meta"], {"sourceMode": "ebook"})

    def test_an_error_response_is_a_failed_job_with_its_message(self):
        job_id = jobs.start(lambda p: Response({"error": "no topic"}, status=400),
                            institute_id="inst-1")
        job = wait_for(job_id)
        self.assertEqual((job["status"], job["httpStatus"], job["error"]),
                         ("failed", 400, "no topic"))

    def test_an_exception_is_a_failed_job_not_a_lost_one(self):
        def run(progress):
            raise RuntimeError("boom")
        job = wait_for(jobs.start(run, institute_id="inst-1"))
        self.assertEqual(job["status"], "failed")
        self.assertIn("boom", job["error"])

    def test_an_institute_cannot_start_unlimited_decks(self):
        import threading
        gate = threading.Event()
        started = [jobs.start(lambda p: (gate.wait(2), Response({}))[1], institute_id="busy")
                   for _ in range(jobs.MAX_RUNNING_PER_INSTITUTE)]
        self.assertTrue(all(started))
        self.assertIsNone(jobs.start(lambda p: Response({}), institute_id="busy"))
        self.assertIsNotNone(jobs.start(lambda p: Response({}), institute_id="other"))
        gate.set()

    def test_a_job_that_stopped_reporting_is_reported_as_interrupted(self):
        jobs._write("old", {"status": "running", "updatedAt": time.time() - jobs.STALE_AFTER_S - 1})
        self.assertEqual(jobs.get("old")["status"], "failed")


class EndpointTests(MemoryStore):
    def setUp(self):
        super().setUp()
        for target in ("ai_services.views.base.tenant_gate",):
            p = mock.patch(target, return_value=None)
            p.start()
            self.addCleanup(p.stop)
        self.factory = APIRequestFactory()

    def _call(self, view, request, institute="inst-1"):
        request.institute_id = institute
        return view(request)

    def test_start_returns_at_once_and_status_carries_progress_then_the_deck(self):
        from ai_services.views import ppt

        def fake(snapshot, progress=None):
            progress("pictures", done=0, total=1, partial={"title": snapshot.data["topic"]})
            return Response({"success": True, "data": {"title": snapshot.data["topic"]}})
        with mock.patch.object(ppt, "_generate_presentation", side_effect=fake):
            started = self._call(ppt.generate_presentation_start, self.factory.post(
                "/ppt/generate/start", {"topic": "Respiration"}, format="json"))
            self.assertEqual(started.status_code, 202)
            job_id = started.data["jobId"]
            wait_for(job_id)
        status = self._call(ppt.generate_presentation_status,
                            self.factory.get("/ppt/generate/status", {"job_id": job_id}))
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.data["result"], {"success": True, "data": {"title": "Respiration"}})
        self.assertEqual(status.data["partial"], {"title": "Respiration"})
        self.assertNotIn("instituteId", status.data)

    def test_another_institute_cannot_see_the_job(self):
        from ai_services.views import ppt
        with mock.patch.object(ppt, "_generate_presentation",
                               return_value=Response({"success": True, "data": {}})):
            job_id = self._call(ppt.generate_presentation_start, self.factory.post(
                "/ppt/generate/start", {"topic": "T"}, format="json")).data["jobId"]
            wait_for(job_id)
        other = self._call(ppt.generate_presentation_status,
                           self.factory.get("/ppt/generate/status", {"job_id": job_id}),
                           institute="inst-2")
        self.assertEqual(other.status_code, 404)

    def test_a_missing_topic_is_refused_before_any_job_starts(self):
        from ai_services.views import ppt
        response = self._call(ppt.generate_presentation_start, self.factory.post(
            "/ppt/generate/start", {}, format="json"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(jobs._memory, {})


class ProgressFromV2Tests(SimpleTestCase):
    """generate_v2 reports its stages and the deck as it is built."""

    def test_v2_reports_the_slides_before_the_pictures_are_in(self):
        from ai_services import tests_ppt_v2_service as svc
        seen = []
        svc.run(on_progress=lambda stage, **kw: seen.append((stage, kw)))
        stages = [s for s, _ in seen]
        self.assertEqual(stages[0], "planning")
        self.assertIn("composing", stages)
        pictures = [kw for s, kw in seen if s == "pictures"]
        first = pictures[0]
        self.assertEqual(first["done"], 0)
        self.assertEqual(len(first["partial"]["slides"]), 2)
        self.assertTrue(any(s["v2"]["visual"].get("pending") for s in first["partial"]["slides"]))
        last = pictures[-1]
        self.assertEqual(last["done"], last["total"])
        self.assertFalse(any(s["v2"]["visual"].get("pending") for s in last["partial"]["slides"]))


class ActivityFeedTests(MemoryStore):
    """What the teacher sees while waiting: one line per step, updated in place."""

    def test_a_step_is_updated_in_place_not_repeated(self):
        jobs._write("j", {"status": "running"})
        p = jobs.Progress("j")
        p.event("outline", "Planning the lesson outline")
        p.event("outline", "Outlined 10 slides", "done")
        p.event("write-0", "Writing slides 1–3")
        feed = jobs.get("j")["activity"]
        self.assertEqual([(e["key"], e["text"], e["state"]) for e in feed],
                         [("outline", "Outlined 10 slides", "done"),
                          ("write-0", "Writing slides 1–3", "active")])

    def test_the_feed_keeps_only_the_latest_steps(self):
        jobs._write("j", {"status": "running"})
        p = jobs.Progress("j")
        for i in range(jobs.Progress.MAX_ACTIVITY + 5):
            p.event("k%d" % i, "step %d" % i, "done")
        feed = jobs.get("j")["activity"]
        self.assertEqual(len(feed), jobs.Progress.MAX_ACTIVITY)
        self.assertEqual(feed[-1]["text"], "step %d" % (jobs.Progress.MAX_ACTIVITY + 4))


class ActivityFromGeneratorsTests(SimpleTestCase):
    def test_v2_says_what_it_is_doing_step_by_step(self):
        from ai_services import tests_ppt_v2_service as svc
        feed = []
        svc.run(on_event=lambda key, text, state: feed.append((key, text, state)))
        keys = [k for k, _, _ in feed]
        self.assertIn("write", keys)
        self.assertIn("check", keys)
        self.assertIn("layout", keys)
        pictures = [(k, t, s) for k, t, s in feed if k.startswith("picture-")]
        self.assertTrue(pictures)
        self.assertTrue(all(t.startswith("Slide ") for _, t, _ in pictures))
        # Every slide that wanted a picture ends on an outcome, not mid-step.
        last = {}
        for k, t, s in pictures:
            last[k] = s
        self.assertTrue(all(s in ("done", "warn") for s in last.values()))

    def test_image_slides_say_which_slide_is_being_painted(self):
        from ai_services.core.ppt_v2 import image_slides
        from ai_services.core.ppt_v2.planner import PlannerResult
        feed = []
        slides = [{"slideType": "concept", "title": "A", "blocks": [{"text": "x"}]},
                  {"slideType": "summary", "title": "B", "blocks": [{"text": "y"}]}]
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        return_value=PlannerResult(title="T", slides=slides)):
            image_slides.generate_image_deck(
                topic="T", slide_count=2, ctx={}, on_event=lambda *a: feed.append(a),
                render=lambda p, **kw: {"ok": True, "url": "u"})
        self.assertIn(("paint-1", "Slide 1: painted", "done"), feed)
        self.assertIn(("paint-2", "Slide 2: painted", "done"), feed)
