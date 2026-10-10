"""Decks run by worker processes from a Redis queue, not threads in the web server.

A deploy, a restart or gunicorn recycling a worker used to kill every deck that
worker was running. These tests drive the queue and the worker on a fake Redis.
"""
import json
import threading
import time
from unittest import mock, skipUnless

from django.test import SimpleTestCase
from rest_framework.response import Response

from ai_services.core.ppt_v2 import job_queue, jobs
from ai_services.core.ppt_v2.job_queue import PENDING, PROCESSING, WORKERS, Worker

try:
    import fakeredis
    HAVE_FAKEREDIS = True
except ImportError:
    HAVE_FAKEREDIS = False


def redis():
    return fakeredis.FakeRedis(decode_responses=True)


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class QueueTests(SimpleTestCase):
    def setUp(self):
        self.r = redis()

    def test_decks_go_to_a_worker_only_while_one_is_alive(self):
        self.assertFalse(job_queue.should_queue(self.r))
        self.r.zadd(WORKERS, {"w1": time.time()})
        self.assertTrue(job_queue.should_queue(self.r))
        self.r.zadd(WORKERS, {"w1": time.time() - 60})            # it stopped beating
        self.assertFalse(job_queue.should_queue(self.r))
        self.r.zadd(WORKERS, {"w2": time.time()})
        with mock.patch.dict("os.environ", {"PPT_JOB_RUNNER": "thread"}):
            self.assertFalse(job_queue.should_queue(self.r))

    def test_jobs_are_taken_oldest_first_and_report_their_place(self):
        for job_id in ("a", "b", "c"):
            job_queue.enqueue(job_id, {"data": {}}, client=self.r)
        self.assertEqual(job_queue.queue_position("a", client=self.r), 0)
        self.assertEqual(job_queue.queue_position("c", client=self.r), 2)
        w = Worker(self.r, handler=lambda *a: None)
        self.assertEqual([w.take(), w.take(), w.take()], ["a", "b", "c"])
        self.assertIsNone(job_queue.queue_position("a", client=self.r))

    def test_a_job_runs_with_its_spec_and_leaves_nothing_behind(self):
        seen = []
        job_queue.enqueue("j1", {"data": {"topic": "Respiration"}}, client=self.r)
        w = Worker(self.r, handler=lambda job_id, spec: seen.append((job_id, spec)))
        w.run_one(w.take())
        self.assertEqual(seen[0][0], "j1")
        self.assertEqual(seen[0][1]["data"], {"topic": "Respiration"})
        self.assertEqual(seen[0][1]["attempts"], 1)
        self.assertEqual(self.r.lrange(PROCESSING, 0, -1), [])
        self.assertFalse(self.r.exists("pptq:spec:j1"))

    def test_a_job_that_raises_is_cleaned_up_and_the_worker_goes_on(self):
        job_queue.enqueue("j1", {}, client=self.r)
        w = Worker(self.r, handler=mock.Mock(side_effect=RuntimeError("boom")))
        w.run_one(w.take())
        self.assertEqual(self.r.lrange(PROCESSING, 0, -1), [])

    def test_a_job_whose_worker_died_goes_back_on_the_queue_first(self):
        job_queue.enqueue("old", {}, client=self.r)
        job_queue.enqueue("next", {}, client=self.r)
        w = Worker(self.r, handler=lambda *a: None)
        self.assertEqual(w.take(), "old")                          # its worker then dies:
        now = time.time()                                          # no beat is ever written
        gave_up = []
        first = job_queue.reap(self.r, on_give_up=gave_up.append, now=now)
        self.assertEqual(first["requeued"], [])                    # perhaps just taken: wait
        again = job_queue.reap(self.r, on_give_up=gave_up.append,
                               now=now + job_queue.ORPHAN_GRACE_S + 1)
        self.assertEqual(again["requeued"], ["old"])
        self.assertEqual(w.take(), "old")                          # ahead of "next"

    def test_a_running_job_is_not_mistaken_for_a_lost_one(self):
        job_queue.enqueue("j1", {}, client=self.r)
        w = Worker(self.r, handler=lambda *a: None)
        w.take()
        self.r.setex("pptq:beat:j1", 60, w.id)
        done = job_queue.reap(self.r, on_give_up=lambda j: None, now=time.time() + 100)
        self.assertEqual(done["requeued"], [])

    def test_a_job_out_of_attempts_is_given_up_on(self):
        job_queue.enqueue("j1", {}, client=self.r)
        spec = json.loads(self.r.get("pptq:spec:j1"))
        spec["attempts"] = job_queue.max_attempts()
        self.r.set("pptq:spec:j1", json.dumps(spec))
        Worker(self.r, handler=lambda *a: None).take()
        gave_up = []
        now = time.time()
        job_queue.reap(self.r, on_give_up=gave_up.append, now=now)
        job_queue.reap(self.r, on_give_up=gave_up.append, now=now + 30)
        self.assertEqual(gave_up, ["j1"])
        self.assertEqual(self.r.lrange(PROCESSING, 0, -1), [])
        self.assertEqual(self.r.lrange(PENDING, 0, -1), [])

    def test_stopping_finishes_the_deck_in_hand_first(self):
        started, finished = threading.Event(), threading.Event()

        def handler(job_id, spec):
            started.set()
            time.sleep(0.3)
            finished.set()
        job_queue.enqueue("j1", {}, client=self.r)
        w = Worker(self.r, handler=handler, take_timeout_s=0.05)
        t = threading.Thread(target=w.run_forever)
        t.start()
        self.assertTrue(started.wait(2))
        self.assertGreater(self.r.zcount(WORKERS, "-inf", "+inf"), 0)  # it says it is alive
        w.stop.set()                                                    # SIGTERM mid-deck
        t.join(3)
        self.assertTrue(finished.is_set())
        self.assertEqual(self.r.lrange(PROCESSING, 0, -1), [])
        self.assertEqual(self.r.zcard(WORKERS), 0)                      # and says it has gone


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class StartTests(SimpleTestCase):
    """jobs.start sends a deck to a worker when one is alive, or runs it here."""

    def setUp(self):
        self.r = redis()
        for target in (mock.patch.object(jobs, "_redis", return_value=self.r),
                       mock.patch.object(job_queue, "_client", return_value=self.r)):
            target.start()
            self.addCleanup(target.stop)

    def test_with_a_live_worker_the_deck_is_queued_not_run_here(self):
        self.r.zadd(WORKERS, {"w1": time.time()})
        run = mock.Mock()
        job_id = jobs.start(run, institute_id="inst-1", spec={"data": {"topic": "T"}})
        time.sleep(0.1)
        run.assert_not_called()
        self.assertEqual(jobs.get(job_id)["status"], "queued")
        self.assertEqual(self.r.lrange(PENDING, 0, -1), [job_id])
        spec = json.loads(self.r.get("pptq:spec:%s" % job_id))
        self.assertEqual(spec["instituteId"], "inst-1")

    def test_without_a_worker_the_deck_runs_in_this_process_as_before(self):
        ran = threading.Event()

        def run(progress):
            ran.set()
            return Response({"success": True, "data": {"slides": []}})
        job_id = jobs.start(run, institute_id="inst-1", spec={"data": {}})
        self.assertTrue(ran.wait(2))
        for _ in range(100):
            if jobs.get(job_id)["status"] == "done":
                break
            time.sleep(0.02)
        self.assertEqual(jobs.get(job_id)["status"], "done")
        self.assertEqual(self.r.lrange(PENDING, 0, -1), [])

    def test_a_long_wait_in_the_queue_is_reported_not_left_spinning(self):
        self.r.zadd(WORKERS, {"w1": time.time()})
        job_id = jobs.start(mock.Mock(), institute_id="inst-1", spec={"data": {}})
        job = jobs.get(job_id)
        job["createdAt"] = time.time() - jobs.QUEUED_MAX_WAIT_S - 1
        self.r.set("ppt_job:%s" % job_id, json.dumps(job))
        self.assertEqual(jobs.get(job_id)["status"], "failed")

    def test_waiting_in_the_queue_is_not_mistaken_for_an_interrupted_deck(self):
        self.r.zadd(WORKERS, {"w1": time.time()})
        job_id = jobs.start(mock.Mock(), institute_id="inst-1", spec={"data": {}})
        job = jobs.get(job_id)
        job["updatedAt"] = time.time() - jobs.STALE_AFTER_S - 1     # quiet, but only waiting
        self.r.set("ppt_job:%s" % job_id, json.dumps(job))
        self.assertEqual(jobs.get(job_id)["status"], "queued")

    def test_a_job_given_up_on_is_failed_and_frees_its_place(self):
        self.r.zadd(WORKERS, {"w1": time.time()})
        job_id = jobs.start(mock.Mock(), institute_id="inst-1", spec={"data": {}})
        jobs.give_up(job_id)
        self.assertEqual(jobs.get(job_id)["status"], "failed")
        self.assertEqual(int(self.r.get("ppt_jobs_running:inst-1")), 0)


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class WorkerRunsTheRealPathTests(SimpleTestCase):
    """A worker runs the same cache-or-generate path, as the teacher who asked."""

    def test_a_queued_deck_is_run_from_its_spec_and_attributed_to_its_teacher(self):
        from ai_services.core import request_context
        from ai_services.views import ppt

        r = redis()
        seen = {}

        def cached_or_generated(snapshot, progress):
            seen["topic"] = snapshot.data["topic"]
            seen["institute"] = snapshot.institute_id
            seen["user"] = request_context.get("user_id")
            return Response({"success": True, "data": {"slides": [{"title": "A"}]}})
        spec = ppt._RequestSnapshot.from_spec({
            "data": {"topic": "Respiration"}, "headers": {"X-User-Id": "teacher-7"},
            "institute_id": "inst-1", "vertical": "school", "board": "cbse"}).spec()
        with mock.patch.object(jobs, "_redis", return_value=r), \
             mock.patch.object(ppt, "_cached_or_generated", side_effect=cached_or_generated):
            jobs._write("j1", {"jobId": "j1", "status": "queued", "instituteId": "inst-1",
                               "createdAt": time.time(), "updatedAt": time.time()})
            ppt.run_queued_job("j1", json.loads(json.dumps(spec)))     # as it comes off Redis
            job = jobs.get("j1")
        self.assertEqual(seen, {"topic": "Respiration", "institute": "inst-1", "user": "teacher-7"})
        self.assertEqual(job["status"], "done")
        self.assertIsNone(request_context.get("user_id"))              # cleared afterwards


class BlockingWaitTests(SimpleTestCase):
    def test_a_wait_for_jobs_never_outlasts_the_connections_read_timeout(self):
        # Found live: the shared pool reads time out at 1s, and a 2s wait made
        # every quiet moment an error.
        client = mock.Mock()
        client.connection_pool.connection_kwargs = {"socket_timeout": 1.0}
        self.assertLess(Worker(client, handler=lambda *a: None, take_timeout_s=2.0).take_timeout_s, 1.0)
        client.connection_pool.connection_kwargs = {"socket_timeout": 15}
        self.assertEqual(Worker(client, handler=lambda *a: None, take_timeout_s=2.0).take_timeout_s, 2.0)


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class BackgroundDeckStatusTests(SimpleTestCase):
    """A teacher can send a deck to the background and come back to it later."""

    def setUp(self):
        self.r = redis()
        p = mock.patch.object(jobs, "_redis", return_value=self.r)
        p.start()
        self.addCleanup(p.stop)

    def status(self, job_id, **params):
        from rest_framework.test import APIRequestFactory
        from ai_services.views import ppt

        req = APIRequestFactory().get("/ppt/generate/status", dict(params, job_id=job_id))
        req.institute_id = "inst-1"
        with mock.patch("ai_services.views.base.tenant_gate", return_value=None):
            return ppt.generate_presentation_status(req).data

    def test_a_finished_deck_is_kept_for_a_day_not_an_hour(self):
        jobs._write("j1", {"jobId": "j1", "status": "done", "instituteId": "inst-1"})
        self.assertGreater(self.r.ttl("ppt_job:j1"), 23 * 3600)

    def test_a_summary_carries_progress_but_never_the_slides(self):
        deck = {"title": "Respiration", "slides": [{"title": "A", "bullets": ["x" * 500]}] * 10,
                "generation": {"version": "image", "cached": False}}
        jobs._write("j1", {"jobId": "j1", "status": "done", "stage": "done", "instituteId": "inst-1",
                           "partial": deck, "result": {"success": True, "data": deck},
                           "activity": [{"key": "paint-3", "text": "Slide 3: painting", "state": "done"}],
                           "createdAt": 1.0, "updatedAt": 2.0})
        out = self.status("j1", summary="1")
        self.assertEqual((out["status"], out["title"], out["slides"], out["version"]),
                         ("done", "Respiration", 10, "image"))
        self.assertNotIn("result", out)
        self.assertNotIn("partial", out)
        self.assertLess(len(json.dumps(out)), 600)
        self.assertIn("result", self.status("j1"))           # the full status is unchanged

    def test_a_summary_says_what_is_happening_now(self):
        jobs._write("j2", {"jobId": "j2", "status": "running", "stage": "pictures", "done": 4,
                           "total": 10, "instituteId": "inst-1", "updatedAt": time.time(),
                           "activity": [{"key": "write", "text": "Wrote slides", "state": "done"},
                                        {"key": "paint-5", "text": "Slide 5: painting the slide",
                                         "state": "active"}]})
        out = self.status("j2", summary="1")
        self.assertEqual((out["done"], out["total"], out["activity"]),
                         (4, 10, "Slide 5: painting the slide"))

    def test_another_schools_deck_is_never_shown(self):
        jobs._write("j3", {"jobId": "j3", "status": "done", "instituteId": "inst-2"})
        self.assertEqual(self.status("j3", summary="1"), {"error": "job not found"})
