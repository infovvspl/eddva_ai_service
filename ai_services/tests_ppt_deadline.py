"""A deck that arrives late is a deck the teacher never sees.

The backend stops waiting at 240s. A live 10-slide deck from a real chapter
took 257s (a 120s plan, then 136s of pictures and vision checks), so it would
have failed in the app. Optional slow steps now give way to the deadline.
"""
import time
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import vision
from ai_services.core.ppt_v2.producers import (AI_IMAGE_MIN_REMAINING_S, ProducerSet,
                                               VISION_MARGIN_S)
from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy


class VisionTimeoutTests(SimpleTestCase):
    def _contract(self):
        c = mock.Mock(is_meaningful=True)
        return c

    def test_no_time_left_means_no_look_and_no_call(self):
        ask = mock.Mock()
        with mock.patch.object(vision, "enabled", return_value=True):
            verdict = vision.validate("http://x/a.png", self._contract(), ask=ask, timeout=2)
        self.assertFalse(verdict.available)
        ask.assert_not_called()

    def test_the_wait_never_exceeds_the_configured_timeout(self):
        with mock.patch.object(vision, "enabled", return_value=True), \
             mock.patch.object(vision, "_ask_with_timeout", return_value=("{}", None)) as ask, \
             mock.patch.object(vision, "build_prompt", return_value="p"),              mock.patch.object(vision, "cache_key", return_value="k"),              mock.patch.object(vision, "interpret", return_value=mock.Mock(passed=True)):
            vision.validate("http://x/a.png", self._contract(), ask=lambda i, p: "", timeout=999)
        self.assertEqual(ask.call_args.args[3], vision.timeout_s())


class ProducerDeadlineTests(SimpleTestCase):
    def _producers(self, left):
        ps = ProducerSet(search_fn=None)
        ps.deadline = time.perf_counter() + left
        return ps

    def test_a_generated_picture_gives_way_when_the_deck_is_running_late(self):
        ps = self._producers(AI_IMAGE_MIN_REMAINING_S - 5)
        with mock.patch("ai_services.core.image_generation.generate_note_image") as gen:
            out = ps.ai_image(VisualSpec(strategy=VisualStrategy.AI_GENERATED_IMAGE,
                                         description="a leaf", image_prompt="a leaf"))
        self.assertFalse(out["ok"])
        self.assertIn("no time left", out["error"])
        gen.assert_not_called()

    def test_with_time_to_spare_it_is_still_made(self):
        ps = self._producers(AI_IMAGE_MIN_REMAINING_S + 60)
        with mock.patch("ai_services.core.image_generation.generate_note_image",
                        return_value={"ok": False, "error": "x"}) as gen:
            ps.ai_image(VisualSpec(strategy=VisualStrategy.AI_GENERATED_IMAGE,
                                   description="a leaf", image_prompt="a leaf"))
        gen.assert_called_once()

    def test_a_vision_check_only_gets_the_time_that_is_left(self):
        ps = self._producers(30)
        self.assertLessEqual(ps._vision_timeout(), 30 - VISION_MARGIN_S + 0.5)
        self.assertIsNone(ProducerSet(search_fn=None)._vision_timeout())   # no deadline


class ServiceDeadlineTests(SimpleTestCase):
    def test_v2_sets_the_deadline_from_the_start_of_the_request(self):
        from ai_services import tests_ppt_v2_service as svc
        seen = {}
        real = svc.generate_v2.__globals__["_resolve_assets"]

        def spy(slides, producers, **kw):
            seen["deadline"] = producers.deadline
            return real(slides, producers, **kw)
        with mock.patch.dict(svc.generate_v2.__globals__, {"_resolve_assets": spy}), \
             mock.patch.dict("os.environ", {"PPT_DEADLINE_S": "150"}):
            t0 = time.perf_counter()
            svc.run()
        self.assertIsNotNone(seen["deadline"])
        self.assertAlmostEqual(seen["deadline"] - t0, 150, delta=5)


class SlideBudgetTests(SimpleTestCase):
    def test_one_slide_cannot_take_the_whole_deck(self):
        from ai_services.core.ppt_v2.producers import SLIDE_PICTURE_BUDGET_S
        ps = ProducerSet(search_fn=None)
        ps.deadline = time.perf_counter() + 1000
        ps.begin_slide()
        self.assertLessEqual(ps._remaining(), SLIDE_PICTURE_BUDGET_S + 0.5)
        ps.end_slide()
        self.assertGreater(ps._remaining(), 900)

    def test_the_deck_deadline_still_wins_when_it_is_sooner(self):
        ps = ProducerSet(search_fn=None)
        ps.deadline = time.perf_counter() + 10
        ps.begin_slide()
        self.assertLessEqual(ps._remaining(), 10.5)

