"""A requested slide count must be honoured, not silently raised.

Asked for a 2-slide deck on 2026-09-29, the endpoint produced 3: slideCount was
clamped by max(3, ...), and on the ungrounded path the coverage plan then re-set
slide_count to len(sub_areas) + 2. Both paths overrode the caller without saying
so, which made "exactly 2 slides" untestable.
"""
import io
import os
from unittest import mock

from django.test import SimpleTestCase

from ai_services.views import ppt


def clamp(requested):
    """The endpoint's clamp, as applied to request.data['slideCount']."""
    return max(ppt._MIN_SLIDES, min(ppt._MAX_SLIDES, int(requested or 5)))


class SlideCountClampTests(SimpleTestCase):
    def test_two_slides_is_allowed(self):
        self.assertEqual(clamp(2), 2)

    def test_a_title_only_deck_is_still_refused(self):
        self.assertEqual(ppt._MIN_SLIDES, 2)
        self.assertEqual(clamp(1), 2)
        self.assertEqual(clamp(0), 5)      # 0 is falsy -> the default of 5
        self.assertEqual(clamp(None), 5)

    def test_decks_are_capped_at_10_slides(self):
        self.assertEqual(clamp(10), 10)
        self.assertEqual(clamp(11), 10)
        self.assertEqual(clamp(99), ppt._MAX_SLIDES)
        self.assertEqual(ppt._MAX_SLIDES, 10)          # decks are capped at 10 (2026-10-09)
        from ai_services.core.ppt_v2 import planner
        self.assertLessEqual(planner.AUTO_MAX_SLIDES, ppt._MAX_SLIDES)

    def test_ordinary_requests_pass_through(self):
        for n in (3, 5, 8, 10):
            self.assertEqual(clamp(n), n)


class CoveragePlanMayOnlyShortenTests(SimpleTestCase):
    """The plan can make a deck shorter (honest for a thin scope) but must never
    make it longer than the caller asked for."""

    @staticmethod
    def planned(slide_count, sub_areas):
        return min(slide_count, len(sub_areas) + 2)

    def test_a_two_slide_request_stays_two(self):
        self.assertEqual(self.planned(2, ["Only one sub-area"]), 2)
        self.assertEqual(self.planned(2, ["a", "b", "c", "d"]), 2)

    def test_a_thin_scope_still_yields_a_shorter_deck(self):
        self.assertEqual(self.planned(10, ["a", "b", "c"]), 5)

    def test_a_rich_scope_never_exceeds_the_request(self):
        self.assertEqual(self.planned(6, ["a", "b", "c", "d", "e", "f", "g"]), 6)

    def test_the_source_uses_min_not_assignment(self):
        src = io.open(os.path.join(os.path.dirname(__file__), "views", "ppt.py"), encoding="utf-8").read()
        self.assertIn("slide_count = min(slide_count, len(sub_areas) + 2)", src)
        self.assertNotIn("slide_count = len(sub_areas) + 2", src)
        self.assertIn("slide_count = max(_MIN_SLIDES, min(_MAX_SLIDES,", src)


class ImageDeckLengthTests(SimpleTestCase):
    """Image decks are always 10 slides; the studio offers no count for them."""

    def run_view(self, version, data):
        from rest_framework.response import Response
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return Response({"success": True, "data": {}})
        request = mock.Mock(data=dict({"topic": "Respiration"}, **data), headers={},
                            institute_id="i", vertical="school", board="CBSE")
        target = "_try_generate_image_deck" if version == "image" else "_try_generate_v2"
        with mock.patch.object(ppt, "ppt_version",
                               return_value=ppt.IMAGE if version == "image" else ppt.V2),              mock.patch.object(ppt, target, side_effect=fake):
            ppt._generate_presentation(request)
        return seen

    def test_an_image_deck_is_always_ten_slides_whatever_is_asked(self):
        for asked in (3, 8, 15, "auto", None):
            seen = self.run_view("image", {"pptVersion": "image", "slideCount": asked})
            self.assertEqual(seen["slide_count"], 10, asked)
            self.assertIsNone(seen["auto_range"], asked)    # no "the content decides" range

    def test_a_designed_deck_keeps_its_own_count_and_auto(self):
        self.assertEqual(self.run_view("v2", {"pptVersion": "v2", "slideCount": 6})["slide_count"], 6)
        auto = self.run_view("v2", {"pptVersion": "v2", "slideCount": "auto"})
        self.assertIsNotNone(auto["auto_range"])
        self.assertLessEqual(auto["auto_range"][1], 10)

    def test_the_image_length_is_a_setting_but_never_above_the_cap(self):
        with mock.patch.dict("os.environ", {"PPT_IMAGE_DECK_SLIDES": "8"}):
            self.assertEqual(ppt.image_deck_slides(), 8)
        with mock.patch.dict("os.environ", {"PPT_IMAGE_DECK_SLIDES": "40"}):
            self.assertEqual(ppt.image_deck_slides(), 10)

