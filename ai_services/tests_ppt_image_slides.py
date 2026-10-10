"""Image slides: every slide painted whole by an image model.

Offered next to the composed V2 deck so the two can be compared. These tests
pin what the option promises: the prompt carries the slide's exact words, a
deck survives a slide that fails, the words stay in the notes, the option is
only reachable through the gated override, and the slide's text is never
OCR-erased the way a figure's is.

No network: the planner and the painter are stubs.
"""
import base64
import os
import tempfile
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core import image_generation
from ai_services.core.ppt_v2 import image_slides
from ai_services.core.ppt_v2.flags import IMAGE, V1, V2, ppt_version
from ai_services.core.ppt_v2.planner import PlannerResult

PROCESS = {
    "slideType": "process", "title": "How Plants Make Food", "emphasisText": "Food",
    "subtitle": "Four steps in the leaf",
    "blocks": [{"kind": "step", "label": "Absorb light", "text": "Chlorophyll traps sunlight."},
               {"kind": "step", "label": "Take in CO2", "text": "Air enters through stomata."}],
    "visual": {"description": "a leaf with sunlight falling on it",
               "annotations": [{"text": "Sunlight", "anchor": [0.2, 0.2]}]},
    "speakerNotes": "Ask what the leaf needs.",
}
COMPARE = {"slideType": "comparison", "title": "Aerobic vs Anaerobic",
           "blocks": [{"kind": "compare_left", "label": "Aerobic", "text": "Uses oxygen."},
                      {"kind": "compare_right", "label": "Anaerobic", "text": "No oxygen."}]}
QUIZ = {"slideType": "quiz", "title": "Quick Check",
        "blocks": [{"kind": "question", "text": "Where does gas exchange happen?"},
                   {"kind": "answer", "text": "Alveoli"}, {"kind": "answer", "text": "Stomach"}]}
FORMULA = {"slideType": "formula", "title": "The Equation",
           "blocks": [{"kind": "equation", "text": "6CO2 + 6H2O -> C6H12O6 + 6O2"},
                      {"kind": "bullet", "label": "Inputs", "text": "Carbon dioxide and water."}]}


def prompt(slide, **kw):
    return image_slides.slide_prompt(slide, number=2, total=5, class_name="Class 8",
                                     subject=kw.get("subject", "Science"), chapter="Photosynthesis")


class PromptTests(SimpleTestCase):
    def test_every_word_on_the_slide_is_quoted_exactly(self):
        p = prompt(PROCESS)
        for text in ('"How Plants Make Food"', '"Four steps in the leaf"', '"Absorb light"',
                     '"Chlorophyll traps sunlight."', '"Sunlight"',
                     '"Class 8 | Science | Photosynthesis"', '"2 / 5"'):
            self.assertIn(text, p)
        self.assertIn("Spell quoted text exactly", p)

    def test_a_quote_inside_the_content_cannot_end_the_quote(self):
        p = prompt(dict(PROCESS, title='The "green" factory'))
        self.assertIn("\"The 'green' factory\"", p)

    def test_the_palette_follows_the_subject_so_a_deck_stays_consistent(self):
        self.assertIn("green", prompt(PROCESS, subject="Science"))
        self.assertIn("orange", prompt(PROCESS, subject="Social Science"))
        self.assertEqual(image_slides.palette_for("Biology"), image_slides.palette_for("Science"))
        # "science" inside a longer subject name must not win.
        for subject in ("Social Science", "Computer Science", "Political Science"):
            self.assertNotEqual(image_slides.palette_for(subject),
                                image_slides.palette_for("Science"), subject)

    def test_a_comparison_is_two_sides_not_a_list(self):
        p = prompt(COMPARE)
        self.assertIn("VS", p)
        self.assertIn('Left: a card: heading "Aerobic"', p)
        self.assertIn('Right: a card: heading "Anaerobic"', p)

    def test_a_quiz_never_marks_the_answer(self):
        p = prompt(QUIZ)
        self.assertIn("do not mark any option as correct", p)
        self.assertIn('"Where does gas exchange happen?"', p)

    def test_an_equation_is_asked_for_with_proper_notation(self):
        p = prompt(FORMULA)
        self.assertIn('"6CO2 + 6H2O -> C6H12O6 + 6O2"', p)
        self.assertIn("subscripts", p)

    def test_a_label_ending_in_a_colon_gets_one_colon(self):
        slide = dict(PROCESS, blocks=[{"kind": "bullet", "label": "Carbon Dioxide:",
                                       "text": "More CO2, more glucose."}])
        self.assertIn('heading "Carbon Dioxide", line "More CO2, more glucose."', prompt(slide))
        self.assertNotIn("Carbon Dioxide:", prompt(slide))
        self.assertEqual(image_slides._bullets(slide), ["Carbon Dioxide: More CO2, more glucose."])

    def test_pictures_stay_at_the_level_of_the_class_book(self):
        p = prompt(PROCESS)
        self.assertIn("at Class 8 textbook level", p)
        self.assertIn("no structural formulas", p)

    def test_no_watermarks_or_invented_text(self):
        self.assertIn("never draw quote marks, asterisks, other text, numbers, watermark or logo",
                      prompt(PROCESS))


def _plan(*slides):
    return PlannerResult(title="Photosynthesis", slides=list(slides), model="stub:planner")


class DeckTests(SimpleTestCase):
    def setUp(self):
        # Lost slides are retried in rounds with a pause between them; a test
        # must not sit through the pauses.
        patcher = mock.patch.object(image_slides, "_sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _deck(self, render, *slides):
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        return_value=_plan(*slides)):
            return image_slides.generate_image_deck(
                topic="Photosynthesis", slide_count=len(slides),
                ctx={"className": "Class 8", "subjectName": "Science",
                     "chapterName": "Photosynthesis"}, render=render)

    def test_every_slide_gets_its_picture(self):
        render = mock.Mock(return_value={"ok": True, "url": "http://x/1.png"})
        data = self._deck(render, PROCESS, COMPARE, QUIZ)
        self.assertEqual(render.call_count, 3)
        self.assertTrue(all(s["slideImage"]["url"] == "http://x/1.png" for s in data["slides"]))
        self.assertEqual(data["generation"]["version"], "image")
        self.assertEqual(data["generation"]["imagesGenerated"], 3)
        self.assertEqual([s["slideNumber"] for s in data["slides"]], [1, 2, 3])

    def test_a_failed_slide_keeps_its_text_and_the_deck_survives(self):
        def render(p, **kw):
            if '"Aerobic vs Anaerobic"' in p:
                return {"ok": False, "error": "429"}
            return {"ok": True, "url": "http://x/ok.png"}
        data = self._deck(render, PROCESS, COMPARE)
        self.assertNotIn("slideImage", data["slides"][1])
        self.assertEqual(data["slides"][1]["title"], "Aerobic vs Anaerobic")
        self.assertTrue(data["slides"][1]["bullets"])
        self.assertEqual(data["generation"]["failedSlides"], [2])

    def test_a_painter_that_raises_costs_one_slide_not_the_deck(self):
        def render(p, **kw):
            raise RuntimeError("boom")
        data = self._deck(render, PROCESS)
        self.assertEqual(data["generation"]["imagesGenerated"], 0)
        self.assertIn("boom", data["slides"][0]["slideImageError"])

    def test_a_slide_lost_to_the_burst_is_repainted_once(self):
        tries = {"n": 0}

        def render(p, **kw):
            if '"Aerobic vs Anaerobic"' in p:
                tries["n"] += 1
                if tries["n"] == 1:
                    return {"ok": False, "error": "429"}
            return {"ok": True, "url": "http://x/ok.png"}
        data = self._deck(render, PROCESS, COMPARE)
        self.assertEqual(tries["n"], 2)
        self.assertIn("slideImage", data["slides"][1])
        self.assertEqual(data["generation"]["failedSlides"], [])

    def test_an_error_that_will_repeat_is_not_repainted(self):
        render = mock.Mock(return_value={"ok": False, "error": "403", "fatal": True})
        self._deck(render, PROCESS)
        self.assertEqual(render.call_count, 1)

    def test_the_words_stay_in_the_notes_because_the_slide_is_pixels(self):
        render = mock.Mock(return_value={"ok": True, "url": "http://x/1.png"})
        notes = self._deck(render, PROCESS)["slides"][0]["speakerNotes"]
        self.assertIn("Ask what the leaf needs.", notes)
        self.assertIn("Absorb light: Chlorophyll traps sunlight.", notes)

    def test_the_configured_model_and_a_16_9_canvas_are_used(self):
        render = mock.Mock(return_value={"ok": True, "url": "u"})
        with mock.patch.dict(os.environ, {"SLIDE_IMAGE_MODEL": "x/model"}):
            self._deck(render, PROCESS)
        kwargs = render.call_args.kwargs
        self.assertEqual(kwargs["model"], "x/model")
        self.assertAlmostEqual(kwargs["width"] / kwargs["height"], 16 / 9, places=2)


class FlagTests(SimpleTestCase):
    def test_image_slides_are_only_reachable_through_the_override(self):
        on = {"PPT_V2_ENABLED": "true", "PPT_V2_ALLOW_OVERRIDE": "true"}
        with mock.patch.dict(os.environ, on, clear=True):
            self.assertEqual(ppt_version(requested="image"), IMAGE)
            self.assertEqual(ppt_version(requested="v2"), V2)
        with mock.patch.dict(os.environ, {"PPT_V2_ENABLED": "true"}, clear=True):
            self.assertNotEqual(ppt_version(requested="image"), IMAGE)
        with mock.patch.dict(os.environ, {"PPT_V2_ALLOW_OVERRIDE": "true"}, clear=True):
            self.assertEqual(ppt_version(requested="image"), V1)   # kill switch wins


class SlideImageTests(SimpleTestCase):
    """The slide's text is its content, so it must never be OCR-erased."""

    def _ok(self):
        r = mock.Mock(ok=True, status_code=200, headers={})
        r.json.return_value = {"data": [{"b64_json": base64.b64encode(b"\x89PNG fake").decode()}]}
        return r

    def test_the_slide_is_saved_untouched(self):
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "k", "NOTES_IMAGE_OUTPUT_DIR": out}), \
             mock.patch.object(image_generation.requests, "post", return_value=self._ok()) as post, \
             mock.patch.object(image_generation, "_strip_embedded_text_from_image") as strip:
            result = image_generation.generate_slide_image("a slide", model="Qwen/Qwen-Image-2.0")
            self.assertTrue(result["ok"])
            strip.assert_not_called()
            body = post.call_args.kwargs["json"]
            self.assertEqual(body["model"], "Qwen/Qwen-Image-2.0")
            self.assertEqual(body["prompt"], "a slide")          # no figure grounding clause
            self.assertNotIn("steps", body)
            with open(result["path"], "rb") as fh:
                self.assertEqual(fh.read(), b"\x89PNG fake")

    def test_no_key_means_no_call(self):
        with mock.patch.dict(os.environ, {"TOGETHER_API_KEY": ""}), \
             mock.patch.object(image_generation.requests, "post") as post:
            self.assertFalse(image_generation.generate_slide_image("x", model="m")["ok"])
            post.assert_not_called()

    def test_a_slide_gets_more_patience_with_429_than_a_figure(self):
        busy = mock.Mock(ok=False, status_code=429, headers={}, text="slow down")
        with mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "k"}),              mock.patch.object(image_generation.requests, "post", return_value=busy) as post,              mock.patch.object(image_generation, "_sleep"):
            result = image_generation.generate_slide_image("x", model="m")
        self.assertFalse(result["ok"])
        self.assertEqual(post.call_count, 5)      # first try + 4 retries
