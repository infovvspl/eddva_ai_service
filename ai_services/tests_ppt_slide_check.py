"""Painted slides are read back and repainted when their text is off.

An image model paints each slide separately; no prompt makes it keep one type
size across a deck (a live deck's card text ran from 3.8% to 5.7% of the slide
height) or stops the odd gibberish card. slide_check reads each painted slide
with OCR, compares it with the rest of the deck, and generate_image_deck
repaints the slides that are off - keeping whichever painting is better.

No network and no real OCR: readings are injected.
"""
import re
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import image_slides, slide_check
# Imported here, not first inside a test: service binds planner.plan_deck when it
# is first imported, and the deck tests below patch plan_deck while
# generate_image_deck imports service - leaving a mock bound for later tests.
from ai_services.core.ppt_v2 import service as _service  # noqa: F401
from ai_services.core.ppt_v2.planner import PlannerResult
from ai_services.core.ppt_v2.slide_check import Reading, judge, read_slide, score, targets

CARDS = ["Nostrils Air enters the body through the nostrils.",
         "Throat The filtered air passes through the throat.",
         "Lungs The passage divides into smaller tubes called bronchi."]


def ocr(lines):
    """A stand-in for _ocr: (top fraction, height %, text)."""
    return lambda path: lines


class ReadSlideTests(SimpleTestCase):
    def test_title_and_card_heights_are_measured_from_the_slides_own_text(self):
        r = read_slide("x.png", title="Path of Air", cards=CARDS, other=["Class 10 | Science"],
                       ocr=ocr([(0.03, 2.0, "Class 10 | Science"),
                                (0.10, 8.0, "Path of Air"),
                                (0.35, 3.6, "Nostrils"),
                                (0.40, 3.0, "Air enters the body through the nostrils."),
                                (0.55, 3.0, "The filtered air passes through the throat."),
                                (0.70, 3.2, "The passage divides into smaller tubes")]))
        self.assertEqual(r.title_h, 8.0)
        self.assertAlmostEqual(r.body_h, 3.1)
        self.assertLess(r.junk, 0.1)
        self.assertGreater(r.coverage, 0.6)

    def test_ocr_misreads_of_real_text_are_not_junk(self):
        r = read_slide("x.png", title="Path of Air", cards=CARDS,
                       ocr=ocr([(0.4, 3.0, "Air enters the bodv through the nostrlls.")]))
        self.assertEqual(r.junk, 0.0)

    def test_gibberish_is_junk(self):
        r = read_slide("x.png", title="Key Points", cards=CARDS,
                       ocr=ocr([(0.4, 3.0, "Krassk armeast tife sraum tanyolc orloosf rnaly"),
                                (0.5, 3.0, "Air enters the body through the nostrils.")]))
        self.assertGreater(r.junk, slide_check.junk_limit())

    def test_no_ocr_means_no_reading_not_an_error(self):
        self.assertIsNone(read_slide("x.png", title="T", cards=CARDS, ocr=lambda p: None))
        self.assertIsNone(read_slide("x.png", title="T", cards=CARDS,
                                     ocr=mock.Mock(side_effect=OSError("unreadable"))))


class JudgeTests(SimpleTestCase):
    def deck(self):
        return {i: Reading(title_h=8.0, body_h=3.0, coverage=0.9, junk=0.05, tokens=40)
                for i in range(1, 7)}

    def test_a_consistent_deck_needs_nothing(self):
        self.assertEqual(judge(self.deck()), {})

    def test_text_too_big_or_small_against_the_deck_is_flagged(self):
        readings = self.deck()
        readings[3] = Reading(title_h=8.0, body_h=4.2, coverage=0.9, junk=0.0, tokens=40)
        readings[5] = Reading(title_h=11.0, body_h=2.2, coverage=0.9, junk=0.0, tokens=40)
        bad = judge(readings)
        self.assertEqual(bad[3], ["card text too big"])
        self.assertEqual(bad[5], ["card text too small", "title too big"])

    def test_garbled_or_missing_text_is_flagged(self):
        readings = self.deck()
        readings[2] = Reading(title_h=8.0, body_h=3.0, coverage=0.9, junk=0.5, tokens=40)
        readings[4] = Reading(title_h=8.0, body_h=3.0, coverage=0.2, junk=0.0, tokens=40)
        bad = judge(readings)
        self.assertIn("garbled text", bad[2])
        self.assertIn("missing text", bad[4])

    def test_the_better_painting_scores_lower(self):
        body_t, title_t = targets(self.deck())
        good = Reading(title_h=8.0, body_h=3.0, coverage=0.9, junk=0.0, tokens=40)
        off = Reading(title_h=8.0, body_h=4.5, coverage=0.9, junk=0.0, tokens=40)
        self.assertLess(score(good, body_t, title_t), score(off, body_t, title_t))
        self.assertGreater(score(None, body_t, title_t), score(off, body_t, title_t))


SLIDE = {"slideType": "concept", "title": "Path of Air",
         "blocks": [{"kind": "bullet", "label": "Nostrils", "text": "Air enters the body."}]}


class DeckTests(SimpleTestCase):
    def _deck(self, readings_by_call, n=6, env=None, framed=()):
        """Paint an n-slide deck. Slide k's painting is "pk" ("pkr" when repainted):
        named by the slide number in its own prompt, since slides paint in parallel
        and the order of render calls is not the order of the slides."""
        import re
        import threading

        calls = {"paint": 0, "read": 0}
        lock = threading.Lock()
        seen = set()

        def render(prompt, **kw):
            k = int(re.search(r'"(\d+) / \d+"', prompt).group(1))
            with lock:
                calls["paint"] += 1
                name = "p%d" % k if k not in seen else "p%dr" % k
                seen.add(k)
            return {"ok": True, "url": "u" + name[1:], "path": name}

        def read(path, **kw):
            calls["read"] += 1
            return readings_by_call(path)
        slides = [dict(SLIDE, slideType="title")] + [dict(SLIDE) for _ in range(n - 1)]
        plan = PlannerResult(title="T", slides=slides)
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck", return_value=plan), \
             mock.patch.object(slide_check, "read_slide", side_effect=read), \
             mock.patch.object(slide_check, "unframe", side_effect=lambda p: p in framed), \
             mock.patch.dict("os.environ", env or {}):
            data = image_slides.generate_image_deck(topic="T", slide_count=n, ctx={},
                                                    render=render)
        return data, calls

    @staticmethod
    def readings(off_paths, repaint_better=True):
        def by_path(path):
            if path in off_paths:
                return Reading(title_h=8.0, body_h=4.5, coverage=0.9, junk=0.0, tokens=40)
            if not repaint_better and path.endswith("r"):
                return Reading(title_h=8.0, body_h=5.0, coverage=0.9, junk=0.0, tokens=40)
            return Reading(title_h=8.0, body_h=3.0, coverage=0.9, junk=0.0, tokens=40)
        return by_path

    def test_an_off_slide_is_repainted_once_and_the_better_painting_kept(self):
        data, calls = self._deck(self.readings({"p3"}))
        self.assertEqual(calls["paint"], 7)                   # six slides + one repaint
        self.assertEqual(data["generation"]["textCheck"]["repainted"], [3])
        self.assertEqual(data["slides"][2]["slideImage"]["url"], "u3r")

    def test_a_repaint_that_is_no_better_keeps_the_first_painting(self):
        data, calls = self._deck(self.readings({"p3"}, repaint_better=False))
        self.assertEqual(data["generation"]["textCheck"]["kept"], [3])
        self.assertEqual(data["slides"][2]["slideImage"]["url"], "u3")

    def test_the_cover_is_never_read_and_a_consistent_deck_costs_nothing_extra(self):
        data, calls = self._deck(self.readings(set()))
        self.assertEqual(calls["paint"], 6)
        self.assertEqual(calls["read"], 5)                    # slides 2-6, not the cover
        self.assertEqual(data["generation"]["textCheck"]["repainted"], [])

    def test_repaints_are_capped(self):
        data, calls = self._deck(self.readings({"p2", "p3", "p4", "p5"}),
                                 env={"PPT_IMAGE_CHECK_MAX_REPAINTS": "1"})
        self.assertEqual(calls["paint"], 7)

    def test_every_painted_slide_has_a_painted_frame_cropped_cover_included(self):
        data, calls = self._deck(self.readings(set()), framed={"p1", "p4"})
        self.assertEqual(data["generation"]["framesCropped"], 2)

    def test_the_check_can_be_switched_off(self):
        data, calls = self._deck(self.readings({"p3"}), env={"PPT_IMAGE_SLIDE_CHECK": "false"})
        self.assertEqual((calls["paint"], calls["read"]), (6, 0))


class QuizAndRepaintWindowTests(SimpleTestCase):
    def test_a_list_of_questions_is_painted_as_questions_with_no_invented_answers(self):
        quiz = {"slideType": "quiz", "title": "Quick Check", "blocks": [
            {"kind": "question", "label": "Glucose", "text": "Where is glucose broken down?"},
            {"kind": "question", "label": "Air", "text": "Where does inhaled air go?"}]}
        p = image_slides.slide_prompt(quiz, number=7, total=8)
        self.assertIn("Question cards, each with a coloured number circle", p)
        self.assertIn(image_slides.SQUARE_GRID, p)
        self.assertIn("Show no answers.", p)
        self.assertNotIn("lettered A, B, C, D", p)

    def test_a_question_with_answers_still_gets_lettered_options(self):
        quiz = {"slideType": "quiz", "title": "Quick Check", "blocks": [
            {"kind": "question", "text": "Where does glycolysis happen?"},
            {"kind": "answer", "text": "Cytoplasm"}, {"kind": "answer", "text": "Nucleus"}]}
        p = image_slides.slide_prompt(quiz, number=7, total=8)
        self.assertIn("lettered A, B, C, D", p)
        self.assertIn('A "Cytoplasm"', p)

    def test_a_flagged_slide_is_retried_later_in_the_deck_than_before(self):
        self.assertEqual(image_slides.repaint_window(), 420.0)
        with mock.patch.dict("os.environ", {"SLIDE_REPAINT_WINDOW_S": "90"}):
            self.assertEqual(image_slides.repaint_window(), 90.0)


class PaintedFrameTests(SimpleTestCase):
    """A live cover came back as a white slide on a grey page, and 10 of 122 painted
    slides on disk had such a frame; the studio showed it as padding round the slide."""

    W, H = 384, 216

    def slide(self, page=None, inset=(20, 10)):
        import numpy as np
        from PIL import Image

        a = np.full((self.H, self.W, 3), 248, dtype=np.uint8)            # a white slide
        a[60:90, 40:170] = (30, 90, 200)                                 # a card
        a[40:180, 230:350] = (120, 200, 160)                             # a picture
        if page is None:
            return a
        x, y = inset
        framed = np.empty_like(a)
        framed[:] = page
        framed[y:self.H - y, x:self.W - x] = np.asarray(
            Image.fromarray(a).resize((self.W - 2 * x, self.H - 2 * y)))
        return framed

    def test_a_slide_on_a_page_is_found(self):
        box = slide_check.find_frame(self.slide(page=(130, 146, 153)))
        self.assertEqual(box, (20, 10, self.W - 20, self.H - 10))

    def test_a_slide_that_fills_the_image_is_left_alone(self):
        self.assertIsNone(slide_check.find_frame(self.slide()))

    def test_content_on_a_plain_background_is_not_mistaken_for_a_frame(self):
        import numpy as np

        a = np.full((self.H, self.W, 3), 30, dtype=np.uint8)              # a dark slide,
        a[8:22, 30:200:3] = (250, 250, 250)                              # its title in the margin,
        a[40:200, 30:350] = (240, 240, 240)                              # one big pale panel
        self.assertIsNone(slide_check.find_frame(a))

    def test_the_frame_is_cropped_in_place_at_the_same_size(self):
        import os
        import tempfile

        from PIL import Image

        big = Image.fromarray(self.slide(page=(130, 146, 153), inset=(24, 14))).resize((1536, 864))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            big.save(path)
            self.assertTrue(slide_check.unframe(path))
            with Image.open(path) as out:
                self.assertEqual(out.size, (1536, 864))
                rgb = out.convert("RGB")
                for corner in ((3, 3), (1532, 3), (3, 860), (1532, 860)):
                    self.assertGreater(min(rgb.getpixel(corner)), 200, corner)
            self.assertFalse(slide_check.unframe(path))                  # nothing left to crop
        finally:
            os.remove(path)

    def test_an_unreadable_file_is_left_as_it_is(self):
        self.assertFalse(slide_check.unframe("no/such/slide.png"))


class FourthRoundTests(SimpleTestCase):
    """A live dark-theme deck: one slide painted the planner's markdown asterisks
    on its title and every card, and a two-card slide's text was visibly bigger
    than the rest but passed the check, because bold headings were measured as
    card text and the size tolerance was 22%."""

    def test_markdown_in_slide_text_is_not_painted(self):
        slide = {"slideType": "concept", "title": "**Aerobic Respiration** in Mitochondria",
                 "emphasisText": "**Aerobic Respiration", "blocks": [
                     {"kind": "bullet", "label": "**Location**",
                      "text": "**Aerobic respiration occurs inside the mitochondria.**"}]}
        p = image_slides.slide_prompt(slide, number=5, total=8)
        quoted = " ".join(re.findall(r'"[^"]*"', p))
        self.assertNotIn("*", quoted)
        self.assertIn('"Aerobic respiration occurs inside the mitochondria."', p)
        self.assertIn('heading "Location"', p)

    def test_headings_are_not_measured_as_card_text(self):
        lines = [(0.10, 8.0, "Respiration"),
                 (0.35, 5.5, "Definition"), (0.40, 4.0, "Respiration uses food material in cells."),
                 (0.55, 5.5, "Importance"), (0.60, 4.0, "Movement and growth depend on energy.")]
        cards = ["Definition Respiration uses food material in cells.",
                 "Importance Movement and growth depend on energy."]
        mixed = read_slide("x.png", title="Respiration", cards=cards, ocr=ocr(lines))
        apart = read_slide("x.png", title="Respiration", cards=cards,
                           headings=["Definition", "Importance"], ocr=ocr(lines))
        self.assertEqual(mixed.body_h, 4.75)          # headings pulled the median up
        self.assertEqual(apart.body_h, 4.0)
        self.assertEqual(apart.coverage, 1.0)         # headings still count as found

    def test_the_plan_tells_the_check_which_words_are_headings(self):
        exp = image_slides._expected_text({"slideType": "concept", "title": "T", "blocks": [
            {"kind": "bullet", "label": "Glucose:", "text": "A sugar."},
            {"kind": "bullet", "text": "No heading here."}]})
        self.assertEqual(exp["headings"], ["Glucose"])

    def test_two_pixels_off_the_deck_is_flagged_one_is_not(self):
        readings = {i: Reading(title_h=8.68, body_h=4.51, coverage=0.9, tokens=40) for i in range(1, 7)}
        readings[3] = Reading(title_h=8.68, body_h=4.86, coverage=0.9, tokens=40)   # +8%
        readings[4] = Reading(title_h=8.68, body_h=5.21, coverage=0.9, tokens=40)   # +15.5%
        readings[5] = Reading(title_h=6.60, body_h=4.51, coverage=0.9, tokens=40)   # title -24%
        bad = judge(readings)
        self.assertNotIn(3, bad)
        self.assertEqual(bad[4], ["card text too big"])
        self.assertEqual(bad[5], ["title too small"])


class PaintEarlyTests(SimpleTestCase):
    """Painting starts as soon as the slides are written; the chapter and fact
    checks run alongside it (on a live deck they took 43s, changed 2 of 8 slides,
    and painting used to wait for both). A slide a check changes is painted
    again, and a painting of unchecked text never reaches the deck."""

    def run_deck(self, *, fact_check=None, scope_check=None, render_fails=(), env=None, n=6):
        import threading

        calls = []
        lock = threading.Lock()

        class Rendered(threading.Event):
            """Set by the first render; ``wait_for(text)``: a render containing ``text`` is done."""
            def wait_for(self, text, timeout=5):
                for _ in range(int(timeout * 100)):
                    with lock:
                        if any(text in c for c in done):
                            return True
                    threading.Event().wait(0.01)
                return False
        first_render = Rendered()
        done = []

        def render(prompt, **kw):
            with lock:
                calls.append(prompt)
                k = len(calls)
            first_render.set()
            # Every painting of the checked text fails - whether it is the first
            # painting (the check finished before the slide's turn came) or a
            # repaint of a slide first painted with the old text.
            if any(marker in prompt for marker in render_fails):
                return {"ok": False, "error": "provider down"}
            with lock:
                done.append(prompt)
            return {"ok": True, "url": "u%d" % k, "path": "p%d" % k}

        slides = [dict(SLIDE, slideType="title", title="Cover")] + [
            {"slideType": "concept", "title": "Slide %d" % (i + 2),
             "blocks": [{"kind": "bullet", "label": "Point", "text": "Original text %d." % (i + 2)}]}
            for i in range(n - 1)]
        plan = PlannerResult(title="T", slides=slides)
        env = dict({"PPT_IMAGE_SLIDE_CHECK": "false"}, **(env or {}))
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck", return_value=plan), \
             mock.patch("ai_services.core.ppt_v2.scope.replan_if_off_scope",
                        side_effect=scope_check or (lambda plan, **kw: plan)), \
             mock.patch("ai_services.core.ppt_v2.service._fact_check",
                        side_effect=(lambda plan, **kw: fact_check(plan, first_render))
                        if fact_check else (lambda plan, **kw: None)), \
             mock.patch.object(slide_check, "unframe", return_value=False), \
             mock.patch.dict("os.environ", env):
            data = image_slides.generate_image_deck(topic="T", slide_count=n, ctx={},
                                                    render=render)
        return data, calls

    def test_painting_runs_while_the_facts_are_checked(self):
        seen = {}

        def fact_check(plan, first_render):
            seen["painting"] = first_render.wait(5)
        data, calls = self.run_deck(fact_check=fact_check)
        self.assertTrue(seen["painting"])
        self.assertEqual(len(calls), 6)                       # nothing changed, nothing repainted
        self.assertTrue(data["generation"]["paintedEarly"])
        self.assertEqual(data["generation"]["repaintedAfterCheck"], [])

    def test_a_slide_the_fact_check_changes_is_painted_again_with_the_checked_text(self):
        def fact_check(plan, first_render):
            first_render.wait_for("Original text 3.")         # painted with the old text first
            plan.slides[2]["blocks"][0]["text"] = "Corrected text 3."
        data, calls = self.run_deck(fact_check=fact_check)
        self.assertEqual(data["generation"]["repaintedAfterCheck"], [3])
        url = data["slides"][2]["slideImage"]["url"]
        self.assertIn("Corrected text 3.", calls[int(url[1:]) - 1])   # the deck shows the repaint
        self.assertEqual(data["generation"]["failedSlides"], [])

    def test_unchecked_text_is_never_shipped_when_the_repaint_fails(self):
        def fact_check(plan, first_render):
            first_render.wait(5)
            plan.slides[2]["blocks"][0]["text"] = "Corrected text 3."
        data, calls = self.run_deck(fact_check=fact_check, render_fails=("Corrected text 3.",),
                                    env={"SLIDE_REPAINT_WINDOW_S": "0"})
        self.assertNotIn("slideImage", data["slides"][2])
        self.assertEqual(data["generation"]["failedSlides"], [3])
        self.assertIn("Corrected text 3.", " ".join(data["slides"][2]["bullets"]))

    def test_a_check_that_changes_the_slide_count_starts_the_pictures_again(self):
        def scope_check(plan, **kw):
            return PlannerResult(title="T", slides=plan.slides[:4])
        data, calls = self.run_deck(scope_check=scope_check)
        self.assertEqual(len(data["slides"]), 4)
        self.assertEqual(data["generation"]["failedSlides"], [])
        for i, s in enumerate(data["slides"]):
            self.assertIn("%d / 4" % (i + 1), calls[int(s["slideImage"]["url"][1:]) - 1])

    def test_painting_can_wait_for_the_checks_as_before(self):
        seen = {}

        def fact_check(plan, first_render):
            seen["painting"] = first_render.is_set()
        data, calls = self.run_deck(fact_check=fact_check, env={"PPT_IMAGE_PAINT_EARLY": "false"})
        self.assertFalse(seen["painting"])
        self.assertFalse(data["generation"]["paintedEarly"])
        self.assertEqual(len(calls), 6)


class CheckRepaintTimeCapTests(SimpleTestCase):
    def test_text_check_repaints_stop_once_the_deck_has_run_long(self):
        self.assertEqual(image_slides.check_window(), 210.0)
        data, calls = DeckTests()._deck(DeckTests.readings({"p3"}),
                                         env={"PPT_IMAGE_CHECK_UNTIL_S": "0.0001"})
        self.assertEqual(calls["paint"], 6)                   # flagged, but no time to repaint


class PaintPerBatchTests(SimpleTestCase):
    """Slides are painted as their planner batch is written: a live 15-slide deck
    waited 76s for all five batches before painting anything, though the first
    was ready after 27s."""

    def make_slides(self, n, text="Text"):
        return [{"slideType": "title" if i == 0 else "concept", "title": "Slide %d" % (i + 1),
                 "blocks": [{"kind": "bullet", "label": "Point", "text": "%s %d." % (text, i + 1)}]}
                for i in range(n)]

    def run_deck(self, plan_deck, render=None, n=6):
        import threading

        calls, lock = [], threading.Lock()
        first_render = threading.Event()

        def default_render(prompt, **kw):
            with lock:
                calls.append(prompt)
                k = len(calls)
            first_render.set()
            return {"ok": True, "url": "u%d" % k, "path": "p%d" % k}
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        side_effect=lambda **kw: plan_deck(kw, first_render)), \
             mock.patch("ai_services.core.ppt_v2.scope.replan_if_off_scope",
                        side_effect=lambda plan, **kw: plan), \
             mock.patch("ai_services.core.ppt_v2.service._fact_check", return_value=None), \
             mock.patch.object(slide_check, "unframe", return_value=False), \
             mock.patch.dict("os.environ", {"PPT_IMAGE_SLIDE_CHECK": "false"}):
            data = image_slides.generate_image_deck(topic="T", slide_count=n, ctx={},
                                                    render=render or default_render)
        return data, calls

    def test_the_first_batch_is_painting_while_the_rest_is_written(self):
        slides = self.make_slides(6)
        seen = {}

        def plan_deck(kw, first_render):
            kw["on_batch"](0, slides[:3], 6)
            seen["painting_before_last_batch"] = first_render.wait(5)
            kw["on_batch"](3, slides[3:], 6)
            return PlannerResult(title="T", slides=slides)
        data, calls = self.run_deck(plan_deck)
        self.assertTrue(seen["painting_before_last_batch"])
        self.assertEqual(len(calls), 6)                       # nothing painted twice
        self.assertEqual(data["generation"]["failedSlides"], [])
        self.assertEqual(data["generation"]["repaintedAfterCheck"], [])

    def test_a_slide_that_changed_after_its_batch_is_painted_again(self):
        import threading

        slides = self.make_slides(6)
        calls, lock = [], threading.Lock()
        old_text_painted = threading.Event()

        def render(prompt, **kw):
            with lock:
                calls.append(prompt)
                k = len(calls)
            if "Text 2." in prompt:
                old_text_painted.set()
            return {"ok": True, "url": "u%d" % k, "path": "p%d" % k}

        def plan_deck(kw, first_render):
            kw["on_batch"](0, [dict(s) for s in slides[:3]], 6)
            old_text_painted.wait(5)        # slide 2 was painted from its batch...
            final = [dict(s) for s in slides]
            final[1] = dict(final[1], blocks=[{"kind": "bullet", "label": "Point",
                                               "text": "Changed text 2."}])
            return PlannerResult(title="T", slides=final)   # ...then the plan changed it
        data, _ = self.run_deck(plan_deck, render=render)
        self.assertEqual(data["generation"]["repaintedAfterCheck"], [2])
        url = data["slides"][1]["slideImage"]["url"]
        self.assertIn("Changed text 2.", calls[int(url[1:]) - 1])

    def test_a_batch_for_a_different_deck_size_is_painted_again_from_the_plan(self):
        slides = self.make_slides(6)

        def plan_deck(kw, first_render):
            kw["on_batch"](0, self.make_slides(3), 8)            # the outline said 8
            first_render.wait(5)
            return PlannerResult(title="T", slides=slides)      # the plan has 6
        data, calls = self.run_deck(plan_deck)
        self.assertEqual(len(data["slides"]), 6)
        self.assertEqual(data["generation"]["failedSlides"], [])
        for i, s in enumerate(data["slides"]):
            self.assertIn("%d / 6" % (i + 1), calls[int(s["slideImage"]["url"][1:]) - 1])

    def test_a_slide_the_image_filter_refuses_is_painted_again_at_once(self):
        slides = self.make_slides(4)
        refused = {"done": False}
        calls = []

        def render(prompt, **kw):
            calls.append(prompt)
            if "Text 3." in prompt and not refused["done"]:
                refused["done"] = True
                return {"ok": False, "error": "Together image API error 400: Invalid content "
                                              "detected. The generated content was flagged and "
                                              "rejected by Alibaba's content moderation system."}
            return {"ok": True, "url": "u%d" % len(calls), "path": "p%d" % len(calls)}

        def plan_deck(kw, first_render):
            return PlannerResult(title="T", slides=slides)
        data, _ = self.run_deck(plan_deck, render=render, n=4)
        self.assertEqual(data["generation"]["failedSlides"], [])
        self.assertEqual(len(calls), 5)                       # one extra painting, at once


class RepaintSoonerTests(SimpleTestCase):
    def test_a_slide_corrected_after_it_was_painted_is_repainted_at_once(self):
        # Live: slide 8 was painted, the fact check then corrected it, and its
        # repaint waited 29s until every other slide had been painted.
        import threading

        order, lock = [], threading.Lock()
        slow_started, release_slow = threading.Event(), threading.Event()
        slide2_painted = threading.Event()
        slides = [{"slideType": "title" if i == 0 else "concept", "title": "S%d" % (i + 1),
                   "blocks": [{"kind": "bullet", "label": "P", "text": "Original %d." % (i + 1)}]}
                  for i in range(4)]

        def render(prompt, **kw):
            with lock:
                order.append(prompt)
            if "Original 4." in prompt:            # slide 4 is still painting...
                slow_started.set()
                release_slow.wait(5)
            if "Original 2." in prompt:
                slide2_painted.set()
            return {"ok": True, "url": "u", "path": "p"}

        def fact_check(plan, **kw):
            slow_started.wait(5)                    # ...when the check corrects slide 2,
            slide2_painted.wait(5)                  # which was painted with the old text
            plan.slides[1]["blocks"][0]["text"] = "Corrected 2."

        def repaint_seen():
            for _ in range(200):
                with lock:
                    if any("Corrected 2." in p for p in order):
                        return True
                threading.Event().wait(0.01)
            return False
        result = {}
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        return_value=PlannerResult(title="T", slides=slides)), \
             mock.patch("ai_services.core.ppt_v2.scope.replan_if_off_scope",
                        side_effect=lambda plan, **kw: plan), \
             mock.patch("ai_services.core.ppt_v2.service._fact_check", side_effect=fact_check), \
             mock.patch.object(slide_check, "unframe", return_value=False), \
             mock.patch.dict("os.environ", {"PPT_IMAGE_SLIDE_CHECK": "false",
                                            "SLIDE_IMAGE_CONCURRENCY": "4"}):
            t = threading.Thread(target=lambda: result.update(data=image_slides.generate_image_deck(
                topic="T", slide_count=4, ctx={}, render=render)))
            t.start()
            seen = repaint_seen()                    # while slide 4 is still painting
            release_slow.set()
            t.join(5)
        self.assertTrue(seen)
        self.assertEqual(result["data"]["generation"]["repaintedAfterCheck"], [2])

    def test_the_text_check_repaints_the_worst_two_by_default(self):
        self.assertEqual(image_slides.max_check_repaints(10), 2)
        self.assertEqual(image_slides.max_check_repaints(3), 1)


class LostSlideRetryTests(SimpleTestCase):
    """A slide the first pass could not paint gets more tries, so a painted deck
    does not come back with text-only slides after a provider hiccup."""

    def run_deck(self, outcomes, env=None, n=4):
        """``outcomes[k]``: what the k-th painting of slide 3 returns (others succeed)."""
        import threading

        calls, lock = {"slide3": 0, "all": 0}, threading.Lock()
        slides = [{"slideType": "title" if i == 0 else "concept", "title": "S%d" % (i + 1),
                   "blocks": [{"kind": "bullet", "label": "P", "text": "Point %d." % (i + 1)}]}
                  for i in range(n)]

        def render(prompt, **kw):
            with lock:
                calls["all"] += 1
                if "Point 3." in prompt:
                    k = calls["slide3"]
                    calls["slide3"] += 1
                    return outcomes[min(k, len(outcomes) - 1)]
            return {"ok": True, "url": "u", "path": "p"}
        sleeps = []
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        return_value=PlannerResult(title="T", slides=slides)), \
             mock.patch("ai_services.core.ppt_v2.scope.replan_if_off_scope",
                        side_effect=lambda plan, **kw: plan), \
             mock.patch("ai_services.core.ppt_v2.service._fact_check", return_value=None), \
             mock.patch.object(slide_check, "unframe", return_value=False), \
             mock.patch.object(image_slides, "_sleep", side_effect=sleeps.append), \
             mock.patch.dict("os.environ", dict({"PPT_IMAGE_SLIDE_CHECK": "false"}, **(env or {}))):
            data = image_slides.generate_image_deck(topic="T", slide_count=n, ctx={}, render=render)
        return data, calls, sleeps

    FAIL = {"ok": False, "error": "Together image API error 503: busy", "fatal": False}
    OK = {"ok": True, "url": "u3", "path": "p3"}

    def test_a_slide_that_keeps_failing_for_a_while_is_painted_in_the_end(self):
        # First pass + 2 rounds fail; the third round paints it.
        data, calls, sleeps = self.run_deck([self.FAIL, self.FAIL, self.FAIL, self.OK])
        self.assertEqual(data["generation"]["failedSlides"], [])
        self.assertEqual(calls["slide3"], 4)
        self.assertEqual(len(sleeps), 2)                     # a pause before rounds 2 and 3

    def test_after_the_last_round_the_slide_stays_text_and_the_deck_still_arrives(self):
        data, calls, _ = self.run_deck([self.FAIL])
        self.assertEqual(data["generation"]["failedSlides"], [3])
        self.assertEqual(calls["slide3"], 1 + image_slides.REPAINT_ATTEMPTS)
        self.assertEqual(len(data["slides"]), 4)
        self.assertIn("slideImage", data["slides"][1])         # the rest are painted

    def test_a_failure_that_would_repeat_is_not_retried(self):
        fatal = {"ok": False, "error": "Together image API error 401: bad key", "fatal": True}
        data, calls, _ = self.run_deck([fatal])
        self.assertEqual(calls["slide3"], 1)
        self.assertEqual(data["generation"]["failedSlides"], [3])

    def test_no_retry_once_the_deck_is_out_of_time(self):
        data, calls, _ = self.run_deck([self.FAIL, self.OK], env={"SLIDE_REPAINT_WINDOW_S": "0"})
        self.assertEqual(calls["slide3"], 1)
        self.assertEqual(data["generation"]["failedSlides"], [3])


class ImageProviderDownTests(SimpleTestCase):
    def test_when_no_slide_can_be_painted_it_does_not_spend_minutes_retrying(self):
        # The standard deck the caller builds instead should not wait behind
        # rounds of retries against a provider that is down.
        t = LostSlideRetryTests()
        calls = {"n": 0}
        slides = [{"slideType": "concept", "title": "S%d" % i,
                   "blocks": [{"kind": "bullet", "label": "P", "text": "Point %d." % i}]} for i in range(4)]

        def render(prompt, **kw):
            calls["n"] += 1
            return {"ok": False, "error": "Together image API error 503: down", "fatal": False}
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck",
                        return_value=PlannerResult(title="T", slides=slides)), \
             mock.patch("ai_services.core.ppt_v2.scope.replan_if_off_scope",
                        side_effect=lambda plan, **kw: plan), \
             mock.patch("ai_services.core.ppt_v2.service._fact_check", return_value=None), \
             mock.patch.object(image_slides, "_sleep") as sleep, \
             mock.patch.dict("os.environ", {"PPT_IMAGE_SLIDE_CHECK": "false"}):
            data = image_slides.generate_image_deck(topic="T", slide_count=4, ctx={}, render=render)
        self.assertEqual(calls["n"], 4)                      # one try each, no rounds
        sleep.assert_not_called()
        self.assertEqual(data["generation"]["imagesGenerated"], 0)   # -> the caller builds a V1 deck
