"""Every layout, at every block count it claims to serve, must be drawable.

The layouts compute geometry arithmetically, so an off-by-one in a gap or a
column divisor produces a box that is slightly off the slide or slightly on
top of its neighbour. Neither raises; both look broken. The sweep below is the
only thing standing between that and a teacher's screen.
"""
from django.test import SimpleTestCase

from ai_services.core.ppt_v2.design import (
    MIN_BODY_CONTRAST,
    MIN_LARGE_CONTRAST,
    PALETTE_WORDS,
    SUBJECT_THEMES,
    build_theme,
    contrast_ratio,
    theme_key_for_subject,
)
from ai_services.core.ppt_v2.layouts import LAYOUTS, select_layout
from ai_services.core.ppt_v2.schema import SlideType

_TEXT_PREFIXES = ("title", "subtitle", "content", "block_", "label",
                  "question", "equation", "footer")


def _is_text(name):
    return any(name == p or name.startswith(p) for p in _TEXT_PREFIXES)


class EveryLayoutTests(SimpleTestCase):
    def test_all_regions_stay_on_the_slide(self):
        for name, layout in LAYOUTS.items():
            for blocks in range(layout.min_blocks, layout.max_blocks + 1):
                for has_visual in (True, False):
                    for slot, region in layout.regions(blocks, has_visual).items():
                        errs = region.errors("%s/%s" % (name, slot))
                        self.assertEqual(
                            errs, [],
                            "%s with %d blocks, visual=%s: %s"
                            % (name, blocks, has_visual, errs))

    def test_text_regions_do_not_collide(self):
        for name, layout in LAYOUTS.items():
            for blocks in range(layout.min_blocks, layout.max_blocks + 1):
                for has_visual in (True, False):
                    regions = layout.regions(blocks, has_visual)
                    slots = sorted(s for s in regions if _is_text(s))
                    for i, a in enumerate(slots):
                        for b in slots[i + 1:]:
                            # "content" is the single-stack alias for block_N.
                            if "content" in (a, b) and (a.startswith("block_")
                                                        or b.startswith("block_")):
                                continue
                            self.assertFalse(
                                regions[a].overlaps(regions[b]),
                                "%s (%d blocks, visual=%s): %s overlaps %s"
                                % (name, blocks, has_visual, a, b))

    def test_the_visual_never_sits_under_the_body_text(self):
        for name, layout in LAYOUTS.items():
            if name == "title_hero":
                continue  # a full-bleed cover image is the point of that one
            for blocks in range(max(layout.min_blocks, 1), layout.max_blocks + 1):
                regions = layout.regions(blocks, True)
                visual = regions.get("visual")
                if visual is None:
                    continue
                for slot, region in regions.items():
                    if slot in ("title", "subtitle", "footer") or not _is_text(slot):
                        continue
                    self.assertFalse(
                        visual.overlaps(region),
                        "%s (%d blocks): visual overlaps %s" % (name, blocks, slot))

    def test_asking_for_a_visual_yields_a_place_to_put_one(self):
        for name, layout in LAYOUTS.items():
            if not layout.wants_visual:
                continue
            blocks = max(layout.min_blocks, 1)
            self.assertIn("visual", layout.regions(blocks, True),
                          "%s claims to take a visual but offers no region" % name)

    def test_every_layout_has_a_title_slot(self):
        for name, layout in LAYOUTS.items():
            self.assertIn("title", layout.regions(max(layout.min_blocks, 1), True), name)

    def test_block_slots_are_produced_for_block_based_layouts(self):
        for name in ("process_steps", "comparison", "cause_effect",
                     "summary_grid", "timeline", "quiz"):
            layout = LAYOUTS[name]
            regions = layout.regions(layout.min_blocks, False)
            got = [s for s in regions if s.startswith("block_")]
            self.assertEqual(len(got), layout.min_blocks,
                             "%s produced %s" % (name, got))


class ProcessLayoutTests(SimpleTestCase):
    def test_few_steps_run_across_and_many_run_down(self):
        across = LAYOUTS["process_steps"].regions(3, False)
        self.assertEqual(across["block_0"].y, across["block_2"].y)
        down = LAYOUTS["process_steps"].regions(6, False)
        self.assertEqual(down["block_0"].x, down["block_5"].x)
        self.assertLess(down["block_0"].y, down["block_5"].y)

    def test_there_is_a_connector_between_each_pair_of_steps(self):
        regions = LAYOUTS["process_steps"].regions(4, False)
        self.assertEqual(sum(1 for s in regions if s.startswith("connector_")), 3)


class SelectionTests(SimpleTestCase):
    def test_slide_type_picks_its_layout(self):
        for slide_type, expected in (
            (SlideType.TITLE, "title_hero"),
            (SlideType.PROCESS, "process_steps"),
            (SlideType.COMPARISON, "comparison"),
            (SlideType.CAUSE_EFFECT, "cause_effect"),
            (SlideType.TIMELINE, "timeline"),
            (SlideType.FORMULA, "formula_equation"),
            (SlideType.SUMMARY, "summary_grid"),
            (SlideType.QUIZ, "quiz"),
        ):
            self.assertEqual(select_layout(slide_type, 2).name, expected)

    def test_a_repeated_layout_is_avoided_when_an_alternative_fits(self):
        first = select_layout(SlideType.CONCEPT, 3)
        second = select_layout(SlideType.CONCEPT, 3, avoid=[first.name])
        self.assertNotEqual(first.name, second.name)

    def test_variety_never_beats_correctness(self):
        # Only one layout serves a quiz, so avoidance must not push it onto
        # something that cannot hold a question.
        self.assertEqual(select_layout(SlideType.QUIZ, 4, avoid=["quiz"]).name, "quiz")

    def test_selection_is_deterministic(self):
        runs = {select_layout(SlideType.CONCEPT, 3, avoid=["x"]).name for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_an_impossible_block_count_still_returns_a_layout(self):
        layout = select_layout(SlideType.COMPARISON, 99)
        self.assertIn(SlideType.COMPARISON, layout.serves)


class ThemeTests(SimpleTestCase):
    def test_body_text_is_readable_on_every_theme(self):
        for slide_type in SlideType:
            for subject in list(SUBJECT_THEMES) + ["Science", "Social Science"]:
                theme = build_theme(subject=subject, slide_type=slide_type)
                background = theme.color("surface")
                for token, minimum in (("ink", MIN_BODY_CONTRAST),
                                       ("ink_muted", MIN_BODY_CONTRAST),
                                       ("heading", MIN_LARGE_CONTRAST)):
                    ratio = contrast_ratio(theme.color(token), background)
                    self.assertGreaterEqual(
                        ratio, minimum,
                        "%s/%s: %s on %s is only %.2f:1"
                        % (subject, slide_type.value, token, background, ratio))

    def test_text_on_a_coloured_fill_is_readable(self):
        for subject in SUBJECT_THEMES:
            theme = build_theme(subject=subject)
            for text, fill in (("on_primary", "primary"), ("on_accent", "accent")):
                self.assertGreaterEqual(
                    contrast_ratio(theme.color(text), theme.color(fill)),
                    MIN_LARGE_CONTRAST, "%s: %s on %s" % (subject, text, fill))

    def test_subjects_map_to_their_palette(self):
        for subject, expected in (("Biology", "biology"), ("Physics", "physics"),
                                  ("Chemistry", "chemistry"), ("Maths", "mathematics"),
                                  ("History", "history"), ("Geography", "geography"),
                                  ("Sanskrit", "default"), ("", "default")):
            self.assertEqual(theme_key_for_subject(subject), expected, subject)

    def test_generic_science_is_split_by_chapter(self):
        # CBSE "Science" is one book covering three subjects, so the chapter
        # is the only thing that says which one a slide is actually in.
        self.assertEqual(theme_key_for_subject("Science", "Metals and Non-metals"),
                         "chemistry")
        self.assertEqual(theme_key_for_subject("Science", "Light and Reflection"),
                         "physics")
        self.assertEqual(theme_key_for_subject("Science", "Cell Structure"), "biology")
        self.assertEqual(theme_key_for_subject("Science", ""), "biology")

    def test_social_science_is_not_mistaken_for_science(self):
        self.assertEqual(theme_key_for_subject("Social Science", "The Mughal Empire"),
                         "history")

    def test_every_palette_has_plain_language_words_for_the_image_prompt(self):
        for key in SUBJECT_THEMES:
            self.assertIn(key, PALETTE_WORDS)

    def test_the_title_slide_inverts_its_text(self):
        theme = build_theme("Biology", slide_type=SlideType.TITLE)
        self.assertTrue(theme.background.startswith("gradient:"))
        self.assertGreaterEqual(contrast_ratio(theme.color("ink"),
                                               theme.color("surface")),
                                MIN_BODY_CONTRAST)
