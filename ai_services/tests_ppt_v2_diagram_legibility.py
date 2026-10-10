"""A diagram nobody can read is worse than no diagram.

Phase 5 rasterised a process slide through real PowerPoint and the generated
figure was unreadable. The cause turned out to be arithmetic, not aesthetics:
the figure was drawn at matplotlib's default 6.4x4.8in and then fitted into a
8.9x1.69in band, a 0.35x reduction, so the prompt's "fontsize 10 or more"
floor arrived on the slide at 3.5pt.

Two changes follow from that, and these tests hold them in place:

  * the figure is drawn at exactly the size it will occupy, so the point
    sizes in the code are the point sizes the class sees;
  * what the code asks for is checked before it is accepted — undersized
    type is repaired mechanically, and the things that need the figure
    redrawn are rejected so the caller can fall through to another visual.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core import diagram as D

# The process_steps visual band, in inches on the 10x5.625in canvas.
BAND = (8.9, 1.69)
PANEL = (4.2, 3.46)


class SizingTests(SimpleTestCase):
    def test_the_figure_is_drawn_at_the_size_it_will_occupy(self):
        # This is the whole fix: a 1:1 fit means the code's point sizes are
        # the slide's point sizes.
        preamble = D._preamble(*BAND, D._MIN_SLIDE_PT)
        self.assertIn("'figure.figsize': (8.9, 1.69)", preamble)

    def test_the_type_scale_is_set_for_classroom_distance(self):
        preamble = D._preamble(*BAND, D._MIN_SLIDE_PT)
        self.assertIn("'font.size': 13.0", preamble)
        self.assertGreaterEqual(D._MIN_SLIDE_PT, 13.0)

    def test_lines_are_heavy_enough_to_see(self):
        self.assertIn("'lines.linewidth': 2.0", D._preamble(*BAND, 13.0))
        # A retry leans harder.
        self.assertIn("'lines.linewidth': 2.6", D._preamble(*BAND, 13.0, strict=True))

    def test_the_saved_geometry_is_kept_exactly(self):
        # A tight crop would make the saved aspect depend on how sparse the
        # drawing happened to be, which is what broke the fit before.
        self.assertIn("_SAVE_BBOX = None", D._preamble(*BAND, 13.0))

    def test_the_label_budget_comes_from_the_space_available(self):
        self.assertEqual(D.label_budget(*BAND), 8)
        self.assertEqual(D.label_budget(*PANEL), 7)
        # Always a sane floor and ceiling, never zero labels or forty.
        self.assertEqual(D.label_budget(0.2, 0.2), D._MIN_LABEL_BUDGET)
        self.assertEqual(D.label_budget(40, 40), D._MAX_LABEL_BUDGET)


class AxisSuppressionTests(SimpleTestCase):
    """A drawing is not a plot."""

    def test_a_drawing_has_no_spines_or_tick_labels(self):
        # A benchmark slide rasterised a plant cell with matplotlib's
        # 0-2-4-6-8-10 tick labels around it.
        preamble = D._preamble(*PANEL, 13.0, axes=False)
        self.assertIn("'axes.spines.left': False", preamble)
        self.assertIn("'xtick.labelbottom': False", preamble)
        self.assertIn("'ytick.labelleft': False", preamble)

    def test_a_chart_keeps_its_axes(self):
        self.assertNotIn("axes.spines", D._preamble(*PANEL, 13.0, axes=True))

    def test_the_default_keeps_axes_for_existing_callers(self):
        self.assertNotIn("axes.spines", D._preamble(*PANEL, 13.0))


class FontRepairTests(SimpleTestCase):
    def test_undersized_type_is_lifted_not_rejected(self):
        # The model reliably writes fontsize=12 out of print-figure habit.
        # Rejecting the figure for it threw away an otherwise good drawing
        # and usually failed the retry the same way.
        self.assertEqual(D.raise_small_fonts('ax.text(0, 0, "A", fontsize=12)'),
                         'ax.text(0, 0, "A", fontsize=13)')

    def test_type_that_is_already_big_enough_is_left_alone(self):
        code = 'ax.text(0, 0, "A", fontsize=20)'
        self.assertEqual(D.raise_small_fonts(code), code)

    def test_tick_label_sizes_are_lifted_too(self):
        self.assertEqual(D.raise_small_fonts("ax.tick_params(labelsize=8)"),
                         "ax.tick_params(labelsize=13)")

    def test_several_sizes_in_one_figure_are_each_handled(self):
        out = D.raise_small_fonts('a(fontsize=6); b(fontsize=14); c(fontsize=9)')
        self.assertEqual(out, 'a(fontsize=13); b(fontsize=14); c(fontsize=13)')

    def test_code_without_any_font_setting_is_untouched(self):
        code = "ax.plot([0, 1], [0, 1])"
        self.assertEqual(D.raise_small_fonts(code), code)

    def test_repaired_code_then_passes_the_check(self):
        self.assertIsNone(D._legibility_problem(
            D.raise_small_fonts('ax.text(0, 0, "x", fontsize=7)'), *BAND))


class LegibilityCheckTests(SimpleTestCase):
    def test_too_many_labels_for_the_space_is_rejected(self):
        problem = D._legibility_problem('ax.text(0,0,"x")' * 14, *BAND)
        self.assertIsNotNone(problem)
        self.assertIn("labels", problem)

    def test_a_label_count_within_budget_passes(self):
        self.assertIsNone(D._legibility_problem('ax.text(0,0,"x")' * 6, *BAND))

    def test_annotate_counts_towards_the_budget_too(self):
        self.assertIsNotNone(
            D._legibility_problem('ax.annotate("x",(0,0))' * 14, *BAND))

    def test_a_legend_is_rejected_in_a_small_space(self):
        problem = D._legibility_problem("ax.legend()", 4.2, 1.69)
        self.assertIsNotNone(problem)
        self.assertIn("legend", problem)

    def test_a_clean_figure_passes(self):
        self.assertIsNone(D._legibility_problem(
            'ax.plot([0,1],[0,1])\nax.text(0,0,"A")', *BAND))

    def test_the_rejection_says_what_to_do_about_it(self):
        # The reason is fed back into the retry, so it has to be actionable.
        problem = D._legibility_problem('ax.text(0,0,"x")' * 14, *BAND)
        self.assertIn("fewer, larger", problem)


class DensityTests(SimpleTestCase):
    def test_a_solid_mass_of_ink_is_rejected(self):
        self.assertLess(D._MIN_INK_FRACTION, D._MAX_INK_FRACTION)
        self.assertLessEqual(D._MAX_INK_FRACTION, 1.0)


class FallbackTests(SimpleTestCase):
    """A diagram that cannot be made readable must not become the asset."""

    def test_an_unreadable_figure_is_not_returned(self):
        # Valid Python, one call per line: invalid code is rejected by syntax
        # screening first and the legibility gate would never run.
        crowded = "\n".join(["ax = plt.gca()"]
                            + ['ax.text(%d, 0, "x")' % i for i in range(20)])
        with patch("ai_services.core.llm_client.get_llm") as get_llm:
            get_llm.return_value.complete.return_value = {"content": crowded}
            result = D.render_diagram("a process", width_in=BAND[0], height_in=BAND[1])
        self.assertFalse(result["success"])
        self.assertIn("labels", result["error"])

    def test_the_caller_can_fall_through_rather_than_ship_it(self):
        # render_diagram never raises, so the producer sees a plain failure
        # and the fallback chain moves on to the next visual strategy.
        with patch("ai_services.core.llm_client.get_llm") as get_llm:
            get_llm.return_value.complete.side_effect = RuntimeError("model down")
            result = D.render_diagram("a process", width_in=BAND[0], height_in=BAND[1])
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_unsized_callers_keep_the_original_behaviour(self):
        # The legibility gate only applies when the caller said how big the
        # figure will be, so existing callers are unaffected.
        with patch("ai_services.core.llm_client.get_llm") as get_llm:
            get_llm.return_value.complete.return_value = {"content": "ax = plt.gca()"}
            result = D.render_diagram("a process")
        # No legibility rejection reason; it failed or succeeded on its own
        # merits as before.
        self.assertNotIn("labels", str(result.get("error") or ""))


class StructuredRoutingTests(SimpleTestCase):
    """Prefer the structured representation that already exists."""

    def test_natively_drawn_slide_types_do_not_fall_back_to_a_raster(self):
        from ai_services.core.ppt_v2.schema import SlideType, VisualStrategy
        from ai_services.core.ppt_v2.visuals import (
            STRUCTURED_NATIVELY, VisualCapabilities, fallback_chain,
        )
        caps = VisualCapabilities(ai_images=True, image_search=True,
                                  diagrams=True, cache=True, ai_image_budget=9)
        for slide_type in STRUCTURED_NATIVELY:
            chain = fallback_chain(VisualStrategy.ICON_COMPOSITION, caps, slide_type)
            self.assertNotIn(VisualStrategy.PROGRAMMATIC_DIAGRAM, chain,
                             slide_type.value)

    def test_a_real_figure_is_still_available_where_it_is_needed(self):
        from ai_services.core.ppt_v2.schema import SlideType, VisualStrategy
        from ai_services.core.ppt_v2.visuals import VisualCapabilities, fallback_chain
        caps = VisualCapabilities(ai_images=True, image_search=True,
                                  diagrams=True, cache=True, ai_image_budget=9)
        # Where plotting is the point, a drawn figure stays in the chain.
        self.assertIn(VisualStrategy.PROGRAMMATIC_DIAGRAM,
                      fallback_chain(VisualStrategy.AI_GENERATED_IMAGE, caps, SlideType.DATA))
        # Asked for directly - a graph, a geometric figure - it is still drawn.
        for slide_type in (SlideType.ANATOMY, SlideType.CONCEPT):
            self.assertEqual(fallback_chain(VisualStrategy.PROGRAMMATIC_DIAGRAM,
                                            caps, slide_type)[0],
                             VisualStrategy.PROGRAMMATIC_DIAGRAM, slide_type.value)
        # As a mere fallback elsewhere it is gone: it cost a live deck 26s and
        # two LLM attempts for a sparse figure on a title slide.
        for slide_type in (SlideType.ANATOMY, SlideType.CONCEPT, SlideType.TITLE):
            self.assertNotIn(VisualStrategy.PROGRAMMATIC_DIAGRAM,
                             fallback_chain(VisualStrategy.AI_GENERATED_IMAGE,
                                            caps, slide_type), slide_type.value)
