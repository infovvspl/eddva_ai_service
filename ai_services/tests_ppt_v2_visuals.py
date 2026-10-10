"""The visual decision engine, and the fallback chain that must never break.

AI image generation is off by default and needs a HuggingFace token that is
not configured in every environment. A planner that assumes it works produces
slides with an empty hole where the diagram should be, and nothing upstream
notices. So the property that actually matters here is not "picks a clever
strategy" - it is "always ends somewhere that renders".

Nothing in these tests touches the network: producers are plain callables.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.ppt_v2.schema import SlideType, VisualSpec, VisualStrategy
from ai_services.core.ppt_v2.visuals import (
    CANONICAL_FALLBACK,
    DeckVisualPlan,
    VisualCapabilities,
    detect_capabilities,
    fallback_chain,
    preferred_strategy,
    resolve_visual,
)

AI = VisualStrategy.AI_GENERATED_IMAGE
SEARCH = VisualStrategy.SEARCHED_IMAGE
DIAGRAM = VisualStrategy.PROGRAMMATIC_DIAGRAM
ICON = VisualStrategy.ICON_COMPOSITION
CACHE = VisualStrategy.CACHED_ASSET
NONE = VisualStrategy.NONE

ALL = VisualCapabilities(ai_images=True, image_search=True, diagrams=True,
                         cache=True, ai_image_budget=10)
NOTHING = VisualCapabilities(ai_images=False, image_search=False,
                             diagrams=False, cache=False)


def ok(url="https://example.test/i.png"):
    return lambda spec: {"ok": True, "url": url}


def fails(error="boom"):
    return lambda spec: {"ok": False, "error": error}


def raises(exc=RuntimeError("provider exploded")):
    def _p(spec):
        raise exc
    return _p


class PreferredStrategyTests(SimpleTestCase):
    def test_a_photographable_object_is_searched(self):
        # A real photo of a volcano teaches better than a generated one, and
        # is cheaper and more honest.
        self.assertIs(preferred_strategy(SlideType.CONCEPT,
                                         "a volcano erupting"), SEARCH)

    def test_something_no_camera_can_see_is_generated(self):
        self.assertIs(preferred_strategy(SlideType.CONCEPT,
                                         "electrons moving through a circuit"), AI)

    def test_microscopic_structures_are_generated_not_searched(self):
        # A search for "stomata" returns a microscope photograph that will not
        # compose with the rest of the slide.
        self.assertIs(preferred_strategy(SlideType.CONCEPT,
                                         "chlorophyll inside a chloroplast"), AI)

    def test_quantitative_content_is_plotted(self):
        self.assertIs(preferred_strategy(SlideType.CONCEPT,
                                         "a graph of distance versus time"), DIAGRAM)
        self.assertIs(preferred_strategy(SlideType.DATA, "rainfall by month"), DIAGRAM)

    def test_a_labelled_figure_is_generated_so_it_matches_its_labels(self):
        self.assertIs(preferred_strategy(SlideType.ANATOMY, "parts of a flower"), AI)

    def test_slides_whose_content_is_already_the_visual_get_no_image(self):
        for slide_type in (SlideType.SUMMARY, SlideType.QUIZ, SlideType.TIMELINE):
            self.assertIs(preferred_strategy(slide_type, "anything at all"), NONE)

    def test_a_process_slide_does_not_change_kind_with_the_wording(self):
        # This used to flip: a description containing a not-photographable
        # noun gave a generated illustration, one without gave shapes. The
        # teaching purpose was identical both times, so the visual kind
        # should have been too. Slide type now decides.
        for description in ("steps of filtering muddy water",
                            "stages of photosynthesis",
                            "how a seed germinates",
                            ""):
            self.assertIs(preferred_strategy(SlideType.PROCESS, description),
                          ICON, description)

    def test_no_description_means_no_visual(self):
        self.assertIs(preferred_strategy(SlideType.CONCEPT, ""), NONE)

    def test_anatomy_is_not_mistaken_for_atom(self):
        # "atom" is a substring of "anatomy"; word-boundary matching only.
        self.assertIs(preferred_strategy(SlideType.CONCEPT,
                                         "the anatomy of a human heart"), SEARCH)


class DeterminismTests(SimpleTestCase):
    """Same teaching purpose and content should give the same visual kind."""

    def test_the_chain_follows_the_slide_type_preference(self):
        # No programmatic diagram for a process slide: the numbered step row
        # IS the structure, drawn natively by both renderers, so a generated
        # raster of it would be strictly worse.
        chain = fallback_chain(ICON, ALL, SlideType.PROCESS)
        self.assertEqual(chain[:3], [ICON, AI, SEARCH])
        self.assertNotIn(DIAGRAM, chain)

    def test_repeated_decisions_on_one_storyboard_agree(self):
        def decide():
            plan = DeckVisualPlan(caps=ALL)
            return [plan.decide(i + 1, t, d).strategy for i, (t, d) in enumerate([
                (SlideType.CONCEPT, "a green leaf in sunlight"),
                (SlideType.PROCESS, "stages of photosynthesis"),
                (SlideType.ANATOMY, "parts of a flower"),
            ])]
        runs = {tuple(decide()) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_a_type_without_a_preference_still_gets_a_chain(self):
        chain = fallback_chain(AI, ALL, SlideType.QUIZ)
        self.assertIs(chain[-1], NONE)
        self.assertIn(ICON, chain)

    def test_the_preference_never_promotes_an_unavailable_strategy(self):
        caps = VisualCapabilities(ai_images=False, image_search=True,
                                  diagrams=False, cache=False)
        self.assertNotIn(AI, fallback_chain(AI, caps, SlideType.PROCESS))
        self.assertNotIn(DIAGRAM, fallback_chain(AI, caps, SlideType.PROCESS))


class FallbackChainTests(SimpleTestCase):
    def test_the_chain_always_ends_somewhere_that_renders(self):
        for primary in VisualStrategy:
            for caps in (ALL, NOTHING):
                chain = fallback_chain(primary, caps)
                self.assertIs(chain[-1], NONE)
                if primary is not NONE:
                    self.assertIn(ICON, chain)

    def test_choosing_no_visual_does_not_fall_back_into_making_one(self):
        # NONE is a decision (a summary slide is better typographic), not a
        # failure to find an image.
        self.assertEqual(fallback_chain(NONE, ALL), [NONE])

    def test_the_preferred_strategy_leads_when_it_is_available(self):
        self.assertIs(fallback_chain(AI, ALL)[0], AI)

    def test_an_unavailable_strategy_is_dropped_from_the_chain(self):
        caps = VisualCapabilities(ai_images=False, image_search=True,
                                  diagrams=True, cache=True)
        self.assertNotIn(AI, fallback_chain(AI, caps))
        self.assertIs(fallback_chain(AI, caps)[0], SEARCH)

    def test_with_nothing_configured_the_chain_is_shapes_then_nothing(self):
        self.assertEqual(fallback_chain(AI, NOTHING), [ICON, NONE])

    def test_the_chain_never_repeats_a_strategy(self):
        for primary in VisualStrategy:
            chain = fallback_chain(primary, ALL)
            self.assertEqual(len(chain), len(set(chain)))

    def test_the_canonical_order_is_respected(self):
        chain = fallback_chain(AI, ALL)
        order = [s for s in CANONICAL_FALLBACK if s in chain]
        self.assertEqual(chain, order)


class BudgetTests(SimpleTestCase):
    def test_ai_images_stop_once_the_budget_is_spent(self):
        plan = DeckVisualPlan(caps=VisualCapabilities(ai_images=True, ai_image_budget=2))
        got = [plan.decide(i + 1, SlideType.ANATOMY, "parts of a cell").strategy
               for i in range(4)]
        # Past the budget an anatomy slide falls to a searched textbook
        # figure, not a drawn one: render_diagram drew a mitochondrion as two
        # bare ellipses with its labels printed over the lines. A searched
        # figure carries a legend and passes the same vision check.
        self.assertEqual(got, [AI, AI, SEARCH, SEARCH])
        self.assertEqual(plan.ai_images_used, 2)

    def test_running_out_of_budget_degrades_rather_than_failing(self):
        plan = DeckVisualPlan(caps=VisualCapabilities(ai_images=True, ai_image_budget=0))
        spec = plan.decide(1, SlideType.ANATOMY, "parts of a cell")
        self.assertIsNot(spec.strategy, AI)
        self.assertTrue(spec.fallbacks)

    def test_budget_is_only_spent_on_slides_that_wanted_an_image(self):
        plan = DeckVisualPlan(caps=VisualCapabilities(ai_images=True, ai_image_budget=2))
        plan.decide(1, SlideType.SUMMARY, "recap")
        plan.decide(2, SlideType.TIMELINE, "dates")
        self.assertEqual(plan.ai_images_used, 0)

    def test_what_the_slide_wanted_is_recorded_even_when_it_is_denied(self):
        # So a log line can say "wanted an AI image, got a search result"
        # rather than quietly reporting only the outcome.
        plan = DeckVisualPlan(caps=VisualCapabilities(ai_images=True, ai_image_budget=0))
        plan.decide(1, SlideType.ANATOMY, "parts of a cell")
        self.assertIs(plan.wanted_for(1), AI)

    def test_degrading_preserves_the_rest_of_the_capabilities(self):
        caps = VisualCapabilities(ai_images=True, ai_image_budget=0,
                                  image_width=1024, image_height=1024)
        plan = DeckVisualPlan(caps=caps)
        plan.decide(1, SlideType.ANATOMY, "parts of a cell")
        self.assertEqual(plan.caps.image_width, 1024)


class ResolveTests(SimpleTestCase):
    def test_the_first_working_producer_wins(self):
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, ICON, NONE])
        out = resolve_visual(spec, {AI: ok("https://a.test/1.png"), SEARCH: ok()})
        self.assertIs(out.resolved_strategy, AI)
        self.assertEqual(out.url, "https://a.test/1.png")

    def test_a_failure_falls_through_to_the_next_strategy(self):
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, ICON, NONE])
        out = resolve_visual(spec, {AI: fails("no token"), SEARCH: ok()})
        self.assertIs(out.resolved_strategy, SEARCH)

    def test_a_producer_that_raises_does_not_take_the_slide_with_it(self):
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, ICON, NONE])
        out = resolve_visual(spec, {AI: raises(), SEARCH: ok()})
        self.assertIs(out.resolved_strategy, SEARCH)

    def test_success_with_no_image_is_treated_as_a_failure(self):
        # A producer reporting ok with nothing to draw would otherwise leave
        # an empty region that no later check can distinguish from a blank.
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, ICON, NONE])
        out = resolve_visual(spec, {AI: lambda s: {"ok": True, "url": ""},
                                    SEARCH: ok()})
        self.assertIs(out.resolved_strategy, SEARCH)

    def test_everything_failing_still_ends_at_a_renderable_slide(self):
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, DIAGRAM, NONE])
        out = resolve_visual(spec, {AI: fails(), SEARCH: fails(), DIAGRAM: fails()})
        self.assertIs(out.resolved_strategy, NONE)

    def test_each_strategy_is_attempted_exactly_once(self):
        # No retry loop here: producers own their own bounded retries, and a
        # second loop on top is how a slow deck becomes a hung one.
        calls = {"n": 0}

        def counting(spec):
            calls["n"] += 1
            return {"ok": False, "error": "nope"}

        resolve_visual(VisualSpec(strategy=AI, fallbacks=[AI, AI, NONE]),
                       {AI: counting})
        self.assertEqual(calls["n"], 1)

    def test_a_missing_producer_is_skipped_not_fatal(self):
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, NONE])
        out = resolve_visual(spec, {SEARCH: ok()})
        self.assertIs(out.resolved_strategy, SEARCH)

    def test_icon_composition_resolves_without_an_asset(self):
        spec = VisualSpec(strategy=ICON, fallbacks=[NONE])
        out = resolve_visual(spec, {ICON: lambda s: {"ok": True}})
        self.assertIs(out.resolved_strategy, ICON)

    def test_every_attempt_is_reported_to_the_observer(self):
        seen = []
        spec = VisualSpec(strategy=AI, fallbacks=[SEARCH, NONE])
        resolve_visual(spec, {AI: fails("no token"), SEARCH: ok()},
                       on_attempt=lambda s, okay, why: seen.append((s, okay, why)))
        self.assertEqual([(s.value, okay) for s, okay, _ in seen],
                         [("ai_generated_image", False), ("searched_image", True)])
        self.assertIn("no token", seen[0][2])


class CapabilityDetectionTests(SimpleTestCase):
    def test_ai_images_report_unavailable_when_generation_is_off(self):
        with patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=False):
            self.assertFalse(detect_capabilities().ai_images)

    def test_ai_images_report_available_when_configured(self):
        with patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=True):
            self.assertTrue(detect_capabilities().ai_images)

    def test_a_broken_image_service_degrades_instead_of_raising(self):
        with patch("ai_services.core.image_generation.can_generate_note_images",
                   side_effect=RuntimeError("config blew up")):
            self.assertFalse(detect_capabilities().ai_images)

    def test_the_generator_canvas_is_read_from_configuration(self):
        with patch("ai_services.core.image_generation.image_generation_config",
                   return_value={"width": 1024, "height": 768}):
            caps = detect_capabilities()
        self.assertEqual((caps.image_width, caps.image_height), (1024, 768))
        self.assertAlmostEqual(caps.image_aspect, 4 / 3)

    def test_a_negative_budget_is_clamped(self):
        self.assertEqual(detect_capabilities(ai_image_budget=-5).ai_image_budget, 0)
