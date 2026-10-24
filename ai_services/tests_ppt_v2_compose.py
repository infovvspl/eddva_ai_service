"""Composition: planner JSON in, renderable slides out.

Two concerns, and they fail differently.

**Tolerance.** The planner is an LLM. It will emit ``slide_type`` instead of
``slideType``, a bare string where an object was asked for, "cycle" instead of
"process", and an emphasis phrase that is not actually in the title. None of
that may raise, because a 500 here costs a teacher their whole deck. Each
test below is a spelling we have actually seen.

**Correctness.** What comes out has to pass structural QA, and the deck has to
not collapse back into V1's single repeated shape.

The Photosynthesis case at the bottom is the reference target. It is checked
structurally, which is all a test without a renderer honestly can do: whether
the deck *looks* like the reference is a human judgement on a rendered file.
"""
import json
from pathlib import Path

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import qa
from ai_services.core.ppt_v2.compose import (
    DeckContext,
    compose_deck,
    slide_from_plan,
)
from ai_services.core.ppt_v2.schema import BlockKind, SlideType, VisualStrategy
from ai_services.core.ppt_v2.visuals import VisualCapabilities

FIXTURE = (Path(__file__).resolve().parents[1]
           / "test" / "class_08" / "science" / "photosynthesis" / "planned_deck.json")

FULL = VisualCapabilities(ai_images=True, image_search=True, diagrams=True,
                          cache=True, ai_image_budget=4)
NO_AI = VisualCapabilities(ai_images=False, image_search=True, diagrams=True,
                           cache=True)


class ToleranceTests(SimpleTestCase):
    """Field spellings the model gets wrong must cost a default, not a 500."""

    def test_an_empty_object_still_produces_a_slide(self):
        spec = slide_from_plan({}, 3)
        self.assertEqual(spec.slide_number, 3)
        self.assertIs(spec.slide_type, SlideType.CONCEPT)

    def test_a_non_dict_does_not_raise(self):
        self.assertEqual(slide_from_plan("not a slide", 1).slide_number, 1)

    def test_snake_case_and_camel_case_both_work(self):
        a = slide_from_plan({"slide_type": "process", "learning_objective": "x"}, 1)
        b = slide_from_plan({"slideType": "process", "learningObjective": "x"}, 1)
        self.assertIs(a.slide_type, b.slide_type)
        self.assertEqual(a.learning_objective, b.learning_objective)

    def test_near_synonyms_for_slide_type_are_understood(self):
        for raw, expected in (("cycle", SlideType.PROCESS),
                              ("steps", SlideType.PROCESS),
                              ("vs", SlideType.COMPARISON),
                              ("difference", SlideType.COMPARISON),
                              ("labelled", SlideType.ANATOMY),
                              ("takeaways", SlideType.SUMMARY),
                              ("mcq", SlideType.QUIZ),
                              ("graph", SlideType.DATA),
                              ("Key Points", SlideType.SUMMARY)):
            self.assertIs(slide_from_plan({"type": raw}, 1).slide_type, expected, raw)

    def test_an_unknown_slide_type_falls_back_rather_than_failing(self):
        self.assertIs(slide_from_plan({"type": "interpretive dance"}, 1).slide_type,
                      SlideType.CONCEPT)

    def test_plain_string_bullets_become_blocks(self):
        spec = slide_from_plan({"bullets": ["one", "two"]}, 1)
        self.assertEqual([b.text for b in spec.content_blocks], ["one", "two"])

    def test_a_visual_given_as_a_bare_string_is_accepted(self):
        spec = slide_from_plan({"visual": "a green leaf"}, 1)
        self.assertEqual(spec.visual.description, "a green leaf")

    def test_an_emphasis_that_is_not_in_the_title_is_dropped_not_rejected(self):
        # Cosmetic, so losing it is better than losing the slide.
        spec = slide_from_plan({"title": "Photosynthesis",
                                "emphasisText": "Respiration"}, 1)
        self.assertEqual(spec.title.emphasis_text, "")
        self.assertTrue(spec.is_valid)

    def test_a_malformed_annotation_anchor_is_clamped(self):
        spec = slide_from_plan(
            {"visual": {"annotations": [{"text": "a", "anchor": [9, -4]},
                                        {"text": "b", "anchor": "nonsense"}]}}, 1)
        self.assertEqual([a.anchor for a in spec.visual.annotations],
                         [(1.0, 0.0), (0.5, 0.5)])

    def test_too_many_blocks_are_trimmed_to_what_a_slide_can_hold(self):
        spec = slide_from_plan({"title": "T",
                                "bullets": ["b%d" % i for i in range(20)]}, 1)
        self.assertTrue(spec.is_valid)
        self.assertLessEqual(len(spec.content_blocks), 6)

    def test_blocks_with_only_a_label_survive(self):
        spec = slide_from_plan({"blocks": [{"label": "Absorb"}]}, 1)
        self.assertEqual(spec.content_blocks[0].label, "Absorb")

    def test_empty_blocks_are_discarded(self):
        spec = slide_from_plan({"blocks": [{"text": "  "}, {"text": "real"}]}, 1)
        self.assertEqual(len(spec.content_blocks), 1)

    def test_a_process_slide_defaults_its_blocks_to_steps(self):
        spec = slide_from_plan({"type": "process", "bullets": ["a", "b"]}, 1)
        self.assertIs(spec.content_blocks[0].kind, BlockKind.STEP)


class CompositionTests(SimpleTestCase):
    def test_every_slide_comes_out_with_a_layout_and_a_theme(self):
        deck = compose_deck([{"type": "concept", "title": "A",
                              "bullets": ["x"]}], DeckContext(subject="Science"),
                            caps=FULL)
        slide = deck.slides[0]
        self.assertIsNotNone(slide.layout)
        self.assertIsNotNone(slide.theme)
        self.assertTrue(slide.layout.regions)

    def test_slide_numbers_are_renumbered_from_one(self):
        deck = compose_deck([{"slideNumber": 7}, {"slideNumber": 2}],
                            DeckContext(), caps=FULL)
        self.assertEqual([s.slide_number for s in deck.slides], [1, 2])

    def test_a_deck_of_similar_slides_does_not_reuse_one_layout(self):
        # This is the V1 failure V2 exists to fix, so it is asserted directly.
        plan = [{"type": "concept", "title": "Slide %d" % i, "bullets": ["a", "b"],
                 "visual": {"description": "a volcano"}} for i in range(4)]
        deck = compose_deck(plan, DeckContext(subject="Science"), caps=FULL)
        names = [s.layout.name for s in deck.slides]
        self.assertGreater(len(set(names)), 1, names)
        self.assertEqual(qa.errors(qa.check_deck(deck.slides)), [])

    def test_the_search_term_is_the_topic_not_the_teaching_headline(self):
        # "How Green Plants Make Their Own Food" is a headline, not a query.
        deck = compose_deck(
            [{"type": "concept", "title": "How Green Plants Make Their Own Food",
              "visual": {"description": "a leaf", "objects": ["leaf"]}}],
            DeckContext(topic="Photosynthesis", subject="Science"), caps=NO_AI)
        terms = deck.slides[0].visual.search_terms
        self.assertTrue(all("How Green Plants" not in t for t in terms), terms)
        self.assertTrue(any("Photosynthesis" in t for t in terms), terms)

    def test_an_ai_slide_gets_a_prompt_and_a_search_fallback(self):
        deck = compose_deck(
            [{"type": "anatomy", "title": "Parts of a cell",
              "bullets": ["a", "b"], "visual": {"description": "a plant cell"}}],
            DeckContext(subject="Science", class_name="Class 8"), caps=FULL)
        visual = deck.slides[0].visual
        self.assertIs(visual.strategy, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertTrue(visual.image_prompt)
        self.assertIn(VisualStrategy.SEARCHED_IMAGE, visual.fallbacks)

    def test_the_prompt_forbids_text_in_the_image(self):
        # Diffusion models cannot spell; labels are added as real text later.
        deck = compose_deck(
            [{"type": "anatomy", "title": "Parts of a cell", "bullets": ["a", "b"],
              "visual": {"description": "a plant cell"}}],
            DeckContext(subject="Science"), caps=FULL)
        self.assertIn("no text", deck.slides[0].visual.image_prompt)

    def test_the_prompt_is_written_for_the_class_level(self):
        def prompt_for(class_name):
            return compose_deck(
                [{"type": "anatomy", "title": "T", "bullets": ["a", "b"],
                  "visual": {"description": "a plant cell"}}],
                DeckContext(subject="Science", class_name=class_name),
                caps=FULL).slides[0].visual.image_prompt

        self.assertNotEqual(prompt_for("Class 3"), prompt_for("Class 11"))
        self.assertIn("cartoon-adjacent", prompt_for("Class 3"))

    def test_with_ai_unavailable_slides_still_get_a_visual_plan(self):
        deck = compose_deck(
            [{"type": "anatomy", "title": "Parts of a cell", "bullets": ["a", "b"],
              "visual": {"description": "a plant cell"}}],
            DeckContext(subject="Science"), caps=NO_AI)
        visual = deck.slides[0].visual
        self.assertIsNot(visual.strategy, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertTrue(visual.fallbacks)
        self.assertEqual(qa.errors(qa.check_deck(deck.slides)), [])

    def test_a_composed_deck_survives_a_json_round_trip(self):
        deck = compose_deck([{"type": "process", "title": "T",
                              "bullets": ["a", "b", "c"],
                              "visual": {"description": "photosynthesis"}}],
                            DeckContext(subject="Science"), caps=FULL)
        restored = json.loads(json.dumps(deck.to_v1_slides()))
        self.assertEqual(restored[0]["v2"]["slideType"], "process")
        self.assertEqual(restored[0]["bullets"], ["a", "b", "c"])


class QualityGateTests(SimpleTestCase):
    def test_a_clean_deck_reports_no_issues(self):
        deck = compose_deck([{"type": "concept", "title": "A", "bullets": ["x"],
                              "visual": {"description": "a volcano"}},
                             {"type": "summary", "title": "B", "bullets": ["y"]}],
                            DeckContext(subject="Science"), caps=FULL)
        self.assertEqual(qa.check_deck(deck.slides), [])

    def test_an_ordinary_bullet_does_not_trip_the_overflow_check(self):
        # The first capacity model used 900 chars per unit area and flagged
        # every normal one-sentence bullet. A warning that fires on correct
        # content gets ignored, and then it is not there when something
        # really does overflow. These are real sentences from the first live
        # V2 run, at the length the planner is asked to produce.
        real = [
            "Photosynthesis is the process by which green plants make food.",
            "This process happens inside the green parts of a leaf.",
            "Oxygen released during photosynthesis is used by animals to breathe.",
            "Roots take in water from the soil and carry it to the leaves.",
        ]
        deck = compose_deck(
            [{"type": "concept", "title": "What is Photosynthesis?",
              "bullets": real, "visual": {"description": "a green leaf"}}],
            DeckContext(topic="Photosynthesis", subject="Science"), caps=NO_AI)
        overflow = [i for i in qa.check_deck(deck.slides)
                    if i.code == "text_overflow"]
        self.assertEqual(overflow, [], [str(i) for i in overflow])

    def test_a_wall_of_text_is_flagged(self):
        deck = compose_deck([{"type": "concept", "title": "A",
                              "bullets": ["word " * 400]}],
                            DeckContext(subject="Science"), caps=NO_AI)
        codes = {i.code for i in qa.check_deck(deck.slides)}
        self.assertIn("text_overflow", codes)

    def test_an_empty_visual_block_does_not_cost_the_slide_its_picture(self):
        # A live run returned a concept slide with an empty visual block, and
        # an empty description reads as "no picture" to the decision engine -
        # so the slide silently lost its hero image. The slide type knows
        # whether it wants one, so the omission must not decide.
        deck = compose_deck(
            [{"type": "concept", "title": "What is Photosynthesis?",
              "learningObjective": "State what a plant needs to make food",
              "highlightedTerms": ["chlorophyll"], "bullets": ["a", "b"],
              "visual": {}}],
            DeckContext(topic="Photosynthesis", subject="Science",
                        class_name="Class 8"), caps=NO_AI)
        slide = deck.slides[0]
        self.assertIsNot(slide.visual.strategy, VisualStrategy.NONE)
        self.assertTrue(slide.visual.description)
        self.assertTrue(slide.visual.search_terms)
        self.assertTrue(any("empty" in n for n in deck.notes))

    def test_a_summary_slide_is_still_allowed_to_have_no_picture(self):
        deck = compose_deck(
            [{"type": "summary", "title": "What We Learned", "bullets": ["a"],
              "visual": {}}],
            DeckContext(subject="Science"), caps=NO_AI)
        self.assertIs(deck.slides[0].visual.strategy, VisualStrategy.NONE)

    def test_every_layout_the_backend_can_choose_has_a_renderer(self):
        # The two registries live in different languages and different repos
        # directories; nothing but this test stops them drifting apart, and a
        # drift shows up as a silently generic-looking slide.
        from ai_services.core.ppt_v2.layouts import LAYOUTS
        self.assertEqual(set(LAYOUTS) - qa.RENDERABLE_LAYOUTS, set())

    def test_every_theme_carries_the_tokens_the_renderer_needs(self):
        from ai_services.core.ppt_v2.design import SUBJECT_THEMES, build_theme
        from ai_services.core.ppt_v2.schema import SlideType
        for subject in SUBJECT_THEMES:
            for slide_type in SlideType:
                theme = build_theme(subject=subject, slide_type=slide_type)
                missing = [t for t in qa.RENDERER_REQUIRED_TOKENS
                           if not theme.colors.get(t)]
                self.assertEqual(missing, [], "%s/%s" % (subject, slide_type.value))

    def test_a_layout_with_no_renderer_is_an_error_not_a_surprise(self):
        deck = compose_deck([{"type": "concept", "title": "A", "bullets": ["x"]}],
                            DeckContext(subject="Science"), caps=NO_AI)
        deck.slides[0].layout.name = "invented_later"
        codes = {i.code for i in qa.check_deck(deck.slides)}
        self.assertIn("unrenderable_layout", codes)

    def test_a_theme_missing_a_renderer_token_is_an_error(self):
        deck = compose_deck([{"type": "concept", "title": "A", "bullets": ["x"]}],
                            DeckContext(subject="Science"), caps=NO_AI)
        deck.slides[0].theme.colors.pop("accent_soft")
        issues = [i for i in qa.check_deck(deck.slides) if i.code == "missing_token"]
        self.assertTrue(issues)
        self.assertIn("accent_soft", issues[0].message)

    def test_the_report_says_it_did_not_look_at_pixels(self):
        # Guards against the report being read as visual sign-off.
        self.assertIn("no visual QA", qa.report([]))
        deck = compose_deck([{"type": "concept", "title": "A",
                              "bullets": ["word " * 400]}],
                            DeckContext(), caps=NO_AI)
        self.assertIn("structural only", qa.report(qa.check_deck(deck.slides)))


class PhotosynthesisReferenceTests(SimpleTestCase):
    """The two-slide reference case, composed from captured planner output."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.plan = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.context = DeckContext(
            topic=cls.plan["topic"], subject=cls.plan["subject"],
            chapter=cls.plan["chapter"], class_name=cls.plan["className"],
            board=cls.plan["board"])

    def compose(self, caps=FULL):
        return compose_deck(self.plan["slides"], self.context, caps=caps)

    def test_it_composes_two_slides_that_pass_structural_qa(self):
        deck = self.compose()
        self.assertEqual(len(deck.slides), 2)
        issues = qa.check_deck(deck.slides)
        self.assertEqual(qa.errors(issues), [], qa.report(issues))

    def test_the_two_slides_do_not_look_the_same(self):
        deck = self.compose()
        self.assertNotEqual(deck.slides[0].layout.name, deck.slides[1].layout.name)

    def test_the_concept_slide_keeps_its_four_labelled_inputs(self):
        slide = self.compose().slides[0]
        self.assertEqual(len(slide.content_blocks), 4)
        self.assertEqual([b.label for b in slide.content_blocks],
                         ["Sunlight", "Water", "Carbon dioxide", "Food and oxygen"])

    def test_the_diagram_annotations_survive_composition(self):
        slide = self.compose().slides[0]
        self.assertEqual(len(slide.visual.annotations), 4)
        self.assertIn("stomata", slide.visual.annotations[1].text)

    def test_the_process_slide_becomes_four_ordered_steps(self):
        slide = self.compose().slides[1]
        self.assertIs(slide.slide_type, SlideType.PROCESS)
        self.assertEqual(slide.layout.name, "process_steps")
        self.assertEqual(sum(1 for s in slide.layout.regions
                             if s.startswith("connector_")), 3)

    def test_it_is_themed_as_biology(self):
        self.assertEqual(self.compose().slides[0].theme.name, "biology")

    def test_each_slide_gets_the_visual_its_teaching_purpose_wants(self):
        # The concept slide needs a figure of something microscopic
        # ("chlorophyll", "stomata", "chloroplast"), which no stock photo can
        # show in context, so it is generated. The process slide carries its
        # sequence in its own step row, which is decided by the slide type
        # rather than by the wording of the description.
        first, second = self.compose().slides
        self.assertIs(first.visual.strategy, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertIs(second.visual.strategy, VisualStrategy.ICON_COMPOSITION)

    def test_the_prompts_describe_the_subject_and_forbid_labels(self):
        slide = self.compose().slides[0]
        prompt = slide.visual.image_prompt.lower()
        self.assertIn("leaf", prompt)
        self.assertIn("class 8", prompt)
        self.assertIn("no text", prompt)
        self.assertIn("green and blue", prompt)

    def test_it_degrades_without_ai_image_generation(self):
        # The common deployment: no HuggingFace token configured.
        deck = self.compose(caps=NO_AI)
        issues = qa.check_deck(deck.slides)
        self.assertEqual(qa.errors(issues), [], qa.report(issues))
        for slide in deck.slides:
            self.assertIsNot(slide.visual.strategy, VisualStrategy.AI_GENERATED_IMAGE)

    def test_it_degrades_with_nothing_configured_at_all(self):
        deck = self.compose(caps=VisualCapabilities(ai_images=False,
                                                    image_search=False,
                                                    diagrams=False, cache=False))
        issues = qa.check_deck(deck.slides)
        self.assertEqual(qa.errors(issues), [], qa.report(issues))

    def test_an_old_renderer_still_gets_a_usable_deck(self):
        slides = self.compose().to_v1_slides()
        self.assertEqual(slides[0]["title"], "How Green Plants Make Their Own Food")
        self.assertEqual(len(slides[0]["bullets"]), 4)
        self.assertTrue(slides[0]["speakerNotes"])
        self.assertTrue(slides[0]["imageSearchTerm"])


class AiBudgetPriorityTests(SimpleTestCase):
    """The AI-image budget goes to the slide that needs a precise figure, not
    to whichever slide comes first. A live deck spent its image on the title
    picture and drew the mitochondrion on slide 8 as two bare ellipses."""

    TITLE = {"slideType": "title", "title": "The Velm", "subtitle": "An overview",
             "visual": {"kind": "illustration", "description": "a velm on a shelf"}}
    PARTS = {"slideType": "anatomy", "contentType": "anatomy_structure",
             "title": "Parts of a velm",
             "blocks": [{"kind": "label", "label": "Shell", "text": "The shell protects the core."},
                        {"kind": "label", "label": "Core", "text": "The core stores the quorrin."},
                        {"kind": "label", "label": "Stalk", "text": "The stalk holds the velm up."}],
             "visual": {"kind": "diagram", "description": "a labelled velm",
                        "mustShow": ["shell", "core", "stalk"]}}

    def _strategies(self, budget):
        caps = VisualCapabilities(ai_images=True, image_search=True, diagrams=True,
                                  cache=True, ai_image_budget=budget)
        deck = compose_deck([self.TITLE, self.PARTS],
                            DeckContext(topic="Velm", subject="Science",
                                        class_name="Class 8"), caps=caps)
        return [s.visual.strategy for s in deck.slides], deck

    def test_one_image_goes_to_the_labelled_figure_not_the_title(self):
        (title, parts), _deck = self._strategies(budget=1)
        self.assertIs(parts, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertIsNot(title, VisualStrategy.AI_GENERATED_IMAGE)

    def test_with_enough_budget_both_get_one(self):
        (title, parts), _deck = self._strategies(budget=2)
        self.assertIs(title, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertIs(parts, VisualStrategy.AI_GENERATED_IMAGE)

    def test_slides_still_come_out_in_their_own_order(self):
        _strategies, deck = self._strategies(budget=1)
        self.assertEqual([s.slide_number for s in deck.slides], [1, 2])
        self.assertEqual([s.title.text for s in deck.slides], ["The Velm", "Parts of a velm"])


class TitlePictureTests(SimpleTestCase):
    def test_a_title_picture_has_nothing_for_a_vision_check_to_verify(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        deck = compose_deck(
            [{"slideType": "title", "title": "Respiration",
              "visual": {"kind": "illustration", "description": "lungs and a leaf",
                         "objects": ["lungs", "leaf"], "mustShow": ["lungs"],
                         "relationships": ["air enters the lungs"]}}],
            DeckContext(topic="Respiration", subject="Science", class_name="Class 10"),
            caps=FULL)
        visual = deck.slides[0].visual
        self.assertEqual((visual.must_show, visual.relationships, visual.icons), ([], [], []))
        self.assertFalse(ProducerSet(search_fn=None).visual_contract(visual).is_meaningful)

