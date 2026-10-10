"""Content analysis, pedagogical strategy, and the semantic label contract.

Three layers, tested where they meet:

  content.analyse     what shape of thing a slide teaches
  pedagogy.decide     how to teach that shape, and which visuals are honest
  label policy        whichever producer wins, the required labels arrive

No network, no LLM, no image provider.
"""
import io
import os
import re
import unittest
from unittest import mock

from ai_services.core.ppt_v2 import content, labels, pedagogy, qa
from ai_services.core.ppt_v2.content import ContentType
from ai_services.core.ppt_v2.pedagogy import PedagogicalStrategy
from ai_services.core.ppt_v2.schema import (
    Annotation, SlideType, VisualSpec, VisualStrategy)

AI = VisualStrategy.AI_GENERATED_IMAGE
DIAGRAM = VisualStrategy.PROGRAMMATIC_DIAGRAM
SEARCH = VisualStrategy.SEARCHED_IMAGE
CACHED = VisualStrategy.CACHED_ASSET
ICONS = VisualStrategy.ICON_COMPOSITION
NONE = VisualStrategy.NONE


def slide(title="", blocks=(), declared=None, legacy=None, objective="",
          visual=None, **extra):
    data = {"title": title, "learningObjective": objective,
            "blocks": [b if isinstance(b, dict) else {"kind": "bullet", "text": b}
                       for b in blocks]}
    if declared:
        data["contentType"] = declared
    if legacy:
        data["slideType"] = legacy
    if visual is not None:
        data["visual"] = visual
    data.update(extra)
    return data


def analyse(data, **ctx):
    ctx.setdefault("subject", "General")
    ctx.setdefault("class_level", "Class 8")
    return content.analyse(data, **ctx)


# ---------------------------------------------------------------------------
# content analysis
# ---------------------------------------------------------------------------


class NormalisationTests(unittest.TestCase):
    def test_every_legacy_slide_type_maps_to_a_content_type(self):
        from ai_services.core.ppt_v2.planner import PLANNER_SLIDE_TYPES
        for legacy in PLANNER_SLIDE_TYPES:
            self.assertIsNotNone(content.normalise_type(legacy), legacy)

    def test_legacy_example_means_a_real_world_case_not_a_worked_solution(self):
        self.assertIs(content.normalise_type("example"), ContentType.CASE_STUDY)
        self.assertIs(content.normalise_type("worked_example"), ContentType.WORKED_EXAMPLE)

    def test_spelling_variants_are_tolerated(self):
        self.assertIs(content.normalise_type("Flow Chart"), None)  # two words, unknown
        self.assertIs(content.normalise_type("flowchart"), ContentType.FLOWCHART_ALGORITHM)
        self.assertIs(content.normalise_type("Geometric-Diagram"), ContentType.GEOMETRIC_DIAGRAM)
        self.assertIs(content.normalise_type("SYSTEM_ARCHITECTURE"), ContentType.SYSTEM_ARCHITECTURE)

    def test_nonsense_is_not_a_type(self):
        for junk in ("", None, "banana_layout", 42, "PHOTOSYNTHESIS_LAYOUT"):
            self.assertIsNone(content.normalise_type(junk), junk)


class GracefulFallbackTests(unittest.TestCase):
    def test_an_unknown_declared_type_falls_back_to_concept_explanation(self):
        a = analyse(slide("Water", ["Water is a liquid at room temperature."],
                          declared="banana_layout"))
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION)

    def test_garbage_input_never_raises(self):
        for junk in (None, [], "text", {"blocks": "not a list"},
                     {"visual": 7, "title": ["a", None]}, {"blocks": [None, 3, {}]}):
            a = content.analyse(junk)
            self.assertIsInstance(a.content_type, ContentType)

    def test_no_cues_means_concept(self):
        a = analyse(slide("What Is Water?", ["Water has no colour.", "Water has no smell."],
                          legacy="concept"))
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION)


class DeclaredTypeValidationTests(unittest.TestCase):
    """A declaration is the starting point, then checked against the content."""

    def test_a_fillable_declaration_is_kept(self):
        a = analyse(slide("Metals and Non-metals",
                          ["Metals conduct heat.", "Non-metals are poor conductors."],
                          declared="comparison"))
        self.assertIs(a.content_type, ContentType.COMPARISON)
        self.assertEqual(a.source, "declared")

    def test_a_timeline_without_dates_is_corrected(self):
        a = analyse(slide("Kinds of Rocks",
                          ["Igneous rocks form from magma.",
                           "Sedimentary rocks form in layers."],
                          declared="timeline"))
        self.assertIsNot(a.content_type, ContentType.TIMELINE)
        self.assertEqual(a.source, "corrected")
        self.assertIn("dated", a.reason)

    def test_a_formula_slide_without_a_formula_is_corrected(self):
        a = analyse(slide("Speed", ["Speed tells us how fast something moves."],
                          declared="formula"))
        self.assertIsNot(a.content_type, ContentType.FORMULA_EQUATION)
        self.assertEqual(a.source, "corrected")

    def test_a_worked_example_needs_worked_numbers(self):
        worked = analyse(slide("Solve 2x + 3 = 11", [
            {"kind": "step", "label": "Step 1", "text": "2x + 3 - 3 = 11 - 3, so 2x = 8"},
            {"kind": "step", "label": "Step 2", "text": "x = 8 / 2 = 4"},
        ], declared="worked_example"))
        self.assertIs(worked.content_type, ContentType.WORKED_EXAMPLE)
        self.assertEqual(len(worked.numerical_steps), 2)

        hollow = analyse(slide("Linear Equations", ["An equation has two sides."],
                               declared="worked_example"))
        self.assertIsNot(hollow.content_type, ContentType.WORKED_EXAMPLE)

    def test_a_comparison_needs_two_sides(self):
        a = analyse(slide("Metals", ["Metals conduct heat."], declared="comparison"))
        self.assertIsNot(a.content_type, ContentType.COMPARISON)

    def test_structural_slides_are_never_reinferred(self):
        # A summary that mentions steps and angles is still a summary.
        a = analyse(slide("Summary", ["The steps of the process.",
                                      "Angles in a triangle add to 180."],
                          legacy="summary"))
        self.assertIs(a.content_type, ContentType.SUMMARY)


class InferenceTests(unittest.TestCase):
    """"concept" is the planner's "unsure" - an invitation to infer."""

    def test_geometry_cues_infer_a_geometric_diagram(self):
        a = analyse(slide("Angles in a Quadrilateral",
                          ["A quadrilateral has four angles.",
                           "Its diagonals join opposite vertices."], legacy="concept"))
        self.assertIs(a.content_type, ContentType.GEOMETRIC_DIAGRAM)
        self.assertEqual(a.source, "inferred")

    def test_two_dated_blocks_infer_a_timeline(self):
        a = analyse(slide("Key Events", ["The first event happened in 1885.",
                                         "The second followed in 1905.",
                                         "A third came in 1920."], legacy="concept"))
        self.assertIs(a.content_type, ContentType.TIMELINE)

    def test_an_algorithm_infers_a_flowchart(self):
        a = analyse(slide("An Algorithm to Find the Larger Number",
                          ["Read two numbers.", "Compare them and print the larger."],
                          legacy="concept"))
        self.assertIs(a.content_type, ContentType.FLOWCHART_ALGORITHM)

    def test_a_cycle_is_not_flattened_into_a_process(self):
        a = analyse(slide("The Water Cycle",
                          ["Water evaporates.", "Vapour condenses into clouds.",
                           "Rain returns to the sea."], legacy="concept"))
        self.assertIs(a.content_type, ContentType.CYCLE)

    def test_the_title_breaks_a_tie_with_body_prose(self):
        # "The Process of X" with numbered steps is a process, even when a
        # bullet mentions something structural in passing.
        a = analyse(slide("The Process of Digestion", [
            {"kind": "step", "label": "1", "text": "Food is broken down in the mouth."},
            {"kind": "step", "label": "2", "text": "The structure of the stomach churns it."},
        ], legacy="concept"))
        self.assertIs(a.content_type, ContentType.PROCESS)

    def test_one_weak_body_cue_does_not_reclassify(self):
        a = analyse(slide("Why Plants Need Light",
                          ["Light is needed by plants.", "We can see a trend here."],
                          legacy="concept"))
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION)


class AnalysisFieldsTests(unittest.TestCase):
    def test_the_structured_fields_are_filled(self):
        a = analyse(slide(
            "Speed", [{"kind": "formula", "text": "v = d / t"},
                      "Speed is distance divided by time."],
            declared="formula", objective="Use the formula for speed",
            highlightedTerms=["speed", "distance"],
            visual={"objects": ["car", "road"], "mustShow": ["car"],
                    "relationships": ["the car moves along the road"],
                    "annotations": [{"text": "distance", "anchor": [0.5, 0.8]}]}))
        self.assertIs(a.content_type, ContentType.FORMULA_EQUATION)
        self.assertIn("v = d / t", a.formulas)
        self.assertIn("speed", a.key_concepts)
        self.assertIn("car", a.required_visual_elements)
        self.assertEqual(a.relationships, ["the car moves along the road"])
        self.assertEqual(a.required_labels, ["distance"])
        self.assertEqual(a.learning_objective, "Use the formula for speed")
        self.assertIn(a.visual_complexity, ("low", "medium", "high"))

    def test_class_constraints_come_from_the_existing_rules(self):
        with mock.patch("ai_services.core.ppt_v2.class_rules.all_forbidden",
                        return_value=["calculus"]) as rules:
            a = analyse(slide("Motion", ["Things move."]), chapter="Motion")
        self.assertEqual(a.class_level_constraints, ["calculus"])
        rules.assert_called_once()

    def test_to_dict_is_json_ready(self):
        import json
        json.dumps(analyse(slide("Water", ["Water is wet."])).to_dict())


# ---------------------------------------------------------------------------
# pedagogical strategy
# ---------------------------------------------------------------------------


class StrategyTableTests(unittest.TestCase):
    def test_every_content_type_has_a_rule(self):
        for ctype in ContentType:
            self.assertIn(ctype, pedagogy.RULES, ctype)

    def test_types_that_existed_before_delegate_to_the_existing_engine(self):
        # So a deck that was right before is decided exactly as before.
        for ctype in (ContentType.PROCESS, ContentType.ANATOMY_STRUCTURE,
                      ContentType.CONCEPT_EXPLANATION, ContentType.CAUSE_EFFECT,
                      ContentType.COMPARISON, ContentType.TIMELINE,
                      ContentType.FORMULA_EQUATION, ContentType.SUMMARY,
                      ContentType.REVISION_QUIZ, ContentType.TITLE,
                      ContentType.CASE_STUDY, ContentType.GRAPH_DATA):
            self.assertIsNone(pedagogy.RULES[ctype].visual, ctype)

    def test_legacy_types_bridge_to_their_own_slide_type(self):
        for legacy, slide_type in (("concept", SlideType.CONCEPT),
                                   ("process", SlideType.PROCESS),
                                   ("anatomy", SlideType.ANATOMY),
                                   ("comparison", SlideType.COMPARISON),
                                   ("cause_effect", SlideType.CAUSE_EFFECT),
                                   ("timeline", SlideType.TIMELINE),
                                   ("formula", SlideType.FORMULA),
                                   ("example", SlideType.EXAMPLE),
                                   ("summary", SlideType.SUMMARY),
                                   ("quiz", SlideType.QUIZ),
                                   ("title", SlideType.TITLE)):
            ctype = content.normalise_type(legacy)
            self.assertIs(pedagogy.RULES[ctype].bridge, slide_type, legacy)

    def test_representations_a_generator_would_falsify_exclude_it(self):
        # A generated map invents geography; a generated figure invents
        # measurements; a generated flowchart invents arrows.
        for ctype in (ContentType.GEOGRAPHIC_MAP, ContentType.GEOMETRIC_DIAGRAM,
                      ContentType.FLOWCHART_ALGORITHM, ContentType.SYSTEM_ARCHITECTURE,
                      ContentType.CYCLE, ContentType.GRAPH_DATA,
                      ContentType.EXPERIMENT_PROCEDURE):
            self.assertIn(AI, pedagogy.RULES[ctype].excluded, ctype)

    def test_none_is_never_excluded(self):
        for rule in pedagogy.RULES.values():
            self.assertNotIn(NONE, rule.excluded)

    def test_planned_labels_make_labels_required_whatever_the_type(self):
        a = analyse(slide("Water", ["Water is wet."],
                          visual={"annotations": [{"text": "droplet"}]}))
        self.assertTrue(pedagogy.decide(a).labels_required)

    def test_the_decision_explains_itself(self):
        d = pedagogy.decide(analyse(slide("Angles in a Quadrilateral",
                                          ["A quadrilateral has four angles.",
                                           "Diagonals join opposite vertices."],
                                          legacy="concept")))
        self.assertIs(d.strategy, PedagogicalStrategy.MEASURED_FIGURE)
        self.assertIn("exact", d.reason)
        self.assertIn("shape cues", d.reason)


class ExclusionFilterTests(unittest.TestCase):
    """Exclusion removes strategies; it never reorders what remains."""

    def _caps(self, **kw):
        from ai_services.core.ppt_v2.visuals import VisualCapabilities
        base = dict(ai_images=True, image_search=True, diagrams=True, cache=True,
                    ai_image_budget=3)
        base.update(kw)
        return VisualCapabilities(**base)

    def test_order_is_fallback_chains_with_the_excluded_removed(self):
        from ai_services.core.ppt_v2.visuals import DeckVisualPlan, fallback_chain

        caps = self._caps()
        full = fallback_chain(DIAGRAM, caps, SlideType.ANATOMY)
        decided = DeckVisualPlan(caps=caps).decide(
            1, SlideType.ANATOMY, "a figure", preferred=DIAGRAM, excluded=(AI,))
        got = [decided.strategy] + decided.fallbacks
        self.assertEqual(got, [s for s in full if s is not AI])
        self.assertNotIn(AI, got)
        self.assertEqual(got[-1], NONE)

    def test_an_excluded_ai_image_does_not_spend_the_budget(self):
        from ai_services.core.ppt_v2.visuals import DeckVisualPlan

        plan = DeckVisualPlan(caps=self._caps(ai_image_budget=1))
        plan.decide(1, SlideType.ANATOMY, "a map", preferred=AI, excluded=(AI,))
        self.assertEqual(plan.ai_images_used, 0)

    def test_without_a_preference_the_existing_engine_decides(self):
        from ai_services.core.ppt_v2.visuals import DeckVisualPlan, preferred_strategy

        caps = self._caps()
        decided = DeckVisualPlan(caps=caps).decide(1, SlideType.PROCESS, "steps",
                                                   block_count=3)
        self.assertIs(decided.strategy,
                      preferred_strategy(SlideType.PROCESS, "steps", None, 3, ""))


class ComposeIntegrationTests(unittest.TestCase):
    def _compose(self, slides, **ctx):
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.visuals import VisualCapabilities

        caps = VisualCapabilities(ai_images=True, image_search=True, diagrams=True,
                                  cache=True, ai_image_budget=3)
        ctx.setdefault("subject", "Mathematics")
        ctx.setdefault("class_name", "Class 8")
        return compose_deck(slides, DeckContext(**ctx), caps=caps)

    def test_a_concept_labelled_geometry_slide_is_drawn_not_generated(self):
        deck = self._compose([slide(
            "Angles in a Quadrilateral",
            ["A quadrilateral has four angles.", "Its diagonals join opposite vertices."],
            legacy="concept", visual={"description": "a quadrilateral with diagonals",
                                      "objects": ["quadrilateral", "diagonal"]})])
        s = deck.slides[0]
        self.assertEqual(s.content_type, "geometric_diagram")
        self.assertIs(s.slide_type, SlideType.ANATOMY)
        self.assertIs(s.visual.strategy, DIAGRAM)
        self.assertNotIn(AI, s.visual.fallbacks)
        self.assertTrue(s.visual.labels_required)
        self.assertEqual(s.pedagogy["pedagogicalStrategy"], "measured_figure")

    def test_the_decision_travels_in_the_v2_payload(self):
        deck = self._compose([slide("Water", ["Water is wet.", "Water flows."])])
        v2 = deck.slides[0].to_dict()
        self.assertEqual(v2["contentType"], "concept_explanation")
        self.assertIn("reason", v2["pedagogy"])
        self.assertIn("labelsRequired", v2["visual"])
        self.assertIn("labelPolicy", v2["visual"])

    def test_a_reclassification_is_noted(self):
        deck = self._compose([slide("Angles in a Quadrilateral",
                                    ["A quadrilateral has four angles.",
                                     "Its diagonals join opposite vertices."],
                                    legacy="concept")])
        self.assertTrue(any("geometric_diagram" in n for n in deck.notes))


# ---------------------------------------------------------------------------
# the semantic label contract
# ---------------------------------------------------------------------------


def _annotated(n=3, **kw):
    spec = VisualSpec(description="a labelled figure", image_prompt="a figure",
                      diagram_spec="a figure", icons=["a"], **kw)
    spec.annotations = [Annotation(text="part %d" % i, anchor=(0.2 * i, 0.5))
                        for i in range(1, n + 1)]
    return spec


class ProducerPolicyTests(unittest.TestCase):
    def test_a_generated_image_is_anchored(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        with mock.patch("ai_services.core.image_generation.generate_note_image",
                        return_value={"ok": True, "url": "https://a.test/g.png",
                                      "provider": "together", "model": "m"}):
            self.assertEqual(ProducerSet().ai_image(_annotated())["labelPolicy"], "anchored")
            self.assertEqual(ProducerSet().ai_image(VisualSpec(image_prompt="x"))["labelPolicy"],
                             "none")

    def test_a_diagram_labels_itself_with_the_required_words(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        with mock.patch("ai_services.core.diagram.render_diagram",
                        return_value={"success": True,
                                      "image_base64": "data:image/png;base64,AAA"}) as draw:
            out = ProducerSet().diagram(_annotated())
        self.assertEqual(out["labelPolicy"], "native")
        source = draw.call_args.args[0]
        for word in ("part 1", "part 2", "part 3"):
            self.assertIn(word, source)
        # Always self-labelled now: the run-2 diagram was told not to label
        # itself and then labelled by nobody.
        self.assertIs(draw.call_args.kwargs["labels"], True)

    def test_a_searched_image_gets_a_legend_never_arrows(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        p = ProducerSet(search_fn=lambda t, ti, c=None, a=None: {
            "imageUrl": "https://s.test/photo.jpg"})
        self.assertEqual(p.search(_annotated())["labelPolicy"], "legend")
        self.assertEqual(ProducerSet(search_fn=lambda t, ti, c=None, a=None: {
            "imageUrl": "https://s.test/other.jpg"}).search(
                VisualSpec(search_terms=["x"]))["labelPolicy"], "none")

    def test_a_cached_asset_keeps_its_policy_or_never_points_into_it(self):
        from ai_services.core.ppt_v2.producers import ProducerSet, cache_key

        spec = _annotated()
        store = {cache_key(spec): {"url": "https://c.test/a.png", "labelPolicy": "native"}}
        self.assertEqual(ProducerSet(cache_get=store.get).cached(spec)["labelPolicy"], "native")
        old = {cache_key(spec): {"url": "https://c.test/a.png"}}   # pre-policy entry
        self.assertEqual(ProducerSet(cache_get=old.get).cached(spec)["labelPolicy"], "legend")

    def test_icons_are_their_own_labels(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        self.assertEqual(ProducerSet().icons(_annotated())["labelPolicy"], "native")


class LegendGeometryTests(unittest.TestCase):
    def test_boxes_stay_inside_the_picture_and_never_overlap(self):
        for n in range(1, 9):
            boxes = labels.legend_boxes(n)
            self.assertEqual(len(boxes), n)
            for x, y, w, h in boxes:
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x + w, 1.0 + 1e-9)
                self.assertLessEqual(y + h, 1.0 + 1e-9)
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    overlap = (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
                               and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])
                    self.assertFalse(overlap, (n, a, b))

    def test_a_legend_sits_along_the_bottom(self):
        for x, y, w, h in labels.legend_boxes(4):
            self.assertGreater(y, 0.5)

    def test_apply_policy_turns_anchored_labels_into_a_key(self):
        spec = _annotated(4)
        spec.label_policy = "legend"
        labels.apply_policy(spec)
        self.assertTrue(all(a.leader is False for a in spec.annotations))
        self.assertEqual([a.box for a in spec.annotations], labels.legend_boxes(4))
        self.assertEqual([a.text for a in spec.annotations],
                         ["part 1", "part 2", "part 3", "part 4"])

    def test_other_policies_leave_annotations_alone(self):
        for policy in ("anchored", "native", "none", ""):
            spec = _annotated(2)
            spec.label_policy = policy
            labels.apply_policy(spec)
            self.assertTrue(all(a.leader is True for a in spec.annotations), policy)


class LabelContractQATests(unittest.TestCase):
    def _slide(self, resolved, policy, annotations=3, required=True):
        from ai_services.core.ppt_v2.schema import SlideSpec, TitleSpec
        s = SlideSpec(slide_number=1, slide_type=SlideType.ANATOMY,
                      title=TitleSpec(text="A figure"))
        s.visual = _annotated(annotations) if annotations else VisualSpec()
        s.visual.labels_required = required
        s.visual.resolved_strategy = resolved
        s.visual.label_policy = policy
        return s

    def test_every_labelled_policy_satisfies_the_contract(self):
        for resolved, policy in ((AI, "anchored"), (DIAGRAM, "native"),
                                 (SEARCH, "legend"), (CACHED, "legend")):
            self.assertEqual(qa.check_label_contract(self._slide(resolved, policy)), [],
                             (resolved, policy))

    def test_a_raster_that_drops_required_labels_is_an_error(self):
        for policy in ("none", ""):
            issues = qa.check_label_contract(self._slide(SEARCH, policy))
            self.assertEqual([i.code for i in issues], ["required_labels_lost"])
            self.assertEqual(issues[0].severity, qa.ERROR)

    def test_it_maps_onto_the_visual_contract_category(self):
        from ai_services.core.ppt_v2 import gate
        self.assertEqual(gate.classify_issue("required_labels_lost"),
                         gate.VISUAL_CONTRACT_ERROR)

    def test_a_stripped_generated_image_with_no_planned_labels_is_flagged(self):
        issues = qa.check_label_contract(self._slide(AI, "none", annotations=0))
        self.assertEqual([i.code for i in issues], ["labels_unspecified"])
        self.assertEqual(issues[0].severity, qa.WARNING)

    def test_nothing_to_check_when_labels_are_not_required_or_no_raster(self):
        self.assertEqual(qa.check_label_contract(self._slide(SEARCH, "none", required=False)), [])
        self.assertEqual(qa.check_label_contract(self._slide(ICONS, "native")), [])
        self.assertEqual(qa.check_label_contract(self._slide(NONE, "")), [])


class FallbackContractTests(unittest.TestCase):
    """AI rejected -> every later producer must still carry the labels."""

    def _chain(self, spec, producers):
        from ai_services.core.ppt_v2.visuals import resolve_visual
        out = resolve_visual(spec, producers)
        labels.apply_policy(out)
        return out

    def test_ai_rejected_then_diagram_keeps_the_labels_native(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        spec = _annotated(5, strategy=AI, fallbacks=[DIAGRAM, SEARCH, ICONS, NONE])
        spec.labels_required = True
        ps = ProducerSet()
        with mock.patch("ai_services.core.diagram.render_diagram",
                        return_value={"success": True,
                                      "image_base64": "data:image/png;base64,AAA"}) as draw:
            out = self._chain(spec, {AI: lambda s: {"ok": False, "error": "vision rejected: x"},
                                     DIAGRAM: ps.diagram})
        self.assertIs(out.resolved_strategy, DIAGRAM)
        self.assertEqual(out.label_policy, "native")
        self.assertIn("part 5", draw.call_args.args[0])
        from ai_services.core.ppt_v2.schema import SlideSpec, TitleSpec
        s = SlideSpec(slide_number=1, slide_type=SlideType.ANATOMY,
                      title=TitleSpec(text="x"), visual=out)
        self.assertEqual(qa.check_label_contract(s), [])

    def test_ai_and_diagram_rejected_then_search_keeps_the_labels_as_a_key(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        spec = _annotated(3, strategy=AI, fallbacks=[DIAGRAM, SEARCH, ICONS, NONE],
                          search_terms=["figure"])
        spec.labels_required = True
        ps = ProducerSet(search_fn=lambda t, ti, c=None, a=None: {
            "imageUrl": "https://s.test/fig.jpg"})
        out = self._chain(spec, {AI: lambda s: {"ok": False, "error": "vision rejected"},
                                 DIAGRAM: lambda s: {"ok": False, "error": "illegible"},
                                 SEARCH: ps.search})
        self.assertIs(out.resolved_strategy, SEARCH)
        self.assertEqual(out.label_policy, "legend")
        self.assertTrue(all(a.leader is False for a in out.annotations))
        self.assertEqual(len(out.annotations), 3)

    def test_a_producer_that_declares_nothing_is_caught(self):
        # A future producer that forgets to declare a policy must not slip a
        # label-less picture past QA.
        from ai_services.core.ppt_v2.schema import SlideSpec, TitleSpec

        spec = _annotated(2, strategy=SEARCH, fallbacks=[NONE])
        spec.labels_required = True
        out = self._chain(spec, {SEARCH: lambda s: {"ok": True, "url": "https://s.test/x.jpg"}})
        s = SlideSpec(slide_number=1, slide_type=SlideType.ANATOMY,
                      title=TitleSpec(text="x"), visual=out)
        self.assertEqual([i.code for i in qa.check_label_contract(s)],
                         ["required_labels_lost"])


# ---------------------------------------------------------------------------
# generalisation guard
# ---------------------------------------------------------------------------


class NoTopicSpecificLogicTests(unittest.TestCase):
    """The engine must decide from shape, not from the name of a chapter."""

    TOPICS = ("photosynthesis", "friction", "parliament", "quadrilateral_layout",
              "demand and supply", "national movement", "water cycle", "dbms",
              "cell structure", "metals and non-metals", "linear equation",
              "photosynthesis_layout", "friction_layout", "cell_layout")

    def _source(self, module):
        with io.open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        # Code only: comments and docstrings may mention examples.
        text = re.sub(r'"""[\s\S]*?"""', "", text)
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines()).lower()

    def test_no_topic_names_in_the_decision_code(self):
        for module in (content, pedagogy):
            code = self._source(module)
            for topic in self.TOPICS:
                self.assertFalse(topic in code, "%r in %s" % (topic, module.__name__))

    def test_no_branching_on_topic_or_chapter_values(self):
        for module in (content, pedagogy):
            code = self._source(module)
            self.assertIsNone(re.search(r"\b(topic|chapter)\s*==", code), module.__name__)


if __name__ == "__main__":
    unittest.main()


class AdvancedTermsDetectorTests(unittest.TestCase):
    """The semantic gate's advanced-terminology check.

    Its regex carried a literal backspace byte where a word boundary was
    meant, so it searched for "\x08calvin cycle" and never found anything:
    every deck passed the semantic layer's terminology check unexamined.
    """

    def _slides(self, *texts):
        from ai_services.core.ppt_v2.schema import (
            BlockKind, ContentBlock, SlideSpec, TitleSpec)
        return [SlideSpec(slide_number=1, slide_type=SlideType.CONCEPT,
                          title=TitleSpec(text="Food in Plants"),
                          content_blocks=[ContentBlock(kind=BlockKind.BULLET, text=t)
                                          for t in texts])]

    def test_it_finds_an_advanced_term(self):
        from ai_services.core.ppt_v2.service import _advanced_terms
        found = _advanced_terms(self._slides("Glucose is made in the Calvin cycle."))
        self.assertIn("calvin cycle", found)

    def test_it_respects_word_starts(self):
        from ai_services.core.ppt_v2.service import _advanced_terms
        # "matp" is not "atp"; the boundary is the point of the regex.
        self.assertEqual(_advanced_terms(self._slides("A matp tool and a mat.")), [])

    def test_clean_text_finds_nothing(self):
        from ai_services.core.ppt_v2.service import _advanced_terms
        self.assertEqual(_advanced_terms(self._slides("Leaves make food using sunlight.")), [])

    def test_no_source_file_in_the_package_carries_a_control_byte(self):
        # The class of bug, not just this instance.
        import ai_services.core.ppt_v2 as pkg
        root = os.path.dirname(pkg.__file__)
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                with open(os.path.join(root, name), "rb") as fh:
                    data = fh.read()
                bad = sorted({b for b in data if b < 9 or 13 < b < 32})
                self.assertEqual(bad, [], name)


class StructuralSignalTests(unittest.TestCase):
    """Shape from STRUCTURE, with no shape vocabulary to lean on.

    Every payload uses invented nouns and avoids every cue word, so the only
    way a test can pass is through the payload's structure: block kinds,
    numbered labels, the visual's kind and relationships. Each positive case
    has a control - the same words without the structural signal - which
    must stay at the concept-explanation default.
    """

    def _infer(self, data):
        data = dict(data)
        data.setdefault("slideType", "concept")
        return content.analyse(data, subject="General", class_level="Class 8")

    def _assert_no_cue_words(self, data):
        # Guard the guard: the payload's WORDING must not contain any lexical
        # cue, or the test proves nothing about structure. Only the text a
        # reader sees is scanned - structural field values such as
        # kind="definition" or a visual kind of "chart" are the signals under
        # test, not vocabulary.
        visual = data.get("visual") or {}
        parts = [data.get("title", ""), data.get("subtitle", ""),
                 data.get("learningObjective", ""), visual.get("description", "")]
        for b in data.get("blocks", []):
            parts += [b.get("label", ""), b.get("text", "")]
        for key in ("objects", "relationships", "mustShow"):
            parts += list(visual.get(key) or [])
        parts += [a.get("text", "") for a in visual.get("annotations") or []]
        text = " ".join(str(p) for p in parts).lower()
        for ctype, cues in content._CUES.items():
            for cue in cues:
                if content.matches_any(text, (cue,)):
                    self.fail("payload contains the %s cue %r" % (ctype.value, cue))

    # ---- ordered sequence -> PROCESS ----------------------------------------

    def _ordered(self, kind="step", labels=("Velm", "Quorrin", "Brask", "Talo")):
        return {"title": "How the Velm Becomes Talo",
                "learningObjective": "Describe what happens to a velm",
                "blocks": [{"kind": kind, "label": lab, "text": txt} for lab, txt in zip(labels, (
                    "The velm is gathered from the high shelf.",
                    "It is pressed until it turns into quorrin.",
                    "The quorrin is warmed and becomes brask.",
                    "The brask cools and hardens into talo."))]}

    def test_marked_steps_infer_a_process(self):
        data = self._ordered()
        self._assert_no_cue_words(data)
        a = self._infer(data)
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)
        self.assertIn("dependent sequence", a.reason)

    def test_numbered_labels_infer_a_process(self):
        data = self._ordered(kind="bullet", labels=("1", "2.", "3)", "04"))
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.PROCESS)

    def test_control_the_same_text_unordered_stays_concept(self):
        data = self._ordered(kind="bullet")
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)

    def test_one_marked_step_is_not_a_sequence(self):
        data = self._ordered()
        data["blocks"] = data["blocks"][:1]
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)

    def test_four_digit_labels_are_years_not_ordinals(self):
        self.assertFalse(content._ORDINAL_LABEL.match("1885"))
        self.assertTrue(content._ORDINAL_LABEL.match("Step 3"))

    # ---- sequence variants ----------------------------------------------------

    def test_a_sequence_that_returns_to_its_start_infers_a_cycle(self):
        data = {"title": "What Happens to Velm",
                "blocks": [
                    {"kind": "step", "label": "Velm", "text": "A small velm sinks into the mud."},
                    {"kind": "step", "label": "Quorrin", "text": "It swells into a soft quorrin."},
                    {"kind": "step", "label": "Brask", "text": "The quorrin hardens into brask."},
                    {"kind": "step", "label": "Talo", "text": "The talo sheds a new velm."}]}
        self._assert_no_cue_words(data)
        a = self._infer(data)
        self.assertIs(a.content_type, ContentType.CYCLE, a.reason)

    def test_control_a_sequence_that_does_not_close_stays_a_process(self):
        data = {"title": "What Happens to Velm",
                "blocks": [
                    {"kind": "step", "label": "Velm", "text": "A small velm sinks into the mud."},
                    {"kind": "step", "label": "Quorrin", "text": "It swells into a soft quorrin."},
                    {"kind": "step", "label": "Brask", "text": "The quorrin hardens into brask."},
                    {"kind": "step", "label": "Talo", "text": "The brask settles as talo."}]}
        self.assertIs(self._infer(data).content_type, ContentType.PROCESS)

    def test_a_branching_sequence_infers_a_flowchart(self):
        data = {"title": "Sorting a Velm",
                "blocks": [
                    {"kind": "step", "label": "1", "text": "Pick up the next velm."},
                    {"kind": "step", "label": "2", "text": "If it is heavy, put it on the left shelf."},
                    {"kind": "step", "label": "3", "text": "Otherwise put it on the right shelf."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.FLOWCHART_ALGORITHM)

    def test_a_sequence_of_worked_arithmetic_infers_a_worked_example(self):
        data = {"title": "Velms in a Crate",
                "blocks": [
                    {"kind": "step", "label": "1", "text": "Each row holds 6 velms and there are 4 rows: 6 × 4 = 24."},
                    {"kind": "step", "label": "2", "text": "Three velms are broken: 24 - 3 = 21."},
                    {"kind": "step", "label": "3", "text": "So 21 velms are whole."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.WORKED_EXAMPLE)

    # ---- term + meaning -> DEFINITION_TERMINOLOGY ----------------------------

    def test_definition_blocks_infer_terminology(self):
        data = {"title": "Three Kinds of Velm Talk",
                "blocks": [
                    {"kind": "definition", "label": "Quorrin", "text": "A word for a slow sound."},
                    {"kind": "definition", "label": "Brask", "text": "A word for a sharp sound."},
                    {"kind": "definition", "label": "Talo", "text": "A word for no sound at all."}]}
        data["title"] = "Velm Talk"          # "kinds of" would be a classification cue
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.DEFINITION_TERMINOLOGY)

    def test_term_and_example_pairs_alone_are_not_terminology(self):
        # V1 counted these as definitions; a held-out probe showed ordinary
        # concept bullets have exactly this shape. Examples alone prove
        # nothing - the text has to be written as a definition.
        data = {"title": "Velm Talk",
                "blocks": [
                    {"kind": "bullet", "label": "Quorrin", "text": "A slow sound, as in 'the quorrin of the hill'."},
                    {"kind": "bullet", "label": "Brask", "text": "A sharp sound, e.g. the brask of a gate."},
                    {"kind": "bullet", "label": "Talo", "text": "No sound at all, as in 'a talo night'."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)

    def test_control_terms_without_examples_stay_concept(self):
        data = {"title": "Velm Talk",
                "blocks": [
                    {"kind": "bullet", "label": "Quorrin", "text": "A slow sound."},
                    {"kind": "bullet", "label": "Brask", "text": "A sharp sound."},
                    {"kind": "bullet", "label": "Talo", "text": "No sound at all."}]}
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)

    # ---- related components -> SYSTEM_ARCHITECTURE ---------------------------

    def _components(self, relationships=True, kind="bullet"):
        visual = {"kind": "diagram",
                  "description": "three boxes for the velm, the quorrin and the brask",
                  "objects": ["velm", "quorrin", "brask"]}
        if relationships:
            visual["relationships"] = ["the velm hands work to the quorrin and the brask"]
        return {"title": "How the Velm Works With Others",
                "blocks": [
                    {"kind": kind, "label": "Velm", "text": "Decides what work must be done."},
                    {"kind": kind, "label": "Quorrin", "text": "Carries out the work it is given."},
                    {"kind": kind, "label": "Brask", "text": "Keeps a record of the finished work."}],
                "visual": visual}

    def test_related_components_infer_a_system(self):
        data = self._components()
        self._assert_no_cue_words(data)
        a = self._infer(data)
        self.assertIs(a.content_type, ContentType.SYSTEM_ARCHITECTURE, a.reason)
        self.assertIn("interacting components", a.reason)

    def test_arrows_in_a_picture_alone_are_not_relationships(self):
        # V1 accepted "joined by arrows" as evidence. Arbitrary arrows say
        # nothing about which component acts on which.
        data = self._components(relationships=False)
        data["visual"]["description"] = "three boxes for the velm, quorrin and brask joined by arrows"
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)

    def test_control_components_with_no_relationships_stay_concept(self):
        # Named parts with nothing connecting them are not a system.
        self.assertIs(self._infer(self._components(relationships=False)).content_type,
                      ContentType.CONCEPT_EXPLANATION)

    def test_control_dependent_stages_through_components_are_a_process(self):
        # When the stages genuinely build on each other, order is the
        # stronger structure: a flow through components is a process.
        data = self._components()
        data["blocks"] = [
            {"kind": "step", "label": "Velm", "text": "The velm decides what work must be done."},
            {"kind": "step", "label": "Quorrin", "text": "It passes that work to the quorrin."},
            {"kind": "step", "label": "Brask", "text": "The quorrin hands the finished work to the brask."}]
        self.assertIs(self._infer(data).content_type, ContentType.PROCESS)

    def test_marked_but_independent_components_are_not_a_sequence(self):
        # The same components marked as steps but not building on each
        # other: the step marking is not order, so the relationships decide.
        self.assertIs(self._infer(self._components(kind="step")).content_type,
                      ContentType.SYSTEM_ARCHITECTURE)

    def test_a_labelled_structure_without_connections_is_not_a_system(self):
        # Parts of one whole, cross-sectioned: no relationships, no
        # connectors - this must stay an anatomy, not become a system.
        data = {"title": "Structure of the Velm",
                "blocks": [{"kind": "bullet", "label": n, "text": "One part of the velm."}
                           for n in ("Rim", "Core", "Shell")],
                "visual": {"kind": "diagram", "description": "a velm cut in half",
                           "objects": ["rim", "core", "shell"]}}
        self.assertIs(self._infer(data).content_type, ContentType.ANATOMY_STRUCTURE)

    # ---- other structural signals ---------------------------------------------

    def test_paired_sides_infer_a_comparison(self):
        data = {"title": "Velm and Quorrin",
                "blocks": [{"kind": "compare_left", "label": "Velm", "text": "Heavy and slow."},
                           {"kind": "compare_right", "label": "Quorrin", "text": "Light and quick."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.COMPARISON)

    def test_a_stated_equation_infers_a_formula_whatever_its_notation(self):
        # Function notation defeats the letter-formula regex; the planner's
        # own "equation" marking does not.
        data = {"title": "Velm Weight",
                "blocks": [{"kind": "equation", "label": "Rule", "text": "W(v) = mass of velm ÷ size of crate"},
                           {"kind": "bullet", "label": "Mass", "text": "How heavy the velm is."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.FORMULA_EQUATION)

    def test_a_chart_visual_infers_quantitative_data(self):
        data = {"title": "Velms Sold Each Month",
                "blocks": [{"kind": "bullet", "label": "Spring", "text": "Most velms were sold in spring."},
                           {"kind": "bullet", "label": "Winter", "text": "Very few were sold in winter."}],
                "visual": {"kind": "chart", "description": "velms sold in each season"}}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.GRAPH_DATA)

    # ---- invariants ------------------------------------------------------------

    def test_structure_never_overrides_a_fillable_declaration(self):
        data = self._ordered()
        data["contentType"] = "comparison"
        a = self._infer(data)
        self.assertIs(a.content_type, ContentType.COMPARISON)
        self.assertEqual(a.source, "declared")

    def test_structural_inference_is_deterministic(self):
        for data in (self._ordered(), self._components()):
            first = self._infer(data)
            for _ in range(5):
                again = self._infer(data)
                self.assertIs(again.content_type, first.content_type)
                self.assertEqual(again.reason, first.reason)
                self.assertEqual(again.scores, first.scores)

    def test_no_structure_and_no_cues_is_still_concept(self):
        data = {"title": "About the Velm",
                "blocks": [{"kind": "bullet", "label": "Size", "text": "A velm fits in one hand."},
                           {"kind": "bullet", "label": "Colour", "text": "Most velms are grey."}]}
        self._assert_no_cue_words(data)
        self.assertIs(self._infer(data).content_type, ContentType.CONCEPT_EXPLANATION)


class SafeStructuralPairsTests(unittest.TestCase):
    """Safe Structural V2: every positive beside a structurally similar negative.

    A false positive picks a wrong representation; CONCEPT_EXPLANATION always
    renders safely. So each pair below keeps the SAME surface - the same
    marking, the same nouns, the same block count - and changes only the one
    structural property that should decide the type.
    """

    _infer = StructuralSignalTests._infer
    _assert_no_cue_words = StructuralSignalTests._assert_no_cue_words

    def _is(self, data, ctype):
        self._assert_no_cue_words(data)
        a = self._infer(data)
        self.assertIs(a.content_type, ctype, a.reason)
        return a

    # 1/2 true ordered process  vs  ordinary numbered list ----------------------

    def test_pair_1_dependent_numbered_stages_are_a_process(self):
        self._is({"title": "Making Dromm",
                  "blocks": [{"kind": "bullet", "label": "1", "text": "Soak the fennow in cold brine."},
                             {"kind": "bullet", "label": "2", "text": "Crush the soaked fennow into a paste."},
                             {"kind": "bullet", "label": "3", "text": "Spread the paste thinly and let it harden."}]},
                 ContentType.PROCESS)

    def test_pair_2_same_numbering_independent_items_stay_concept(self):
        self._is({"title": "Making Dromm",
                  "blocks": [{"kind": "bullet", "label": "1", "text": "Fennow grows near cold hills."},
                             {"kind": "bullet", "label": "2", "text": "Brine is sold in small jars."},
                             {"kind": "bullet", "label": "3", "text": "Paste keeps well in a dark room."}]},
                 ContentType.CONCEPT_EXPLANATION)

    def test_pair_2b_step_kind_alone_is_not_order(self):
        # Marked as steps by the planner, but nothing builds on anything.
        self._is({"title": "Looking After a Velm",
                  "blocks": [{"kind": "step", "label": "Shade", "text": "Keep the velm out of strong light."},
                             {"kind": "step", "label": "Water", "text": "Give fresh water every morning."},
                             {"kind": "step", "label": "Space", "text": "Leave room around the pot."}]},
                 ContentType.CONCEPT_EXPLANATION)

    def test_parallel_framing_repeated_in_every_stage_is_not_progression(self):
        # "brask" in every line is framing, not one stage feeding the next.
        self._is({"title": "Velm Habits",
                  "blocks": [{"kind": "step", "label": "1", "text": "The brask wakes before the sun."},
                             {"kind": "step", "label": "2", "text": "The brask hums near the water."},
                             {"kind": "step", "label": "3", "text": "The brask sleeps under leaves."}]},
                 ContentType.CONCEPT_EXPLANATION)

    # 3 worked example  vs  numbered numbers ------------------------------------

    def test_pair_3_numbered_worked_answer_is_a_worked_example(self):
        self._is({"title": "Velms for the Fair",
                  "blocks": [{"kind": "bullet", "label": "1", "text": "Five crates of 9 velms: 5 × 9 = 45."},
                             {"kind": "bullet", "label": "2", "text": "Seven are sold early: 45 - 7 = 38."},
                             {"kind": "bullet", "label": "3", "text": "So 38 velms are left for the fair."}]},
                 ContentType.WORKED_EXAMPLE)

    def test_pair_3_control_numbered_sums_that_carry_nothing_stay_concept(self):
        # Same marking, same operators - but no line uses an earlier result.
        self._is({"title": "Velms for the Fair",
                  "blocks": [{"kind": "bullet", "label": "1", "text": "Red crates: 5 × 9 = 45."},
                             {"kind": "bullet", "label": "2", "text": "Blue crates: 4 × 6 = 24."},
                             {"kind": "bullet", "label": "3", "text": "Green crates: 2 × 7 = 14."}]},
                 ContentType.CONCEPT_EXPLANATION)

    def test_pair_3_control_a_numbered_list_of_amounts_stays_concept(self):
        self._is({"title": "Where the Velms Went",
                  "blocks": [{"kind": "bullet", "label": "1", "text": "Half of the velms went away."},
                             {"kind": "bullet", "label": "2", "text": "A quarter of them stayed home."},
                             {"kind": "bullet", "label": "3", "text": "About 10% were never found."}]},
                 ContentType.CONCEPT_EXPLANATION)

    def test_worked_is_read_before_process(self):
        # Dependent AND worked: the more specific reading wins outright.
        a = self._is({"title": "Velms for the Fair",
                      "blocks": [{"kind": "step", "label": "1", "text": "Five crates of 9 velms: 5 × 9 = 45."},
                                 {"kind": "step", "label": "2", "text": "Seven of these velms are sold: 45 - 7 = 38."},
                                 {"kind": "step", "label": "3", "text": "So 38 of these velms are left."}]},
                     ContentType.WORKED_EXAMPLE)
        self.assertNotIn("process", a.scores)

    def test_a_numbered_worked_example_is_fillable_when_declared(self):
        # The shared marking: numbered labels count for the declared check too.
        data = {"title": "Velms for the Fair", "contentType": "worked_example",
                "blocks": [{"kind": "bullet", "label": "1", "text": "5 × 9 = 45."},
                           {"kind": "bullet", "label": "2", "text": "45 - 7 = 38."}]}
        a = self._infer(data)
        self.assertIs(a.content_type, ContentType.WORKED_EXAMPLE)
        self.assertEqual(a.source, "declared")

    # 4/5 term->meaning  vs  concept examples -----------------------------------

    def test_pair_4_terms_written_as_definitions_are_terminology(self):
        self._is({"title": "Velm Talk",
                  "blocks": [{"kind": "bullet", "label": "Quorrin", "text": "Quorrin is a slow, low sound."},
                             {"kind": "bullet", "label": "Brask", "text": "Brask is a short, sharp sound."},
                             {"kind": "bullet", "label": "Talo", "text": "Talo is the silence after a sound."}]},
                 ContentType.DEFINITION_TERMINOLOGY)

    def test_pair_5_same_terms_as_properties_with_examples_stay_concept(self):
        self._is({"title": "Velm Talk",
                  "blocks": [{"kind": "bullet", "label": "Quorrin", "text": "Carries far over water, as in the bay."},
                             {"kind": "bullet", "label": "Brask", "text": "Startles birds, e.g. at dawn."},
                             {"kind": "bullet", "label": "Talo", "text": "Calms the herd, such as at night."}]},
                 ContentType.CONCEPT_EXPLANATION)

    def test_one_definition_is_not_a_terminology_slide(self):
        self._is({"title": "Velm Talk",
                  "blocks": [{"kind": "bullet", "label": "Quorrin", "text": "Quorrin is a slow sound."},
                             {"kind": "bullet", "label": "Brask", "text": "Startles birds at dawn."},
                             {"kind": "bullet", "label": "Talo", "text": "Calms the herd at night."}]},
                 ContentType.CONCEPT_EXPLANATION)

    # 6/7/8 component interaction  vs  parts of one object  vs  forces on one ---

    def _three(self, relationships, title="The Velm Works", description="three boxes"):
        return {"title": title,
                "blocks": [{"kind": "bullet", "label": "Quorrin", "text": "Gathers the raw fennow."},
                           {"kind": "bullet", "label": "Brask", "text": "Sorts what it is given."},
                           {"kind": "bullet", "label": "Talo", "text": "Stores what is sorted."}],
                "visual": {"kind": "diagram", "description": description,
                           "objects": ["quorrin", "brask", "talo"],
                           "relationships": relationships}}

    def test_pair_6_components_acting_on_each_other_are_a_system(self):
        self._is(self._three(["the quorrin sends fennow to the brask",
                              "the brask passes sorted fennow to the talo"]),
                 ContentType.SYSTEM_ARCHITECTURE)

    def test_pair_7_the_same_parts_framed_as_one_object_are_not_a_system(self):
        data = self._three(["the quorrin sends fennow to the brask",
                            "the brask passes sorted fennow to the talo"],
                           title="Parts of the Velm",
                           description="a velm cut open to show the quorrin, brask and talo")
        a = self._infer(data)
        self.assertIsNot(a.content_type, ContentType.SYSTEM_ARCHITECTURE, a.reason)
        self.assertIs(a.content_type, ContentType.ANATOMY_STRUCTURE)

    def test_pair_7b_something_passing_through_parts_is_not_component_interaction(self):
        # The subject is a substance moving through the parts, not a part
        # acting on another part.
        self._is(self._three(["fennow passes through the quorrin and brask into the talo"]),
                 ContentType.CONCEPT_EXPLANATION)

    def test_pair_8_several_actors_on_one_passive_object_are_not_a_system(self):
        data = {"title": "Why the Velm Stays Still",
                "blocks": [{"kind": "bullet", "label": "Quorrin", "text": "The quorrin leans on the velm."},
                           {"kind": "bullet", "label": "Brask", "text": "The brask tugs the velm back."},
                           {"kind": "bullet", "label": "Velm", "text": "The velm stays still between them."}],
                "visual": {"kind": "illustration", "description": "a velm between a quorrin and a brask",
                           "objects": ["velm", "quorrin", "brask"],
                           "relationships": ["the quorrin leans on the velm",
                                             "the brask tugs the velm back"]}}
        self._is(data, ContentType.CONCEPT_EXPLANATION)

    # 9/10 hierarchy  vs  independent labelled structure ------------------------

    def test_pair_9_a_hierarchy_of_roles_is_a_system(self):
        self._is({"title": "Who Guides Whom",
                  "blocks": [{"kind": "bullet", "label": "Elder", "text": "Sets the plan for the whole group."},
                             {"kind": "bullet", "label": "Warden", "text": "Turns the plan into daily tasks."},
                             {"kind": "bullet", "label": "Runner", "text": "Carries out the daily tasks."}],
                  "visual": {"kind": "diagram", "description": "the elder above the wardens above the runners",
                             "objects": ["elder", "warden", "runner"],
                             "relationships": ["the elder guides the wardens",
                                               "each warden directs the runners"]}},
                 ContentType.SYSTEM_ARCHITECTURE)

    def test_pair_10_the_same_roles_unconnected_stay_concept(self):
        self._is({"title": "Who Guides Whom",
                  "blocks": [{"kind": "bullet", "label": "Elder", "text": "Sets the plan for the whole group."},
                             {"kind": "bullet", "label": "Warden", "text": "Turns the plan into daily tasks."},
                             {"kind": "bullet", "label": "Runner", "text": "Carries out the daily tasks."}],
                  "visual": {"kind": "diagram", "description": "the elder, a warden and a runner",
                             "objects": ["elder", "warden", "runner"]}},
                 ContentType.CONCEPT_EXPLANATION)

    # evidence balancing ---------------------------------------------------------

    def test_a_narrow_structural_lead_over_a_rival_is_ambiguous(self):
        from ai_services.core.ppt_v2.content import _choose
        chosen, _score, note = _choose({ContentType.SYSTEM_ARCHITECTURE: 3,
                                        ContentType.ANATOMY_STRUCTURE: 2})
        self.assertIsNone(chosen)
        self.assertIn("ambiguous", note)

    def test_a_heading_named_type_is_not_overridden_by_a_narrow_lead(self):
        from ai_services.core.ppt_v2.content import _choose
        chosen, _s, _n = _choose({ContentType.SYSTEM_ARCHITECTURE: 3,
                                  ContentType.ANATOMY_STRUCTURE: 2},
                                 heading={ContentType.ANATOMY_STRUCTURE: 1})
        self.assertIs(chosen, ContentType.ANATOMY_STRUCTURE)

    def test_a_clear_lead_still_wins(self):
        from ai_services.core.ppt_v2.content import _choose
        chosen, _s, _n = _choose({ContentType.SYSTEM_ARCHITECTURE: 5,
                                  ContentType.ANATOMY_STRUCTURE: 2},
                                 heading={ContentType.ANATOMY_STRUCTURE: 1})
        self.assertIs(chosen, ContentType.SYSTEM_ARCHITECTURE)

    def test_an_unopposed_type_at_threshold_is_accepted(self):
        from ai_services.core.ppt_v2.content import _choose
        chosen, _s, _n = _choose({ContentType.PROCESS: 2, ContentType.CYCLE: 1})
        self.assertIs(chosen, ContentType.PROCESS)


class SpecificityOverrideTests(unittest.TestCase):
    """Phase 11: generic PROCESS versus the specific sequence readings.

    Invented nouns throughout. Where a test needs a specific cue ("cycle",
    "experiment") it is the heading that carries it, because that is the
    policy under test: a specific reading the heading names takes the
    sequence's points; a passing mention in the body does not.
    """

    def _infer(self, data):
        data = dict(data)
        data.setdefault("slideType", "concept")
        return content.analyse(data, subject="General", class_level="Class 8")

    def _stages(self, labels, texts, kind="step"):
        return [{"kind": kind, "label": lab, "text": txt} for lab, txt in zip(labels, texts)]

    # a dependent sequence whose last stage does NOT name the first
    DEPENDENT = ("The velm is gathered from the high shelf.",
                 "It is pressed until it turns into quorrin.",
                 "The quorrin is warmed until it becomes brask.",
                 "The brask cools and hardens into talo.")
    # a dependent sequence whose last stage names the first stage again
    CLOSING = ("A small velm sinks into the mud.",
               "It swells into a soft quorrin.",
               "The quorrin hardens into brask.",
               "The brask cracks open and a new velm crawls out.")

    # ---- Part 1: label normalisation ------------------------------------------

    def test_numbering_is_stripped_for_comparison_only(self):
        from ai_services.core.ppt_v2.content import normalise_labels
        blocks = [{"label": l} for l in ("1. Velm", "2) Quorrin", "Step 3: Brask")]
        self.assertEqual(normalise_labels(blocks), ["Velm", "Quorrin", "Brask"])
        self.assertEqual(blocks[0]["label"], "1. Velm")        # untouched

    def test_a_counting_series_of_bare_numbers_is_numbering(self):
        from ai_services.core.ppt_v2.content import normalise_labels
        self.assertEqual(normalise_labels([{"label": l} for l in ("01 Velm", "02 Quorrin", "03 Talo")]),
                         ["Velm", "Quorrin", "Talo"])

    def test_meaningful_numbers_are_kept(self):
        from ai_services.core.ppt_v2.content import normalise_labels
        for labels in (("1857 Velm", "1858 Quorrin"),      # years
                       ("12 Velms", "3 Quorrins"),          # quantities, not counting from 1
                       ("3D Velm",), ("03 Velm",)):         # lone bare number: content
            self.assertEqual(normalise_labels([{"label": l} for l in labels]), list(labels))

    def test_a_pure_ordinal_names_nothing(self):
        from ai_services.core.ppt_v2.content import normalise_labels
        self.assertEqual(normalise_labels([{"label": "1"}, {"label": "Step 2"}]), ["", ""])

    def test_the_original_label_is_what_the_slide_renders(self):
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        deck = compose_deck([{"slideType": "concept", "title": "What Happens to Velm",
                              "blocks": self._stages(("1. Velm", "2. Quorrin", "3. Brask", "4. Talo"),
                                                     self.CLOSING)}],
                            DeckContext(subject="General", class_name="Class 8"))
        self.assertEqual([b.label for b in deck.slides[0].content_blocks][0], "1. Velm")

    # ---- Part 4: cycle closing on normalised labels ---------------------------

    def test_pos_numbered_cycle_whose_first_label_is_1_dot_x(self):
        a = self._infer({"title": "What Happens to Velm",
                         "blocks": self._stages(("1. Velm", "2. Quorrin", "3. Brask", "4. Talo"),
                                                self.CLOSING)})
        self.assertIs(a.content_type, ContentType.CYCLE, a.reason)

    def test_pos_cycle_closes_on_the_first_stage_by_plural(self):
        a = self._infer({"title": "What Happens to Velm",
                         "blocks": self._stages(("1. Velm", "2. Quorrin", "3. Brask", "4. Talo"),
                                                self.CLOSING[:3] + ("The talo lays many velms.",))})
        self.assertIs(a.content_type, ContentType.CYCLE, a.reason)

    def test_neg_same_numbered_stages_that_do_not_close_stay_process(self):
        a = self._infer({"title": "What Happens to Velm",
                         "blocks": self._stages(("1. Velm", "2. Quorrin", "3. Brask", "4. Talo"),
                                                self.DEPENDENT)})
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)

    # ---- Part 2: specificity hierarchy ----------------------------------------

    def test_pos_a_heading_named_cycle_takes_the_sequence_points(self):
        # No structural loop, but the heading names a cycle: the dependent
        # sequence is read as the cycle, not as a generic process.
        a = self._infer({"title": "The Velm Cycle",
                         "blocks": self._stages(("Velm", "Quorrin", "Brask", "Talo"), self.DEPENDENT)})
        self.assertIs(a.content_type, ContentType.CYCLE, a.reason)
        self.assertIn("more specific", a.reason)
        self.assertNotIn("process", a.scores)

    def test_neg_a_weak_cycle_hint_does_not_override_a_process(self):
        # One passing mention in the body, and a heading that names steps.
        a = self._infer({"title": "How Velm Is Made",
                         "learningObjective": "Put the steps in order",
                         "blocks": self._stages(("Velm", "Quorrin", "Brask", "Talo"),
                                                self.DEPENDENT[:3] +
                                                ("The brask cools into talo, ending its cycle.",))})
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)

    def test_pos_a_heading_named_experiment_takes_the_sequence_points(self):
        a = self._infer({"title": "Velm Dust Experiment",
                         "blocks": self._stages(("Velm", "Quorrin", "Brask", "Talo"), self.DEPENDENT)})
        self.assertIs(a.content_type, ContentType.EXPERIMENT_PROCEDURE, a.reason)

    def test_neg_the_same_procedure_without_the_heading_is_a_process(self):
        a = self._infer({"title": "Velm Dust",
                         "blocks": self._stages(("Velm", "Quorrin", "Brask", "Talo"), self.DEPENDENT)})
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)

    def test_pos_numbered_branching_sequence_is_a_flowchart(self):
        a = self._infer({"title": "Sorting Velms",
                         "blocks": self._stages(("1. Pick", "2. Check", "3. Place"), (
                             "Pick up the next velm from the pile.",
                             "If it is heavy, put it on the left shelf.",
                             "Otherwise put it on the right shelf."))})
        self.assertIs(a.content_type, ContentType.FLOWCHART_ALGORITHM, a.reason)

    def test_neg_a_weak_flowchart_hint_does_not_override_a_process(self):
        a = self._infer({"title": "How Velm Is Made",
                         "learningObjective": "Put the steps in order",
                         "blocks": self._stages(("Velm", "Quorrin", "Brask", "Talo"),
                                                self.DEPENDENT[:3] +
                                                ("The brask cools into talo in a loop of air.",))})
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)

    def test_pos_numbered_worked_example_with_carried_result(self):
        a = self._infer({"title": "Velms for the Fair",
                         "blocks": self._stages(("1)", "2)", "3)"), (
                             "Five crates of 9 velms: 5 × 9 = 45.",
                             "Seven are sold: 45 - 7 = 38.",
                             "So 38 velms are left."), kind="bullet")})
        self.assertIs(a.content_type, ContentType.WORKED_EXAMPLE, a.reason)

    def test_neg_ordinary_numbered_list_stays_concept(self):
        a = self._infer({"title": "About Velms",
                         "blocks": self._stages(("1. Size", "2. Colour", "3. Home"), (
                             "A velm fits in one hand.", "Most velms are grey.",
                             "Velms live under flat stones."), kind="bullet")})
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION, a.reason)

    def test_neg_independent_numbered_stages_stay_concept(self):
        a = self._infer({"title": "Looking After a Velm",
                         "blocks": self._stages(("1. Shade", "2. Water", "3. Space"), (
                             "Keep the velm out of strong light.", "Give fresh water every morning.",
                             "Leave room around the pot."))})
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION, a.reason)

    def test_neg_repeated_framing_is_not_a_cycle_even_when_numbered(self):
        a = self._infer({"title": "Velm Habits",
                         "blocks": self._stages(("1. Brask", "2. Brask", "3. Brask"), (
                             "The brask wakes before the sun.", "The brask hums near the water.",
                             "The brask sleeps under leaves."))})
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION, a.reason)

    def test_neg_parts_of_one_object_stay_anatomy_with_numbered_labels(self):
        a = self._infer({"title": "Parts of the Velm",
                         "blocks": [{"kind": "bullet", "label": "%d. %s" % (i, n), "text": "One part of the velm."}
                                    for i, n in enumerate(("Rim", "Core", "Shell"), 1)],
                         "visual": {"kind": "diagram", "description": "a velm cut in half",
                                    "objects": ["rim", "core", "shell"],
                                    "relationships": ["the rim holds the core inside the shell"]}})
        self.assertIs(a.content_type, ContentType.ANATOMY_STRUCTURE, a.reason)

    def test_neg_an_unrelated_component_list_stays_concept(self):
        a = self._infer({"title": "Things in a Velm Den",
                         "blocks": [{"kind": "bullet", "label": n, "text": "Found in most dens."}
                                    for n in ("Quorrin", "Brask", "Talo")],
                         "visual": {"kind": "illustration", "description": "a den",
                                    "objects": ["quorrin", "brask", "talo"]}})
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION, a.reason)

    # ---- Part 3: generic planner declarations ---------------------------------

    def _declared(self, declared, title, texts, labels=("1. Velm", "2. Quorrin", "3. Brask", "4. Talo"),
                  **extra):
        data = {"contentType": declared, "slideType": "process", "title": title,
                "blocks": self._stages(labels, texts)}
        data.update(extra)
        return content.analyse(data, subject="General", class_level="Class 8")

    def test_pos_declared_process_with_strong_cycle_is_upgraded(self):
        a = self._declared("process", "The Velm Cycle", self.CLOSING)
        self.assertIs(a.content_type, ContentType.CYCLE, a.reason)
        self.assertEqual(a.source, "upgraded")

    def test_neg_declared_process_with_a_weak_cycle_hint_is_kept(self):
        a = self._declared("process", "How Velm Is Made",
                           self.DEPENDENT[:3] + ("The brask cools into talo, ending its cycle.",))
        self.assertIs(a.content_type, ContentType.PROCESS, a.reason)
        self.assertEqual(a.source, "declared")

    def test_neg_declared_process_with_no_specific_evidence_is_kept(self):
        a = self._declared("process", "How Velm Is Made", self.DEPENDENT)
        self.assertIs(a.content_type, ContentType.PROCESS)
        self.assertEqual(a.source, "declared")

    def test_neg_a_specific_declaration_is_never_overridden(self):
        # Anatomy declared; the structure also supports a system reading.
        # A specific declaration is the planner committing to a shape, so it
        # stands even against strong rival evidence.
        a = content.analyse({
            "contentType": "anatomy_structure", "title": "The Velm Works",
            "blocks": [{"kind": "bullet", "label": n, "text": t} for n, t in (
                ("Quorrin", "Gathers the raw fennow."), ("Brask", "Sorts what it is given."),
                ("Talo", "Stores what is sorted."))],
            "visual": {"kind": "diagram", "description": "three boxes",
                       "objects": ["quorrin", "brask", "talo"],
                       "relationships": ["the quorrin sends fennow to the brask",
                                         "the brask passes sorted fennow to the talo"]}},
            subject="General", class_level="Class 8")
        self.assertIs(a.content_type, ContentType.ANATOMY_STRUCTURE)
        self.assertEqual(a.source, "declared")

    def test_neg_concept_declaration_still_infers_and_still_falls_back_safely(self):
        a = content.analyse({"contentType": "concept_explanation", "title": "About Velms",
                             "blocks": [{"kind": "bullet", "label": "Size", "text": "A velm fits in one hand."},
                                        {"kind": "bullet", "label": "Colour", "text": "Most velms are grey."}]},
                            subject="General", class_level="Class 8")
        self.assertIs(a.content_type, ContentType.CONCEPT_EXPLANATION)
