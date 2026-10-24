"""The validation policy, the three QA layers, and the contract fixtures.

Two ideas are under test here.

**Spend a vision call only where it buys something.** The deterministic
score already separates clearly-right from clearly-wrong for nothing. What
it cannot separate is the middle: an on-topic caption on a reputable host
whose picture may still teach nothing. So HIGH is accepted as is, LOW is
rejected, and only MEDIUM is worth a model call.

**Say which question failed.** "QA failed" conflates three different
questions answered by different machinery at different cost. A deck can be
structurally perfect and semantically wrong - the
photosynthesis-and-respiration diagram was - or pass both and fail visually,
like the lamp starter schematic on a Light slide. Reporting one verdict hides
which happened, so the layers are reported separately and `skipped` is kept
distinct from `pass`.

The contract fixtures live in test/class_08/science/visual_contracts.json so
the benchmark harness and these tests cannot drift apart.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import gate, relevance as R, vision as V

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "test" / "class_08" / "science"
     / "visual_contracts.json").read_text(encoding="utf-8")
)["contracts"]

ON = {"PPT_V2_VISION_VALIDATION": "true"}


def contract_of(name):
    f = FIXTURES[name]
    return V.build_contract(
        learning_objective=f["purpose"], objects=f["required_objects"],
        relationships=f["required_relationships"], avoid=f["avoid"],
        visual_kind=f["visual_kind"])


class ConfidencePolicyTests(SimpleTestCase):
    def test_the_three_bands(self):
        self.assertEqual(R.confidence(5.0), R.HIGH)
        self.assertEqual(R.confidence(1.5), R.MEDIUM)
        self.assertEqual(R.confidence(-1.0), R.LOW)

    def test_only_the_middle_band_is_worth_a_vision_call(self):
        self.assertFalse(R.needs_vision(5.0))     # already confident
        self.assertTrue(R.needs_vision(1.5))      # ambiguous
        self.assertFalse(R.needs_vision(-1.0))    # already rejected

    def test_the_bands_are_configurable(self):
        with patch.dict(os.environ, {"PPT_V2_RELEVANCE_CONFIDENT": "10"}):
            self.assertEqual(R.confidence(5.0), R.MEDIUM)
        with patch.dict(os.environ, {"PPT_V2_RELEVANCE_ACCEPT": "4"}):
            self.assertEqual(R.confidence(2.0), R.LOW)

    def test_a_nonsense_setting_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"PPT_V2_RELEVANCE_CONFIDENT": "banana"}):
            self.assertEqual(R.confidence(5.0), R.HIGH)

    def test_a_confident_bar_below_the_accept_bar_is_not_honoured(self):
        # That would mean "accept everything without looking", the opposite
        # of what the band is for.
        with patch.dict(os.environ, {"PPT_V2_RELEVANCE_ACCEPT": "3",
                                     "PPT_V2_RELEVANCE_CONFIDENT": "1"}):
            accept, confident = R.thresholds()
            self.assertGreaterEqual(confident, accept)

    def test_a_verdict_reports_its_own_band(self):
        spec = R.build_spec(topic="Photosynthesis",
                            required=["green leaf", "sunlight", "chlorophyll"],
                            class_name="Class 8")
        rich = R.score_candidate(
            "A green leaf in sunlight showing chlorophyll - diagram", spec)
        plain = R.score_candidate("Photosynthesis diagram", spec)
        self.assertEqual(rich.confidence, R.HIGH)
        self.assertFalse(rich.needs_vision)
        self.assertEqual(plain.confidence, R.MEDIUM)
        self.assertTrue(plain.needs_vision)


class PolicyInProducerTests(SimpleTestCase):
    """The policy, through the real producer."""

    @staticmethod
    def _spec():
        from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy
        return VisualSpec(
            strategy=VisualStrategy.SEARCHED_IMAGE,
            search_terms=["reflection plane mirror diagram"],
            icons=["incident ray", "plane mirror", "reflected ray"],
            relationships=["the incident ray strikes the mirror"],
            objective="Explain how a light ray reflects from a plane mirror")

    def _run(self, score, vision_reply=None):
        from ai_services.core.ppt_v2.producers import ProducerSet
        calls = {"n": 0}

        def search(term, title, ctx=None, accept=None):
            return {"imageUrl": "https://x.test/a.png",
                    "relevanceScore": score,
                    "relevanceConfidence": R.confidence(score)}

        def ask(image, prompt):
            calls["n"] += 1
            return json.dumps(vision_reply or {})

        producers = ProducerSet(search_fn=search, vision_ask=ask,
                                vision_model="test-vision")
        with patch.dict(os.environ, ON):
            out = producers.search(self._spec())
        return out, calls["n"]

    def test_a_high_confidence_image_is_accepted_without_a_vision_call(self):
        out, calls = self._run(5.0)
        self.assertTrue(out["ok"])
        self.assertEqual(calls, 0)

    def test_a_medium_confidence_image_is_sent_for_a_look(self):
        out, calls = self._run(1.5, {
            "visible_objects": ["incident ray", "plane mirror", "reflected ray"],
            "shown_relationships": ["strikes the mirror"], "conflicts": []})
        self.assertTrue(out["ok"])
        self.assertEqual(calls, 1)

    def test_a_medium_confidence_image_that_fails_the_look_is_dropped(self):
        out, calls = self._run(1.5, {"visible_objects": ["lamp"],
                                     "shown_relationships": [], "conflicts": []})
        self.assertFalse(out["ok"])
        self.assertIn("vision rejected", out["error"])
        self.assertEqual(calls, 2)       # bounded, not endless

    def test_a_chapter_with_class_level_risks_is_looked_at_anyway(self):
        """HIGH confidence means on topic, not at the right level.

        Phase 7's Friction figure had a good caption and a reputable host and
        carried mu_s*N: the deterministic layer cannot see that, so for a
        chapter with known class-level risks the image is checked whatever
        its score.
        """
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy
        calls = {"n": 0}

        def search(term, title, ctx=None, accept=None):
            return {"imageUrl": "https://x.test/a.png", "relevanceScore": 9.0,
                    "relevanceConfidence": "high"}

        def ask(image, prompt):
            calls["n"] += 1
            return json.dumps({"visible_concepts": ["friction"],
                               "advanced_terms": [], "equations": [],
                               "advanced_notation": [], "class_level_concerns": [],
                               "required_objects_seen": ["block", "surface"],
                               "required_relationships_seen": ["slides"],
                               "conflicts": []})

        spec = VisualSpec(strategy=VisualStrategy.SEARCHED_IMAGE,
                          search_terms=["friction diagram"],
                          icons=["block", "surface"],
                          relationships=["the block slides over the surface"],
                          objective="Explain friction as a force opposing motion")
        producers = ProducerSet(search_fn=search, vision_ask=ask,
                                class_name="Class 8", subject="Science",
                                chapter="Friction")
        with patch.dict(os.environ, ON):
            out = producers.search(spec)
        self.assertTrue(out["ok"])
        self.assertEqual(calls["n"], 1)

    def test_the_advanced_friction_figure_is_rejected_at_high_confidence(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy

        def search(term, title, ctx=None, accept=None):
            return {"imageUrl": "https://x.test/a.png", "relevanceScore": 9.0,
                    "relevanceConfidence": "high"}

        def ask(image, prompt):
            return json.dumps({"visible_concepts": ["friction"],
                               "advanced_terms": ["mu_s N", "mu_k N"],
                               "equations": ["f_s,max = mu_s N"],
                               "advanced_notation": ["vector arrows"],
                               "class_level_concerns": ["static and kinetic friction graph"],
                               "required_objects_seen": ["block", "surface"],
                               "required_relationships_seen": ["slides"],
                               "conflicts": []})

        spec = VisualSpec(strategy=VisualStrategy.SEARCHED_IMAGE,
                          search_terms=["friction diagram"],
                          icons=["block", "surface"],
                          objective="Explain friction as a force opposing motion")
        producers = ProducerSet(search_fn=search, vision_ask=ask,
                                class_name="Class 8", subject="Science",
                                chapter="Friction")
        with patch.dict(os.environ, ON):
            out = producers.search(spec)
        self.assertFalse(out["ok"])
        self.assertIn("above the class", out["error"])

    def test_a_chapter_without_class_rules_still_skips_at_high_confidence(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy
        calls = {"n": 0}

        def search(term, title, ctx=None, accept=None):
            return {"imageUrl": "https://x.test/a.png", "relevanceScore": 9.0,
                    "relevanceConfidence": "high"}

        def ask(image, prompt):
            calls["n"] += 1
            return "{}"

        producers = ProducerSet(search_fn=search, vision_ask=ask,
                                class_name="Class 99", subject="Art",
                                chapter="Still Life")
        with patch.dict(os.environ, ON):
            producers.search(VisualSpec(strategy=VisualStrategy.SEARCHED_IMAGE,
                                        search_terms=["still life"],
                                        icons=["bowl"], objective="Draw a bowl"))
        self.assertEqual(calls["n"], 0)

    def test_a_low_confidence_image_never_reaches_the_validator(self):
        # It was rejected by the free layer before anything was downloaded.
        from ai_services.core.ppt_v2.producers import ProducerSet
        calls = {"n": 0}

        def search(term, title, ctx=None, accept=None):
            return {}        # nothing cleared the deterministic bar

        def ask(image, prompt):
            calls["n"] += 1
            return "{}"

        producers = ProducerSet(search_fn=search, vision_ask=ask)
        with patch.dict(os.environ, ON):
            out = producers.search(self._spec())
        self.assertFalse(out["ok"])
        self.assertEqual(calls["n"], 0)

    def test_a_search_with_no_score_still_gets_looked_at(self):
        # An older search function that does not report a score must not
        # silently skip validation.
        out, calls = self._run(None, {
            "visible_objects": ["incident ray", "plane mirror", "reflected ray"],
            "shown_relationships": ["strikes"], "conflicts": []})
        self.assertTrue(out["ok"])
        self.assertEqual(calls, 1)

    def test_a_native_shape_composition_is_never_validated(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec
        calls = {"n": 0}

        def ask(image, prompt):
            calls["n"] += 1
            return "{}"

        producers = ProducerSet(vision_ask=ask)
        with patch.dict(os.environ, ON):
            out = producers.icons(VisualSpec(icons=["sun", "water"]))
        self.assertTrue(out["ok"])
        self.assertEqual(calls["n"], 0)

    def test_a_programmatic_diagram_is_never_validated(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec
        calls = {"n": 0}

        def ask(image, prompt):
            calls["n"] += 1
            return "{}"

        producers = ProducerSet(vision_ask=ask)
        with patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": True,
                                 "image_base64": "data:image/png;base64,AAA"}):
            with patch.dict(os.environ, ON):
                out = producers.diagram(VisualSpec(diagram_spec="a lever"))
        self.assertTrue(out["ok"])
        self.assertEqual(calls["n"], 0)


class QaLayerTests(SimpleTestCase):
    def _deck(self, **kw):
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.visuals import VisualCapabilities
        plan = kw.pop("plan", [{"type": "concept", "title": "What is Light?",
                                "learningObjective": "Describe how light travels",
                                "bullets": ["Light travels in straight lines."],
                                "visual": {"description": "a ray of light",
                                           "objects": ["light ray"]}}])
        deck = compose_deck(plan, DeckContext(topic="Light", subject="Science",
                                              class_name="Class 8"),
                            caps=VisualCapabilities(ai_images=False,
                                                    image_search=True,
                                                    diagrams=True, cache=False))
        # Stand in for asset resolution. The gate is evaluated after assets
        # are resolved in the real service, and it correctly warns about a
        # slide that wanted a picture and never got one.
        if kw.pop("resolved", True):
            for slide in deck.slides:
                if slide.visual.strategy is not None:
                    slide.visual.resolved_strategy = slide.visual.strategy
                    slide.visual.url = "https://x.test/a.png"
        return deck

    def test_the_three_layers_are_reported_separately(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides))
        payload = result.to_dict()
        self.assertEqual(set(payload) & {"structural", "semantic", "visual"},
                         {"structural", "semantic", "visual"})

    def test_a_clean_deck_passes_structurally_and_semantically(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides))
        self.assertEqual(result.structural.status, gate.PASS)
        self.assertEqual(result.semantic.status, gate.PASS)

    def test_an_uninspected_visual_is_skipped_not_passed(self):
        # A pass would claim a check that never happened.
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides))
        self.assertEqual(result.visual.status, gate.SKIPPED)

    def test_an_inspected_and_accepted_visual_passes(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        verdict = V.VisionVerdict(passed=True, available=True, reason="ok")
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides),
                               vision_verdicts=[verdict])
        self.assertEqual(result.visual.status, gate.PASS)

    def test_a_rejected_visual_warns_but_does_not_block_the_deck(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        verdict = V.VisionVerdict(passed=False, available=True, reason="missing mirror")
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides),
                               vision_verdicts=[verdict])
        self.assertEqual(result.visual.status, gate.WARN)
        self.assertIn(gate.VISION_VALIDATION_ERROR, result.codes)
        self.assertTrue(result.passed)      # plainer deck, still usable

    def test_terminology_above_the_class_fails_the_semantic_layer(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides),
                               advanced_terms=["calvin cycle"])
        self.assertEqual(result.semantic.status, gate.FAIL)
        self.assertIn(gate.CONTENT_SCOPE_ERROR, result.codes)

    def test_only_a_structural_failure_blocks(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._deck()
        deck.slides[0].layout.name = "invented_later"
        result = gate.evaluate(deck.slides, qa.check_deck(deck.slides))
        self.assertEqual(result.structural.status, gate.FAIL)
        self.assertFalse(result.passed)
        self.assertIn(gate.LAYOUT_ERROR, result.codes)

    def test_the_existing_api_key_is_still_present(self):
        # Backward compatibility: the new block is added beside the old one.
        from ai_services.core.ppt_v2 import qa
        from ai_services.core.ppt_v2.service import V2Result
        deck = self._deck()
        issues = qa.check_deck(deck.slides)
        result = V2Result(title="T", slides=deck.slides, issues=issues,
                          gate=gate.evaluate(deck.slides, issues))
        payload = result.to_payload()["generation"]
        self.assertIn("structuralIssues", payload)
        self.assertIn("qa", payload)


class FailureTaxonomyTests(SimpleTestCase):
    def test_structural_issue_codes_map_to_the_taxonomy(self):
        self.assertEqual(gate.classify_issue("text_overflow"), gate.TEXT_OVERFLOW)
        self.assertEqual(gate.classify_issue("overlap"), gate.LAYOUT_ERROR)
        self.assertEqual(gate.classify_issue("unrenderable_layout"), gate.LAYOUT_ERROR)

    def test_service_reasons_map_to_the_taxonomy(self):
        self.assertEqual(gate.classify_reason("llm_error"), gate.PROVIDER_ERROR)
        self.assertEqual(gate.classify_reason("malformed_json"), gate.CONTENT_ERROR)
        self.assertEqual(gate.classify_reason("slide_count"), gate.CONTENT_ERROR)

    def test_an_unknown_code_still_gets_classified(self):
        self.assertTrue(gate.classify_issue("something_new"))
        self.assertTrue(gate.classify_reason("something_new"))


class ContractFixtureTests(SimpleTestCase):
    """Every fixture must accept its good example and reject its bad one."""

    def test_all_four_contracts_are_checkable(self):
        self.assertEqual(sorted(FIXTURES), ["force_and_pressure", "photosynthesis",
                                            "reflection", "sound_production"])
        for name in FIXTURES:
            self.assertTrue(contract_of(name).is_meaningful, name)

    def test_each_contract_accepts_its_good_visual(self):
        for name, f in FIXTURES.items():
            verdict = V.interpret(f["accept_example"]["vision"], contract_of(name))
            self.assertTrue(verdict.passed, "%s: %s" % (name, verdict.reason))

    def test_each_contract_rejects_its_bad_visual(self):
        for name, f in FIXTURES.items():
            verdict = V.interpret(f["reject_example"]["vision"], contract_of(name))
            self.assertFalse(verdict.passed, "%s should have been rejected" % name)

    def test_the_deterministic_layer_catches_what_it_can_for_free(self):
        # Vision is the second line of defence. Where a caption or a host
        # already gives it away, no call should be needed.
        cases = {
            "photosynthesis": R.build_spec(topic="Photosynthesis",
                                           required=["green leaf"],
                                           class_name="Class 8"),
            "force_and_pressure": R.build_spec(topic="Force and Pressure",
                                               required=["applied force"],
                                               class_name="Class 8"),
        }
        for name, spec in cases.items():
            bad = FIXTURES[name]["reject_example"]
            verdict = R.score_candidate(bad["caption"], spec, bad["url"])
            self.assertFalse(verdict.accepted,
                             "%s: %s scored %s" % (name, bad["url"], verdict.score))

    def test_a_reflection_slide_rejects_a_lamp_and_accepts_a_ray_diagram(self):
        # The single most important case in the visual-contract work.
        c = contract_of("reflection")
        lamp = V.interpret({"visible_objects": ["lamp", "bulb"],
                            "shown_relationships": [], "conflicts": []}, c)
        rays = V.interpret({"visible_objects": ["incident ray", "plane mirror",
                                                "reflected ray", "normal"],
                            "shown_relationships": ["the incident ray strikes the mirror"],
                            "conflicts": []}, c)
        self.assertFalse(lamp.passed)
        self.assertTrue(rays.passed, rays.reason)
