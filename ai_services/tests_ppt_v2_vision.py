"""Teaching relevance: is this picture actually usable for THIS slide?

Deterministic scoring (tests_ppt_v2_relevance) reads captions and hosts, and
it is free. It catches the picture about the wrong thing. It cannot catch the
picture about the right thing that still teaches nothing — which the real
four-topic run produced twice over:

  * a "What is Light?" slide was given a fluorescent lamp starter schematic.
    Caption on topic, host reputable, figure useless for the objective;
  * a photosynthesis slide was given a photosynthesis-and-respiration diagram.

The first needs someone to look at the image. So there is one narrow vision
check on the single candidate the slide chose — never per search result.

Everything here runs offline: the vision provider is a stub. The two
regression cases at the bottom are the point of the whole phase.
"""
import os
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import vision as V

ON = {"PPT_V2_VISION_VALIDATION": "true"}


def contract(**kw):
    kw.setdefault("learning_objective",
                  "Explain how a light ray reflects from a plane mirror")
    kw.setdefault("objects", ["incident ray", "plane mirror", "reflected ray"])
    kw.setdefault("relationships", ["the incident ray strikes the mirror"])
    kw.setdefault("avoid", ["refraction"])
    return V.build_contract(**kw)


def answering(payload):
    """A vision provider that returns one canned JSON reply."""
    import json
    return lambda image, prompt: json.dumps(payload)


class ContractTests(SimpleTestCase):
    def test_it_is_built_from_fields_the_planner_already_fills(self):
        c = contract()
        self.assertIn("plane mirror", c.required_objects)
        self.assertIn("the incident ray strikes the mirror", c.relationships)
        self.assertIn("refraction", c.avoid)
        self.assertTrue(c.purpose)

    def test_must_show_is_merged_ahead_of_objects_and_deduplicated(self):
        # must_show leads: it is "the detail the picture is useless without",
        # so when the list is capped it must survive ahead of a generic
        # object. Case-insensitive dedupe keeps "Mirror" and "mirror" as one.
        c = V.build_contract(objects=["mirror", "ray"], must_show=["Mirror", "normal"])
        self.assertEqual([o.lower() for o in c.required_objects],
                         ["mirror", "normal", "ray"])

    def test_the_contract_asks_for_a_core_not_an_inventory(self):
        # A live Friction slide listed five objects spanning four different
        # examples; no single picture holds them, three came back missing,
        # and a good textbook figure was discarded.
        c = V.build_contract(objects=["block on rough floor", "block on smooth floor",
                                      "wheel on road", "boat in water",
                                      "block slides when pulled"])
        self.assertEqual(len(c.required_objects), 4)

    def test_a_four_object_contract_tolerates_two_missing(self):
        # The half-threshold has to leave room for a real figure that shows
        # the subject without every listed extra.
        c = V.build_contract(objects=["block", "surface", "arrow", "hand"],
                             class_name="Class 8", subject="Science",
                             chapter="Friction")
        verdict = V.interpret({"adv": [], "seen": ["block", "surface"],
                               "rel": [], "conf": []}, c)
        self.assertTrue(verdict.passed, verdict.reason)

    def test_a_contract_with_nothing_checkable_is_not_worth_a_call(self):
        # A vague question returns a vague yes, so the call is skipped.
        self.assertFalse(V.build_contract(learning_objective="Understand light").is_meaningful)
        self.assertTrue(V.build_contract(objects=["mirror"]).is_meaningful)

    def test_the_prompt_asks_only_what_is_visible(self):
        prompt = V.build_prompt(contract())
        self.assertIn("plane mirror", prompt)
        self.assertIn("the incident ray strikes the mirror", prompt)
        self.assertIn("refraction", prompt)
        # Explicitly not an aesthetic judgement, and explicitly not a verdict:
        # the model reports observations and the application decides.
        # Whitespace-normalised because the prompt is line-wrapped.
        import re
        flat = re.sub(r"\s+", " ", prompt)
        self.assertIn("do not judge how attractive", flat)
        self.assertIn("not deciding", flat)
        self.assertIn("No verdict", flat)


class InterpretationTests(SimpleTestCase):
    def test_a_picture_showing_what_is_needed_passes(self):
        verdict = V.interpret({"visible_objects": ["incident ray", "plane mirror",
                                                   "reflected ray"],
                               "shown_relationships": ["ray strikes the mirror"],
                               "conflicts": []}, contract())
        self.assertTrue(verdict.passed)

    def test_a_picture_missing_most_of_the_objects_fails(self):
        verdict = V.interpret({"visible_objects": ["lamp"],
                               "shown_relationships": [], "conflicts": []},
                              contract())
        self.assertFalse(verdict.passed)
        self.assertIn("plane mirror", verdict.missing)

    def test_a_conflicting_concept_fails_it(self):
        verdict = V.interpret({"visible_objects": ["incident ray", "plane mirror",
                                                   "reflected ray"],
                               "shown_relationships": ["ray strikes the mirror"],
                               "conflicts": ["refraction"]}, contract())
        self.assertFalse(verdict.passed)

    def test_all_objects_present_but_no_relationship_is_accepted(self):
        # This was a rejection until a live run showed the cost: the planner
        # can ask for a relationship no picture can show ("each organelle
        # labelled with its function"), and two good plant-cell diagrams were
        # discarded for failing it. With everything the slide names visible,
        # that is enough; a missing relationship is now only decisive when
        # objects are thin too.
        verdict = V.interpret({"visible_objects": ["incident ray", "plane mirror",
                                                   "reflected ray"],
                               "shown_relationships": [], "conflicts": []},
                              contract())
        self.assertTrue(verdict.passed, verdict.reason)
        self.assertIn("no relationship was reported", verdict.reason)

    def test_the_decision_is_ours_not_the_models_usable_flag(self):
        # A model asked for a boolean hands one over cheerfully; what it is
        # good at is listing what it can see.
        verdict = V.interpret({"visible_objects": [], "shown_relationships": [],
                               "conflicts": ["refraction"], "usable": True},
                              contract())
        self.assertFalse(verdict.passed)

    def test_partial_object_coverage_is_tolerated(self):
        verdict = V.interpret({"visible_objects": ["plane mirror", "reflected ray"],
                               "shown_relationships": ["reflects off the mirror"],
                               "conflicts": []}, contract())
        self.assertTrue(verdict.passed)


class SafetyTests(SimpleTestCase):
    """No failure path may cost the deck."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(V.enabled())
            verdict = V.validate("https://x.test/a.png", contract(),
                                 ask=answering({"visible_objects": []}))
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)      # never looked => not a failure

    def test_no_provider_is_not_a_failure(self):
        with patch.dict(os.environ, ON):
            verdict = V.validate("https://x.test/a.png", contract(), ask=None)
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)

    def test_a_provider_exception_is_not_a_failure(self):
        def boom(image, prompt):
            raise RuntimeError("vision provider timed out")
        with patch.dict(os.environ, ON):
            verdict = V.validate("https://x.test/a.png", contract(), ask=boom)
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)

    def test_unparseable_json_is_not_a_failure(self):
        with patch.dict(os.environ, ON):
            verdict = V.validate("https://x.test/a.png", contract(),
                                 ask=lambda i, p: "I think the picture is nice!")
        self.assertFalse(verdict.available)

    def test_json_wrapped_in_prose_is_still_read(self):
        reply = 'Sure! {"visible_objects": ["plane mirror"], "shown_relationships": ["strikes"], "conflicts": []} hope that helps'
        with patch.dict(os.environ, ON):
            verdict = V.validate("https://x.test/a.png", contract(),
                                 ask=lambda i, p: reply)
        self.assertTrue(verdict.available)

    def test_a_fenced_reply_is_still_read(self):
        reply = '```json\n{"visible_objects": ["plane mirror"], "shown_relationships": ["x"], "conflicts": []}\n```'
        with patch.dict(os.environ, ON):
            self.assertTrue(V.validate("https://x.test/a.png", contract(),
                                       ask=lambda i, p: reply).available)

    def test_no_image_is_not_a_failure(self):
        with patch.dict(os.environ, ON):
            self.assertFalse(V.validate("", contract(),
                                        ask=answering({})).available)


class CostTests(SimpleTestCase):
    def test_a_cached_verdict_costs_no_call(self):
        calls = {"n": 0}

        def counting(image, prompt):
            calls["n"] += 1
            import json
            return json.dumps({"visible_objects": ["plane mirror", "reflected ray"],
                               "shown_relationships": ["strikes"], "conflicts": []})

        store = {}
        with patch.dict(os.environ, ON):
            first = V.validate("https://x.test/a.png", contract(), ask=counting,
                               cache_get=store.get, cache_put=store.__setitem__)
            second = V.validate("https://x.test/a.png", contract(), ask=counting,
                                cache_get=store.get, cache_put=store.__setitem__)
        self.assertEqual(calls["n"], 1)
        self.assertTrue(first.passed and second.passed)
        self.assertTrue(second.cached)

    def test_the_cache_key_is_the_picture_and_the_contract_together(self):
        a = V.cache_key("https://x.test/a.png", contract())
        self.assertEqual(a, V.cache_key("https://x.test/a.png", contract()))
        self.assertNotEqual(a, V.cache_key("https://x.test/b.png", contract()))
        self.assertNotEqual(a, V.cache_key("https://x.test/a.png",
                                           contract(objects=["lens"])))

    def test_at_most_two_validations_per_slide(self):
        self.assertEqual(V.MAX_VALIDATIONS_PER_SLIDE, 2)

    def test_usage_is_reported_for_telemetry(self):
        seen = {}
        with patch.dict(os.environ, ON):
            V.validate("https://x.test/a.png", contract(),
                       ask=answering({"visible_objects": ["plane mirror", "reflected ray"],
                                      "shown_relationships": ["strikes"], "conflicts": []}),
                       model="test-vision",
                       on_usage=lambda **kw: seen.update(kw))
        self.assertTrue(seen.get("ok"))
        self.assertEqual(seen.get("model"), "test-vision")
        self.assertIn("latency_ms", seen)

    def test_drawn_visuals_are_not_worth_looking_at(self):
        # A diagram and an icon composition are drawn from our own spec, so a
        # vision model can only confirm what we already know.
        self.assertNotIn("programmatic_diagram", V.VALIDATED_STRATEGIES)
        self.assertNotIn("icon_composition", V.VALIDATED_STRATEGIES)
        self.assertIn("searched_image", V.VALIDATED_STRATEGIES)


class ProducerIntegrationTests(SimpleTestCase):
    """The bounded check, through the real producer."""

    @staticmethod
    def _spec():
        from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy
        return VisualSpec(
            strategy=VisualStrategy.SEARCHED_IMAGE,
            search_terms=["reflection of light plane mirror diagram", "light"],
            icons=["incident ray", "plane mirror", "reflected ray"],
            relationships=["the incident ray strikes the mirror"],
            objective="Explain how a light ray reflects from a plane mirror",
            description="a light ray reflecting off a plane mirror")

    def _producers(self, ask, urls):
        from ai_services.core.ppt_v2.producers import ProducerSet
        pool = iter(urls)

        def search(term, title, ctx=None, accept=None):
            return {"imageUrl": next(pool, "")}

        return ProducerSet(search_fn=search, vision_ask=ask,
                           vision_model="test-vision"), search

    def test_a_rejected_picture_is_replaced_by_the_next_candidate(self):
        import json
        seen = []

        def ask(image, prompt):
            seen.append(image)
            if "lamp" in image:
                return json.dumps({"visible_objects": ["lamp"],
                                   "shown_relationships": [], "conflicts": []})
            return json.dumps({"visible_objects": ["incident ray", "plane mirror",
                                                   "reflected ray"],
                               "shown_relationships": ["strikes the mirror"],
                               "conflicts": []})

        producers, _ = self._producers(ask, ["https://x.test/lamp.png",
                                             "https://x.test/mirror.png"])
        with patch.dict(os.environ, ON):
            out = producers.search(self._spec())
        self.assertTrue(out["ok"])
        self.assertEqual(out["url"], "https://x.test/mirror.png")
        self.assertEqual(len(seen), 2)

    def test_it_gives_up_after_two_looks_and_lets_the_chain_continue(self):
        import json
        calls = {"n": 0}

        def ask(image, prompt):
            calls["n"] += 1
            return json.dumps({"visible_objects": ["lamp"],
                               "shown_relationships": [], "conflicts": []})

        producers, _ = self._producers(ask, ["https://x.test/a.png",
                                             "https://x.test/b.png",
                                             "https://x.test/c.png"])
        with patch.dict(os.environ, ON):
            out = producers.search(self._spec())
        self.assertFalse(out["ok"])
        self.assertEqual(calls["n"], 2)          # never a third
        self.assertIn("vision rejected", out["error"])

    def test_with_validation_off_the_deterministic_choice_stands(self):
        calls = {"n": 0}

        def ask(image, prompt):
            calls["n"] += 1
            return "{}"

        producers, _ = self._producers(ask, ["https://x.test/lamp.png"])
        with patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "false"}):
            out = producers.search(self._spec())
        self.assertTrue(out["ok"])
        self.assertEqual(calls["n"], 0)


class LightReflectionRegressionTests(SimpleTestCase):
    """The lamp problem: on topic, reputable host, useless for the lesson.

    The most important case in this phase. A reflection slide must not accept
    a generic lamp photograph as its teaching visual, and must accept a real
    reflection diagram.
    """

    CONTRACT = dict(
        learning_objective="Explain how a light ray reflects from a plane mirror",
        objects=["incident ray", "plane mirror", "reflected ray"],
        relationships=["the incident ray strikes the mirror and reflects"],
        avoid=["refraction"])

    def test_a_generic_lamp_photograph_is_rejected(self):
        verdict = V.interpret(
            {"visible_objects": ["lamp", "bulb", "table"],
             "shown_relationships": [], "conflicts": []},
            V.build_contract(**self.CONTRACT))
        self.assertFalse(verdict.passed)
        self.assertIn("plane mirror", verdict.missing)

    def test_a_lamp_starter_schematic_is_rejected(self):
        # The exact asset the live run chose.
        verdict = V.interpret(
            {"visible_objects": ["circuit", "starter", "fluorescent tube"],
             "shown_relationships": [], "conflicts": []},
            V.build_contract(**self.CONTRACT))
        self.assertFalse(verdict.passed)

    def test_a_real_reflection_diagram_is_accepted(self):
        verdict = V.interpret(
            {"visible_objects": ["incident ray", "plane mirror", "reflected ray",
                                 "normal"],
             "shown_relationships": ["the incident ray strikes the mirror and reflects"],
             "conflicts": []},
            V.build_contract(**self.CONTRACT))
        self.assertTrue(verdict.passed, verdict.reason)

    def test_a_refraction_diagram_is_rejected_for_a_reflection_slide(self):
        verdict = V.interpret(
            {"visible_objects": ["light ray", "glass block"],
             "shown_relationships": ["the ray bends entering the glass"],
             "conflicts": ["refraction"]},
            V.build_contract(**self.CONTRACT))
        self.assertFalse(verdict.passed)


class PhotosynthesisRegressionTests(SimpleTestCase):
    """Semantic correctness, not URL identity."""

    CONTRACT = dict(
        learning_objective="State what a plant needs in order to make its own food",
        objects=["green leaf", "sunlight", "carbon dioxide", "water"],
        relationships=["sunlight falls on the leaf"],
        avoid=["respiration", "mitochondria"])

    def test_a_photosynthesis_figure_is_accepted(self):
        verdict = V.interpret(
            {"visible_objects": ["green leaf", "sunlight", "carbon dioxide", "water"],
             "shown_relationships": ["sunlight falls on the leaf"],
             "conflicts": []},
            V.build_contract(**self.CONTRACT))
        self.assertTrue(verdict.passed, verdict.reason)

    def test_a_respiration_heavy_figure_is_rejected(self):
        verdict = V.interpret(
            {"visible_objects": ["mitochondrion", "glucose"],
             "shown_relationships": ["glucose is broken down"],
             "conflicts": ["respiration", "mitochondria"]},
            V.build_contract(**self.CONTRACT))
        self.assertFalse(verdict.passed)

    def test_the_deterministic_layer_rejects_it_first_and_for_free(self):
        # Vision is the second line of defence, not the first: this case is
        # already caught by caption scoring at no cost.
        from ai_services.core.ppt_v2 import relevance as R
        spec = R.build_spec(topic="Photosynthesis",
                            required=["green leaf", "sunlight"],
                            class_name="Class 8")
        self.assertFalse(R.score_candidate(
            "Photosynthesis and Respiration in Plants", spec).accepted)


class ClassAwareContractTests(SimpleTestCase):
    """The audience is part of the question, not an afterthought."""

    @staticmethod
    def _friction(class_name="Class 8"):
        return V.build_contract(
            learning_objective="Explain friction as a force opposing motion",
            objects=["block", "surface", "friction arrow"],
            relationships=["the block slides over the surface"],
            class_name=class_name, board="CBSE", subject="Science",
            chapter="Friction")

    def test_the_prompt_names_the_audience(self):
        prompt = V.build_prompt(self._friction())
        self.assertIn("Class 8", prompt)
        self.assertIn("Science", prompt)
        self.assertIn("Friction", prompt)

    def test_the_prompt_asks_for_observations_not_a_verdict(self):
        # Compact keys: the shared vision helper caps its output, and an
        # eight-field schema came back truncated mid-string from a real call.
        prompt = V.build_prompt(self._friction())
        for field in ('"adv"', '"seen"', '"rel"', '"conf"'):
            self.assertIn(field, prompt, field)
        self.assertIn("No verdict", prompt)
        self.assertIn("no extra keys", prompt)

    def test_the_long_field_names_are_still_understood(self):
        # A model that volunteers the verbose spellings must still work.
        verdict = V.interpret(
            {"advanced_terms": [], "required_objects_seen": ["block", "surface",
                                                             "friction arrow"],
             "required_relationships_seen": ["slides"], "conflicts": []},
            self._friction())
        self.assertTrue(verdict.passed, verdict.reason)

    def test_the_compact_field_names_are_understood(self):
        verdict = V.interpret(
            {"adv": [], "seen": ["block", "surface", "friction arrow"],
             "rel": ["slides"], "conf": []}, self._friction())
        self.assertTrue(verdict.passed, verdict.reason)

    def test_the_compact_advanced_list_still_rejects(self):
        verdict = V.interpret(
            {"adv": ["mu_s N", "static region"], "seen": ["block", "surface"],
             "rel": ["slides"], "conf": []}, self._friction())
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.decision, V.CLASS_LEVEL_VIOLATION)

    def test_the_audience_is_part_of_the_cache_key(self):
        # The same picture is fine for one class and wrong for another, so
        # the two verdicts must not collide in the cache.
        self.assertNotEqual(V.cache_key("u", self._friction("Class 8")),
                            V.cache_key("u", self._friction("Class 12")))


class DeterministicDecisionTests(SimpleTestCase):
    """The model observes; the application decides."""

    def _c(self, class_name="Class 8"):
        return ClassAwareContractTests._friction(class_name)

    GOOD = {"visible_concepts": ["friction", "rough surface"],
            "advanced_terms": [], "equations": [], "advanced_notation": [],
            "class_level_concerns": [],
            "required_objects_seen": ["block", "surface", "friction arrow"],
            "required_relationships_seen": ["the block slides over the surface"],
            "conflicts": []}

    def test_a_plain_class_8_friction_figure_passes(self):
        verdict = V.interpret(self.GOOD, self._c())
        self.assertTrue(verdict.passed, verdict.reason)
        self.assertEqual(verdict.decision, V.PASSED)

    def test_the_advanced_friction_figure_is_a_class_level_violation(self):
        # The Phase 7 defect: genuinely about friction, useless to Class 8.
        bad = dict(self.GOOD, advanced_terms=["mu_s N", "mu_k N"],
                   equations=["f_s,max = mu_s N"],
                   advanced_notation=["vector arrows over F"],
                   class_level_concerns=["static and kinetic friction graph"])
        verdict = V.interpret(bad, self._c())
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.decision, V.CLASS_LEVEL_VIOLATION)

    def test_the_same_figure_is_acceptable_for_a_senior_class(self):
        # Proof the rule is about the audience rather than the picture.
        bad = dict(self.GOOD, advanced_terms=[], equations=[],
                   advanced_notation=[], class_level_concerns=[],
                   visible_concepts=["coefficient of friction"])
        self.assertTrue(V.interpret(bad, self._c("Class 12")).passed)

    def test_missing_objects_give_their_own_code(self):
        verdict = V.interpret(dict(self.GOOD, required_objects_seen=["surface"]),
                              self._c())
        self.assertEqual(verdict.decision, V.REQUIRED_CONCEPT_MISSING)

    def test_a_missing_relationship_with_objects_missing_too_is_rejected(self):
        # Two of three objects: past the object threshold, so the missing
        # relationship is what decides.
        verdict = V.interpret(dict(self.GOOD, required_relationships_seen=[],
                                   required_objects_seen=["block", "surface"]),
                              self._c())
        self.assertEqual(verdict.decision, V.RELATIONSHIP_MISSING)

    def test_a_missing_relationship_alone_does_not_discard_a_good_figure(self):
        # A live run rejected two perfectly good plant-cell diagrams because
        # the planner asked for "each organelle labeled with its function" -
        # a requirement about the lesson, not something a picture can show.
        verdict = V.interpret(dict(self.GOOD, required_relationships_seen=[]),
                              self._c())
        self.assertTrue(verdict.passed, verdict.reason)
        self.assertEqual(verdict.decision, V.PASSED)

    def test_the_planner_is_told_relationships_must_be_depictable(self):
        from ai_services.core.ppt_v2.planner import build_planner_prompts
        system, _ = build_planner_prompts("Cell Structure", 2)
        self.assertIn("VISIBLE IN A PICTURE", system)

    def test_a_competing_concept_gives_its_own_code(self):
        verdict = V.interpret(dict(self.GOOD, conflicts=["gravity"]), self._c())
        self.assertEqual(verdict.decision, V.COMPETING_CONCEPT)

    def test_the_models_own_verdict_is_ignored(self):
        bad = dict(self.GOOD, advanced_terms=["mu_s N"], usable=True,
                   pass_=True, score=10)
        self.assertFalse(V.interpret(bad, self._c()).passed)

    def test_the_topic_word_is_not_mistaken_for_an_advanced_concept(self):
        # "friction" must not match the rule "coefficient of friction" - it
        # did, and rejected every ordinary friction figure.
        self.assertTrue(V.interpret(self.GOOD, self._c()).passed)

    def test_the_observations_are_kept_for_debugging(self):
        verdict = V.interpret(self.GOOD, self._c())
        self.assertIn("objects_seen", verdict.observations)
        self.assertIn("advanced", verdict.observations)


class ClassRuleTests(SimpleTestCase):
    def test_rules_merge_class_subject_and_chapter(self):
        from ai_services.core.ppt_v2 import class_rules
        merged = class_rules.lookup("Class 8", "Science", "Friction")
        # chapter-specific
        self.assertIn("static region", merged["forbidden_visual_terms"])
        # subject-level
        self.assertIn("calvin cycle", merged["forbidden_visual_terms"])
        # class-level notation
        self.assertIn("μ", merged["forbidden_notation"])

    def test_the_table_is_not_specific_to_friction(self):
        from ai_services.core.ppt_v2 import class_rules
        for chapter, expected in (("Light", "lens formula"),
                                  ("Sound", "wave equation"),
                                  ("Metals and Non-metals", "electrochemical series")):
            self.assertIn(expected,
                          class_rules.lookup("Class 8", "Science", chapter)["forbidden_visual_terms"],
                          chapter)

    def test_an_unknown_class_has_no_rules(self):
        from ai_services.core.ppt_v2 import class_rules
        self.assertEqual(class_rules.all_forbidden("Class 99", "Science", "X"), [])

    def test_violations_match_the_models_paraphrase(self):
        from ai_services.core.ppt_v2 import class_rules
        found = class_rules.violations(
            ["coefficients of friction shown as mu_s and mu_k"],
            "Class 8", "Science", "Friction")
        self.assertIn("coefficient of friction", found)

    def test_a_shorter_observation_does_not_match_a_longer_rule(self):
        from ai_services.core.ppt_v2 import class_rules
        self.assertEqual(
            class_rules.violations(["friction"], "Class 8", "Science", "Friction"),
            [])


class ProviderConfigTests(SimpleTestCase):
    """A provider that is misconfigured must cost the picture, never the deck.

    Measured on 2026-10-03: both Groq Llama 4 vision IDs return 404 on this
    account, so every validation spent four failed round-trips before the
    Gemini fallback did the work. Groq is now skipped unless configured.
    """

    def test_groq_vision_is_skipped_unless_configured(self):
        import re
        from pathlib import Path
        source = (Path(__file__).resolve().parent / "views" / "bridge.py").read_text(
            encoding="utf-8")
        # The default must be empty rather than a deprecated model id.
        self.assertRegex(source, r'os\.getenv\("GROQ_VISION_MODELS",\s*""\)')
        # ...and the deprecated ids must not be the fallback either.
        self.assertNotIn("llama-4-scout-17b-16e-instruct,", source)

    def test_the_model_list_is_still_configurable(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parent / "views" / "bridge.py").read_text(
            encoding="utf-8")
        self.assertIn("GROQ_VISION_MODELS", source)

    def test_a_provider_404_is_not_a_semantic_rejection(self):
        # The distinction that matters: nobody looked, so nothing was
        # rejected. Treating an outage as a rejection would throw away a
        # perfectly good picture.
        def four_oh_four(image, prompt):
            raise RuntimeError("Error code: 404 - model does not exist")

        with patch.dict(os.environ, ON):
            verdict = V.validate("https://x.test/a.png", contract(),
                                 ask=four_oh_four)
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.decision, V.PASSED)

    def test_a_provider_outage_leaves_the_visual_layer_skipped(self):
        from ai_services.core.ppt_v2 import gate
        verdict = V.unavailable("provider error")
        result = gate.evaluate([], [], vision_verdicts=[verdict])
        self.assertEqual(result.visual.status, gate.SKIPPED)


class TimeoutTests(SimpleTestCase):
    """A slow provider must cost the picture, not the request.

    A live run measured one Gemini vision call at 156s - more than twice the
    whole deck's normal generation time - and the provider's own timeout did
    not bound it.
    """

    def test_a_slow_provider_is_abandoned(self):
        import time as _time

        def glacial(image, prompt):
            _time.sleep(5)
            return "{}"

        with patch.dict(os.environ, dict(ON, PPT_V2_VISION_TIMEOUT="5")):
            # Floor is 5s, so this races; assert the mechanism not the clock.
            started = _time.perf_counter()
            verdict = V.validate("https://x.test/a.png", contract(), ask=glacial)
            elapsed = _time.perf_counter() - started
        self.assertLess(elapsed, 12)
        if not verdict.available:
            self.assertIn("timed out", verdict.reason)

    def test_a_timeout_is_not_a_semantic_rejection(self):
        def glacial(image, prompt):
            import time as _t
            _t.sleep(30)
            return "{}"

        with patch.dict(os.environ, dict(ON, PPT_V2_VISION_TIMEOUT="5")):
            verdict = V.validate("https://x.test/a.png", contract(), ask=glacial)
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)     # nobody looked => nothing rejected
        self.assertEqual(verdict.decision, V.PASSED)

    def test_the_budget_is_configurable_with_a_floor(self):
        with patch.dict(os.environ, {"PPT_V2_VISION_TIMEOUT": "40"}):
            self.assertEqual(V.timeout_s(), 40.0)
        with patch.dict(os.environ, {"PPT_V2_VISION_TIMEOUT": "0.1"}):
            self.assertGreaterEqual(V.timeout_s(), 5.0)
        with patch.dict(os.environ, {"PPT_V2_VISION_TIMEOUT": "junk"}):
            self.assertEqual(V.timeout_s(), V.TIMEOUT_S)

    def test_a_fast_provider_is_unaffected(self):
        import json as _json
        with patch.dict(os.environ, ON):
            verdict = V.validate(
                "https://x.test/a.png", contract(),
                ask=lambda i, p: _json.dumps({"adv": [], "seen": ["plane mirror",
                                                                  "reflected ray"],
                                              "rel": ["strikes"], "conf": []}))
        self.assertTrue(verdict.available)
        self.assertTrue(verdict.passed)


class TaughtBySlideTests(SimpleTestCase):
    """The model's own "advanced" list is an opinion. On a live Class 10
    Respiration slide it rejected a figure for showing CO2, O2 and H2O - the
    very things the slide teaches."""

    def _contract(self, **kw):
        return V.build_contract(class_name="Class 10", subject="Science",
                                chapter="Life Processes", **kw)

    def test_what_the_slide_teaches_is_not_above_the_slide(self):
        c = self._contract(learning_objective="State that respiration releases CO2 and H2O",
                           objects=["leaf", "arrows"])
        verdict = V.interpret({"adv": ["CO₂ symbol", "H₂O molecule"],
                               "seen": ["leaf", "arrows"], "rel": [], "conf": []}, c)
        self.assertTrue(verdict.passed, verdict.reason)

    def test_an_opinion_about_something_the_slide_does_not_teach_still_rejects(self):
        c = self._contract(learning_objective="Name the parts of the lungs",
                           objects=["lungs", "trachea"])
        verdict = V.interpret({"adv": ["Krebs cycle"], "seen": ["lungs", "trachea"],
                               "rel": [], "conf": []}, c)
        self.assertFalse(verdict.passed)

    def test_a_class_rule_is_never_overridden_by_the_slide(self):
        c = V.build_contract(class_name="Class 8", subject="Science", chapter="Friction",
                             learning_objective="Use mu_k to compare surfaces",
                             objects=["block", "surface"])
        verdict = V.interpret({"adv": ["mu_k"], "seen": ["block", "surface"],
                               "rel": [], "conf": []}, c)
        self.assertFalse(verdict.passed)
