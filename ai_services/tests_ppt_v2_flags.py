"""V2 must never turn itself on, and the producers must never loop.

The flag tests exist because the failure is silent and expensive: V2 calls a
paid image generator, so a default that leans the wrong way bills every deck
in production before anyone notices the slides look different.

The producer tests cover the one behaviour that turns a slow deck into a hung
one - repeating a call that has already told us it will never work.
"""
import os
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.ppt_v2.flags import V1, V2, ai_image_budget, ppt_version
from ai_services.core.ppt_v2.producers import ProducerSet, cache_key
from ai_services.core.ppt_v2.schema import VisualSpec

ON = {"PPT_V2_ENABLED": "true"}


def env(**kw):
    return patch.dict(os.environ, kw, clear=False)


class FlagTests(SimpleTestCase):
    def test_v1_is_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ppt_version(), V1)

    def test_the_master_switch_beats_everything_else(self):
        with env(PPT_V2_ENABLED="false", PPT_V2_ROLLOUT_PERCENT="100",
                 PPT_V2_ALLOW_OVERRIDE="true"):
            self.assertEqual(ppt_version(requested="v2"), V1)

    def test_enabling_it_switches_to_v2(self):
        with env(**ON):
            self.assertEqual(ppt_version(), V2)

    def test_a_client_cannot_opt_in_unless_the_override_is_allowed(self):
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="0",
                 PPT_V2_ALLOW_OVERRIDE="false"):
            self.assertEqual(ppt_version(requested="v2"), V1)
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="0",
                 PPT_V2_ALLOW_OVERRIDE="true"):
            self.assertEqual(ppt_version(requested="v2"), V2)

    def test_a_client_can_always_be_ignored_if_it_asks_for_nonsense(self):
        with env(PPT_V2_ENABLED="true", PPT_V2_ALLOW_OVERRIDE="true"):
            self.assertEqual(ppt_version(requested="v99"), V2)

    def test_zero_percent_rollout_keeps_everyone_on_v1(self):
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="0"):
            self.assertTrue(all(ppt_version(deck_key="deck-%d" % i) == V1
                                for i in range(50)))

    def test_the_rollout_bucket_is_stable_for_a_deck(self):
        # A retry of the same deck must not switch versions halfway.
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="50"):
            got = {ppt_version(deck_key="Science/Photosynthesis") for _ in range(20)}
            self.assertEqual(len(got), 1)

    def test_a_partial_rollout_actually_splits_traffic(self):
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="50"):
            versions = [ppt_version(deck_key="deck-%d" % i) for i in range(200)]
        self.assertGreater(versions.count(V2), 50)
        self.assertGreater(versions.count(V1), 50)

    def test_a_junk_rollout_value_does_not_crash(self):
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="banana"):
            self.assertEqual(ppt_version(), V2)
        with env(PPT_V2_ENABLED="true", PPT_V2_ROLLOUT_PERCENT="-40"):
            self.assertEqual(ppt_version(), V1)

    def test_the_image_budget_is_clamped(self):
        for value, expected in (("0", 0), ("5", 5), ("999", 20), ("-2", 0),
                                ("junk", 3)):
            with env(PPT_V2_AI_IMAGE_BUDGET=value):
                self.assertEqual(ai_image_budget(), expected, value)


class ProducerTests(SimpleTestCase):
    def test_a_fatal_generation_error_stops_the_whole_deck_asking(self):
        # "Not configured" fails identically on slide 10 as on slide 1.
        calls = {"n": 0}

        def never_configured(prompt):
            calls["n"] += 1
            return {"ok": False, "error": "not configured", "fatal": True}

        producers = ProducerSet()
        with patch("ai_services.core.image_generation.generate_note_image",
                   never_configured):
            for _ in range(5):
                producers.ai_image(VisualSpec(image_prompt="a leaf"))
        self.assertEqual(calls["n"], 1)

    def test_a_transient_error_does_not_disable_later_slides(self):
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            return {"ok": False, "error": "timeout", "fatal": False}

        producers = ProducerSet()
        with patch("ai_services.core.image_generation.generate_note_image", flaky):
            for _ in range(3):
                producers.ai_image(VisualSpec(image_prompt="a leaf"))
        self.assertEqual(calls["n"], 3)

    def test_a_successful_generation_is_mapped_to_the_producer_contract(self):
        with patch("ai_services.core.image_generation.generate_note_image",
                   return_value={"ok": True, "url": "https://x.test/a.png",
                                 "provider": "huggingface", "model": "flux"}):
            out = ProducerSet().ai_image(VisualSpec(image_prompt="a leaf"))
        # labelPolicy joined the contract with the semantic label rules: a
        # generated image with no planned labels carries none.
        self.assertEqual(out, {"ok": True, "url": "https://x.test/a.png",
                               "source": "huggingface", "model": "flux",
                               "labelPolicy": "none"})

    def test_generation_without_a_prompt_is_refused_before_the_call(self):
        self.assertFalse(ProducerSet().ai_image(VisualSpec())["ok"])

    def test_the_diagram_data_uri_is_returned_as_base64(self):
        with patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": True, "image_base64": "data:image/png;base64,AAA"}):
            out = ProducerSet(subject="Physics").diagram(VisualSpec(diagram_spec="v-t graph"))
        self.assertTrue(out["ok"])
        self.assertTrue(out["base64"].startswith("data:image/png"))

    def test_a_failed_diagram_reports_why(self):
        with patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": False, "error": "sandbox timed out"}):
            out = ProducerSet().diagram(VisualSpec(diagram_spec="x"))
        self.assertEqual(out, {"ok": False, "error": "sandbox timed out"})

    def test_search_walks_its_queries_until_one_lands(self):
        seen = []

        def search(term, title, ctx=None):
            seen.append(term)
            return {"imageUrl": "https://x.test/i.png"} if term == "second" else {}

        out = ProducerSet(search_fn=search).search(
            VisualSpec(search_terms=["first", "second", "third"]))
        self.assertTrue(out["ok"])
        self.assertEqual(seen, ["first", "second"])

    def test_search_without_a_function_fails_cleanly(self):
        # core must not import views, so the search is injected or absent.
        self.assertFalse(ProducerSet().search(VisualSpec(search_terms=["x"]))["ok"])

    def test_a_produced_asset_is_written_to_the_cache(self):
        store = {}
        producers = ProducerSet(cache_put=store.__setitem__, cache_get=store.get)
        spec = VisualSpec(image_prompt="a leaf")
        with patch("ai_services.core.image_generation.generate_note_image",
                   return_value={"ok": True, "url": "https://x.test/a.png"}):
            producers.ai_image(spec)
        self.assertTrue(producers.cached(spec)["ok"])

    def test_a_broken_cache_never_breaks_the_slide(self):
        def explode(key):
            raise RuntimeError("redis is down")

        self.assertFalse(ProducerSet(cache_get=explode).cached(
            VisualSpec(image_prompt="a leaf"))["ok"])

    def test_the_cache_key_follows_the_prompt(self):
        a = cache_key(VisualSpec(image_prompt="a leaf"))
        self.assertEqual(a, cache_key(VisualSpec(image_prompt="a leaf")))
        self.assertNotEqual(a, cache_key(VisualSpec(image_prompt="a root")))

    def test_icon_composition_needs_something_to_draw(self):
        self.assertTrue(ProducerSet().icons(VisualSpec(icons=["sun"]))["ok"])
        self.assertFalse(ProducerSet().icons(VisualSpec())["ok"])

    def test_the_map_covers_every_strategy_that_produces_an_asset(self):
        from ai_services.core.ppt_v2.schema import VisualStrategy
        produced = set(ProducerSet().as_map())
        self.assertEqual(produced, set(VisualStrategy) - {VisualStrategy.NONE})
