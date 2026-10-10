"""V2 orchestration, and the switch that decides whether it runs at all.

The property this file exists to defend is that **turning V2 on cannot cost a
teacher their deck**. V2 calls a paid image generator, a search provider, a
sandboxed plotter and an LLM, over a request a human is waiting on. Any of
those can be down. So:

  * every stage failure has a named reason and a decision attached to it;
  * a V2 failure falls back to V1 rather than 502-ing, and says so in the log,
    because a fallback nobody can see is a bug nobody can find;
  * assets resolve concurrently, since generated images are the slowest step
    by an order of magnitude and serialising two of them is how a 40-second
    deck becomes a timeout;
  * with the flag off, none of this code runs.

External providers are mocked throughout. The Photosynthesis case at the
bottom is the Phase 2 integration target.
"""
import contextlib
import os
import threading
import time
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from ai_services.core.ppt_v2.schema import VisualStrategy
from ai_services.core.ppt_v2.service import V2Error, generate_v2
from ai_services.middleware import invalidate_institute_cache
from ai_services.models import Institute

PHOTOSYNTHESIS = {
    "title": "Photosynthesis",
    "slides": [
        {
            "slideType": "concept",
            "title": "What is Photosynthesis?",
            "emphasisText": "Photosynthesis",
            "learningObjective": "State what a plant needs to make its own food",
            "blocks": [
                {"kind": "bullet", "label": "Sunlight",
                 "text": "Chlorophyll in the leaf traps energy from sunlight.", "icon": "sun"},
                {"kind": "bullet", "label": "Water",
                 "text": "Roots absorb water from the soil and carry it upward.", "icon": "water"},
                {"kind": "bullet", "label": "Carbon dioxide",
                 "text": "Stomata let carbon dioxide move into the leaf.", "icon": "air"},
            ],
            "highlightedTerms": ["chlorophyll", "glucose"],
            "visual": {
                "kind": "illustration",
                "description": "a broad green leaf in sunlight with water rising through the stem",
                "objects": ["green leaf", "sunlight", "water"],
                "mustShow": ["chlorophyll"],
                "annotations": [{"text": "Sunlight absorbed", "anchor": [0.5, 0.2]}],
            },
            "speakerNotes": "Ask what a plant eats before revealing it makes its own food.",
        },
        {
            "slideType": "process",
            "title": "The Process of Photosynthesis",
            "learningObjective": "Put the stages of photosynthesis in order",
            "blocks": [
                {"kind": "step", "label": "1. Absorb", "text": "Chlorophyll absorbs sunlight."},
                {"kind": "step", "label": "2. Intake", "text": "Carbon dioxide enters the leaf."},
                {"kind": "step", "label": "3. Form", "text": "Glucose is formed inside the leaf."},
                {"kind": "step", "label": "4. Release", "text": "Oxygen leaves through the stomata."},
            ],
            "visual": {"kind": "illustration",
                       "description": "energy conversion inside a chloroplast",
                       "objects": ["chloroplast"]},
            "speakerNotes": "Stress that the oxygen comes from water, not carbon dioxide.",
        },
    ],
}

CTX = {"className": "Class 8", "subjectName": "Science",
       "chapterName": "Photosynthesis", "topicName": ""}


class _StubLLM:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload or PHOTOSYNTHESIS, error, 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return {"content": self.payload, "model": "stub-model",
                "tokens_input": 100, "tokens_output": 400, "latency_ms": 20}


def stub_providers(ai_ok=True, ai_enabled=True, diagram_ok=False):
    """Patch every outbound provider the pipeline can reach.

    render_diagram has to be in here. It is a real link in the fallback chain,
    and left unpatched it asks an LLM to write matplotlib code and then runs
    it — which an earlier version of this file did, turning a 0.1s unit test
    into 70s of billed API calls.
    """
    gen = (lambda prompt: {"ok": True, "url": "https://img.test/a.png",
                           "provider": "huggingface", "model": "flux"}) if ai_ok \
        else (lambda prompt: {"ok": False, "error": "provider down"})
    drawn = ({"success": True, "image_base64": "data:image/png;base64,AAA"}
             if diagram_ok else
             {"success": False, "image_base64": None, "error": "diagram off in tests"})
    return (
        patch("ai_services.core.image_generation.can_generate_note_images",
              return_value=ai_enabled),
        patch("ai_services.core.image_generation.image_generation_config",
              return_value={"width": 768, "height": 512, "model": "flux"}),
        patch("ai_services.core.image_generation.generate_note_image", gen),
        patch("ai_services.core.diagram.render_diagram", return_value=drawn),
    )


def both_concept():
    """The fixture with both slides as concept slides.

    Asset behaviour (budget, concurrency, dedupe, fallback) needs two slides
    that both fetch a picture. The real fixture's second slide is a process
    slide, and a process slide is deliberately carried by its own step row
    rather than an image - so it is the wrong shape for those tests, though
    exactly right for the process tests below.
    """
    import copy
    payload = copy.deepcopy(PHOTOSYNTHESIS)
    second = payload["slides"][1]
    second["slideType"] = "concept"
    # Concept-SHAPED, not only concept-labelled. Relabelling the step-by-step
    # process slide used to be enough; content analysis now reads "The
    # Process of ..." with four numbered steps as the process it is, so the
    # slide has to actually be a concept for these tests to have two
    # picture-fetching slides.
    second["title"] = "Where Photosynthesis Happens"
    second["learningObjective"] = "Name the part of the leaf cell where food is made"
    second["blocks"] = [
        {"kind": "bullet", "label": "Chloroplasts",
         "text": "Chloroplasts are the green parts of a leaf cell."},
        {"kind": "bullet", "label": "Chlorophyll",
         "text": "Chlorophyll inside them traps energy from sunlight."},
    ]
    second["visual"]["description"] = "a chloroplast inside a leaf cell"
    return _StubLLM(payload=payload)


def passing_vision(_image, _prompt):
    """A vision provider that reports everything the contract asks for.

    Needed because a generated image for a chapter with class-level risks -
    Photosynthesis is one - now has to be inspected before it can be used,
    and an image nobody could inspect is refused. These tests are about
    asset resolution, not about vision, so they supply a provider that
    always confirms and let the asset logic be what is under test.
    """
    return '{"adv": [], "seen": ["chlorophyll", "green leaf", "sunlight", '            '"water", "chloroplast"], "rel": [], "conf": []}'


def run(llm=None, ai_ok=True, ai_enabled=True, search=None, diagram_ok=False,
        vision_ask=passing_vision, **kw):
    """generate_v2 with every external provider stubbed."""
    with contextlib.ExitStack() as stack:
        for provider in stub_providers(ai_ok, ai_enabled, diagram_ok):
            stack.enter_context(provider)
        stack.enter_context(patch.dict(
            os.environ, {"PPT_V2_VISION_VALIDATION": "true"}))
        return generate_v2(topic="Photosynthesis", slide_count=2, ctx=dict(CTX),
                           board="CBSE", llm=llm or _StubLLM(),
                           search_fn=search, vision_ask=vision_ask, **kw)


class PipelineTests(SimpleTestCase):
    def test_it_produces_exactly_the_requested_number_of_slides(self):
        self.assertEqual(len(run().slides), 2)

    def test_a_planner_failure_is_a_named_v2_error(self):
        with self.assertRaises(V2Error) as ctx:
            run(llm=_StubLLM(error=RuntimeError("down")))
        self.assertEqual(ctx.exception.reason, "llm_error")

    def test_malformed_planner_output_is_a_named_v2_error(self):
        with self.assertRaises(V2Error) as ctx:
            run(llm=_StubLLM(payload="{broken"))
        self.assertEqual(ctx.exception.reason, "malformed_json")

    def test_a_deck_of_the_wrong_length_never_reaches_the_caller(self):
        short = {"slides": [PHOTOSYNTHESIS["slides"][0]]}
        with self.assertRaises(V2Error) as ctx:
            run(llm=_StubLLM(payload=short))
        self.assertEqual(ctx.exception.reason, "slide_count")

    def test_each_stage_is_timed(self):
        timings = run().timings
        self.assertEqual(set(timings) & {"planner", "compose", "assets", "total"},
                         {"planner", "compose", "assets", "total"})

    def test_the_two_slides_get_different_layouts(self):
        slides = run().slides
        self.assertNotEqual(slides[0].layout.name, slides[1].layout.name)

    def test_structural_qa_runs_and_passes(self):
        from ai_services.core.ppt_v2 import qa
        self.assertEqual(qa.errors(run().issues), [])


class AssetTests(SimpleTestCase):
    def test_ai_images_are_used_when_available(self):
        result = run(llm=both_concept(), ai_ok=True)
        self.assertEqual(result.ai_images_made, 2)
        for slide in result.slides:
            self.assertEqual(slide.visual.url, "https://img.test/a.png")

    def test_a_failed_ai_image_falls_through_to_search(self):
        hits = {"n": 0}

        def search(term, title, ctx=None):
            hits["n"] += 1
            return {"imageUrl": "https://search.test/b.png"}

        result = run(llm=both_concept(), ai_ok=False, search=search)
        self.assertGreater(hits["n"], 0)
        for slide in result.slides:
            self.assertIs(slide.visual.resolved_strategy, VisualStrategy.SEARCHED_IMAGE)

    def test_with_ai_generation_disabled_the_deck_still_builds(self):
        # The common deployment: no HuggingFace token configured.
        result = run(ai_enabled=False,
                     search=lambda t, ti, c=None: {"imageUrl": "https://s.test/c.png"})
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(result.ai_images_made, 0)

    def test_without_a_picture_a_concept_slide_takes_the_instant_icon_row(self):
        # Not a drawn diagram: as a fallback it was slow (two LLM attempts)
        # and sparse. The icon row is instant and was the clearer slide.
        result = run(llm=both_concept(), ai_ok=False, diagram_ok=True,
                     search=lambda t, ti, c=None: {})
        for slide in result.slides:
            self.assertIs(slide.visual.resolved_strategy,
                          VisualStrategy.ICON_COMPOSITION)

    def test_with_every_provider_down_the_deck_still_builds(self):
        result = run(ai_ok=False, diagram_ok=False, search=lambda t, ti, c=None: {})
        self.assertEqual(len(result.slides), 2)
        for slide in result.slides:
            self.assertIn(slide.visual.resolved_strategy,
                          (VisualStrategy.ICON_COMPOSITION, VisualStrategy.NONE))

    def test_one_slide_failing_does_not_affect_the_other(self):
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first call explodes")
            return {"ok": True, "url": "https://img.test/ok.png"}

        with patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=True), \
             patch("ai_services.core.image_generation.image_generation_config",
                   return_value={"width": 768, "height": 512, "model": "flux"}), \
             patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": False, "error": "off in tests"}), \
             patch("ai_services.core.image_generation.generate_note_image", flaky),              patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}):
            result = generate_v2(topic="Photosynthesis", slide_count=2, ctx=dict(CTX),
                                 llm=both_concept(), vision_ask=passing_vision,
                                 search_fn=lambda t, ti, c=None: {})
        resolved = [s.visual.resolved_strategy for s in result.slides]
        self.assertIn(VisualStrategy.AI_GENERATED_IMAGE, resolved)

    def test_assets_resolve_concurrently_not_one_after_another(self):
        # Two generated images taking 0.3s each must cost ~0.3s, not ~0.6s.
        # A Barrier is the assertion: it only clears if both threads are in
        # the producer at the same moment.
        barrier = threading.Barrier(2, timeout=5)

        def slow(prompt):
            barrier.wait()
            return {"ok": True, "url": "https://img.test/a.png"}

        with patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=True), \
             patch("ai_services.core.image_generation.image_generation_config",
                   return_value={"width": 768, "height": 512, "model": "flux"}), \
             patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": False, "error": "off in tests"}), \
             patch("ai_services.core.image_generation.generate_note_image", slow),              patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}):
            started = time.perf_counter()
            result = generate_v2(topic="Photosynthesis", slide_count=2,
                                 ctx=dict(CTX), llm=both_concept(),
                                 vision_ask=passing_vision)
        self.assertEqual(result.ai_images_made, 2)
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_two_slides_do_not_end_up_with_the_same_picture(self):
        # The first live run put the identical photograph on both slides of a
        # two-slide deck, because two slides on one topic generate near
        # identical queries. Showing the same image twice reads as a bug.
        pool = iter(["https://s.test/same.png", "https://s.test/same.png",
                     "https://s.test/other.png", "https://s.test/other.png"])

        def search(term, title, ctx=None):
            return {"imageUrl": next(pool, "")}

        result = run(llm=both_concept(), ai_ok=False, search=search)
        urls = [s.visual.url for s in result.slides if s.visual.url]
        self.assertEqual(len(urls), len(set(urls)), urls)

    def test_when_no_unique_image_exists_the_duplicate_beats_a_blank_shape(self):
        # Preferring uniqueness must not mean dropping to a generic shape
        # composition: an on-topic picture shown twice still teaches more.
        result = run(llm=both_concept(), ai_ok=False,
                     search=lambda t, ti, c=None: {"imageUrl": "https://s.test/one.png"})
        self.assertEqual(len(result.slides), 2)
        for slide in result.slides:
            self.assertIs(slide.visual.resolved_strategy,
                          VisualStrategy.SEARCHED_IMAGE)
            self.assertEqual(slide.visual.url, "https://s.test/one.png")

    def test_every_asset_attempt_is_recorded_for_the_log(self):
        seen = []
        run(ai_ok=False, search=lambda t, ti, c=None: {},
            on_asset=lambda n, s, ok, d: seen.append((n, s.value, ok)))
        self.assertTrue(seen)
        self.assertTrue(any(not ok for _, _, ok in seen))


class CacheTests(SimpleTestCase):
    class _Cache:
        def __init__(self):
            self.store, self.scopes = {}, []

        def get(self, institute_id, feature, key, vertical="base"):
            self.scopes.append((institute_id, feature, vertical))
            return self.store.get((institute_id, feature, key, vertical))

        def set(self, institute_id, feature, key, value, vertical="base"):
            self.store[(institute_id, feature, key, vertical)] = value

    def test_assets_are_cached_under_the_tenant_not_the_bare_prompt(self):
        # Two schools generating from their own chapter text must not be able
        # to read each other's assets out of a shared key.
        cache = self._Cache()
        run(cache=cache, institute_id="school-a", vertical="school")
        self.assertTrue(cache.store)
        for institute_id, feature, vertical in cache.scopes:
            self.assertEqual((institute_id, feature, vertical),
                             ("school-a", "ppt_v2_asset", "school"))

    def test_no_cache_is_used_without_a_tenant(self):
        cache = self._Cache()
        run(cache=cache, institute_id=None)
        self.assertEqual(cache.scopes, [])

    def test_a_broken_cache_does_not_break_the_deck(self):
        class Exploding:
            def get(self, *a, **kw):
                raise RuntimeError("redis is down")

            def set(self, *a, **kw):
                raise RuntimeError("redis is down")

        self.assertEqual(len(run(cache=Exploding(), institute_id="s").slides), 2)


class ResponseContractTests(SimpleTestCase):
    def test_the_payload_keeps_the_v1_shape(self):
        data = run().to_payload()
        self.assertEqual(set(data) & {"title", "slides"}, {"title", "slides"})
        for slide in data["slides"]:
            for key in ("slideNumber", "type", "title", "bullets",
                        "speakerNotes", "imageSearchTerm", "imageUrl"):
                self.assertIn(key, slide)

    def test_the_v2_specification_rides_underneath(self):
        slide = run().to_payload()["slides"][0]
        self.assertEqual(slide["v2"]["version"], "2.0")
        self.assertTrue(slide["v2"]["layout"]["regions"])
        self.assertTrue(slide["v2"]["theme"]["colors"])

    def test_the_report_separates_what_was_wanted_from_what_was_possible(self):
        # With no HuggingFace token the deck still builds, but the report has
        # to say the slide wanted a generated figure and the deployment could
        # not make one. Collapsing wanted/planned/resolved into one value
        # hides exactly the fact an operator needs.
        strategies = run(llm=both_concept(), ai_enabled=False,
                         search=lambda t, ti, c=None: {"imageUrl": "https://s.test/a.png"}
                         ).to_payload()["generation"]["visualStrategies"]
        for entry in strategies:
            self.assertEqual(entry["wanted"], "ai_generated_image")
            self.assertEqual(entry["planned"], "searched_image")
            self.assertEqual(entry["resolved"], "searched_image")

    def test_wanted_and_resolved_agree_when_nothing_is_in_the_way(self):
        for entry in run(llm=both_concept(), ai_ok=True).to_payload()["generation"]["visualStrategies"]:
            self.assertEqual(entry["wanted"], "ai_generated_image")
            self.assertEqual(entry["resolved"], "ai_generated_image")

    def test_the_response_explains_how_it_was_made(self):
        generation = run().to_payload()["generation"]
        self.assertEqual(generation["version"], "v2")
        self.assertEqual(len(generation["visualStrategies"]), 2)
        self.assertEqual(generation["plannerModel"], "stub-model")

    def test_the_payload_is_json_serialisable(self):
        import json
        json.dumps(run().to_payload())


class PhotosynthesisIntegrationTests(SimpleTestCase):
    """Class 8 / CBSE / Science / Photosynthesis / 2 slides."""

    def test_the_full_pipeline_produces_a_renderable_two_slide_deck(self):
        from ai_services.core.ppt_v2 import qa

        result = run()
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(qa.errors(result.issues), [])

        first, second = result.slides
        self.assertEqual(first.title.text, "What is Photosynthesis?")
        self.assertEqual(second.title.text, "The Process of Photosynthesis")
        self.assertEqual(second.layout.name, "process_steps")
        self.assertEqual(len(second.content_blocks), 4)
        self.assertEqual(first.theme.name, "biology")

        for slide in result.slides:
            self.assertTrue(slide.learning_objective)
            self.assertTrue(slide.visual.description)
            self.assertIsNotNone(slide.visual.resolved_strategy)
            self.assertTrue(slide.layout.regions)

    def test_each_slide_gets_the_visual_its_teaching_purpose_wants(self):
        # Not one blanket answer: the concept slide wants a figure of
        # something microscopic, which has to be generated; the process slide
        # is carried by its own ordered step row.
        first, second = run().slides
        self.assertIs(first.visual.strategy, VisualStrategy.AI_GENERATED_IMAGE)
        self.assertIs(second.visual.strategy, VisualStrategy.ICON_COMPOSITION)

    def test_the_image_prompts_forbid_baked_in_text(self):
        for slide in run().slides:
            self.assertIn("no text", slide.visual.image_prompt)

    def test_the_search_fallback_gets_topic_terms_not_the_slide_headline(self):
        terms = run().slides[0].visual.search_terms
        self.assertTrue(any("Photosynthesis" in t for t in terms), terms)
        self.assertFalse(any("What is Photosynthesis?" == t for t in terms), terms)


class EndpointSwitchTests(TestCase):
    """The flag, exercised over the real HTTP path."""

    API_KEY = "ppt-v2-switch-test-key"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="V2 Switch", slug="v2-switch",
                                 api_key=self.API_KEY, vertical="school",
                                 is_active=True)

    def post(self, **extra):
        return self.client.post(
            "/ppt/generate",
            data={"topic": "Photosynthesis", "slideCount": 2,
                  "className": "Class 8", "subjectName": "Science",
                  "chapterName": "Photosynthesis", **extra},
            content_type="application/json",
            HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school",
        )

    def test_with_the_flag_off_v2_never_runs(self):
        with self.settings(), patch.dict("os.environ", {"PPT_V2_ENABLED": "false"}), \
             patch("ai_services.core.ppt_v2.service.generate_v2") as v2, \
             patch("ai_services.views.ppt.get_llm", return_value=_StubLLM(
                 payload={"title": "T", "slides": [{"title": "A", "bullets": ["x"]},
                                                   {"title": "B", "bullets": ["y"]}]})), \
             patch("ai_services.views.ppt._fetch_image_for_slide",
                   return_value={"imageUrl": None, "imageBase64": None}), \
             patch("ai_services.views.ppt._decompose_ppt_coverage", return_value=None):
            response = self.post()
        v2.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("generation", response.json()["data"])

    def test_with_the_flag_on_v2_runs_and_returns_its_payload(self):
        with patch.dict("os.environ", {"PPT_V2_ENABLED": "true"}), \
             patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=False), \
             patch("ai_services.core.diagram.render_diagram",
                   return_value={"success": False, "error": "off in tests"}), \
             patch("ai_services.views.ppt._fetch_image_for_slide",
                   return_value={"imageUrl": "https://s.test/a.png"}), \
             patch("ai_services.core.llm_client.get_llm", return_value=_StubLLM()):
            # The planner resolves get_llm from core.llm_client, not from the
            # view, so patching the view's name leaves it calling the real
            # provider — 27s and real spend inside a unit test.
            response = self.post()
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["generation"]["version"], "v2")
        self.assertEqual(len(data["slides"]), 2)
        self.assertIn("v2", data["slides"][0])

    def test_a_v2_failure_falls_back_to_v1_rather_than_failing(self):
        v1 = {"title": "V1 deck", "slides": [{"title": "A", "bullets": ["x"]},
                                             {"title": "B", "bullets": ["y"]}]}
        with patch.dict("os.environ", {"PPT_V2_ENABLED": "true",
                                       "PPT_V2_FALLBACK_TO_V1": "true"}), \
             patch("ai_services.core.ppt_v2.service.generate_v2",
                   side_effect=V2Error("llm_error", "planner down")), \
             patch("ai_services.views.ppt._decompose_ppt_coverage", return_value=None), \
             patch("ai_services.views.ppt._fetch_image_for_slide",
                   return_value={"imageUrl": None, "imageBase64": None}), \
             patch("ai_services.views.ppt.get_llm", return_value=_StubLLM(payload=v1)):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("generation", response.json()["data"])

    def test_the_fallback_can_be_turned_off_for_debugging(self):
        with patch.dict("os.environ", {"PPT_V2_ENABLED": "true",
                                       "PPT_V2_FALLBACK_TO_V1": "false"}), \
             patch("ai_services.core.ppt_v2.service.generate_v2",
                   side_effect=V2Error("qa_failed", "text off the slide")):
            response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertIn("qa_failed", response.json()["error"])

    def test_an_unexpected_v2_exception_still_falls_back(self):
        v1 = {"title": "V1", "slides": [{"title": "A", "bullets": ["x"]},
                                        {"title": "B", "bullets": ["y"]}]}
        with patch.dict("os.environ", {"PPT_V2_ENABLED": "true"}), \
             patch("ai_services.core.ppt_v2.service.generate_v2",
                   side_effect=ZeroDivisionError("bug")), \
             patch("ai_services.views.ppt._decompose_ppt_coverage", return_value=None), \
             patch("ai_services.views.ppt._fetch_image_for_slide",
                   return_value={"imageUrl": None, "imageBase64": None}), \
             patch("ai_services.views.ppt.get_llm", return_value=_StubLLM(payload=v1)):
            response = self.post()
        self.assertEqual(response.status_code, 200)
