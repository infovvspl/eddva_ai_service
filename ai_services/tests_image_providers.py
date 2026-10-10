"""Provider selection for ``generate_note_image``, and validation of what it
returns.

Two things are under test here and they meet in one place:

  * Part 1 - the provider split. Hugging Face must behave exactly as it did
    before the split, and Together must speak the API Together documents.
  * Part 2 - a generated image is checked against the slide's visual
    contract before it reaches a slide, through the same vision/class-rule
    machinery the searched-image path already uses.

No test here reaches the network. Every provider call is a patched
``requests.post``; the assertions are made against the request that would
have been sent.
"""
import base64
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from unittest import mock

from ai_services.core import image_generation


# Every environment variable the module reads, so one test cannot inherit
# another's configuration - or the developer's .env.
_IMAGE_ENV_KEYS = (
    "NOTES_IMAGE_PROVIDER", "NOTES_IMAGE_GENERATION_ENABLED",
    "NOTES_IMAGE_STRIP_EMBEDDED_TEXT", "NOTES_IMAGE_OUTPUT_DIR",
    "NOTES_IMAGE_PUBLIC_BASE_URL", "AI_PUBLIC_BASE_URL",
    "NOTES_IMAGE_WIDTH", "NOTES_IMAGE_HEIGHT", "NOTES_IMAGE_MODEL",
    "HF_IMAGE_WIDTH", "HF_IMAGE_HEIGHT", "HF_IMAGE_MODEL",
    "HF_IMAGE_ROUTER_BASE_URL", "HF_IMAGE_TIMEOUT_SECONDS",
    "HF_IMAGE_TOKEN", "HUGGINGFACE_IMAGE_TOKEN", "HF_TOKEN",
    "HUGGINGFACE_API_TOKEN", "HUGGINGFACE_IMAGE_MODEL",
    "TOGETHER_API_KEY", "TOGETHER_IMAGE_MODEL", "TOGETHER_IMAGE_STEPS",
    "TOGETHER_IMAGE_TIMEOUT_SECONDS", "TOGETHER_IMAGE_BASE_URL",
)

# Obvious non-secrets. Nothing here is read from the environment.
FAKE_TOGETHER_KEY = "test-together-key-not-a-secret"
FAKE_HF_KEY = "test-hf-key-not-a-secret"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake image payload"


@contextmanager
def image_env(**overrides):
    """A clean slate for the image configuration, restored on exit."""
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in _IMAGE_ENV_KEYS:
            os.environ.pop(key, None)
        # The OCR strip needs easyocr and cv2 and rewrites the file; it is
        # not what these tests are about.
        os.environ["NOTES_IMAGE_STRIP_EMBEDDED_TEXT"] = "false"
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield


@contextmanager
def together_env(**overrides):
    with tempfile.TemporaryDirectory() as out:
        settings = {
            "NOTES_IMAGE_PROVIDER": "together",
            "NOTES_IMAGE_GENERATION_ENABLED": "true",
            "TOGETHER_API_KEY": FAKE_TOGETHER_KEY,
            "NOTES_IMAGE_OUTPUT_DIR": out,
        }
        settings.update(overrides)
        # Retries wait for real in production; never in a test.
        with image_env(**settings), mock.patch.object(image_generation, "_sleep"):
            yield


def cache_key_for(spec):
    from ai_services.core.ppt_v2.producers import cache_key
    return cache_key(spec)


class FakeResponse:
    """The parts of ``requests.Response`` this module touches."""

    def __init__(self, status_code=200, json_body=None, text="",
                 headers=None, content=b""):
        self.status_code = status_code
        self._json = json_body
        self.text = text or ""
        self.headers = headers or {}
        self.content = content

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._json is None:
            raise ValueError("response body is not JSON")
        return self._json


def together_ok_response(payload_bytes=PNG_BYTES):
    return FakeResponse(json_body={
        "id": "img-1", "model": "black-forest-labs/FLUX.2-dev",
        "object": "list",
        "data": [{"index": 0, "type": "b64_json",
                  "b64_json": base64.b64encode(payload_bytes).decode("ascii")}],
    })


# ---------------------------------------------------------------------------
# Part 1 - provider selection
# ---------------------------------------------------------------------------


class ProviderSelectionTests(unittest.TestCase):
    def test_the_default_provider_is_still_hugging_face(self):
        # The whole point of the split: a deployment that never heard of
        # NOTES_IMAGE_PROVIDER must behave exactly as it did before.
        with image_env():
            cfg = image_generation.image_generation_config()
        self.assertEqual(cfg["provider"], "huggingface")
        self.assertEqual(cfg["model"], "black-forest-labs/FLUX.1-schnell")
        self.assertTrue(cfg["url"].startswith("https://router.huggingface.co/"))

    def test_an_empty_provider_value_is_hugging_face_too(self):
        with image_env(NOTES_IMAGE_PROVIDER=""):
            self.assertEqual(
                image_generation.image_generation_config()["provider"], "huggingface")

    def test_selecting_together_switches_model_and_endpoint(self):
        with together_env():
            cfg = image_generation.image_generation_config()
            self.assertTrue(image_generation.can_generate_note_images())
        self.assertEqual(cfg["provider"], "together")
        # Verified live against GET /v1/models before being made the default.
        self.assertEqual(cfg["model"], "black-forest-labs/FLUX.2-dev")
        self.assertEqual(cfg["url"], "https://api.together.ai/v1/images/generations")
        self.assertTrue(cfg["has_key"])

    def test_the_together_model_is_configurable(self):
        with together_env(TOGETHER_IMAGE_MODEL="black-forest-labs/FLUX.2-flex"):
            self.assertEqual(image_generation.image_generation_config()["model"],
                             "black-forest-labs/FLUX.2-flex")

    def test_an_unknown_provider_disables_generation_rather_than_guessing(self):
        with together_env(NOTES_IMAGE_PROVIDER="togethr"):
            cfg = image_generation.image_generation_config()
            self.assertFalse(cfg["has_key"])
            self.assertEqual(cfg["model"], "")
            self.assertFalse(image_generation.can_generate_note_images())
            result = image_generation.generate_note_image("a leaf")
        self.assertFalse(result["ok"])
        self.assertTrue(result["fatal"])

    def test_together_without_a_key_is_not_configured(self):
        with together_env(TOGETHER_API_KEY=""):
            self.assertFalse(image_generation.can_generate_note_images())
            result = image_generation.generate_note_image("a leaf")
        self.assertEqual(result["error"], "together_image_generation_not_configured")
        self.assertTrue(result["fatal"])

    def test_generation_stays_off_until_explicitly_enabled(self):
        with together_env(NOTES_IMAGE_GENERATION_ENABLED="false"):
            result = image_generation.generate_note_image("a leaf")
        self.assertEqual(result["error"], "image_generation_disabled")

    def test_together_steps_and_timeout_are_configurable(self):
        with together_env(TOGETHER_IMAGE_STEPS="12",
                          TOGETHER_IMAGE_TIMEOUT_SECONDS="45"):
            cfg = image_generation.image_generation_config()
        self.assertEqual(cfg["steps"], 12)
        self.assertEqual(cfg["timeout"], 45)


# ---------------------------------------------------------------------------
# Part 1 - the Together request
# ---------------------------------------------------------------------------


class TogetherRequestTests(unittest.TestCase):
    def _send(self, response=None, **env):
        with together_env(**env):
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=response or together_ok_response()) as post:
                result = image_generation.generate_note_image("a leaf cross section")
        self.assertEqual(post.call_count, 1)
        return result, post.call_args

    def test_it_posts_to_the_documented_endpoint(self):
        _result, call = self._send()
        self.assertEqual(call.args[0], "https://api.together.ai/v1/images/generations")

    def test_it_authenticates_with_the_existing_together_key(self):
        _result, call = self._send()
        self.assertEqual(call.kwargs["headers"]["Authorization"],
                         "Bearer %s" % FAKE_TOGETHER_KEY)

    def test_it_sends_the_configured_canvas(self):
        _result, call = self._send()
        body = call.kwargs["json"]
        self.assertEqual(body["width"], 768)
        self.assertEqual(body["height"], 512)
        self.assertEqual(body["model"], "black-forest-labs/FLUX.2-dev")
        self.assertEqual(body["n"], 1)

    def test_dimensions_are_rounded_to_a_multiple_of_eight(self):
        # "Both values should be multiples of 8" - rounding down here beats
        # a 400 from the provider on every slide of the deck.
        _result, call = self._send(NOTES_IMAGE_WIDTH="770", NOTES_IMAGE_HEIGHT="515")
        self.assertEqual(call.kwargs["json"]["width"], 768)
        self.assertEqual(call.kwargs["json"]["height"], 512)

    def test_it_asks_for_base64_not_a_url(self):
        _result, call = self._send()
        self.assertEqual(call.kwargs["json"]["response_format"], "base64")

    def test_it_asks_for_png_because_the_provider_default_is_jpeg(self):
        _result, call = self._send()
        self.assertEqual(call.kwargs["json"]["output_format"], "png")

    def test_steps_are_sent_when_configured_and_omitted_when_zero(self):
        _result, call = self._send(TOGETHER_IMAGE_STEPS="30")
        self.assertEqual(call.kwargs["json"]["steps"], 30)
        _result, call = self._send(TOGETHER_IMAGE_STEPS="0")
        self.assertNotIn("steps", call.kwargs["json"])

    def test_the_prompt_keeps_its_grounding_clause(self):
        # Provider-independent: the clause is applied before dispatch, so
        # both providers get the same instruction.
        _result, call = self._send()
        prompt = call.kwargs["json"]["prompt"]
        self.assertTrue(prompt.startswith("a leaf cross section"))
        self.assertIn("Do not add unsupported facts", prompt)


class TogetherResponseTests(unittest.TestCase):
    def _generate(self, response, **env):
        with together_env(**env) as _:
            out_dir = os.environ["NOTES_IMAGE_OUTPUT_DIR"]
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=response):
                result = image_generation.generate_note_image("a leaf")
            listing = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
            saved = None
            if result.get("ok"):
                with open(result["path"], "rb") as handle:
                    saved = handle.read()
        return result, listing, saved

    def test_b64_json_is_decoded_and_saved(self):
        result, listing, saved = self._generate(together_ok_response())
        self.assertTrue(result["ok"], result)
        self.assertEqual(saved, PNG_BYTES)
        self.assertEqual(len(listing), 1)
        self.assertEqual(result["provider"], "together")
        self.assertEqual(result["model"], "black-forest-labs/FLUX.2-dev")
        self.assertEqual(result["mime_type"], "image/png")
        self.assertEqual(result["image_size"], "768x512")
        self.assertTrue(result["url"].endswith(listing[0]))

    def test_a_missing_b64_json_is_a_failure_not_a_blank_image(self):
        result, listing, _ = self._generate(FakeResponse(json_body={"data": [{"index": 0}]}))
        self.assertFalse(result["ok"])
        self.assertFalse(result["fatal"])
        self.assertIn("b64_json", result["error"])
        self.assertEqual(listing, [])

    def test_an_empty_data_array_is_a_failure(self):
        result, _listing, _ = self._generate(FakeResponse(json_body={"data": []}))
        self.assertFalse(result["ok"])

    def test_undecodable_base64_is_a_failure(self):
        result, listing, _ = self._generate(FakeResponse(json_body={
            "data": [{"index": 0, "b64_json": "not base64 at all !!!"}]}))
        self.assertFalse(result["ok"])
        self.assertIn("undecodable", result["error"])
        self.assertEqual(listing, [])

    def test_a_url_only_response_is_rejected_rather_than_followed(self):
        # Asking for base64 and being handed a link is not papered over by
        # quietly downloading it: that reintroduces the URL lifetime the
        # base64 request exists to avoid.
        result, listing, _ = self._generate(FakeResponse(json_body={
            "data": [{"index": 0, "type": "url", "url": "https://cdn.test/a.png"}]}))
        self.assertFalse(result["ok"])
        self.assertFalse(result["fatal"])
        self.assertIn("url", result["error"])
        self.assertEqual(listing, [])

    def test_a_non_json_body_is_a_failure(self):
        result, _listing, _ = self._generate(FakeResponse(text="<html>oops</html>"))
        self.assertFalse(result["ok"])
        self.assertIn("non-JSON", result["error"])

    def test_a_transport_error_is_not_fatal(self):
        with together_env():
            with mock.patch.object(image_generation.requests, "post",
                                   side_effect=OSError("connection reset")):
                result = image_generation.generate_note_image("a leaf")
        self.assertFalse(result["ok"])
        self.assertFalse(result["fatal"])

    def test_the_key_never_reaches_an_error_string(self):
        echoed = FakeResponse(status_code=400,
                              text="bad request for key %s" % FAKE_TOGETHER_KEY)
        result, _listing, _ = self._generate(echoed)
        self.assertNotIn(FAKE_TOGETHER_KEY, result["error"])
        self.assertIn("<redacted>", result["error"])


class TogetherFatalClassificationTests(unittest.TestCase):
    def _status(self, code):
        with together_env():
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=FakeResponse(status_code=code,
                                                             text="error body")):
                return image_generation.generate_note_image("a leaf")

    def test_401_is_fatal(self):
        self.assertTrue(self._status(401)["fatal"])

    def test_403_is_fatal(self):
        self.assertTrue(self._status(403)["fatal"])

    def test_404_is_fatal(self):
        self.assertTrue(self._status(404)["fatal"])

    def test_429_is_not_fatal(self):
        # Rate limiting is the deck going too fast, not a dead provider.
        self.assertFalse(self._status(429)["fatal"])

    def test_500_is_not_fatal(self):
        self.assertFalse(self._status(500)["fatal"])

    def test_503_is_not_fatal(self):
        self.assertFalse(self._status(503)["fatal"])

    def test_the_hugging_face_wording_rules_are_not_applied_to_together(self):
        # "does not exist" is Hugging Face phrasing. A 429 that happens to
        # contain it must still be retryable.
        with together_env():
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=FakeResponse(
                                       status_code=429,
                                       text="slow down; that queue does not exist")):
                result = image_generation.generate_note_image("a leaf")
        self.assertFalse(result["fatal"])


# ---------------------------------------------------------------------------
# Part 1 - Hugging Face, unchanged
# ---------------------------------------------------------------------------


class HuggingFaceStillWorksTests(unittest.TestCase):
    @contextmanager
    def _env(self, **overrides):
        with tempfile.TemporaryDirectory() as out:
            settings = {"NOTES_IMAGE_GENERATION_ENABLED": "true",
                        "HF_IMAGE_TOKEN": FAKE_HF_KEY,
                        "NOTES_IMAGE_OUTPUT_DIR": out}
            settings.update(overrides)
            with image_env(**settings):
                yield

    def test_it_still_posts_the_router_shape_and_saves_the_bytes(self):
        response = FakeResponse(headers={"content-type": "image/png"},
                                content=PNG_BYTES)
        with self._env():
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=response) as post:
                result = image_generation.generate_note_image("a leaf")
            with open(result["path"], "rb") as handle:
                saved = handle.read()
        body = post.call_args.kwargs["json"]
        self.assertIn("inputs", body)                       # not "prompt"
        self.assertEqual(body["parameters"]["width"], 768)
        self.assertEqual(post.call_args.kwargs["headers"]["Accept"], "image/png")
        self.assertEqual(saved, PNG_BYTES)
        self.assertEqual(result["provider"], "huggingface")

    def test_a_cold_model_is_still_retried_once(self):
        cold = FakeResponse(status_code=503, json_body={"estimated_time": 0.01})
        warm = FakeResponse(headers={"content-type": "image/png"}, content=PNG_BYTES)
        with self._env():
            with mock.patch.object(image_generation.requests, "post",
                                   side_effect=[cold, warm]) as post:
                result = image_generation.generate_note_image("a leaf")
        self.assertEqual(post.call_count, 2)
        self.assertTrue(result["ok"])

    def test_a_missing_token_keeps_its_existing_error_string(self):
        with self._env(HF_IMAGE_TOKEN=""):
            result = image_generation.generate_note_image("a leaf")
        self.assertEqual(result["error"], "huggingface_image_generation_not_configured")
        self.assertTrue(result["fatal"])

    def test_hugging_face_wording_still_classifies_fatally(self):
        with self._env():
            with mock.patch.object(image_generation.requests, "post",
                                   return_value=FakeResponse(
                                       status_code=400,
                                       text="model is gated")):
                result = image_generation.generate_note_image("a leaf")
        self.assertTrue(result["fatal"])


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class ImageTelemetryTests(unittest.TestCase):
    def test_usage_is_labelled_with_the_live_provider(self):
        from ai_services.views.ppt import _image_usage_model

        with together_env():
            self.assertEqual(_image_usage_model(),
                             "together:black-forest-labs/FLUX.2-dev")

    def test_hugging_face_telemetry_is_unchanged(self):
        from ai_services.views.ppt import _image_usage_model

        with image_env(HF_IMAGE_TOKEN=FAKE_HF_KEY):
            self.assertEqual(_image_usage_model(),
                             "huggingface:black-forest-labs/FLUX.1-schnell")


# ---------------------------------------------------------------------------
# Part 2 - a generated image is validated before it reaches a slide
# ---------------------------------------------------------------------------


class GeneratedImageValidationTests(unittest.TestCase):
    """The AI branch now goes through the same contract/vision/class-rule
    machinery the searched-image branch already used."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.image_path = os.path.join(self._dir.name, "generated.png")
        # A real, decodable PNG: the vision reference is re-encoded through
        # PIL, and the provider tests' PNG_BYTES is only a magic number.
        from PIL import Image
        Image.new("RGB", (768, 512), (40, 150, 70)).save(self.image_path)
        self.generated = {"ok": True, "url": "https://ai.test/generated.png",
                          "path": self.image_path, "mime_type": "image/png",
                          "provider": "together",
                          "model": "black-forest-labs/FLUX.2-dev"}

    def _spec(self, **overrides):
        from ai_services.core.ppt_v2.schema import VisualSpec

        fields = {"image_prompt": "a leaf cross section showing chloroplasts",
                  "icons": ["leaf", "sunlight"],
                  "must_show": ["chloroplast"],
                  "relationships": ["sunlight striking the leaf surface"],
                  "avoid_concepts": ["calvin cycle"],
                  "objective": "where photosynthesis happens in a leaf"}
        fields.update(overrides)
        return VisualSpec(**fields)

    @contextmanager
    def _vision_on(self):
        with mock.patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}):
            yield

    def _producers(self, ask, store=None):
        from ai_services.core.ppt_v2.producers import ProducerSet

        store = {} if store is None else store
        return ProducerSet(subject="Science", class_name="Class 8",
                           board="CBSE", chapter="Photosynthesis",
                           cache_get=store.get, cache_put=store.__setitem__,
                           vision_ask=ask, vision_model="test-vision"), store

    def _run(self, ask, spec=None, store=None):
        producers, store = self._producers(ask, store)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = producers.ai_image(spec or self._spec())
        return out, producers, store

    def test_a_generated_image_is_looked_at(self):
        seen = {}

        def ask(image, prompt):
            seen["image"] = image
            seen["prompt"] = prompt
            return '{"adv": [], "seen": ["chloroplast", "leaf", "sunlight"], ' \
                   '"rel": ["sunlight striking the leaf surface"], "conf": []}'

        out, producers, _store = self._run(ask)
        self.assertTrue(out["ok"], out)
        self.assertIn("image", seen, "the generated image was never validated")
        # The saved file, inlined - not a round trip to our own static route -
        # and encoded as the URL path encodes a searched image.
        self.assertTrue(seen["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(producers.vision_verdicts), 1)

    def test_an_accepted_image_reaches_the_asset_and_the_cache(self):
        from ai_services.core.ppt_v2.producers import cache_key

        def ask(_image, _prompt):
            return '{"adv": [], "seen": ["chloroplast", "leaf", "sunlight"], ' \
                   '"rel": ["sunlight striking the leaf surface"], "conf": []}'

        spec = self._spec()
        out, _producers, store = self._run(ask, spec=spec)
        self.assertTrue(out["ok"])
        self.assertEqual(out["url"], "https://ai.test/generated.png")
        self.assertEqual(out["source"], "together")
        self.assertEqual(out["model"], "black-forest-labs/FLUX.2-dev")
        self.assertIn(cache_key(spec), store)

    def test_a_rejected_image_fails_the_producer_and_is_not_cached(self):
        from ai_services.core.ppt_v2.producers import cache_key

        def ask(_image, _prompt):
            # Class 11 chemistry on a Class 8 slide.
            return '{"adv": ["calvin cycle", "ATP"], "seen": ["leaf"], ' \
                   '"rel": [], "conf": ["calvin cycle"]}'

        spec = self._spec()
        out, _producers, store = self._run(ask, spec=spec)
        self.assertFalse(out["ok"])
        self.assertIn("vision rejected", out["error"])
        # Caching it would hand the same rejected picture back through
        # CACHED_ASSET on a later slide.
        self.assertNotIn(cache_key(spec), store)

    def test_a_rejected_image_lets_the_chain_continue_to_search(self):
        from ai_services.core.ppt_v2.schema import VisualStrategy
        from ai_services.core.ppt_v2.visuals import resolve_visual

        def ask(_image, _prompt):
            return '{"adv": ["calvin cycle"], "seen": [], "rel": [], ' \
                   '"conf": ["calvin cycle"]}'

        producers, _store = self._producers(ask)
        spec = self._spec(strategy=VisualStrategy.AI_GENERATED_IMAGE,
                          fallbacks=[VisualStrategy.SEARCHED_IMAGE,
                                     VisualStrategy.ICON_COMPOSITION,
                                     VisualStrategy.NONE])
        producer_map = dict(producers.as_map())
        producer_map[VisualStrategy.SEARCHED_IMAGE] = lambda _s: {
            "ok": True, "url": "https://img.test/photo.jpg", "source": "search"}

        attempts = []
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = resolve_visual(spec, producer_map,
                                     on_attempt=lambda s, ok, detail:
                                     attempts.append((s, ok, detail)))

        # The ordering is unchanged: AI is still tried first, and only the
        # rejection moves the chain on to the searched image.
        self.assertEqual(attempts[0][0], VisualStrategy.AI_GENERATED_IMAGE)
        self.assertFalse(attempts[0][1])
        self.assertIn("vision rejected", attempts[0][2])
        self.assertEqual(out.resolved_strategy, VisualStrategy.SEARCHED_IMAGE)
        self.assertEqual(out.url, "https://img.test/photo.jpg")

    def test_a_plain_contract_is_not_worth_a_vision_call(self):
        # "Do not blindly run expensive vision calls on every AI image."
        # Nothing to verify beyond the objects themselves, and a chapter with
        # no class-level risks: accept it and keep the money.
        calls = []

        def ask(image, prompt):
            calls.append(image)
            return "{}"

        spec = self._spec(relationships=[], avoid_concepts=[])
        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.ppt_v2.class_rules.all_forbidden",
                            return_value=[]):
                with mock.patch("ai_services.core.image_generation.generate_note_image",
                                return_value=dict(self.generated)):
                    out = producers.ai_image(spec)
        self.assertTrue(out["ok"])
        self.assertEqual(calls, [])

    def test_a_class_level_risk_alone_is_enough_to_look(self):
        calls = []

        def ask(image, _prompt):
            calls.append(image)
            return '{"adv": [], "seen": ["chloroplast", "leaf", "sunlight"], '                    '"rel": [], "conf": []}'

        spec = self._spec(relationships=[], avoid_concepts=[])
        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.ppt_v2.class_rules.all_forbidden",
                            return_value=["calvin cycle"]):
                with mock.patch("ai_services.core.image_generation.generate_note_image",
                                return_value=dict(self.generated)):
                    out = producers.ai_image(spec)
        self.assertTrue(out["ok"])
        self.assertEqual(len(calls), 1)

    def test_no_vision_provider_means_the_image_is_refused(self):
        # Fail-closed. This asserted the opposite until a live run showed what
        # fail-open costs: FLUX returned an image, the check could not be
        # made, and the picture went onto a Class 8 slide unexamined.
        producers, store = self._producers(ask=None)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = producers.ai_image(self._spec())
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])
        self.assertEqual(store, {})

    def test_vision_switched_off_entirely_also_refuses_the_image(self):
        # An operator who turns validation off does not thereby authorise
        # unvalidated generated imagery; they turn generated imagery off.
        def ask(_image, _prompt):
            raise AssertionError("vision must not be called when disabled")

        producers, _store = self._producers(ask)
        with mock.patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "false"}):
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = producers.ai_image(self._spec())
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])

    def test_an_unparseable_vision_reply_refuses_the_image(self):
        # The exact first-live-run failure: a nine-second call whose answer
        # did not parse, reported as available=False.
        def ask(_image, _prompt):
            return "I'm sorry, I can't help with that."

        producers, store = self._producers(ask)
        spec = self._spec()
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = producers.ai_image(spec)
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])
        self.assertIn("unparseable", out["error"])
        self.assertNotIn(cache_key_for(spec), store)

    def test_a_vision_timeout_refuses_the_image(self):
        def ask(_image, _prompt):
            time.sleep(0.5)
            return '{"adv": [], "seen": ["green leaf"], "rel": [], "conf": []}'

        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.ppt_v2.vision.timeout_s",
                            return_value=0.05):
                with mock.patch("ai_services.core.image_generation.generate_note_image",
                                return_value=dict(self.generated)):
                    out = producers.ai_image(self._spec())
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])
        self.assertIn("timed out", out["error"])

    def test_a_vision_provider_error_refuses_the_image(self):
        def ask(_image, _prompt):
            raise RuntimeError("vision backend down")

        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = producers.ai_image(self._spec())
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])

    def test_a_semantic_rejection_refuses_the_image_and_the_chain_continues(self):
        # Objects the slide names are simply not in the picture.
        from ai_services.core.ppt_v2.schema import VisualStrategy
        from ai_services.core.ppt_v2.visuals import resolve_visual

        def ask(_image, _prompt):
            return '{"adv": [], "seen": [], "rel": [], "conf": []}'

        producers, store = self._producers(ask)
        spec = self._spec(strategy=VisualStrategy.AI_GENERATED_IMAGE,
                          fallbacks=[VisualStrategy.PROGRAMMATIC_DIAGRAM,
                                     VisualStrategy.SEARCHED_IMAGE,
                                     VisualStrategy.NONE])
        producer_map = dict(producers.as_map())
        producer_map[VisualStrategy.PROGRAMMATIC_DIAGRAM] = lambda _s: {
            "ok": False, "error": "diagram off in tests"}
        producer_map[VisualStrategy.SEARCHED_IMAGE] = lambda _s: {
            "ok": True, "url": "https://img.test/photo.jpg", "source": "search"}

        attempts = []
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                out = resolve_visual(spec, producer_map,
                                     on_attempt=lambda st, ok, d:
                                     attempts.append((st, ok, d)))
        self.assertFalse(attempts[0][1])
        self.assertIn("vision rejected", attempts[0][2])
        # Ordering is untouched: diagram is still tried before search.
        self.assertEqual([a[0] for a in attempts[:3]],
                         [VisualStrategy.AI_GENERATED_IMAGE,
                          VisualStrategy.PROGRAMMATIC_DIAGRAM,
                          VisualStrategy.SEARCHED_IMAGE])
        self.assertEqual(out.resolved_strategy, VisualStrategy.SEARCHED_IMAGE)
        self.assertNotIn(cache_key_for(spec), store)

    def test_a_refused_image_never_comes_back_as_a_cached_asset(self):
        def ask(_image, _prompt):
            return "not json"

        producers, store = self._producers(ask)
        spec = self._spec()
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)):
                self.assertFalse(producers.ai_image(spec)["ok"])
        # The cached-asset producer is the next deck's route back to the same
        # picture. A refused image must not be waiting there.
        self.assertFalse(producers.cached(spec).get("ok"))
        self.assertEqual(store, {})

    def test_the_search_path_still_accepts_an_unparseable_reply(self):
        # Deliberately NOT changed. A searched image has provenance a
        # generated one does not - a caption, a host, a deterministic score -
        # so "nobody looked" leaves the earlier decision standing.
        def ask(_image, _prompt):
            return "I'm sorry, I can't help with that."

        producers, _store = self._producers(ask)
        producers._search_fn = lambda term, title, ctx=None, accept=None: {
            "imageUrl": "https://img.test/leaf.jpg",
            "relevanceScore": 2.0, "relevanceConfidence": "medium"}
        spec = self._spec(search_terms=["green leaf photosynthesis"])
        with self._vision_on():
            out = producers.search(spec)
        self.assertTrue(out["ok"])
        self.assertEqual(out["url"], "https://img.test/leaf.jpg")

    def test_diagnostics_separate_generated_text_from_erased_text(self):
        # Three different questions, three numbers. The first live run could
        # only report one of them.
        def ask(_image, _prompt):
            return '{"adv": [], "seen": ["chloroplast", "green leaf", "sunlight"], ' \
                   '"rel": ["sunlight striking the leaf surface"], "conf": []}'

        generated = dict(self.generated)
        generated.update({"embedded_text_detected": 16, "embedded_text_removed": 16,
                          "embedded_text_samples": ["CO2", "H2O", "O2"]})
        producers, _store = self._producers(ask)
        spec = self._spec()
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=generated):
                producers.ai_image(spec)
        self.assertEqual(len(producers.image_diagnostics), 1)
        entry = producers.image_diagnostics[0]
        self.assertEqual(entry["embedded_text_generated"], 16)
        self.assertEqual(entry["embedded_text_removed"], 16)
        self.assertEqual(entry["annotations_overlaid"], len(spec.annotations or []))
        self.assertIn("CO2", entry["embedded_text_samples"])

    def test_a_rejection_pays_for_exactly_one_guided_redraw(self):
        # One look used to be final, and a live deck lost two of three pictures
        # to specific, fixable misses. Now: one redraw, told what was wrong.
        def ask(_image, _prompt):
            return '{"adv": ["calvin cycle"], "seen": [], "rel": [], ' \
                   '"conf": ["calvin cycle"]}'

        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)) as generate:
                out = producers.ai_image(self._spec())
        self.assertFalse(out["ok"])
        self.assertNotIn("_verdict", out)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("must NOT show: calvin cycle", generate.call_args_list[1].args[0])

    def test_no_redraw_when_the_deck_is_short_of_time(self):
        import time as _t
        from ai_services.core.ppt_v2.producers import AI_IMAGE_MIN_REMAINING_S

        def ask(_image, _prompt):
            return '{"adv": ["calvin cycle"], "seen": [], "rel": [], ' \
                   '"conf": ["calvin cycle"]}'

        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)) as generate:
                # Enough to start, not enough for a second attempt afterwards.
                producers.deadline = _t.perf_counter() + AI_IMAGE_MIN_REMAINING_S + 0.5

                def slow(prompt):
                    _t.sleep(0.6)
                    return dict(self.generated)
                generate.side_effect = slow
                producers.ai_image(self._spec())
        self.assertEqual(generate.call_count, 1)

    def test_no_redraw_when_nobody_could_look(self):
        def ask(_image, _prompt):
            raise RuntimeError("vision down")

        producers, _store = self._producers(ask)
        with self._vision_on():
            with mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value=dict(self.generated)) as generate:
                producers.ai_image(self._spec())
        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()


class TogetherRateLimitTests(unittest.TestCase):
    """A 10-slide deck asked for three images at once; Together answered two
    with 429 and, with no retry, both slides lost their picture."""

    def _run(self, responses, **env):
        with together_env(**env):
            with mock.patch.object(image_generation.requests, "post",
                                   side_effect=list(responses)) as post,                  mock.patch.object(image_generation, "_sleep") as sleep:
                result = image_generation.generate_note_image("a leaf")
        return result, post, sleep

    def test_a_429_is_retried_and_the_image_still_arrives(self):
        result, post, sleep = self._run([FakeResponse(status_code=429, text="slow down"),
                                         together_ok_response()])
        self.assertTrue(result["ok"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_the_providers_retry_after_is_honoured_and_capped(self):
        _r, _p, sleep = self._run([FakeResponse(status_code=429, headers={"Retry-After": "3"}),
                                   together_ok_response()])
        self.assertEqual(sleep.call_args.args[0], 3.0)
        _r, _p, sleep = self._run([FakeResponse(status_code=429, headers={"Retry-After": "600"}),
                                   together_ok_response()])
        self.assertEqual(sleep.call_args.args[0], image_generation._TOGETHER_MAX_RETRY_WAIT_S)

    def test_a_persistent_429_gives_up_after_the_retries_and_stays_non_fatal(self):
        result, post, _sleep = self._run([FakeResponse(status_code=429)] * 3)
        self.assertFalse(result["ok"])
        self.assertFalse(result["fatal"])
        self.assertEqual(post.call_count, 3)          # first try + 2 retries

    def test_a_server_error_is_retried_too(self):
        result, post, _sleep = self._run([FakeResponse(status_code=503), together_ok_response()])
        self.assertTrue(result["ok"])
        self.assertEqual(post.call_count, 2)

    def test_a_credential_or_policy_error_is_never_retried(self):
        result, post, sleep = self._run([FakeResponse(status_code=403, text="data sharing")])
        self.assertTrue(result["fatal"])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_retries_can_be_switched_off(self):
        _result, post, _sleep = self._run([FakeResponse(status_code=429)],
                                          TOGETHER_IMAGE_RETRIES="0")
        self.assertEqual(post.call_count, 1)

    def test_image_calls_queue_one_at_a_time_by_default(self):
        import threading
        import time as _time
        active, peak, lock = [0], [0], threading.Lock()

        def slow_post(*_a, **_k):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            _time.sleep(0.05)
            with lock:
                active[0] -= 1
            return together_ok_response()

        with together_env():
            with mock.patch.object(image_generation.requests, "post", side_effect=slow_post):
                threads = [threading.Thread(target=image_generation.generate_note_image,
                                            args=("a leaf",)) for _ in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        self.assertEqual(peak[0], 1)
