"""Regression tests for inspecting a GENERATED image with the vision model.

The first two live Together runs both reported the AI image's vision verdict
as "unparseable reply". The log carried the reply itself:

    {"adv": [], "seen": ["green leaf", "sunlight", "carbon dioxide entering",
     "water rising stem"], "rel": ["sunlight falling leaf", "

A correct reading of the picture, cut off mid-string. Not a transport fault -
Gemini received the bytes and read them - but an output-budget one: on the
Gemini 2.5+ family thinking tokens count against max_output_tokens, and the
1024 cap sized for transcriptions left ~35 tokens for the answer once the
model had reasoned about a richer contract.

These tests pin:
  * the PPT validator asks for a structured, full-budget answer;
  * every other caller of the shared helper is untouched;
  * a truncated answer still fails parsing and still rejects the AI image -
    the fix widens the budget, it does not loosen the parser;
  * the generated image reaches the helper encoded the way a searched one
    does.

No network: every model call is patched.
"""
import base64
import os
import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

try:
    import google.genai  # noqa: F401  (the request config these tests inspect)
    HAVE_GENAI = True
except ImportError:  # CI installs requirements.ci.txt, which leaves it out
    HAVE_GENAI = False

needs_genai = unittest.skipUnless(HAVE_GENAI, "google-genai is not installed")

# The reply a live run actually received, verbatim (130 chars).
LIVE_TRUNCATED_REPLY = (
    '{"adv": [], "seen": ["green leaf", "sunlight", "carbon dioxide entering", '
    '"water rising stem"], "rel": ["sunlight falling leaf", "'
)


def _png(path, size=(768, 512), mode="RGBA"):
    from PIL import Image
    Image.new(mode, size, (30, 160, 60, 255) if mode == "RGBA" else (30, 160, 60)).save(path)
    return path


@needs_genai
class StructuredBudgetTests(unittest.TestCase):
    def test_flash_gets_a_large_cap_and_no_thinking(self):
        from ai_services.views.bridge import _structured_vision_config

        cfg = _structured_vision_config("gemini-2.5-flash")
        self.assertEqual(cfg["max_output_tokens"], 8192)
        self.assertEqual(cfg["thinking_config"].thinking_budget, 0)

    def test_flash_lite_is_flash_too(self):
        from ai_services.views.bridge import _structured_vision_config

        cfg = _structured_vision_config("gemini-2.5-flash-lite")
        self.assertEqual(cfg["thinking_config"].thinking_budget, 0)

    def test_models_that_cannot_disable_thinking_get_only_the_cap(self):
        # Pro rejects a zero budget; Gemini 3 controls thinking differently.
        # Guessing a setting for them would turn a truncation into a 400.
        from ai_services.views.bridge import _structured_vision_config

        for model in ("gemini-2.5-pro", "gemini-3-flash-preview",
                      "gemini-flash-latest", ""):
            cfg = _structured_vision_config(model)
            self.assertEqual(cfg["max_output_tokens"], 8192, model)
            self.assertNotIn("thinking_config", cfg, model)

    def test_a_max_tokens_finish_is_recognised(self):
        from ai_services.views.bridge import _gemini_reply_truncated

        hit = SimpleNamespace(candidates=[SimpleNamespace(
            finish_reason=SimpleNamespace(name="MAX_TOKENS"))])
        done = SimpleNamespace(candidates=[SimpleNamespace(
            finish_reason=SimpleNamespace(name="STOP"))])
        self.assertTrue(_gemini_reply_truncated(hit))
        self.assertFalse(_gemini_reply_truncated(done))
        self.assertFalse(_gemini_reply_truncated(object()))


@needs_genai
class GeminiCallConfigTests(unittest.TestCase):
    """What `_call_gemini_vision` actually sends to the SDK."""

    def _call(self, **kwargs):
        from ai_services.views import bridge

        seen = {}

        def fake_generate(api_key, model, contents, config):
            seen["model"] = model
            seen["config"] = config
            return SimpleNamespace(text='{"adv": [], "seen": [], "rel": [], "conf": []}',
                                   candidates=[SimpleNamespace(
                                       finish_reason=SimpleNamespace(name="STOP"))])

        with mock.patch.object(bridge, "gemini_generate", fake_generate), \
             mock.patch("ai_services.core.gemini_keys.get_rotated_gemini_keys",
                        return_value=[(1, "test-key-not-a-secret")]), \
             mock.patch("ai_services.core.gemini_keys.resolve_gemini_model",
                        side_effect=lambda key, model: model), \
             mock.patch.object(bridge, "_image_to_bytes_and_mime",
                               return_value=(b"\xff\xd8fake", "image/jpeg")), \
             mock.patch.dict(os.environ, {"GEMINI_VISION_MODEL": "gemini-2.5-flash"}):
            os.environ.pop("NOTES_IMAGE_OVERLAY_GEMINI_MODEL", None)
            out = bridge._call_gemini_vision("data:image/jpeg;base64,AAAA", "prompt", **kwargs)
        return out, seen

    def test_structured_calls_send_the_full_budget(self):
        out, seen = self._call(structured=True)
        self.assertTrue(out)
        self.assertEqual(seen["config"].max_output_tokens, 8192)
        self.assertEqual(seen["config"].thinking_config.thinking_budget, 0)
        self.assertEqual(seen["config"].temperature, 0.0)

    def test_every_other_caller_is_byte_for_byte_unchanged(self):
        # Doubt-solving and grading call the same helper without the flag.
        _out, seen = self._call()
        self.assertEqual(seen["config"].max_output_tokens, 1024)
        self.assertIsNone(seen["config"].thinking_config)
        self.assertEqual(seen["config"].temperature, 0.0)


class HelperThreadingTests(unittest.TestCase):
    def test_the_flag_reaches_gemini_through_the_shared_helper(self):
        from ai_services.views import bridge

        captured = {}

        def fake_gemini(image, prompt, **kwargs):
            captured.update(kwargs)
            return "<transcription>{}</transcription>"

        with mock.patch.object(bridge, "_call_gemini_vision", fake_gemini), \
             mock.patch.object(bridge, "get_rotated_groq_keys",
                               return_value=[("test-groq-key", 1)]), \
             mock.patch.dict(os.environ, {"GROQ_VISION_MODELS": ""}):
            bridge._vision_text_from_image("data:image/jpeg;base64,AAAA", "p",
                                           structured=True)
        self.assertIs(captured.get("structured"), True)

    def test_the_default_does_not_set_the_flag(self):
        from ai_services.views import bridge

        captured = {}

        def fake_gemini(image, prompt, **kwargs):
            captured.update(kwargs)
            return "<transcription>x</transcription>"

        with mock.patch.object(bridge, "_call_gemini_vision", fake_gemini), \
             mock.patch.object(bridge, "get_rotated_groq_keys",
                               return_value=[("test-groq-key", 1)]), \
             mock.patch.dict(os.environ, {"GROQ_VISION_MODELS": ""}):
            bridge._vision_text_from_image("data:image/jpeg;base64,AAAA", "p")
        self.assertIs(captured.get("structured"), False)

    def test_the_ppt_validator_asks_for_a_structured_answer(self):
        from ai_services.views import ppt

        with mock.patch("ai_services.views.bridge._vision_text_from_image",
                        return_value="{}") as helper:
            ppt._ppt_vision_ask("data:image/jpeg;base64,AAAA", "prompt")
        self.assertEqual(helper.call_args.kwargs.get("structured"), True)


class TruncationStillFailsClosedTests(unittest.TestCase):
    """The fix widens the budget. It must not loosen the parser."""

    def test_the_live_truncated_reply_is_still_unparseable(self):
        from ai_services.core.ppt_v2 import vision

        contract = vision.build_contract(
            learning_objective="what enters and leaves a leaf",
            objects=["green leaf", "sunlight"],
            relationships=["sunlight falling on the leaf"],
            class_name="Class 8", subject="Science", chapter="Photosynthesis")
        with mock.patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}):
            verdict = vision.validate("data:image/jpeg;base64,AAAA", contract,
                                      ask=lambda _i, _p: LIVE_TRUNCATED_REPLY)
        self.assertFalse(verdict.available)
        self.assertEqual(verdict.reason, "unparseable reply")

    def test_and_a_generated_image_with_that_reply_is_rejected(self):
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.schema import VisualSpec

        with tempfile.TemporaryDirectory() as tmp:
            path = _png(os.path.join(tmp, "g.png"))
            producers = ProducerSet(subject="Science", class_name="Class 8",
                                    chapter="Photosynthesis",
                                    vision_ask=lambda _i, _p: LIVE_TRUNCATED_REPLY)
            spec = VisualSpec(image_prompt="a green leaf in sunlight",
                              icons=["green leaf", "sunlight"],
                              relationships=["sunlight falling on the leaf"])
            with mock.patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}), \
                 mock.patch("ai_services.core.image_generation.generate_note_image",
                            return_value={"ok": True, "url": "https://ai.test/g.png",
                                          "path": path, "mime_type": "image/png",
                                          "provider": "together", "model": "flux"}):
                out = producers.ai_image(spec)
        self.assertFalse(out["ok"])
        self.assertIn("could not inspect", out["error"])

    def test_the_same_reading_complete_is_accepted(self):
        # The control: the identical observation, untruncated, passes. So the
        # only thing between the live run and an accepted image was the budget.
        from ai_services.core.ppt_v2 import vision

        complete = ('{"adv": [], "seen": ["green leaf", "sunlight"], '
                    '"rel": ["sunlight falling on the leaf"], "conf": []}')
        contract = vision.build_contract(
            learning_objective="what enters and leaves a leaf",
            objects=["green leaf", "sunlight"],
            relationships=["sunlight falling on the leaf"],
            class_name="Class 8", subject="Science", chapter="Photosynthesis")
        with mock.patch.dict(os.environ, {"PPT_V2_VISION_VALIDATION": "true"}):
            verdict = vision.validate("data:image/jpeg;base64,AAAA", contract,
                                      ask=lambda _i, _p: complete)
        self.assertTrue(verdict.available)
        self.assertTrue(verdict.passed)


class GeneratedImageEncodingTests(unittest.TestCase):
    def test_a_generated_png_reaches_vision_as_the_url_path_would_send_it(self):
        from PIL import Image

        from ai_services.core.ppt_v2.producers import ProducerSet

        with tempfile.TemporaryDirectory() as tmp:
            path = _png(os.path.join(tmp, "g.png"))
            ref = ProducerSet._vision_reference({"path": path,
                                                 "mime_type": "image/png",
                                                 "url": "http://localhost:8000/x.png"})
        self.assertTrue(ref.startswith("data:image/jpeg;base64,"))
        with Image.open(BytesIO(base64.b64decode(ref.split(",", 1)[1]))) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.mode, "RGB")          # alpha flattened
            self.assertEqual(img.size, (768, 512))     # under the cap: untouched

    def test_an_oversized_image_is_capped_at_1024(self):
        from PIL import Image

        from ai_services.core.ppt_v2.producers import _jpeg_data_uri

        with tempfile.TemporaryDirectory() as tmp:
            path = _png(os.path.join(tmp, "big.png"), size=(2048, 1024), mode="RGB")
            ref = _jpeg_data_uri(path)
        with Image.open(BytesIO(base64.b64decode(ref.split(",", 1)[1]))) as img:
            self.assertEqual(max(img.size), 1024)

    def test_a_missing_file_falls_back_to_the_url(self):
        from ai_services.core.ppt_v2.producers import ProducerSet

        ref = ProducerSet._vision_reference({"path": "Z:/does/not/exist.png",
                                             "url": "https://ai.test/g.png"})
        self.assertEqual(ref, "https://ai.test/g.png")


if __name__ == "__main__":
    unittest.main()
