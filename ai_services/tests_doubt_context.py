"""
Regression tests for curriculum-primed doubt image transcription.

Observed on DEV: a student submitted an image-only doubt (question_text was the
empty string, so the image WAS the question). The vision model transcribed the
handwritten name "Horace" as "Florence", and the resolver - correctly, given what
it was handed - answered "insufficient information to determine the reason for
Florence's arrest". The question was Class 10 CBSE English Communicative, chapter
"A Question of Trust", where Horace Danby is the main character.

Nothing was wrong with the model. The class, board and subject were all loaded by
the backend and simply never forwarded, so transcription had no priors.

These tests pin the prompt builder: context in => constrained transcription, no
context => byte-identical to the old generic prompt.

The function is executed from the real bridge.py source rather than imported,
because importing bridge.py pulls in Django settings and a DB connection. The
code under test is the real code, not a copy.
"""
import io
import os
import unittest


def _load_builder():
    """Exec the real _build_doubt_vision_prompt out of bridge.py, no Django."""
    path = os.path.join(os.path.dirname(__file__), "views", "bridge.py")
    src = io.open(path, encoding="utf-8").read()

    const_start = src.index("_DOUBT_VISION_PROMPT = (")
    # The prompt text itself contains "(e.g. x^2 + 3x = 0)", so close on the
    # dedented ")" that ends the assignment, not the first ")" found.
    const_end = src.index(chr(10) + ")" + chr(10), const_start) + 3
    fn_start = src.index("def _build_doubt_vision_prompt")
    fn_end = src.index("return header + _DOUBT_VISION_PROMPT", fn_start) + len(
        "return header + _DOUBT_VISION_PROMPT"
    )

    ns = {}
    exec(src[const_start:const_end], ns)          # the generic prompt
    exec(src[fn_start:fn_end], ns)                # the builder
    return ns["_build_doubt_vision_prompt"], ns["_DOUBT_VISION_PROMPT"]


BUILD, GENERIC = _load_builder()


class VisionPromptContextTests(unittest.TestCase):
    """The prompt must carry curriculum context when it is known."""

    def test_no_context_is_byte_identical_to_the_old_prompt(self):
        # Callers that pass nothing must see exactly today's behaviour.
        self.assertEqual(BUILD(), GENERIC)
        self.assertEqual(BUILD(subject="", class_name="", board="", chapter=""), GENERIC)

    def test_full_context_names_board_class_subject_and_chapter(self):
        out = BUILD(
            subject="English Communicative",
            class_name="Class 10",
            board="cbse",
            chapter="A Question of Trust",
        )
        self.assertIn("CBSE", out)                      # board upper-cased
        self.assertIn("Class 10", out)
        self.assertIn("English Communicative", out)
        self.assertIn("A Question of Trust", out)
        # the generic instructions must still be present, not replaced
        self.assertIn(GENERIC, out)

    def test_context_precedes_the_generic_instructions(self):
        out = BUILD(subject="Physics", class_name="Class 12", board="cbse")
        self.assertTrue(out.startswith("CONTEXT:"))
        self.assertLess(out.index("CONTEXT:"), out.index(GENERIC))

    def test_partial_context_still_helps(self):
        # Subject alone is common: chapter is only known for lecture-linked doubts.
        out = BUILD(subject="Mathematics")
        self.assertIn("Mathematics", out)
        self.assertNotEqual(out, GENERIC)

    def test_chapter_alone_is_enough_to_prime(self):
        out = BUILD(chapter="A Question of Trust")
        self.assertIn("A Question of Trust", out)
        self.assertNotEqual(out, GENERIC)

    def test_it_instructs_transcription_not_answering(self):
        # The vision step must never try to answer - that is the solver's job.
        out = BUILD(subject="English Communicative", class_name="Class 10")
        self.assertIn("do not answer", out.lower())

    def test_it_biases_toward_syllabus_readings(self):
        # This is the sentence that turns "Florence" back into "Horace".
        out = BUILD(subject="English Communicative", class_name="Class 10", board="cbse")
        self.assertIn("syllabus", out.lower())
        self.assertIn("proper nouns", out.lower())

    def test_whitespace_only_context_is_treated_as_absent(self):
        self.assertEqual(BUILD(subject="   ", class_name="  ", board=" ", chapter="  "), GENERIC)

    def test_none_values_do_not_raise(self):
        # studentContext fields arrive as None when the client omits them.
        self.assertEqual(BUILD(subject=None, class_name=None, board=None, chapter=None), GENERIC)


class VisionCallSiteTests(unittest.TestCase):
    """Assertions against the real call site in resolve_doubt."""

    def _resolver_src(self) -> str:
        path = os.path.join(os.path.dirname(__file__), "views", "bridge.py")
        src = io.open(path, encoding="utf-8").read()
        start = src.index('logger.info("[DOUBT] Image detected')
        return src[start:start + 2600]

    def test_the_vision_call_uses_the_context_primed_prompt(self):
        body = self._resolver_src()
        self.assertIn("_build_doubt_vision_prompt(", body)
        self.assertIn("_vision_text_from_image(image_source, _vision_prompt, language)", body)
        # the bare constant must no longer be passed straight in
        self.assertNotIn("_vision_text_from_image(image_source, _DOUBT_VISION_PROMPT", body)

    def test_context_is_read_from_studentContext_and_the_board_header(self):
        body = self._resolver_src()
        self.assertIn('data.get("studentContext")', body)
        self.assertIn('_vctx.get("className")', body)
        self.assertIn('_vctx.get("chapterName")', body)
        self.assertIn('getattr(request, "board", "")', body)

    def test_the_vision_step_is_observable(self):
        # It runs inside @metered("doubt_resolver") so it emits no usage event of
        # its own; elapsed time and length must at least reach the log.
        body = self._resolver_src()
        self.assertIn("_vision_started", body)
        self.assertIn("vision OK", body)
        self.assertIn("elapsed", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
