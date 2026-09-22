"""
Tests for drawn diagrams (ai_services.core.diagram + core._plot_runner).

This covers the case textbook figures cannot: a question whose diagram is
invented along with the question, so no book contains it, but the question's own
data determines it completely.

Two things here carry real risk and get the most attention:

  screen_code   — generated code is screened by AST before it is ever launched.
                  The sandbox would contain a hostile script anyway, but this
                  has to be the kind of check that cannot be written around, so
                  it is an allowlist over the tree and never a text search.
  _execute_plot_code
                — the process boundary: a hard timeout, a secret-stripped
                  environment, and no credential reachable from drawn code.

Runs standalone (no Django, no database, no model calls):
    python -m unittest ai_services.tests_diagram
"""
import base64
import unittest

from ai_services.core import diagram as dg


GOOD_CODE = """
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot([1, 4, 4, 1], [1, 1, 5, 1], "k-", lw=1.8)
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.grid(True, ls=":")
plt.tight_layout()
"""


class ScreenCodeTests(unittest.TestCase):
    """The static gate in front of the sandbox."""

    def test_accepts_ordinary_plotting_code(self):
        self.assertIsNone(dg.screen_code(GOOD_CODE))

    def test_accepts_numpy_and_loops(self):
        code = (
            "xs = np.linspace(0, 10, 100)\n"
            "ys = [x ** 2 for x in range(5)]\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(xs, np.sin(xs))\n"
        )
        self.assertIsNone(dg.screen_code(code))

    def test_rejects_every_import(self):
        # The namespace already holds everything a figure needs, so an import
        # is a red flag rather than a convenience.
        for code in ("import os", "import os, sys", "from os import system",
                     "from subprocess import run", "import socket"):
            self.assertIsNotNone(dg.screen_code(code), code)

    def test_rejects_filesystem_and_interpreter_builtins(self):
        for code in ("open('/etc/passwd').read()", "eval('1+1')", "exec('x=1')",
                     "compile('1', '<s>', 'eval')", "__import__('os')",
                     "globals()", "vars()", "getattr(plt, 'show')()"):
            self.assertIsNotNone(dg.screen_code(code), code)

    def test_rejects_attribute_escape_paths(self):
        # The classic route out of a restricted namespace.
        for code in ("().__class__.__bases__[0].__subclasses__()",
                     "plt.__globals__", "(1).__class__.__mro__",
                     "print.__self__.__dict__"):
            self.assertIsNotNone(dg.screen_code(code), code)

    def test_screening_is_not_defeated_by_string_tricks(self):
        # An allowlist over the AST, so obfuscating the text changes nothing:
        # the call node is still a call to a forbidden builtin.
        self.assertIsNotNone(dg.screen_code("o = open\no('x')"))
        self.assertIsNotNone(dg.screen_code("exec('import os')"))

    def test_rejects_syntactically_invalid_code(self):
        reason = dg.screen_code("fig, ax = plt.subplots(")
        self.assertIsNotNone(reason)
        self.assertIn("syntax", reason)

    def test_rejects_empty_and_oversized_code(self):
        self.assertIsNotNone(dg.screen_code(""))
        self.assertIsNotNone(dg.screen_code("   \n  "))
        self.assertIsNotNone(dg.screen_code("x = 1\n" * 5000))

    def test_ordinary_variable_names_are_not_rejected(self):
        # A blocklist over text would trip on a variable called `open_price`.
        self.assertIsNone(dg.screen_code("open_price = 5\nfig, ax = plt.subplots()"))


class CleanCodeTests(unittest.TestCase):
    def test_strips_markdown_fences(self):
        cleaned = dg._clean_code("```python\nfig, ax = plt.subplots()\n```")
        self.assertEqual(cleaned, "fig, ax = plt.subplots()")

    def test_strips_bare_fences(self):
        self.assertEqual(dg._clean_code("```\nx = 1\n```"), "x = 1")

    def test_removes_plt_show(self):
        # Never wanted: the sandbox collects the figure itself.
        cleaned = dg._clean_code("fig, ax = plt.subplots()\nplt.show()")
        self.assertNotIn("plt.show", cleaned)
        self.assertIn("plt.subplots", cleaned)

    def test_passes_plain_code_through(self):
        self.assertEqual(dg._clean_code("x = 1\ny = 2"), "x = 1\ny = 2")

    def test_unwraps_the_brace_wrapper_the_model_actually_emits(self):
        # Observed against the live model: gpt-oss-120b returns perfectly good
        # plotting code wrapped in braces even with json_mode off. Two
        # characters made every diagram fail with "invalid syntax (line 2)".
        wrapped = "{\nplt.figure()\nplt.plot([1, 4], [1, 5])\nplt.xlim(0, 6)\n}"
        cleaned = dg._clean_code(wrapped)
        self.assertFalse(cleaned.startswith("{"))
        self.assertIsNone(dg.screen_code(cleaned))
        self.assertIn("plt.plot", cleaned)

    def test_unwrapping_leaves_a_genuine_dict_alone(self):
        # The braces come off ONLY when what is inside them parses as Python,
        # so a real dict literal cannot be silently turned into something else.
        text = "{'a': 1, 'b': 2}"
        self.assertEqual(dg._unwrap_braces(text), text)

    def test_unwrapping_ignores_unbalanced_or_empty_braces(self):
        self.assertEqual(dg._unwrap_braces("{"), "{")
        self.assertEqual(dg._unwrap_braces("{}"), "{}")
        self.assertEqual(dg._unwrap_braces("plt.plot([1,2])"), "plt.plot([1,2])")

    def test_handles_braces_inside_a_fence(self):
        cleaned = dg._clean_code("```python\n{\nplt.figure()\n}\n```")
        self.assertEqual(cleaned, "plt.figure()")


class ExecutePlotCodeTests(unittest.TestCase):
    """The process boundary."""

    def test_draws_a_figure_and_returns_a_png(self):
        result = dg._execute_plot_code(GOOD_CODE)
        self.assertTrue(result.get("success"), result.get("error"))
        graphs = result.get("graphs") or []
        self.assertEqual(len(graphs), 1)
        self.assertTrue(graphs[0].startswith("data:image/png;base64,"))
        png = base64.b64decode(graphs[0].split(",", 1)[1])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_the_figure_is_not_blank(self):
        result = dg._execute_plot_code(GOOD_CODE)
        png = base64.b64decode(result["graphs"][0].split(",", 1)[1])
        self.assertGreater(dg._ink_fraction(png), dg._MIN_INK_FRACTION)

    def test_code_that_draws_nothing_returns_no_figure(self):
        result = dg._execute_plot_code("x = 1 + 1")
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("graphs"), [])

    def test_an_exception_is_reported_not_raised(self):
        result = dg._execute_plot_code("raise ValueError('boom')")
        self.assertFalse(result.get("success"))
        self.assertIn("boom", result.get("error") or "")

    def test_no_credential_is_reachable_from_drawn_code(self):
        # The child runs with a secret-stripped environment. If this ever
        # regresses, generated code could read provider keys out of os.environ.
        probe = (
            "import os\n"
            "leaked = [k for k in os.environ "
            "if 'GROQ' in k or 'GEMINI' in k or 'DATABASE' in k or 'REDIS' in k "
            "or 'SECRET' in k or 'AWS' in k or 'R2' in k]\n"
            "fig, ax = plt.subplots()\n"
            "ax.text(0.5, 0.5, str(len(leaked)))\n"
            "print('LEAKED=' + str(len(leaked)))\n"
        )
        # screen_code refuses this outright — that is the first line of defence.
        self.assertIsNotNone(dg.screen_code(probe))
        # And even run directly, the environment carries nothing to find.
        result = dg._execute_plot_code(probe)
        if result.get("success"):
            self.assertIn("LEAKED=0", result.get("stdout") or "")

    def test_rendering_can_be_disabled(self):
        import os
        os.environ["DIAGRAM_EXEC_ENABLED"] = "false"
        try:
            result = dg._execute_plot_code(GOOD_CODE)
        finally:
            os.environ.pop("DIAGRAM_EXEC_ENABLED", None)
        self.assertFalse(result.get("success"))
        self.assertIn("disabled", result.get("error") or "")

    def test_only_one_figure_comes_back(self):
        # One question gets one figure; extras are dropped rather than guessed
        # between.
        code = (
            "for i in range(4):\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot([0, 1], [0, i + 1])\n"
        )
        result = dg._execute_plot_code(code)
        self.assertTrue(result.get("success"), result.get("error"))
        self.assertLessEqual(len(result.get("graphs") or []), 1)


class InkFractionTests(unittest.TestCase):
    def test_a_blank_png_reads_as_no_ink(self):
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.new("RGB", (80, 80), "white").save(buf, format="PNG")
        self.assertLess(dg._ink_fraction(buf.getvalue()), dg._MIN_INK_FRACTION)

    def test_a_drawn_png_reads_as_ink(self):
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.new("RGB", (80, 80), "black").save(buf, format="PNG")
        self.assertGreater(dg._ink_fraction(buf.getvalue()), 0.9)

    def test_unreadable_bytes_are_zero_not_an_exception(self):
        self.assertEqual(dg._ink_fraction(b"not a png"), 0.0)


class RenderDiagramTests(unittest.TestCase):
    """The public entry point, with the model call stubbed."""

    def _with_llm(self, responses):
        """Patch get_llm so no network call happens; returns the call log."""
        calls = []

        class _FakeLLM:
            def complete(self, system_prompt, user_prompt, model, json_mode):
                calls.append({"system": system_prompt, "user": user_prompt})
                index = min(len(calls) - 1, len(responses) - 1)
                return {"content": responses[index]}

        import ai_services.core.llm_client as llm_module
        original = llm_module.get_llm
        llm_module.get_llm = lambda: _FakeLLM()
        return calls, (lambda: setattr(llm_module, "get_llm", original))

    def test_rejects_an_empty_or_oversized_spec(self):
        self.assertFalse(dg.render_diagram("")["success"])
        self.assertFalse(dg.render_diagram("x" * 5000)["success"])

    def test_draws_the_figure_a_spec_describes(self):
        calls, restore = self._with_llm([GOOD_CODE])
        try:
            result = dg.render_diagram(
                "right-angled triangle A(1,1) B(4,1) C(4,5)",
                subject="Mathematics", class_name="Class 10", board="cbse",
            )
        finally:
            restore()
        self.assertTrue(result["success"], result.get("error"))
        self.assertTrue(result["image_base64"].startswith("data:image/png;base64,"))
        self.assertEqual(result["attempts"], 1)
        # Curriculum context reaches the prompt, so the figure is drawn at the
        # right level and in the right notation.
        self.assertIn("CBSE", calls[0]["user"])
        self.assertIn("Class 10", calls[0]["user"])
        self.assertIn("Mathematics", calls[0]["user"])

    def test_retries_once_with_the_failure_fed_back(self):
        # An unguided retry of a deterministic model reproduces the same
        # failure, so the reason has to go back in the prompt.
        calls, restore = self._with_llm(["import os\nos.system('x')", GOOD_CODE])
        try:
            result = dg.render_diagram("a triangle")
        finally:
            restore()
        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual(result["attempts"], 2)
        self.assertIn("previous attempt was rejected", calls[1]["user"])
        self.assertIn("import", calls[1]["user"])

    def test_gives_up_after_the_attempt_limit(self):
        calls, restore = self._with_llm(["import os"])
        try:
            result = dg.render_diagram("a triangle")
        finally:
            restore()
        self.assertFalse(result["success"])
        self.assertEqual(len(calls), dg._MAX_ATTEMPTS)
        self.assertIsNone(result["image_base64"])

    def test_a_blank_figure_is_treated_as_a_failure(self):
        # Code that runs but draws nothing would otherwise reach a student as
        # an empty white box on their exam paper.
        calls, restore = self._with_llm(["fig, ax = plt.subplots()\nax.axis('off')"])
        try:
            result = dg.render_diagram("a triangle")
        finally:
            restore()
        self.assertFalse(result["success"])
        self.assertIn("blank", (result.get("error") or "").lower())

    def test_code_producing_no_figure_is_a_failure(self):
        calls, restore = self._with_llm(["x = 1 + 1"])
        try:
            result = dg.render_diagram("a triangle")
        finally:
            restore()
        self.assertFalse(result["success"])
        self.assertIn("no figure", (result.get("error") or "").lower())

    def test_a_model_outage_is_reported_not_raised(self):
        class _BrokenLLM:
            def complete(self, **_kwargs):
                raise RuntimeError("provider unreachable")

        import ai_services.core.llm_client as llm_module
        original = llm_module.get_llm
        llm_module.get_llm = lambda: _BrokenLLM()
        try:
            result = dg.render_diagram("a triangle")
        finally:
            llm_module.get_llm = original
        self.assertFalse(result["success"])

    def test_the_system_prompt_forbids_drawing_the_answer(self):
        # The figure is the question's given information, not its solution.
        self.assertIn("never annotate the answer", dg._SYSTEM_PROMPT.lower())
        self.assertIn("no import", dg._SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
