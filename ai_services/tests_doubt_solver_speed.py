"""
Doubt resolver: scientific-solver gating, code-generation prompt shaping and
token accounting.

Observed on 2026-09-15: a teacher's "Draft with AI" for "What are whole numbers?"
(subject Mathematics) took ~15.5s. "Mathematics" was hard-mapped to type
"derivation", so the definition went through the scientific solver: Python code
generation (invalid, retried), a sandboxed subprocess run and two explanation
calls. The generated code was invalid because LLMClient prepended a prefix
ending "START YOUR RESPONSE DIRECTLY WITH '{'" to the code-generation prompt.
The usage row recorded 0+0 tokens.
"""
import asyncio
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.views.bridge import _looks_computational


class ComputationalQuestionTests(SimpleTestCase):
    def test_definition_and_explanation_questions_do_not_need_the_solver(self):
        for q in (
            "What are whole numbers? (At segment timestamp: 13:38) Lecture: Real Number",
            "What are whole numbers?",
            "Define rational numbers",
            "Explain Newton's first law of motion",
            "What is the difference between speed and velocity?",
            "Why is the sky blue? Lecture: Light 2",
            "What is photosynthesis",
        ):
            self.assertFalse(_looks_computational(q), q)

    def test_calculation_proof_and_derivation_questions_do(self):
        for q in (
            "A ball is thrown upward at 20 m/s. How high does it go?",
            "Solve 2x + 3 = 7",
            "Find the HCF of 96 and 404",
            "Prove that root 2 is irrational",
            "Calculate the molarity of 4 g NaOH in 500 ml",
            "Differentiate sin(x) with respect to x",
            "What is the value of 3/4 of 12?",
            "Derive the equation of motion v = u + at",
        ):
            self.assertTrue(_looks_computational(q), q)

    def test_lecture_context_is_ignored(self):
        self.assertFalse(_looks_computational("What is a prime? (At segment timestamp: 12:30) Lecture: Class 10 = Maths"))


class ResolverGateSourceTests(SimpleTestCase):
    def _src(self):
        return io.open(os.path.join(os.path.dirname(__file__), "views", "bridge.py"), encoding="utf-8").read()

    def test_solver_requires_a_computational_question_type(self):
        src = self._src()
        self.assertIn("if subject in _SOLVER_SUBJECTS and qtype in _SOLVER_QTYPES:", src)
        self.assertNotIn('if subject in ("physics", "chemistry", "mathematics", "math", "science"):', src)
        self.assertIn("_looks_computational(question_text)", src)

    def test_usage_row_uses_solver_tokens_and_actual_model(self):
        src = self._src()
        self.assertIn('pop("_usage", None)', src)
        self.assertIn("solve_result.get('provider_model')", src)


class _Resp(dict):
    pass


def _llm_result(content, tin=10, tout=20, model="openai/gpt-oss-120b"):
    return {"content": content, "model": model, "latency_ms": 1, "tokens_input": tin, "tokens_output": tout,
            "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}}


class SolverTests(SimpleTestCase):
    def _solver(self, fake):
        import ai_services.solver.scientific_solver as ss

        solver = ss.ScientificSolver.__new__(ss.ScientificSolver)
        solver.llm = fake
        return ss, solver

    def test_code_generation_is_sent_unshaped(self):
        seen = []

        class Fake:
            def complete(self, **kw):
                seen.append(kw)
                return _llm_result("not python {")

        ss, solver = self._solver(Fake())
        with patch.object(ss.formula_retriever, "retrieve", lambda *a, **k: []):
            out = asyncio.run(solver.solve("Solve 2x + 3 = 7", "detailed", "school"))
        self.assertEqual(len(seen), 2)                               # first try + one retry
        self.assertTrue(all(kw["legacy_prompt_shaping"] is False for kw in seen))
        self.assertTrue(all(kw["json_mode"] is False for kw in seen))
        self.assertFalse(out["success"])
        self.assertEqual(out["_usage"], {"tokens_input": 20, "tokens_output": 40, "model": "openai/gpt-oss-120b"})

    def test_tokens_are_summed_across_generation_and_explanations(self):
        class Fake:
            def complete(self, **kw):
                return _llm_result("FINAL_RESULT = 2", tin=100, tout=30, model="together:openai/gpt-oss-120b")

            def parallel_complete_many(self, tasks, model=None, json_mode=False, **kw):
                return [_llm_result({"brief": {"final_answer": "2"}}, tin=50, tout=40, model="together:openai/gpt-oss-120b"),
                        _llm_result({"detailed": {"explanation": "x=2"}}, tin=60, tout=70, model="together:openai/gpt-oss-120b")]

        ss, solver = self._solver(Fake())
        exec_ok = {"success": True, "stdout": "", "results": {"FINAL_RESULT": 2}, "error": None, "graphs": []}
        with patch.object(ss.formula_retriever, "retrieve", lambda *a, **k: []), \
             patch.object(ss.ScientificSolver, "_execute_code", lambda self, code: exec_ok):
            out = asyncio.run(solver.solve("Solve 2x + 3 = 7", "detailed", "school"))
        self.assertEqual(out["_usage"], {"tokens_input": 210, "tokens_output": 140,
                                         "model": "together:openai/gpt-oss-120b"})
        self.assertEqual(out["brief"], {"final_answer": "2"})

    def test_lone_closing_brace_from_a_wrapped_reply_is_removed(self):
        _, solver = self._solver(None)
        wrapped = "{\ndef f():\n    return 1\nFINAL_RESULT = f()\n}"
        code = solver._clean_generated_code(wrapped)
        self.assertIsNone(solver._code_is_valid_python(code))
        self.assertFalse(code.rstrip().endswith("}"))

    def test_valid_code_ending_in_a_brace_is_left_alone(self):
        _, solver = self._solver(None)
        code = 'FINAL_RESULT = {\n    "a": 1\n}'
        self.assertEqual(solver._clean_generated_code(code), code)


class GroqUnshapedPromptTests(SimpleTestCase):
    def test_groq_path_honours_legacy_prompt_shaping_false(self):
        import groq as groq_mod
        import ai_services.core.llm_client as lc
        from ai_services.core import routing

        sent = []

        class FakeCompletions:
            def create(self, **kw):
                sent.append(kw)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="print(1)"))],
                                       usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        class FakeGroq:
            def __init__(self, api_key=None, **kw):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        routing.reset_router(routing.ModelRouter(routing.load_config({})))
        self.addCleanup(routing.reset_router)
        with patch.object(groq_mod, "Groq", FakeGroq), patch.object(lc, "GROQ_API_KEYS", ["k1"]), \
             patch.object(lc, "_DISABLED_GROQ_KEYS", set()):
            lc.LLMClient().complete(system_prompt="RAW CODEGEN PROMPT", user_prompt="u",
                                    model="openai/gpt-oss-120b", json_mode=False, legacy_prompt_shaping=False)
            lc.LLMClient().complete(system_prompt="RAW CODEGEN PROMPT", user_prompt="u",
                                    model="openai/gpt-oss-120b", json_mode=False)
        self.assertEqual(sent[0]["messages"][0]["content"], "RAW CODEGEN PROMPT")
        self.assertTrue(sent[1]["messages"][0]["content"].endswith("RAW CODEGEN PROMPT"))
        self.assertNotEqual(sent[1]["messages"][0]["content"], "RAW CODEGEN PROMPT")   # default unchanged


class DoubtAnswerShapeTests(SimpleTestCase):
    """Teacher/student school doubt pages read brief.final_answer / detailed.explanation;
    LLM answers wrote brief.answer / detailed.solution, so Brief and Detailed were empty."""

    def test_llm_shape_gains_the_solver_field_names(self):
        from ai_services.views.bridge import _normalize_doubt_answer

        brief = {"answer": "Neurons carry signals.", "question_nature": "theory"}
        detailed = {"solution": "**(i) Structure**\n- dendrites", "final_answer": "Neurons transmit signals.",
                    "verification": "None", "key_concept": "none"}
        _normalize_doubt_answer(brief, detailed)
        self.assertEqual(brief["final_answer"], "Neurons carry signals.")
        self.assertEqual(detailed["explanation"], "**(i) Structure**\n- dendrites")
        self.assertEqual((detailed["verification"], detailed["key_concept"]), ("", ""))

    def test_solver_shape_gains_the_llm_field_names_and_keeps_its_own(self):
        from ai_services.views.bridge import _normalize_doubt_answer

        brief = {"final_answer": "HCF = 4"}
        detailed = {"explanation": "Apply Euclid's lemma", "verification": "4 divides both", "key_concept": "Euclid"}
        _normalize_doubt_answer(brief, detailed)
        self.assertEqual((brief["answer"], brief["final_answer"]), ("HCF = 4", "HCF = 4"))
        self.assertEqual((detailed["solution"], detailed["explanation"]), ("Apply Euclid's lemma",) * 2)
        self.assertEqual(detailed["verification"], "4 divides both")

    def test_existing_values_are_never_overwritten(self):
        from ai_services.views.bridge import _normalize_doubt_answer

        brief = {"answer": "short", "final_answer": "final"}
        detailed = {"solution": "sol", "explanation": "exp"}
        _normalize_doubt_answer(brief, detailed)
        self.assertEqual((brief["answer"], brief["final_answer"], detailed["solution"], detailed["explanation"]),
                         ("short", "final", "sol", "exp"))

    def test_resolve_doubt_calls_the_normalizer(self):
        src = io.open(os.path.join(os.path.dirname(__file__), "views", "bridge.py"), encoding="utf-8").read()
        self.assertIn("    _normalize_doubt_answer(brief_obj, detailed_obj)\n", src)
