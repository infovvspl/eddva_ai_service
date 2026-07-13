"""
End-to-end-ish test for the X-Vertical contract used by the NestJS school module.

Drives the REAL HTTP path — TenantAuthMiddleware (API-key auth + vertical
resolution) → bridge.generate_topic_content — with only the LLM call mocked, so
it runs offline and fast. Proves that `X-Vertical: school` (what the school
module sends) produces school-framed content while the default stays coaching.
"""

from unittest.mock import patch
from django.test import TestCase

from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache
from ai_services.views.bridge import (
    _has_incomplete_mcq_options,
    _normalize_generated_math_markdown,
    _normalize_pyq_exam_tags,
)


class _FakeLLM:
    """Captures the system prompt the view builds; returns a canned completion."""
    def __init__(self, sink):
        self._sink = sink

    def complete(self, system_prompt, user_prompt, **kwargs):
        self._sink["system_prompt"] = system_prompt
        self._sink["user_prompt"] = user_prompt
        self._sink["kwargs"] = kwargs
        return {"content": "# Mock content", "model": "mock", "latency_ms": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class ContentVerticalContractTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"  # matches NestJS AI_API_KEY default

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(
            name="Test Tenant", slug="test-tenant",
            api_key=self.API_KEY, vertical="coaching", is_active=True,
        )

    def _generate(self, vertical_header=None, content_type="lesson"):
        sink = {}
        headers = {"HTTP_X_API_KEY": self.API_KEY}
        if vertical_header:
            headers["HTTP_X_VERTICAL"] = vertical_header
        with patch("ai_services.views.bridge.get_llm", return_value=_FakeLLM(sink)):
            resp = self.client.post(
                "/content/generate",
                data={"topicName": "Photosynthesis", "subjectName": "Biology",
                      "chapterName": "Life Processes", "contentType": content_type},
                content_type="application/json",
                **headers,
            )
        return resp, sink

    def test_school_vertical_uses_class10_framing(self):
        resp, sink = self._generate(vertical_header="school")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "school")
        sys = sink["system_prompt"]
        self.assertIn("Class 10 Board Exam", sys)       # school-level constraint injected
        self.assertNotIn("TARGET EXAM: JEE", sys)       # no competitive framing
        self.assertNotIn("NEET", sys)

    def test_default_vertical_stays_coaching(self):
        resp, sink = self._generate(vertical_header=None)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "coaching")
        # Coaching with no exam target => no Class 10 school constraint block.
        self.assertNotIn("Class 10 Board Exam", sink["system_prompt"])

    def test_coaching_dpp_and_pyq_prompts(self):
        # Coaching DPP
        resp, sink = self._generate(vertical_header="coaching", content_type="dpp")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        self.assertIn("Section A — Single Correct MCQ (4 marks each, –1 for wrong)", user_prompt)
        self.assertIn("Section B — Integer Type Numericals", user_prompt)
        self.assertNotIn("Section A -- Multiple Choice (1 mark each)", user_prompt)

        # Coaching PYQ
        resp, sink = self._generate(vertical_header="coaching", content_type="pyq")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        self.assertIn("PYQ Practice", user_prompt)
        # Coaching PYQ must include multi-correct MCQs (a JEE-pattern feature).
        # Don't couple to the exact count — it's derived (max(1, round(q_count*0.20))).
        self.assertIn("Multi-correct MCQ (", user_prompt)
        self.assertNotIn("school Board Exam", user_prompt)

    def test_school_dpp_and_pyq_prompts(self):
        # School DPP
        resp, sink = self._generate(vertical_header="school", content_type="dpp")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        self.assertIn("Section A -- Multiple Choice (1 mark each)", user_prompt)
        self.assertNotIn("Section A — Single Correct MCQ (4 marks each, –1 for wrong)", user_prompt)

        # School PYQ
        resp, sink = self._generate(vertical_header="school", content_type="pyq")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        self.assertIn("PYQ Practice Set", user_prompt)
        self.assertIn("Do NOT generate JEE, NEET, Olympiad", user_prompt)

    def test_pyq_requires_balanced_katex_and_allows_full_solution_budget(self):
        resp, sink = self._generate(vertical_header="coaching", content_type="pyq")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("MANDATORY OUTPUT VALIDATION", sink["user_prompt"])
        self.assertIn("matching closing delimiter", sink["user_prompt"])
        self.assertIn("only inside EXAMTAG", sink["user_prompt"])
        self.assertEqual(sink["kwargs"]["max_tokens"], 8192)

    def test_generated_math_normalizer_repairs_detailed_solution_math(self):
        raw = """```markdown
## Detailed Solutions

Step 1
x = \\frac{6}{3 + \\sqrt{2}}

Step 2
x² + 2x = 3

Step 3
$y = mx + c

Step 4
Therefore, z = \\sqrt{5}
```"""
        normalized = _normalize_generated_math_markdown(raw)

        self.assertNotIn("```", normalized)
        self.assertIn("$x = \\frac{6}{3 + \\sqrt{2}}$", normalized)
        self.assertIn("$x^{2} + 2x = 3$", normalized)
        self.assertIn("$y = mx + c$", normalized)
        self.assertIn("Therefore, $z = \\sqrt{5}$", normalized)

    def test_generated_math_normalizer_converts_latex_code_fence(self):
        normalized = _normalize_generated_math_markdown(
            "## Detailed Solutions\n\n```latex\nx = \\frac{1}{2}\n```"
        )
        self.assertNotIn("```", normalized)
        self.assertIn("$$\nx = \\frac{1}{2}\n$$", normalized)

    def test_generated_math_normalizer_closes_unbalanced_delimiters(self):
        normalized = _normalize_generated_math_markdown(
            "## Detailed Solutions\n\n$x = \\sqrt{5}\n\nTherefore the value follows."
        )
        self.assertIn("$x = \\sqrt{5}$", normalized)
        self.assertEqual(len([m for m in normalized if m == '$']), 2)

    def test_generated_math_normalizer_prepends_missing_opening_delimiter(self):
        normalized = _normalize_generated_math_markdown(
            "## Detailed Solutions\n\n$f\n\n(0) = 0^2 - 3(0) + 2$"
        )
        self.assertIn("$f$", normalized)
        self.assertIn("$(0) = 0^2 - 3(0) + 2$", normalized)
        self.assertNotIn("+ 2$$", normalized)

    def test_pyq_exam_year_is_moved_into_badge_tag_only(self):
        normalized = _normalize_pyq_exam_tags(
            "Q1. CBSE Class 10 2021: Find the roots.\n"
            "Q2. Find the value. (JEE Main 2019)\n"
            "3. [EXAMTAG: NEET 2021] NEET 2021: Find the answer."
        )
        self.assertIn("1. [EXAMTAG: CBSE Class 10 2021] Find the roots.", normalized)
        self.assertIn("2. [EXAMTAG: JEE Main 2019] Find the value.", normalized)
        self.assertIn("3. [EXAMTAG: NEET 2021] Find the answer.", normalized)
        self.assertNotIn("2021: Find", normalized)

    def test_incomplete_mcq_option_detector_rejects_bare_labels(self):
        broken = """# DPP — Photosynthesis

## Section A — Single Correct MCQ

1. Where does photosynthesis occur?
A
B
C
D

2. What is chlorophyll?
A. A pigment
B. A sugar
C. A gas
D. A protein

## Answer Key
1. A
""" + ("Supporting content. " * 12)
        complete = broken.replace("A\nB\nC\nD\n", "A. Chloroplast\nB. Nucleus\nC. Ribosome\nD. Vacuole\n")

        self.assertTrue(_has_incomplete_mcq_options(broken))
        self.assertFalse(_has_incomplete_mcq_options(complete))

    def _generate_test(self, vertical_header=None, exam_target=None):
        sink = {}
        headers = {"HTTP_X_API_KEY": self.API_KEY}
        if vertical_header:
            headers["HTTP_X_VERTICAL"] = vertical_header
        data = {
            "topic": "Photosynthesis",
            "subject": "Biology",
            "chapter": "Life Processes",
            "difficulty": "medium",
            "type": "mcq",
            "num_questions": 2,
        }
        if exam_target:
            data["exam_target"] = exam_target
        with patch("ai_services.views.test.get_llm", return_value=_FakeLLM(sink)):
            resp = self.client.post(
                "/test/generate/",
                data=data,
                content_type="application/json",
                **headers,
            )
        return resp, sink

    def test_school_question_generation_defaults_to_cbse(self):
        resp, sink = self._generate_test(vertical_header="school")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        # With school vertical and no exam_target, it should default to class 10 (CBSE)
        self.assertIn("CBSE board standard", user_prompt)
        self.assertNotIn("competitive JEE Mains / NEET level", user_prompt)

    def test_coaching_question_generation_defaults_to_competitive(self):
        resp, sink = self._generate_test(vertical_header="coaching")
        self.assertEqual(resp.status_code, 200)
        user_prompt = sink["user_prompt"]
        # With coaching vertical and no exam_target, it should default to JEE/NEET competitive level
        self.assertIn("competitive JEE Mains / NEET level", user_prompt)
        self.assertNotIn("CBSE board standard", user_prompt)

    def _tutor_session(self, vertical_header=None, endpoint="/tutor/session", context=""):
        sink = {}
        headers = {"HTTP_X_API_KEY": self.API_KEY}
        if vertical_header:
            headers["HTTP_X_VERTICAL"] = vertical_header
        
        data = {
            "studentId": "test-student-id",
            "topicId": "test-topic-id",
            "context": context,
            "sessionId": "test-session-id",
            "studentMessage": "hello",
        }
        
        with patch("ai_services.views.bridge.get_llm", return_value=_FakeLLM(sink)):
            resp = self.client.post(
                endpoint,
                data=data,
                content_type="application/json",
                **headers,
            )
        return resp, sink

    @patch("ai_services.views.bridge.log_usage")
    def test_tutor_session_vertical_passing(self, mock_log_usage):
        # 1. School start session
        resp, sink = self._tutor_session(vertical_header="school", endpoint="/tutor/session", context="short context")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "school")
        kwargs = mock_log_usage.call_args[1]
        self.assertEqual(kwargs.get("institute_id"), "test-tenant")
        self.assertEqual(kwargs.get("institute_type"), "school")
        self.assertEqual(kwargs.get("feature_id"), "ai_lecture_notes")
        self.assertEqual(kwargs.get("success"), True)
        self.assertEqual(kwargs.get("user_id"), "test-student-id")

        mock_log_usage.reset_mock()

        # 2. Coaching start session
        resp, sink = self._tutor_session(vertical_header="coaching", endpoint="/tutor/session", context="short context")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "coaching")
        kwargs = mock_log_usage.call_args[1]
        self.assertEqual(kwargs.get("institute_id"), "test-tenant")
        self.assertEqual(kwargs.get("institute_type"), "coaching")
        self.assertEqual(kwargs.get("feature_id"), "ai_lecture_notes")
        self.assertEqual(kwargs.get("success"), True)
        self.assertEqual(kwargs.get("user_id"), "test-student-id")

    @patch("ai_services.views.bridge.log_usage")
    def test_tutor_continue_vertical_passing(self, mock_log_usage):
        # 1. School continue session
        resp, sink = self._tutor_session(vertical_header="school", endpoint="/tutor/continue")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "school")
        kwargs = mock_log_usage.call_args[1]
        self.assertEqual(kwargs.get("institute_id"), "test-tenant")
        self.assertEqual(kwargs.get("institute_type"), "school")
        self.assertEqual(kwargs.get("feature_id"), "doubt_resolver")
        self.assertEqual(kwargs.get("success"), True)
        self.assertEqual(kwargs.get("user_id"), "test-student-id")

        mock_log_usage.reset_mock()

        # 2. Coaching continue session
        resp, sink = self._tutor_session(vertical_header="coaching", endpoint="/tutor/continue")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Vertical"], "coaching")
        kwargs = mock_log_usage.call_args[1]
        self.assertEqual(kwargs.get("institute_id"), "test-tenant")
        self.assertEqual(kwargs.get("institute_type"), "coaching")
        self.assertEqual(kwargs.get("feature_id"), "doubt_resolver")
        self.assertEqual(kwargs.get("success"), True)
        self.assertEqual(kwargs.get("user_id"), "test-student-id")
