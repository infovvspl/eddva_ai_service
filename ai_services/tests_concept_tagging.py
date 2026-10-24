"""
Concept tagging for /test/generate/, used by the school learning path to find
which concepts inside a topic a student gets wrong. The LLM call is mocked.
"""

from unittest.mock import patch
from django.test import TestCase

from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache
from ai_services.views.test import _clean_concepts, parse_ai_result


class _FakeLLM:
    def __init__(self, sink, content=None):
        self._sink = sink
        self._content = content if content is not None else {"questions": []}

    def complete(self, system_prompt, user_prompt, **kwargs):
        self._sink["user_prompt"] = user_prompt
        return {"content": self._content, "model": "mock", "latency_ms": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class ConceptTaggingPromptTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _generate(self, extra=None, content=None):
        sink = {}
        data = {"topic": "Quadratic Equations", "subject": "Mathematics", "chapter": "Quadratic Equations",
                "difficulty": "medium", "type": "mcq", "num_questions": 4}
        data.update(extra or {})
        with patch("ai_services.views.test.get_llm", return_value=_FakeLLM(sink, content)), \
             patch("ai_services.views.test.question_bank.save"):
            resp = self.client.post("/test/generate/", data=data, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, sink

    def test_concepts_are_listed_and_required_per_question(self):
        resp, sink = self._generate({"concepts": ["Discriminant", "Nature of roots"]})
        self.assertEqual(resp.status_code, 200)
        prompt = sink["user_prompt"]
        self.assertIn("CONCEPT TAGGING (MANDATORY)", prompt)
        self.assertIn("  - Discriminant\n  - Nature of roots", prompt)
        self.assertIn("'concept' (copied exactly from the CONCEPT TAGGING list)", prompt)

    def test_prompt_is_unchanged_without_concepts(self):
        resp, sink = self._generate()
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("CONCEPT TAGGING", sink["user_prompt"])
        self.assertNotIn("'concept'", sink["user_prompt"])

    def test_concept_and_subtopic_reach_the_caller(self):
        content = {"questions": [{"question": "What is b^2 - 4ac called?", "options": ["Discriminant", "Root", "Vertex", "Axis"],
                                  "answer": "A", "explanation": "By definition.", "concept": "Discriminant",
                                  "subtopic": "discriminant definition"}]}
        resp, _ = self._generate({"concepts": ["Discriminant"]}, content)
        q = resp.json()["questions"][0]
        self.assertEqual(q["concept"], "Discriminant")
        self.assertEqual(q["subtopic"], "discriminant definition")
        self.assertNotIn("answerDefaulted", q)


class ParseAndCleanTests(TestCase):
    def test_flags_a_guessed_answer_key(self):
        parsed = parse_ai_result({"content": {"questions": [
            {"question": "Q1?", "options": ["a", "b", "c", "d"], "answer": "the second one"},
            {"question": "Q2?", "options": ["a", "b", "c", "d"], "answer": "C"},
        ]}}, "t", "medium", "mcq")["questions"]
        self.assertEqual(parsed[0]["answer"], "A")
        self.assertTrue(parsed[0]["answerDefaulted"])
        self.assertNotIn("answerDefaulted", parsed[1])

    def test_clean_concepts_strips_dedupes_and_caps(self):
        raw = ["  Roots\n of equations ", "roots of equations", 'Say "hi"', "", None] + [f"C{i}" for i in range(20)]
        cleaned = _clean_concepts(raw)
        self.assertEqual(cleaned[0], "Roots of equations")
        self.assertEqual(cleaned[1], "Say hi")
        self.assertEqual(len(cleaned), 12)
        self.assertTrue(all("\n" not in c for c in cleaned))

    def test_clean_concepts_ignores_non_lists(self):
        self.assertEqual(_clean_concepts("Discriminant"), [])
        self.assertEqual(_clean_concepts(None), [])
