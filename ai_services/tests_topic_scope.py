"""/learning/topic-scope: is a question asked in a topic's chat about that topic? LLM mocked."""

from unittest.mock import patch
from django.test import TestCase

from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache
from ai_services.views.learning import build_scope_prompt, parse_scope


class _FakeLLM:
    def __init__(self, sink, content=None, error=None):
        self._sink, self._content, self._error = sink, content, error

    def complete(self, system_prompt, user_prompt, **kwargs):
        if self._error:
            raise RuntimeError(self._error)
        self._sink["user_prompt"] = user_prompt
        return {"content": self._content, "model": "mock", "latency_ms": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class ParseScopeTests(TestCase):
    def test_reads_true_and_false(self):
        self.assertTrue(parse_scope({"on_topic": True}))
        self.assertFalse(parse_scope({"on_topic": False}))
        self.assertFalse(parse_scope({"on_topic": "false"}))

    def test_no_clear_answer(self):
        self.assertIsNone(parse_scope({}))
        self.assertIsNone(parse_scope({"on_topic": "maybe"}))
        self.assertIsNone(parse_scope("not json"))


class BuildScopePromptTests(TestCase):
    def test_lists_the_topic_its_subtopics_and_recent_messages(self):
        prompt = build_scope_prompt("Science", "Magnetic Effects", "Magnetic Field and Field Lines",
                                    ["Bar magnet field", "Compass"], [{"role": "student", "content": "What is a field line?"}], "why?")
        for part in ("Topic: Magnetic Field and Field Lines", "Subtopics: Bar magnet field; Compass",
                     "- student: What is a field line?", "New message from the student: why?", "follow-up"):
            self.assertIn(part, prompt)


class TopicScopeViewTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, data, content=None, error=None):
        sink = {}
        with patch("ai_services.views.learning.get_llm", return_value=_FakeLLM(sink, content, error)):
            resp = self.client.post("/learning/topic-scope", data=data, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, sink

    BODY = {"topic": "Magnetic Field and Field Lines", "chapter": "Magnetic Effects of Electric Current", "subject": "Science",
            "subtopics": ["Bar magnet field"], "history": [{"role": "tutor", "content": "Field lines show direction."}],
            "message": "Who won the cricket world cup?"}

    def test_off_topic(self):
        resp, sink = self._post(self.BODY, {"on_topic": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"on_topic": False})
        self.assertIn("Who won the cricket world cup?", sink["user_prompt"])
        self.assertIn("- tutor: Field lines show direction.", sink["user_prompt"])

    def test_on_topic(self):
        resp, _ = self._post({**self.BODY, "message": "Why do field lines never cross?"}, {"on_topic": True})
        self.assertEqual(resp.json(), {"on_topic": True})

    def test_lets_the_question_through_when_the_model_is_unclear(self):
        resp, _ = self._post(self.BODY, {"something": "else"})
        self.assertEqual(resp.json(), {"on_topic": True})

    def test_errors(self):
        resp, _ = self._post({**self.BODY, "message": " "}, {"on_topic": True})
        self.assertEqual(resp.status_code, 400)
        resp, _ = self._post(self.BODY, error="provider down")
        self.assertEqual(resp.status_code, 502)
