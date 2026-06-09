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


class _FakeLLM:
    """Captures the system prompt the view builds; returns a canned completion."""
    def __init__(self, sink):
        self._sink = sink

    def complete(self, system_prompt, user_prompt, **kwargs):
        self._sink["system_prompt"] = system_prompt
        self._sink["user_prompt"] = user_prompt
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

    def _generate(self, vertical_header=None):
        sink = {}
        headers = {"HTTP_X_API_KEY": self.API_KEY}
        if vertical_header:
            headers["HTTP_X_VERTICAL"] = vertical_header
        with patch("ai_services.views.bridge.get_llm", return_value=_FakeLLM(sink)):
            resp = self.client.post(
                "/content/generate",
                data={"topicName": "Photosynthesis", "subjectName": "Biology",
                      "chapterName": "Life Processes", "contentType": "lesson"},
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
