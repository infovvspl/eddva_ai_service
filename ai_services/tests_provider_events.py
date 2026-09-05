"""
Unit tests for P0-5 provider-event telemetry + P1-6 request context.
Pure logic (no DB) — SimpleTestCase so CI runs them without a test database.
"""
import time
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core import provider_events, request_context


class RequestContextTests(SimpleTestCase):
    def tearDown(self):
        request_context.clear()

    def test_set_get_clear(self):
        request_context.set_context(
            request_id="r1", user_id="u1", user_role="teacher",
            institute_id="i1", vertical="school",
        )
        self.assertEqual(request_context.get("request_id"), "r1")
        self.assertEqual(request_context.get("user_id"), "u1")
        self.assertEqual(request_context.get("user_role"), "teacher")
        request_context.clear()
        self.assertIsNone(request_context.get("user_id"))
        self.assertEqual(request_context.get("missing", "d"), "d")


class ProviderEventTests(SimpleTestCase):
    def tearDown(self):
        request_context.clear()

    def test_key_fingerprint_is_short_and_deterministic(self):
        a = provider_events.key_fingerprint("secret-key-123")
        b = provider_events.key_fingerprint("secret-key-123")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertNotIn("secret", a)              # never leaks the key
        self.assertEqual(provider_events.key_fingerprint(""), "")

    def test_invalid_event_type_is_dropped(self):
        with mock.patch.object(provider_events.threading, "Thread") as T:
            provider_events.emit(event_type="not_valid")
            T.assert_not_called()

    def test_valid_event_posts_expected_payload(self):
        captured = {}

        def fake_post(payload):
            captured.update(payload)

        with mock.patch.dict("os.environ", {"NESTJS_INTERNAL_URL": "http://x", "INTERNAL_API_KEY": "k"}), \
             mock.patch.object(provider_events, "_post", fake_post):
            request_context.set_context(request_id="rq", institute_id="inst")
            provider_events.emit(
                event_type="429", provider="groq", model="openai/gpt-oss-120b",
                status_code=429, attempt_number=2, key_hash="deadbeef",
            )
            time.sleep(0.2)  # let the daemon thread run

        self.assertEqual(captured.get("eventType"), "429")
        self.assertEqual(captured.get("provider"), "groq")
        self.assertEqual(captured.get("statusCode"), 429)
        self.assertEqual(captured.get("attemptNumber"), 2)
        # pulled from request context
        self.assertEqual(captured.get("requestId"), "rq")
        self.assertEqual(captured.get("instituteId"), "inst")
