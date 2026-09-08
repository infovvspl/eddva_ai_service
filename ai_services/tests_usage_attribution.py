"""
Regression tests for P1-6 user attribution in usage_logger.log_usage.

Root cause these lock down: views/bridge.py derives user_id from the request body
with an `or ''` tail, so worker/background calls passed user_id=''. The old
fallback was `if user_id is None`, which '' does not satisfy, so the authenticated
request context was ignored and the empty string became NULL in ai_usage_events —
while user_role/request_id (never passed, therefore None) fell back correctly.

Plain unittest.TestCase on purpose: no Django settings and no database are needed,
so these run standalone AND under the CI test runner.
"""
import unittest
from unittest import mock

from ai_services.core import request_context, usage_logger


class _InlineThread:
    """Runs the target inline so log_usage's fire-and-forget send is observable."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class _InlineThreading:
    Thread = _InlineThread


def _log(**overrides):
    """Call log_usage with the send captured; returns the kwargs it would ship."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    payload = dict(
        institute_id="c259cd4e-b018-45e2-8e46-52a497ca49a1",
        institute_type="school",
        feature_id="ai_lecture_notes",
        feature_category="teacher",
        model_used="openai/gpt-oss-120b",
        tokens_input=0,          # keeps the Redis budget path out of the test
        tokens_output=0,
    )
    payload.update(overrides)

    with mock.patch.object(usage_logger, "log_ai_usage_sync", _capture), \
            mock.patch.object(usage_logger, "threading", _InlineThreading):
        usage_logger.log_usage(**payload)
    return captured


AUTH_USER = "3d0eabde-0695-4935-9dd9-da21ae1dced8"
AUTH_ROLE = "TEACHER,INSTITUTE_ADMIN"
AUTH_REQ = "354b4245-96c7-4425-bb18-6898fec9461d"


class UsageAttributionTests(unittest.TestCase):
    def tearDown(self):
        request_context.clear()

    def _authenticate(self):
        request_context.set_context(
            request_id=AUTH_REQ, user_id=AUTH_USER, user_role=AUTH_ROLE,
            institute_id="c259cd4e-b018-45e2-8e46-52a497ca49a1", vertical="school",
        )

    # (a) the exact production defect: body userId='' from views/bridge.py
    def test_empty_body_user_id_falls_back_to_authenticated_context(self):
        self._authenticate()
        sent = _log(user_id="")
        self.assertEqual(sent["user_id"], AUTH_USER)

    def test_whitespace_body_user_id_falls_back_to_authenticated_context(self):
        self._authenticate()
        sent = _log(user_id="   ")
        self.assertEqual(sent["user_id"], AUTH_USER)

    # (b) caller omits user_id entirely
    def test_missing_body_user_id_falls_back_to_authenticated_context(self):
        self._authenticate()
        sent = _log()
        self.assertEqual(sent["user_id"], AUTH_USER)

    # (c) authenticated identity is not overridable by a client-supplied body value
    def test_context_user_id_wins_over_conflicting_body_user_id(self):
        self._authenticate()
        sent = _log(user_id="11111111-2222-3333-4444-555555555555")
        self.assertEqual(sent["user_id"], AUTH_USER)

    # (d) + (e) role and request id keep propagating from the context
    def test_user_role_propagates_from_context(self):
        self._authenticate()
        self.assertEqual(_log(user_id="")["user_role"], AUTH_ROLE)

    def test_request_id_propagates_from_context(self):
        self._authenticate()
        self.assertEqual(_log(user_id="")["request_id"], AUTH_REQ)

    def test_context_role_and_request_id_win_over_caller_values(self):
        self._authenticate()
        sent = _log(user_role="STUDENT", request_id="00000000-0000-0000-0000-000000000000")
        self.assertEqual(sent["user_role"], AUTH_ROLE)
        self.assertEqual(sent["request_id"], AUTH_REQ)

    # (f) no authenticated context — legacy/unauthenticated callers unchanged
    def test_without_context_body_user_id_is_still_used(self):
        sent = _log(user_id="9f8e7d6c-1111-2222-3333-444455556666")
        self.assertEqual(sent["user_id"], "9f8e7d6c-1111-2222-3333-444455556666")

    def test_without_context_empty_user_id_becomes_none_not_empty_string(self):
        sent = _log(user_id="")
        self.assertIsNone(sent["user_id"])
        self.assertIsNone(sent["user_role"])
        self.assertIsNone(sent["request_id"])

    def test_without_context_missing_user_id_is_none(self):
        self.assertIsNone(_log()["user_id"])

    # the non-attribution payload must be untouched by this change
    def test_other_fields_unchanged(self):
        self._authenticate()
        sent = _log(feature_id="lecture_transcription", model_used="whisper-large-v3-turbo")
        self.assertEqual(sent["feature_id"], "lecture_transcription")
        self.assertEqual(sent["model_used"], "whisper-large-v3-turbo")
        self.assertEqual(sent["institute_type"], "school")
        self.assertTrue(sent["success"])


class WebhookPayloadTests(unittest.TestCase):
    """End-to-end: the JSON actually posted to the usage webhook carries the
    authenticated identity, reproducing the exact production call shape
    (views/bridge.py passes user_id='' for worker-initiated STT/notes calls)."""

    def tearDown(self):
        request_context.clear()

    def test_posted_payload_carries_authenticated_user(self):
        request_context.set_context(
            request_id=AUTH_REQ, user_id=AUTH_USER, user_role=AUTH_ROLE,
            institute_id="c259cd4e-b018-45e2-8e46-52a497ca49a1", vertical="school",
        )
        posted = {}

        class _Resp:
            def raise_for_status(self):
                return None

        def _post(url, json=None, headers=None, timeout=None):
            posted.update(json or {})
            return _Resp()

        with mock.patch.object(usage_logger, "threading", _InlineThreading),                 mock.patch.object(usage_logger, "httpx", mock.Mock(post=_post)),                 mock.patch.dict("os.environ", {"NESTJS_INTERNAL_URL": "http://internal"}):
            usage_logger.log_usage(
                institute_id="c259cd4e-b018-45e2-8e46-52a497ca49a1",
                institute_type="school",
                feature_id="lecture_transcription",
                feature_category="teacher",
                model_used="whisper-large-v3-turbo",
                tokens_input=0,
                tokens_output=0,
                user_id="",  # <- the production defect
            )

        self.assertEqual(posted.get("userId"), AUTH_USER)
        self.assertEqual(posted.get("userRole"), AUTH_ROLE)
        self.assertEqual(posted.get("requestId"), AUTH_REQ)


class FirstPresentTests(unittest.TestCase):
    def test_absent_values_are_skipped(self):
        self.assertIsNone(usage_logger._first_present(None, "", "   "))

    def test_first_real_value_wins_and_is_stripped(self):
        self.assertEqual(usage_logger._first_present(None, "", " abc ", "def"), "abc")

    def test_no_values(self):
        self.assertIsNone(usage_logger._first_present())


if __name__ == "__main__":
    unittest.main()
