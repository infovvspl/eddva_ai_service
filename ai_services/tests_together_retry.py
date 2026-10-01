"""A rate limit must not reach the student as a 502.

Together's limits are dynamic - set per organization and per model, and raised
by sustained successful traffic - so a new account has a low ceiling and a burst
reads as abuse. Measured on 2026-09-30: a 170-call suite fired back-to-back
collapsed into 429s and lost 65 cases; the same calls paced 3s apart completed
with zero failures. With fallback disabled, every one of those 429s would have
been a 502 for a student.

The retry is deliberately narrow, because retry amplification against a provider
that is already rate-limiting you makes the rate limit worse:
  * only rate_limited / server_error / network are retried;
  * timeouts are NOT - the budget is already spent and a human is waiting;
  * 4xx other than 429 are NOT - the same request fails the same way;
  * the provider's own Retry-After wins over any backoff we would guess;
  * the whole sequence stays inside the caller's timeout.
"""
import os
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.routing.errors import (
    NonRetryableProviderError,
    ProviderConfigError,
    RetryableProviderError,
    error_for_status,
)
from ai_services.core.routing.providers import TogetherAdapter

OK = (200, "answer", {"prompt_tokens": 1, "completion_tokens": 2}, "m", "stop")


def rate_limited(retry_after_ms=None):
    return RetryableProviderError("Together 429", provider="together", model="m",
                                  status_code=429, kind="rate_limited",
                                  retry_after_ms=retry_after_ms)


class _Adapter(TogetherAdapter):
    """Counts attempts and never really sleeps."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.slept = []

    def _send(self, *a, **kw):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def run(script, timeout=60.0, attempts="2"):
    a = _Adapter(script)
    with patch.dict(os.environ, {"TOGETHER_RETRY_ATTEMPTS": attempts}), \
         patch.object(TogetherAdapter, "_post_completion", _Adapter._send), \
         patch.object(TogetherAdapter, "_stream_completion", _Adapter._send), \
         patch("time.sleep", side_effect=lambda s: a.slept.append(round(s, 2))):
        out = a._request_with_retry({}, "k", "m", timeout, "", False)
    return out, a


class RetryOnRateLimitTests(SimpleTestCase):
    def test_a_429_is_retried_and_can_succeed(self):
        out, a = run([rate_limited(), OK])
        self.assertEqual(out, OK)
        self.assertEqual(a.calls, 2)
        self.assertEqual(len(a.slept), 1)

    def test_retries_are_bounded(self):
        with self.assertRaises(RetryableProviderError):
            run([rate_limited(), rate_limited(), rate_limited(), rate_limited()])

    def test_attempts_can_be_disabled(self):
        a = None
        with self.assertRaises(RetryableProviderError):
            _, a = run([rate_limited(), OK], attempts="0")

    def test_a_server_error_is_retried(self):
        out, a = run([error_for_status(503, "down", provider="together", model="m"), OK])
        self.assertEqual(out, OK)
        self.assertEqual(a.calls, 2)

    def test_a_network_error_is_retried(self):
        net = RetryableProviderError("net", provider="together", model="m", kind="network")
        out, a = run([net, OK])
        self.assertEqual(a.calls, 2)


class WhatMustNotBeRetriedTests(SimpleTestCase):
    def test_a_timeout_is_not_retried(self):
        # The budget is already spent; retrying doubles the wait a human sits through.
        timeout = RetryableProviderError("timed out", provider="together", model="m",
                                         kind="timeout")
        with self.assertRaises(RetryableProviderError) as ctx:
            run([timeout, OK])
        self.assertEqual(ctx.exception.kind, "timeout")

    def test_a_bad_request_is_not_retried(self):
        bad = error_for_status(400, "nope", provider="together", model="m")
        with self.assertRaises(NonRetryableProviderError):
            run([bad, OK])

    def test_auth_and_missing_model_are_not_retried(self):
        for status in (401, 403, 404):
            with self.assertRaises(ProviderConfigError):
                run([error_for_status(status, "x", provider="together", model="m"), OK])


class BackoffTests(SimpleTestCase):
    def test_the_providers_retry_after_wins(self):
        _, a = run([rate_limited(retry_after_ms=4500), OK])
        self.assertEqual(a.slept, [4.5])

    def test_backoff_grows_when_the_provider_says_nothing(self):
        _, a = run([rate_limited(), rate_limited(), OK])
        self.assertEqual(len(a.slept), 2)
        self.assertGreater(a.slept[1], a.slept[0])
        self.assertLessEqual(a.slept[1], 30.0)

    def test_a_retry_is_not_started_without_budget_for_it(self):
        # 2s left, provider asks for 10s: raise now rather than blow the deadline.
        with self.assertRaises(RetryableProviderError):
            run([rate_limited(retry_after_ms=10000), OK], timeout=2.0)

    def test_retry_after_is_capped(self):
        _, a = run([rate_limited(retry_after_ms=600000), OK], timeout=600.0)
        self.assertEqual(a.slept, [30.0])


class ConfigTests(SimpleTestCase):
    def test_attempts_are_clamped_and_survive_junk(self):
        for value, expected in (("0", 0), ("2", 2), ("99", 5), ("-1", 0), ("abc", 2), ("", 2)):
            with patch.dict(os.environ, {"TOGETHER_RETRY_ATTEMPTS": value}):
                self.assertEqual(TogetherAdapter._retry_attempts(), expected, value)
