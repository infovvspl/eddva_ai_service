"""Admission control for endpoints that generate their own content.

These endpoints previously logged what a tenant spent while enforcing nothing.
The tests that matter most here are the fail-open ones: a bug in metering must
degrade to "allowed", never to a 403 for a paying school mid-lesson.
"""
from unittest.mock import patch

from django.test import TestCase

from ai_services.core.rate_limiter import UsageLimiter, get_shared_limiter
from ai_services.models import Institute
from ai_services.views.base import metered, tenant_gate


class _Req:
    """Minimal stand-in for the DRF request TenantAuthMiddleware decorates."""

    def __init__(self, institute=None, institute_id="school-a", vertical="school"):
        self.institute = institute
        self.institute_id = institute_id
        self.vertical = vertical


def _institute(**kwargs):
    defaults = dict(
        name="Test School", slug="test-school", api_key="k-test-metering",
        vertical="school", plan="free",
    )
    defaults.update(kwargs)
    return Institute.objects.create(**defaults)


class TenantGateTests(TestCase):
    def setUp(self):
        get_shared_limiter()._memory.clear()

    def test_allows_by_default(self):
        """A tenant with no feature config must not be blocked.

        features_enabled defaults to {} and every cap is generous, so a freshly
        onboarded school has to work before anyone configures anything.
        """
        inst = _institute()
        self.assertIsNone(tenant_gate(_Req(inst), "content_generate"))

    def test_allows_when_no_institute_row(self):
        """Service-account and unmatched traffic must still flow."""
        self.assertIsNone(tenant_gate(_Req(None), "content_generate"))

    def test_denies_403_when_feature_off_for_plan(self):
        inst = _institute(features_enabled={"content_generate": False})
        resp = tenant_gate(_Req(inst), "content_generate")
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("not enabled", resp.data["error"])

    def test_other_features_unaffected_when_one_is_off(self):
        inst = _institute(features_enabled={"content_generate": False})
        self.assertIsNone(tenant_gate(_Req(inst), "doubt_resolver"))

    def test_denies_429_over_daily_hard_cap(self):
        inst = _institute(daily_soft_cap=100, daily_hard_cap=200)
        get_shared_limiter().record_usage("school-a", 250)
        resp = tenant_gate(_Req(inst), "content_generate")
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.data["limit"], 200)

    def test_allows_under_hard_cap(self):
        inst = _institute(daily_soft_cap=100, daily_hard_cap=200)
        get_shared_limiter().record_usage("school-a", 150)  # over soft, under hard
        self.assertIsNone(tenant_gate(_Req(inst), "content_generate"))

    def test_budget_is_per_tenant(self):
        inst = _institute(daily_hard_cap=200)
        get_shared_limiter().record_usage("school-b", 999)
        self.assertIsNone(tenant_gate(_Req(inst, institute_id="school-a"), "content_generate"))

    def test_fails_open_when_limiter_raises(self):
        """An outage in metering must not read as "denied"."""
        inst = _institute()
        with patch.object(UsageLimiter, "check_budget", side_effect=RuntimeError("redis down")):
            self.assertIsNone(tenant_gate(_Req(inst), "content_generate"))


class MeteredDecoratorTests(TestCase):
    def setUp(self):
        get_shared_limiter()._memory.clear()

    def test_calls_view_when_allowed(self):
        inst = _institute()
        calls = []

        @metered("content_generate")
        def view(request):
            calls.append(1)
            return "generated"

        self.assertEqual(view(_Req(inst)), "generated")
        self.assertEqual(len(calls), 1)

    def test_view_not_called_when_denied(self):
        inst = _institute(features_enabled={"content_generate": False})
        calls = []

        @metered("content_generate")
        def view(request):
            calls.append(1)
            return "generated"

        resp = view(_Req(inst))
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(calls, [], "the view must not run once denied")

    def test_slot_released_when_view_raises(self):
        """A leaked slot would shrink the tenant's concurrency until the lease expired."""
        inst = _institute()
        released = []

        @metered("content_generate")
        def view(request):
            raise ValueError("generation blew up")

        with patch.object(UsageLimiter, "acquire_concurrency_slot", return_value=True), \
             patch.object(UsageLimiter, "release_concurrency_slot",
                          side_effect=lambda iid: released.append(iid)):
            with self.assertRaises(ValueError):
                view(_Req(inst))
        self.assertEqual(released, ["school-a"])

    def test_proceeds_when_concurrency_gate_errors(self):
        inst = _institute()

        @metered("content_generate")
        def view(request):
            return "generated"

        with patch.object(UsageLimiter, "acquire_concurrency_slot",
                          side_effect=RuntimeError("redis down")):
            self.assertEqual(view(_Req(inst)), "generated")

    def test_refusal_without_redis_does_not_block(self):
        """Without Redis the gate is per-worker, so a refusal proves nothing."""
        inst = _institute()

        @metered("content_generate")
        def view(request):
            return "generated"

        limiter = get_shared_limiter()
        with patch.object(UsageLimiter, "acquire_concurrency_slot", return_value=False), \
             patch.object(limiter, "_redis", None):
            self.assertEqual(view(_Req(inst)), "generated")

    def test_preserves_view_identity(self):
        @metered("content_generate")
        def some_view(request):
            """Docstring kept."""

        self.assertEqual(some_view.__name__, "some_view")
        self.assertEqual(some_view.__doc__, "Docstring kept.")


class UsageAccountingTests(TestCase):
    def setUp(self):
        get_shared_limiter()._memory.clear()

    def test_log_usage_books_tokens_against_the_budget(self):
        """Every generating endpoint already calls log_usage, so booking there is
        what makes the daily cap reflect all traffic rather than only ai_call()."""
        from ai_services.core.usage_logger import log_usage

        with patch("ai_services.core.usage_logger.log_ai_usage_sync"):
            log_usage(
                institute_id="school-a", institute_type="school",
                feature_id="content_dpp", feature_category="content",
                model_used="gemini-2.5-flash", tokens_input=1000, tokens_output=500,
            )
        self.assertEqual(get_shared_limiter()._get_usage("school-a"), 1500)

    def test_booked_once_not_twice(self):
        """ai_call() used to book tokens itself as well as via log_usage, which
        charged every metered call to the budget twice."""
        from ai_services.core.usage_logger import log_usage

        with patch("ai_services.core.usage_logger.log_ai_usage_sync"):
            log_usage(
                institute_id="school-a", institute_type="school",
                feature_id="quiz_generate", feature_category="content",
                model_used="llama-3.3-70b-versatile",
                tokens_input=800, tokens_output=200,
            )
        self.assertEqual(get_shared_limiter()._get_usage("school-a"), 1000)

    def test_accounting_failure_never_breaks_generation(self):
        from ai_services.core.usage_logger import log_usage

        with patch("ai_services.core.usage_logger.log_ai_usage_sync"), \
             patch("ai_services.core.rate_limiter.get_shared_limiter",
                   side_effect=RuntimeError("redis down")):
            log_usage(  # must not raise
                institute_id="school-a", institute_type="school",
                feature_id="content_dpp", feature_category="content",
                model_used="gemini-2.5-flash", tokens_input=10, tokens_output=10,
            )

    def test_shared_limiter_is_a_singleton(self):
        """A new UsageLimiter per call would open a Redis connection each time."""
        self.assertIs(get_shared_limiter(), get_shared_limiter())


class GateCoverageTests(TestCase):
    def test_every_generating_endpoint_is_gated(self):
        """Guards the regression this change fixes: a new endpoint added without
        metering is exactly how the 17 unmetered ones appeared."""
        import re

        src = open("ai_services/views/bridge.py", encoding="utf-8").read()
        lines = src.split("\n")
        defs = [(i, re.match(r"def (\w+)", l).group(1))
                for i, l in enumerate(lines) if l.startswith("def ")]

        ungated = []
        for n, (i, name) in enumerate(defs):
            ctx = "\n".join(lines[max(0, i - 6):i])
            if "@api_view" not in ctx:
                continue
            end = defs[n + 1][0] if n + 1 < len(defs) else len(lines)
            body = "\n".join(lines[i:end])
            gated = "@metered(" in ctx or re.search(r"\bai_call(_text)?\(", body)
            if not gated:
                ungated.append(name)

        # A health probe must answer even when the tenant is over budget,
        # otherwise monitoring goes dark exactly when it is needed.
        self.assertEqual(
            ungated, ["ai_engine_health"],
            f"these endpoints generate without metering: {ungated}",
        )
