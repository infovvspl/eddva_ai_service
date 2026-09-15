"""
Tests for the EDVA AI model / provider router (feature/ai-model-router).

No network and no database: providers are fake adapters or a patched httpx
transport. The compatibility tests matter most — with default configuration the
router must reproduce the pre-router Groq call exactly.

Model ids containing "test" / "cfg" below are fixtures, not real provider ids.
"""
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from ai_services.core import routing
from ai_services.core.routing import (
    AIRequest,
    FallbackExhaustedError,
    ModelRouter,
    NonRetryableProviderError,
    NoRouteError,
    ProviderConfigError,
    RetryableProviderError,
    load_config,
)
from ai_services.core.routing import benchmark
from ai_services.core.routing.errors import classify_status
from ai_services.core.routing.providers import GeminiAdapter, ProviderCall, TogetherAdapter, scrub
from ai_services.core.routing.registry import STRUCTURED_NATIVE, default_registry, unmet_requirements

TOGETHER_ID = "test/fixture-gpt-oss-120b"
FALLBACK_ENV = {"AI_ROUTER_FALLBACK_ENABLED": "true", "TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID}
SECRET = "tk-live-SUPERSECRET-abcdef123456"


# ── fixtures ─────────────────────────────────────────────────────────────────
def ok(model="m", content=None):
    return {
        "content": {"ok": True} if content is None else content,
        "model": model,
        "latency_ms": 5,
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


class FakeAdapter:
    def __init__(self, name, *, configured=True, script=None):
        self.name = name
        self.configured = configured
        self.script = list(script or [])
        self.calls = []

    def is_configured(self):
        return self.configured

    def complete(self, spec, call):
        self.calls.append((spec, call))
        step = self.script.pop(0) if self.script else ok(call.model_id)
        if isinstance(step, BaseException):
            raise step
        return dict(step)


def make_router(env=None, groq=None, together=None, gemini=None):
    adapters = {
        "groq": groq or FakeAdapter("groq"),
        "together": together or FakeAdapter("together"),
        "gemini": gemini or FakeAdapter("gemini"),
    }
    return ModelRouter(load_config(env or {}), adapters=adapters), adapters


def req(**kw):
    base = dict(system_prompt="s", user_prompt="u", model="openai/gpt-oss-120b", provider="groq")
    base.update(kw)
    return AIRequest(**base)


def retryable(status=503, kind="server_error"):
    return RetryableProviderError(
        f"Error code: {status}", provider="groq", model="openai/gpt-oss-120b", status_code=status, kind=kind,
    )


class _Resp:
    def __init__(self, status, payload=None, text=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _chat(content, prompt_tokens=2, completion_tokens=3):
    return _Resp(200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def _together_spec(**changes):
    return replace(default_registry()["together/gpt-oss-120b"], provider_model_id=TOGETHER_ID, **changes)


# ── 1. registry ──────────────────────────────────────────────────────────────
class RegistryTests(SimpleTestCase):
    def test_1_default_registry_loads_every_intended_model(self):
        cfg = load_config({})
        for rid in (
            "groq/gpt-oss-120b", "groq/gpt-oss-20b", "gemini/gemini-2.5-flash",
            "together/gpt-oss-120b", "together/qwen3.8-flash", "together/glm-5.3-flash",
            "together/qwen3.7-max", "together/deepseek-v4-flash",
        ):
            self.assertIn(rid, cfg.models)
        self.assertEqual(cfg.warnings, ())

    def test_1b_only_repository_proven_ids_are_built_in(self):
        m = default_registry()
        self.assertEqual(m["groq/gpt-oss-120b"].provider_model_id, "openai/gpt-oss-120b")
        self.assertEqual(m["groq/gpt-oss-20b"].provider_model_id, "openai/gpt-oss-20b")
        self.assertEqual(m["gemini/gemini-2.5-flash"].provider_model_id, "gemini-2.5-flash")

    def test_1c_no_together_model_id_is_invented(self):
        for rid, spec in default_registry().items():
            if spec.provider == "together":
                self.assertIsNone(spec.provider_model_id, rid)
                self.assertTrue(spec.model_env, rid)
                self.assertFalse(spec.is_configured, rid)

    def test_1d_premium_is_never_a_default_route(self):
        cfg = load_config({})
        self.assertEqual([n for n, f in cfg.features.items() if f.capability == "premium"], [])
        for cap, p in cfg.policies.items():
            if cap != "premium":
                self.assertNotIn("together/qwen3.7-max", (p.primary, *p.fallbacks), cap)

    def test_1e_default_primaries_are_todays_production_models(self):
        p = load_config({}).policies
        self.assertEqual(p["reasoning"].primary, "groq/gpt-oss-120b")
        self.assertEqual(p["lightweight"].primary, "groq/gpt-oss-20b")
        self.assertEqual(p["content"].primary, "groq/gpt-oss-120b")
        self.assertEqual(p["grounded"].primary, "gemini/gemini-2.5-flash")
        # Qwen is a benchmark candidate for content, not a route.
        self.assertIn("together/qwen3.8-flash", p["content"].candidates)
        self.assertNotIn("together/qwen3.8-flash", (p["content"].primary, *p["content"].fallbacks))

    def test_1f_router_on_and_fallback_off_by_default(self):
        cfg = load_config({})
        self.assertTrue(cfg.enabled)
        self.assertFalse(cfg.fallback_enabled)


# ── 2 / 3 / 13. selection ────────────────────────────────────────────────────
class RoutingSelectionTests(SimpleTestCase):
    def test_2_feature_maps_to_capability(self):
        r, _ = make_router()
        self.assertEqual(r.plan(req(feature="doubt_resolve")).capability, "reasoning")
        self.assertEqual(
            r.plan(req(feature="content_recommend", model="openai/gpt-oss-20b")).capability, "lightweight"
        )

    def test_2b_explicit_capability_wins_over_feature(self):
        r, _ = make_router()
        self.assertEqual(r.plan(req(feature="doubt_resolve", capability="content")).capability, "content")

    def test_3_pinned_model_is_primary_and_sent_unchanged(self):
        r, a = make_router()
        out = r.execute(req(model="quiz"))
        spec, call = a["groq"].calls[0]
        self.assertEqual(call.model_id, "quiz")               # alias string untouched
        self.assertEqual(spec.id, "groq/gpt-oss-120b")        # understood as what it resolves to
        self.assertEqual(out["provider"], "groq")
        self.assertFalse(out["fallback_used"])

    def test_3b_unpinned_request_uses_policy_primary(self):
        r, a = make_router()
        r.execute(req(model=None, capability="lightweight"))
        self.assertEqual(a["groq"].calls[0][1].model_id, "openai/gpt-oss-20b")

    def test_3c_pin_outside_registry_is_still_honoured(self):
        r, a = make_router()
        r.execute(req(model="gemma2-9b-it"))
        self.assertEqual(a["groq"].calls[0][1].model_id, "gemma2-9b-it")

    def test_13_unknown_feature_uses_the_pinned_models_capability(self):
        r, _ = make_router()
        plan = r.plan(req(feature="no_such_feature", model="openai/gpt-oss-20b"))
        self.assertEqual(plan.capability, "lightweight")
        self.assertEqual(plan.candidates[0].model_id, "openai/gpt-oss-20b")

    def test_13b_unknown_capability_is_rejected(self):
        r, _ = make_router()
        with self.assertRaises(NoRouteError):
            r.plan(req(capability="telepathy"))

    def test_13c_unknown_feature_without_model_defaults_to_reasoning(self):
        r, _ = make_router()
        self.assertEqual(r.plan(req(feature="zzz", model=None)).capability, "reasoning")


# ── 4 / 5. fallback + error classes ──────────────────────────────────────────
class FallbackTests(SimpleTestCase):
    def setUp(self):
        p = patch("ai_services.core.provider_events.emit")
        self.emit = p.start()
        self.addCleanup(p.stop)

    def test_4_retryable_primary_failure_falls_back_to_together(self):
        r, a = make_router(
            FALLBACK_ENV,
            groq=FakeAdapter("groq", script=[retryable()]),
            together=FakeAdapter("together", script=[ok(TOGETHER_ID)]),
        )
        out = r.execute(req(feature="doubt_resolve"))
        self.assertTrue(out["fallback_used"])
        self.assertEqual(out["provider"], "together")
        self.assertEqual(out["registry_id"], "together/gpt-oss-120b")
        self.assertEqual(a["together"].calls[0][1].model_id, TOGETHER_ID)
        self.assertEqual([x["outcome"] for x in out["route_attempts"]], ["error", "success"])

    def test_4b_fallback_flag_off_means_primary_error_unchanged(self):
        err = retryable()
        r, a = make_router({"TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID}, groq=FakeAdapter("groq", script=[err]))
        with self.assertRaises(RetryableProviderError) as cm:
            r.execute(req())
        self.assertIs(cm.exception, err)
        self.assertEqual(a["together"].calls, [])

    def test_5_deterministic_error_never_falls_back(self):
        err = NonRetryableProviderError("json_validate_failed", provider="groq", status_code=400,
                                        kind="structured_output")
        r, a = make_router(FALLBACK_ENV, groq=FakeAdapter("groq", script=[err]))
        with self.assertRaises(NonRetryableProviderError) as cm:
            r.execute(req())
        self.assertIs(cm.exception, err)
        self.assertEqual(a["together"].calls, [])

    def test_5b_configuration_error_never_falls_back(self):
        err = ProviderConfigError("No active GROQ keys left", provider="groq")
        r, a = make_router(FALLBACK_ENV, groq=FakeAdapter("groq", script=[err]))
        with self.assertRaises(ProviderConfigError):
            r.execute(req())
        self.assertEqual(a["together"].calls, [])

    def test_5c_unclassified_exception_never_falls_back(self):
        r, a = make_router(FALLBACK_ENV, groq=FakeAdapter("groq", script=[ValueError("bug")]))
        with self.assertRaises(ValueError):
            r.execute(req())
        self.assertEqual(a["together"].calls, [])

    def test_5d_every_retryable_class_falls_back(self):
        for err in (retryable(429, "rate_limited"), retryable(500, "server_error"),
                    RetryableProviderError("timed out", provider="groq", kind="timeout"),
                    RetryableProviderError("conn refused", provider="groq", kind="network"),
                    RetryableProviderError("rotation exhausted", provider="groq", kind="exhausted")):
            r, _ = make_router(
                FALLBACK_ENV,
                groq=FakeAdapter("groq", script=[err]),
                together=FakeAdapter("together", script=[ok(TOGETHER_ID)]),
            )
            self.assertTrue(r.execute(req())["fallback_used"], err.kind)

    def test_5e_http_status_classification(self):
        cases = {
            429: (True, "rate_limited"), 500: (True, "server_error"), 503: (True, "server_error"),
            408: (True, "timeout"), None: (True, "network"),
            400: (False, "bad_request"), 422: (False, "bad_request"), 401: (False, "auth"),
            403: (False, "auth"), 404: (False, "config"), 413: (False, "request_too_large"),
        }
        for status, expected in cases.items():
            self.assertEqual(classify_status(status), expected, status)

    def test_5f_all_providers_failing_raises_exhausted_with_attempts(self):
        r, _ = make_router(
            FALLBACK_ENV,
            groq=FakeAdapter("groq", script=[retryable()]),
            together=FakeAdapter("together", script=[
                RetryableProviderError("Together 503", provider="together", status_code=503, kind="server_error"),
            ]),
        )
        with self.assertRaises(FallbackExhaustedError) as cm:
            r.execute(req())
        self.assertEqual(len(cm.exception.attempts), 2)
        self.assertIsInstance(cm.exception, RuntimeError)

    def test_5g_no_retry_amplification(self):
        r, a = make_router(
            FALLBACK_ENV,
            groq=FakeAdapter("groq", script=[retryable()]),
            together=FakeAdapter("together", script=[retryable()]),
        )
        with self.assertRaises(FallbackExhaustedError):
            r.execute(req())
        self.assertEqual((len(a["groq"].calls), len(a["together"].calls)), (1, 1))

    def test_5h_fallback_skipped_once_primary_passed_the_deadline(self):
        r, a = make_router(
            {**FALLBACK_ENV, "AI_ROUTE_REASONING_FALLBACK_DEADLINE_S": "10"},
            groq=FakeAdapter("groq", script=[retryable()]),
        )
        with patch("ai_services.core.routing.router.time.monotonic", side_effect=[0.0, 50.0, 50.0, 50.0]):
            with self.assertRaises(RetryableProviderError):
                r.execute(req())
        self.assertEqual(a["together"].calls, [])


# ── 6. structured output ─────────────────────────────────────────────────────
class StructuredOutputTests(SimpleTestCase):
    def setUp(self):
        p = patch("ai_services.core.provider_events.emit")
        p.start()
        self.addCleanup(p.stop)

    def _no_structured_env(self):
        doc = {"models": {"together/gpt-oss-120b": {"provider_model_id": TOGETHER_ID, "structured_output": "none"}}}
        return {"AI_ROUTER_FALLBACK_ENABLED": "true", "AI_ROUTING_CONFIG": json.dumps(doc)}

    def test_6_model_without_structured_output_is_not_a_json_fallback(self):
        r, _ = make_router(self._no_structured_env())
        plan = r.plan(req(json_mode=True))
        self.assertEqual(len(plan.candidates), 1)
        self.assertIn(("together/gpt-oss-120b", "unmet:structured_output"), plan.skipped)

    def test_6b_text_request_does_not_require_structured_output(self):
        r, _ = make_router(self._no_structured_env())
        self.assertEqual(len(r.plan(req(json_mode=False)).candidates), 2)

    def test_6c_response_format_sent_only_to_verified_native_models(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": SECRET}), patch("httpx.post") as post:
            post.return_value = _chat('{"a": 1}')
            TogetherAdapter().complete(_together_spec(), ProviderCall("s", "u", TOGETHER_ID, json_mode=True))
            self.assertNotIn("response_format", post.call_args.kwargs["json"])
            TogetherAdapter().complete(
                _together_spec(structured_output=STRUCTURED_NATIVE),
                ProviderCall("s", "u", TOGETHER_ID, json_mode=True),
            )
            self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_6d_invalid_json_from_together_is_deterministic(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": SECRET}), patch("httpx.post", return_value=_chat("not json")):
            with self.assertRaises(NonRetryableProviderError) as cm:
                TogetherAdapter().complete(_together_spec(), ProviderCall("s", "u", TOGETHER_ID, json_mode=True))
        self.assertEqual(cm.exception.kind, "structured_output")

    def test_6e_fallback_gets_the_same_shaped_system_prompt_as_groq(self):
        from ai_services.core.llm_client import build_effective_system_prompt

        with patch.dict(os.environ, {"TOGETHER_API_KEY": SECRET}), patch("httpx.post") as post:
            post.return_value = _chat('{"a": 1}')
            TogetherAdapter().complete(_together_spec(), ProviderCall("SYS", "u", TOGETHER_ID, json_mode=True))
        sent = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(sent, build_effective_system_prompt("SYS", True, None))


# ── 7 / 8. capability requirements ───────────────────────────────────────────
class CapabilityRequirementTests(SimpleTestCase):
    def test_7_vision_request_skips_text_only_fallback(self):
        doc = {"policies": {"vision": {"fallbacks": ["groq/gpt-oss-120b"]}}}
        r, _ = make_router({"AI_ROUTER_FALLBACK_ENABLED": "true", "AI_ROUTING_CONFIG": json.dumps(doc)})
        plan = r.plan(req(model=None, capability="vision", requires_vision=True))
        self.assertEqual(plan.candidates[0].spec.id, "gemini/gemini-2.5-flash")
        self.assertIn(("groq/gpt-oss-120b", "unmet:vision"), plan.skipped)

    def test_7b_router_chosen_primary_must_support_vision(self):
        r, _ = make_router({"AI_ROUTE_VISION_PRIMARY": "groq/gpt-oss-120b"})
        with self.assertRaises(NoRouteError):
            r.plan(req(model=None, capability="vision", requires_vision=True))

    def test_7c_unverified_candidates_default_to_no_vision(self):
        self.assertFalse(default_registry()["together/qwen3.8-flash"].multimodal)

    def test_8_long_context_grounded_request_excludes_groq(self):
        doc = {"policies": {"grounded": {"fallbacks": ["groq/gpt-oss-120b"]}}}
        r, _ = make_router({"AI_ROUTER_FALLBACK_ENABLED": "true", "AI_ROUTING_CONFIG": json.dumps(doc)})
        plan = r.plan(req(model=None, capability="grounded", requires_long_context=True, requires_grounding=True))
        self.assertEqual([c.spec.id for c in plan.candidates], ["gemini/gemini-2.5-flash"])
        self.assertIn(("groq/gpt-oss-120b", "unmet:long_context,grounding"), plan.skipped)

    def test_8b_min_context_respects_groqs_request_ceiling(self):
        spec = default_registry()["groq/gpt-oss-120b"]
        self.assertEqual(unmet_requirements(spec, req(min_context_tokens=30_000)), ["context_tokens"])
        self.assertEqual(unmet_requirements(spec, req(min_context_tokens=8_000)), [])

    def test_8c_candidate_long_context_enabled_only_by_config(self):
        doc = {"models": {"together/qwen3.8-flash": {"provider_model_id": "cfg-qwen", "long_context": True}}}
        spec = load_config({"AI_ROUTING_CONFIG": json.dumps(doc)}).models["together/qwen3.8-flash"]
        self.assertTrue(spec.long_context)
        self.assertEqual(spec.metadata_source, "config")


# ── 9. Gemini stays available ────────────────────────────────────────────────
class GeminiAvailabilityTests(SimpleTestCase):
    SPEC = default_registry()["gemini/gemini-2.5-flash"]

    def test_9_gemini_registered_and_primary_for_grounded_and_vision(self):
        p = load_config({}).policies
        self.assertEqual(p["grounded"].primary, "gemini/gemini-2.5-flash")
        self.assertEqual(p["vision"].primary, "gemini/gemini-2.5-flash")
        self.assertTrue(self.SPEC.supports_grounding and self.SPEC.multimodal and self.SPEC.long_context)

    def test_9b_gemini_adapter_uses_the_existing_client(self):
        with patch("ai_services.core.gemini_client.complete_text", return_value={
            "content": "hi", "model": "gemini-2.5-flash", "latency_ms": 9, "tokens_input": 4, "tokens_output": 5,
        }) as ct:
            out = GeminiAdapter().complete(self.SPEC, ProviderCall("s", "u", "gemini-2.5-flash", json_mode=False))
        self.assertEqual(ct.call_args.kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(out["usage"]["total_tokens"], 9)

    def test_9c_gemini_errors_are_classified(self):
        from ai_services.core import gemini_client as gc

        cases = [
            (gc.GeminiUnavailable("No Gemini API key is configured"), ProviderConfigError, "config"),
            (RuntimeError("Gemini returned malformed JSON"), NonRetryableProviderError, "structured_output"),
            (RuntimeError("Gemini returned an empty response"), NonRetryableProviderError, "empty_response"),
            (RuntimeError("All 5 Gemini key(s) failed for JSON completion: 503 UNAVAILABLE"),
             RetryableProviderError, "exhausted"),
        ]
        for exc, cls, kind in cases:
            with patch("ai_services.core.gemini_client.complete_json", side_effect=exc):
                with self.assertRaises(cls) as cm:
                    GeminiAdapter().complete(self.SPEC, ProviderCall("s", "u", "gemini-2.5-flash", json_mode=True))
            self.assertEqual(cm.exception.kind, kind, str(exc))

    def test_9d_gemini_model_follows_existing_env_convention(self):
        cfg = load_config({"GEMINI_TEXT_MODEL": "gemini-flash-latest"})
        self.assertEqual(cfg.models["gemini/gemini-2.5-flash"].provider_model_id, "gemini-flash-latest")


# ── 10 / 11. attribution + telemetry ─────────────────────────────────────────
class AttributionAndTelemetryTests(SimpleTestCase):
    INSTITUTE = "11111111-1111-1111-1111-111111111111"

    def setUp(self):
        from ai_services.core import request_context

        request_context.set_context(request_id="req-123", user_id="user-9", user_role="TEACHER",
                                    institute_id=self.INSTITUTE, vertical="school")
        self.addCleanup(request_context.clear)

    def _fallback_router(self):
        return make_router(
            FALLBACK_ENV,
            groq=FakeAdapter("groq", script=[retryable()]),
            together=FakeAdapter("together", script=[ok(TOGETHER_ID)]),
        )

    def test_10_institute_reaches_provider_and_context_is_untouched(self):
        from ai_services.core import request_context as rc

        r, a = make_router()
        r.execute(req(institute_id="inst-1"))
        self.assertEqual(a["groq"].calls[0][1].institute_id, "inst-1")
        self.assertEqual(
            (rc.get("request_id"), rc.get("user_id"), rc.get("user_role"), rc.get("institute_id")),
            ("req-123", "user-9", "TEACHER", self.INSTITUTE),
        )

    def test_10b_failover_event_carries_request_institute_and_feature(self):
        captured = []

        def run_inline(target, args, daemon):
            return SimpleNamespace(start=lambda: target(*args))

        with patch("ai_services.core.provider_events._post", side_effect=captured.append), \
             patch("ai_services.core.provider_events.threading.Thread", side_effect=run_inline):
            r, _ = self._fallback_router()
            r.execute(req(feature="doubt_resolve", institute_id="default"))
        failovers = [p for p in captured if p["eventType"] == "failover"]
        self.assertEqual(len(failovers), 1)
        ev = failovers[0]
        self.assertEqual(
            (ev["requestId"], ev["instituteId"], ev["feature"], ev["provider"], ev["statusCode"]),
            ("req-123", self.INSTITUTE, "doubt_resolve", "groq", 503),
        )

    def test_11_result_carries_route_metadata(self):
        with patch("ai_services.core.provider_events.emit"):
            r, _ = self._fallback_router()
            out = r.execute(req())
        for key in ("provider", "registry_id", "capability", "fallback_used", "route_attempts", "route_latency_ms"):
            self.assertIn(key, out)
        self.assertEqual(out["usage"]["total_tokens"], 3)

    def test_11b_one_logical_request_never_writes_usage_rows_from_the_router(self):
        with patch("ai_services.core.usage_logger.log_usage") as lu, \
             patch("ai_services.core.usage_logger.log_ai_usage_sync") as lus, \
             patch("ai_services.core.provider_events.emit"):
            r, _ = self._fallback_router()
            r.execute(req())
        lu.assert_not_called()
        lus.assert_not_called()

    def test_11c_result_without_usage_is_normalised(self):
        gem = FakeAdapter("gemini", script=[{"content": "x", "model": "gemini-2.5-flash", "latency_ms": 1,
                                             "tokens_input": 10, "tokens_output": 20}])
        r, _ = make_router(gemini=gem)
        out = r.execute(req(model=None, capability="grounded", json_mode=False))
        self.assertEqual(out["usage"], {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})

    def test_11d_attempt_logs_carry_request_id_and_never_prompts(self):
        with patch("ai_services.core.provider_events.emit"), \
             self.assertLogs("ai_services.routing", level="DEBUG") as logs:
            r, _ = self._fallback_router()
            r.execute(req(user_prompt="SECRET-STUDENT-TEXT", system_prompt="SYSTEM-TEXT"))
        joined = "\n".join(logs.output)
        self.assertIn("request_id=req-123", joined)
        self.assertNotIn("SECRET-STUDENT-TEXT", joined)
        self.assertNotIn("SYSTEM-TEXT", joined)


# ── 12. configuration ────────────────────────────────────────────────────────
class ConfigurationTests(SimpleTestCase):
    def test_12_env_overrides_policy_fallbacks_and_timeout(self):
        cfg = load_config({"AI_ROUTE_CONTENT_FALLBACKS": "gemini/gemini-2.5-flash", "AI_ROUTE_CONTENT_TIMEOUT_S": "45"})
        self.assertEqual(cfg.policies["content"].fallbacks, ("gemini/gemini-2.5-flash",))
        self.assertEqual(cfg.policies["content"].timeout_s, 45.0)

    def test_12b_feature_override_replaces_the_pinned_primary(self):
        doc = {"features": {"content_generate": {
            "capability": "content", "primary": "together/qwen3.8-flash", "fallbacks": ["gemini/gemini-2.5-flash"],
        }}}
        r, _ = make_router({
            "AI_ROUTING_CONFIG": json.dumps(doc), "TOGETHER_MODEL_QWEN3_8_FLASH": "cfg-qwen",
            "AI_ROUTER_FALLBACK_ENABLED": "true",
        })
        plan = r.plan(req(feature="content_generate"))
        self.assertEqual((plan.candidates[0].model_id, plan.candidates[0].source), ("cfg-qwen", "feature_override"))
        self.assertEqual(plan.candidates[1].spec.id, "gemini/gemini-2.5-flash")

    def test_12c_invalid_json_keeps_defaults_and_warns(self):
        cfg = load_config({"AI_ROUTING_CONFIG": "{not json"})
        self.assertEqual(cfg.policies["reasoning"].primary, "groq/gpt-oss-120b")
        self.assertTrue(any("invalid JSON" in w for w in cfg.warnings))

    def test_12d_unknown_model_in_override_is_ignored_with_warning(self):
        cfg = load_config({"AI_ROUTE_REASONING_PRIMARY": "nope/model"})
        self.assertEqual(cfg.policies["reasoning"].primary, "groq/gpt-oss-120b")
        self.assertTrue(any("nope/model" in w for w in cfg.warnings))

    def test_12e_config_file_is_loaded(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"policies": {"lightweight": {"timeout_s": 12}}}, fh)
        self.addCleanup(os.unlink, fh.name)
        self.assertEqual(load_config({"AI_ROUTING_CONFIG_FILE": fh.name}).policies["lightweight"].timeout_s, 12.0)

    def test_12f_kill_switch(self):
        self.assertFalse(load_config({"AI_ROUTER_ENABLED": "false"}).enabled)

    def test_12g_env_takes_precedence_over_json(self):
        cfg = load_config({
            "AI_ROUTING_CONFIG": json.dumps({"policies": {"content": {"timeout_s": 90}}}),
            "AI_ROUTE_CONTENT_TIMEOUT_S": "30",
        })
        self.assertEqual(cfg.policies["content"].timeout_s, 30.0)

    def test_12h_invalid_number_is_rejected(self):
        cfg = load_config({"AI_ROUTE_CONTENT_TIMEOUT_S": "-5"})
        self.assertEqual(cfg.policies["content"].timeout_s, 120.0)
        self.assertTrue(cfg.warnings)

    def test_12i_together_model_id_from_env(self):
        cfg = load_config({"TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID})
        self.assertEqual(cfg.models["together/gpt-oss-120b"].provider_model_id, TOGETHER_ID)


# ── 14. missing provider configuration ───────────────────────────────────────
class MissingProviderConfigurationTests(SimpleTestCase):
    def test_14_fallback_without_credentials_is_skipped_not_attempted(self):
        err = retryable()
        r, a = make_router(FALLBACK_ENV, groq=FakeAdapter("groq", script=[err]),
                           together=FakeAdapter("together", configured=False))
        self.assertIn(("together/gpt-oss-120b", "provider_not_configured"), r.plan(req()).skipped)
        with self.assertRaises(RetryableProviderError) as cm:
            r.execute(req())
        self.assertIs(cm.exception, err)
        self.assertEqual(a["together"].calls, [])

    def test_14b_fallback_without_model_id_is_skipped(self):
        r, _ = make_router({"AI_ROUTER_FALLBACK_ENABLED": "true"})
        self.assertIn(("together/gpt-oss-120b", "model_id_not_configured"), r.plan(req()).skipped)

    def test_14c_unconfigured_policy_primary_names_the_env_var(self):
        r, _ = make_router()
        with self.assertRaises(ProviderConfigError) as cm:
            r.plan(req(model=None, capability="premium"))
        self.assertIn("TOGETHER_MODEL_QWEN3_7_MAX", str(cm.exception))

    def test_14d_together_without_key_is_a_config_error(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": ""}):
            with self.assertRaises(ProviderConfigError):
                TogetherAdapter().complete(_together_spec(), ProviderCall("s", "u", TOGETHER_ID))


# ── 15. no secret leakage ────────────────────────────────────────────────────
class NoSecretLeakageTests(SimpleTestCase):
    def setUp(self):
        p = patch("ai_services.core.provider_events.emit")
        self.emit = p.start()
        self.addCleanup(p.stop)

    def _call(self, response=None, side_effect=None):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": SECRET}), \
             patch("httpx.post", return_value=response, side_effect=side_effect):
            TogetherAdapter().complete(_together_spec(), ProviderCall("s", "u", TOGETHER_ID, json_mode=False))

    def test_15_error_body_echoing_the_key_is_scrubbed(self):
        body = json.dumps({"error": f"invalid key {SECRET}", "auth": f"Bearer {SECRET}"})
        with self.assertRaises(ProviderConfigError) as cm:
            self._call(_Resp(401, text=body))
        self.assertNotIn(SECRET, str(cm.exception))

    def test_15b_timeout_and_network_errors_carry_no_key_or_chained_request(self):
        import httpx

        for exc in (httpx.ReadTimeout(f"timed out {SECRET}"), httpx.ConnectError(f"refused {SECRET}")):
            with self.assertRaises(RetryableProviderError) as cm:
                self._call(side_effect=exc)
            self.assertNotIn(SECRET, str(cm.exception))
            self.assertIsNone(cm.exception.__cause__)

    def test_15c_scrub_removes_bearer_tokens_and_known_secrets(self):
        text = scrub(f"Authorization: Bearer abc.def-123 and {SECRET}", SECRET)
        self.assertNotIn("abc.def-123", text)
        self.assertNotIn(SECRET, text)

    def test_15d_config_document_carrying_credentials_is_refused(self):
        doc = {"models": {"together/gpt-oss-120b": {"provider_model_id": "x", "api_key": SECRET}}}
        cfg = load_config({"AI_ROUTING_CONFIG": json.dumps(doc)})
        self.assertIsNone(cfg.models["together/gpt-oss-120b"].provider_model_id)
        self.assertTrue(any("credential-like" in w for w in cfg.warnings))
        self.assertFalse(any(SECRET in w for w in cfg.warnings))

    def test_15e_failure_events_send_only_a_key_fingerprint(self):
        with self.assertRaises(RetryableProviderError):
            self._call(_Resp(429, text="slow down"))
        kwargs = self.emit.call_args.kwargs
        self.assertEqual(kwargs["key_hash"], hashlib.sha256(SECRET.encode()).hexdigest()[:12])
        self.assertFalse(any(SECRET in str(v) for v in kwargs.values()))

    def test_15f_benchmark_list_never_prints_credentials(self):
        routing.reset_router(ModelRouter(load_config({"TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID})))
        self.addCleanup(routing.reset_router)
        out = StringIO()
        with patch.dict(os.environ, {"TOGETHER_API_KEY": SECRET}):
            call_command("ai_benchmark", "--list", stdout=out)
        self.assertNotIn(SECRET, out.getvalue())
        self.assertIn("together/gpt-oss-120b", out.getvalue())


# ── compatibility with the pre-router LLMClient ──────────────────────────────
class LLMClientCompatibilityTests(SimpleTestCase):
    def setUp(self):
        routing.reset_router(ModelRouter(load_config({})))
        self.addCleanup(routing.reset_router)

    @staticmethod
    def _fake_groq(sink, content='{"answer": 42}'):
        class FakeCompletions:
            def create(self, **kw):
                sink.append(kw)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
                )

        class FakeGroq:
            def __init__(self, api_key=None, **kw):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        return FakeGroq

    @staticmethod
    def _patches(fake_groq, keys=("k1",)):
        import groq as groq_mod
        import ai_services.core.llm_client as lc

        return [
            patch.object(groq_mod, "Groq", fake_groq),
            patch.object(lc, "GROQ_API_KEYS", list(keys)),
            patch.object(lc, "_DISABLED_GROQ_KEYS", set()),
            patch.object(lc.time, "sleep", lambda *_a, **_k: None),
        ]

    def _run(self, fn, sink, content='{"answer": 42}', keys=("k1",)):
        with ExitStack() as stack:
            for p in self._patches(self._fake_groq(sink, content), keys):
                stack.enter_context(p)
            return fn()

    def test_default_routing_sends_the_identical_groq_request(self):
        import ai_services.core.llm_client as lc

        kwargs = dict(system_prompt="SYS", user_prompt="USR", model="quiz", temperature=0.2,
                      max_tokens=321, json_mode=True)
        via_router, direct = [], []
        out = self._run(lambda: lc.LLMClient().complete(**kwargs, feature="quiz_generate"), via_router)
        self._run(lambda: lc.LLMClient()._complete_groq(**kwargs), direct)
        self.assertEqual(via_router, direct)
        self.assertEqual(via_router[0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(out["content"], {"answer": 42})
        self.assertEqual((out["provider"], out["fallback_used"]), ("groq", False))

    def test_kill_switch_bypasses_the_router(self):
        import ai_services.core.llm_client as lc

        routing.reset_router(ModelRouter(load_config({"AI_ROUTER_ENABLED": "false"})))
        sink = []
        with patch.object(ModelRouter, "execute") as execute:
            out = self._run(lambda: lc.LLMClient().complete(system_prompt="s", user_prompt="u",
                                                            model="openai/gpt-oss-20b"), sink)
        execute.assert_not_called()
        self.assertNotIn("provider", out)
        self.assertEqual(sink[0]["model"], "openai/gpt-oss-20b")

    def test_missing_model_still_means_GROQ_MODEL(self):
        import ai_services.core.llm_client as lc

        sink = []
        self._run(lambda: lc.LLMClient().complete(system_prompt="s", user_prompt="u", model=None), sink)
        self.assertEqual(sink[0]["model"], lc._resolve_model(lc.GROQ_MODEL))

    def test_every_new_error_is_still_a_RuntimeError(self):
        for cls in (RetryableProviderError, NonRetryableProviderError, ProviderConfigError,
                    FallbackExhaustedError, NoRouteError):
            self.assertTrue(issubclass(cls, RuntimeError), cls)

    def test_no_groq_keys_is_a_config_error_with_the_original_message(self):
        import ai_services.core.llm_client as lc

        with self.assertRaises(ProviderConfigError) as cm:
            self._run(lambda: lc.LLMClient().complete(system_prompt="s", user_prompt="u",
                                                      model="openai/gpt-oss-120b"), [], keys=())
        self.assertIn("No GROQ_API_KEY configured", str(cm.exception))

    def test_repeated_unparseable_json_is_not_retryable(self):
        import ai_services.core.llm_client as lc

        with self.assertRaises(NonRetryableProviderError) as cm:
            self._run(lambda: lc.LLMClient().complete(system_prompt="s", user_prompt="u",
                                                      model="openai/gpt-oss-120b", json_mode=True),
                      [], content="definitely not json", keys=("k1", "k2"))
        self.assertEqual(cm.exception.kind, "structured_output")


class BaseViewFeatureHintTests(SimpleTestCase):
    """ai_call / ai_call_text already know their feature; they now pass it."""

    FAKE = {"content": {"x": 1}, "model": "openai/gpt-oss-120b", "latency_ms": 1,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def test_ai_call_passes_feature(self):
        from ai_services.views import base

        with patch.object(base, "_cache") as cache, patch.object(base, "_limiter") as lim, \
             patch.object(base, "_log_usage_to_db"), \
             patch.object(base._llm, "complete", return_value=self.FAKE) as comp:
            cache.get.return_value = None
            lim.check_budget.return_value = (True, False, 0)
            base._do_ai_call(None, "inst", "feedback_generate", "prompt", 0.5, False)
        self.assertEqual(comp.call_args.kwargs["feature"], "feedback_generate")

    def test_ai_call_text_passes_feature(self):
        from ai_services.views import base

        request = SimpleNamespace(institute=None, institute_id="inst", vertical="base", board="", profile=None)
        fake = {**self.FAKE, "content": "text"}
        with patch.object(base, "_cache") as cache, patch.object(base, "_limiter") as lim, \
             patch.object(base, "_log_usage_to_db"), \
             patch.object(base._llm, "complete", return_value=fake) as comp:
            cache.get.return_value = None
            lim.acquire_concurrency_slot.return_value = True
            lim.check_budget.return_value = (True, False, 0)
            base.ai_call_text(request, "feedback_generate", "prompt", wrap_fn=lambda t: {"text": t})
        self.assertEqual(comp.call_args.kwargs["feature"], "feedback_generate")


# ── benchmark support ────────────────────────────────────────────────────────
class BenchmarkTests(SimpleTestCase):
    def test_unconfigured_candidate_is_blocked_not_run(self):
        r, a = make_router()
        recs = benchmark.benchmark_candidate(r, "together/qwen3.8-flash", system_prompt="s", user_prompt="u")
        self.assertEqual(recs[0]["status"], benchmark.BLOCKED)
        self.assertIn("TOGETHER_MODEL_QWEN3_8_FLASH", recs[0]["reason"])
        self.assertEqual(a["together"].calls, [])

    def test_missing_credentials_are_blocked(self):
        r, _ = make_router({"TOGETHER_MODEL_QWEN3_8_FLASH": "cfg-qwen"},
                           together=FakeAdapter("together", configured=False))
        recs = benchmark.benchmark_candidate(r, "together/qwen3.8-flash", system_prompt="s", user_prompt="u")
        self.assertEqual(recs[0]["status"], benchmark.BLOCKED)
        self.assertIn("credentials", recs[0]["reason"])

    def test_benchmark_runs_the_candidate_directly_and_changes_no_routing(self):
        together = FakeAdapter("together", script=[
            ok("cfg-qwen", content="notes"),
            RetryableProviderError("503", provider="together", status_code=503, kind="server_error"),
        ])
        r, a = make_router({"TOGETHER_MODEL_QWEN3_8_FLASH": "cfg-qwen"}, together=together)
        before = dict(r.config.policies)
        recs = benchmark.benchmark_candidate(r, "together/qwen3.8-flash", system_prompt="s",
                                             user_prompt="u", repeat=2)
        self.assertEqual([x["status"] for x in recs], [benchmark.PASS, benchmark.FAIL])
        self.assertEqual(a["groq"].calls, [])          # no fallback in a benchmark
        self.assertEqual(dict(r.config.policies), before)
        self.assertNotIn("output", recs[0])            # outputs only on request

    def test_capability_selector_lists_primary_then_candidates(self):
        r, _ = make_router()
        self.assertEqual(
            benchmark.candidate_ids_for(r, "capability:content"),
            ["groq/gpt-oss-120b", "together/qwen3.8-flash", "together/glm-5.3-flash", "gemini/gemini-2.5-flash"],
        )

    def test_summary_is_computed_only_from_real_runs(self):
        rows = benchmark.summarize([{"registry_id": "x", "status": benchmark.BLOCKED, "reason": "no key"}])
        self.assertEqual(rows[0]["runs"], 0)
        self.assertIsNone(rows[0]["median_wall_ms"])
        self.assertTrue(rows[0]["blocked"])

    def test_plan_command_reports_skipped_fallback(self):
        routing.reset_router(ModelRouter(load_config({})))
        self.addCleanup(routing.reset_router)
        out = StringIO()
        call_command("ai_benchmark", "--plan", "--feature", "doubt_resolve", "--model", "openai/gpt-oss-120b",
                     stdout=out)
        text = out.getvalue()
        self.assertIn("capability=reasoning", text)
        self.assertIn("fallback_disabled", text)
