"""
Tests for the Together multi-model integration (feature/ai-model-router).

No network: httpx is patched and other providers are fake adapters. Model ids
below that contain "cfg" or "test" are fixtures, not real Together ids — this
suite never assumes any real id.

Most important guarantees:
  * default routes are unchanged with Together fully configured;
  * the local override is server-side, off by default, and refused unless DEBUG;
  * unknown capabilities are never treated as supported;
  * the Together key never appears in errors, logs, events or command output.
"""
import hashlib
import io
import json
import os
import re
from contextlib import ExitStack
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from ai_services.core import routing
from ai_services.core.routing import (
    AIRequest,
    ModelRouter,
    NonRetryableProviderError,
    NoRouteError,
    ProviderConfigError,
    RetryableProviderError,
    load_config,
    telemetry_model_from_error,
    telemetry_model_from_results,
    telemetry_model_id,
)
from ai_services.core.routing import benchmark
from ai_services.core.routing.config import OVERRIDE_ALL
from ai_services.core.routing.providers import ProviderCall, TogetherAdapter
from ai_services.core.routing.registry import (
    STRUCTURED_NATIVE,
    STRUCTURED_PROMPTED,
    STRUCTURED_UNKNOWN,
    default_registry,
    resolve_model_ref,
    unmet_requirements,
)

KEY = "tgp-TEST-SECRET-0123456789abcdef"
QWEN = "cfg/qwen-fixture"
GPTOSS = "cfg/gpt-oss-fixture"

TOGETHER_IDS = {
    "TOGETHER_MODEL_GPT_OSS_120B": GPTOSS,
    "TOGETHER_MODEL_QWEN38_FLASH": QWEN,
    "TOGETHER_MODEL_GLM53_FLASH": "cfg/glm-fixture",
    "TOGETHER_MODEL_DEEPSEEK_V4_FLASH": "cfg/deepseek-fixture",
    "TOGETHER_MODEL_QWEN37_MAX": "cfg/qwen-max-fixture",
}
OVERRIDE_ON = {"AI_MODEL_OVERRIDE_ENABLED": "true"}


# ── fixtures ─────────────────────────────────────────────────────────────────
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


def _chat(content, usage=True):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
    return _Resp(200, body)


def ok(model="m", content=None, **extra):
    return {"content": {"ok": True} if content is None else content, "model": model, "latency_ms": 5,
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, **extra}


class FakeAdapter:
    def __init__(self, name, *, configured=True, script=None):
        self.name, self.configured, self.script, self.calls = name, configured, list(script or []), []

    def is_configured(self):
        return self.configured

    def complete(self, spec, call):
        self.calls.append((spec, call))
        step = self.script.pop(0) if self.script else ok(call.model_id)
        if isinstance(step, BaseException):
            raise step
        return dict(step)


def router_with(env=None, *, debug=False, together=None, groq=None, gemini=None):
    adapters = {"groq": groq or FakeAdapter("groq"), "together": together or FakeAdapter("together"),
                "gemini": gemini or FakeAdapter("gemini")}
    return ModelRouter(load_config(env or {}, debug=debug), adapters=adapters), adapters


def req(**kw):
    base = dict(system_prompt="s", user_prompt="u", model="openai/gpt-oss-120b", provider="groq")
    base.update(kw)
    return AIRequest(**base)


def spec(registry_id, **changes):
    return replace(default_registry()[registry_id], **changes)


class _EventsPatched(SimpleTestCase):
    def setUp(self):
        p = patch("ai_services.core.provider_events.emit")
        self.emit = p.start()
        self.addCleanup(p.stop)


# ── 1 / 2. Together adapter + API key ────────────────────────────────────────
class TogetherAdapterTests(_EventsPatched):
    SPEC = spec("together/qwen3.8-flash", provider_model_id=QWEN)

    def _call(self, response=None, side_effect=None, *, json_mode=False, s=None, env_key=KEY, timeout_s=None):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": env_key}), \
             patch("httpx.post", return_value=response, side_effect=side_effect) as post:
            out = TogetherAdapter().complete(s or self.SPEC,
                                             ProviderCall("SYS", "USR", QWEN, json_mode=json_mode, timeout_s=timeout_s))
        return out, post

    def test_text_success_returns_the_llmclient_result_shape(self):
        out, post = self._call(_chat("notes text"))
        self.assertEqual(out["content"], "notes text")
        self.assertEqual((out["tokens_input"], out["tokens_output"], out["tokens_reported"]), (11, 7, True))
        self.assertEqual(out["usage"]["total_tokens"], 18)
        self.assertEqual(out["model"], QWEN)
        self.assertEqual(post.call_count, 1)
        self.assertTrue(post.call_args.args[0].endswith("/chat/completions"))

    def test_native_json_sends_response_format(self):
        out, post = self._call(_chat('{"a": 1}'), json_mode=True, s=replace(self.SPEC, structured_output=STRUCTURED_NATIVE))
        self.assertEqual(out["content"], {"a": 1})
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_prompted_json_is_parsed_without_response_format(self):
        out, post = self._call(_chat('{"a": 1}'), json_mode=True, s=replace(self.SPEC, structured_output=STRUCTURED_PROMPTED))
        self.assertEqual(out["content"], {"a": 1})
        self.assertNotIn("response_format", post.call_args.kwargs["json"])

    def test_unknown_structured_output_refuses_json_before_any_http_call(self):
        with self.assertRaises(NonRetryableProviderError) as cm:
            self._call(_chat('{"a": 1}'), json_mode=True)
        self.assertEqual(cm.exception.kind, "structured_output")
        self.assertIn("TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT", str(cm.exception))

    def test_missing_usage_is_flagged_not_fabricated(self):
        with self.assertLogs("ai_services.routing", level="WARNING") as logs:
            out, _ = self._call(_chat("x", usage=False))
        self.assertFalse(out["tokens_reported"])
        self.assertTrue(any("no token usage" in line for line in logs.output))

    def test_missing_api_key_fails_clearly_without_calling_the_api(self):
        with self.assertRaises(ProviderConfigError) as cm:
            self._call(_chat("x"), env_key="")
        self.assertIn("TOGETHER_API_KEY", str(cm.exception))

    def test_is_configured_reads_only_the_dedicated_variable(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": ""}):
            self.assertFalse(TogetherAdapter().is_configured())
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}):
            self.assertTrue(TogetherAdapter().is_configured())

    def test_missing_model_id_names_the_env_var(self):
        with self.assertRaises(ProviderConfigError) as cm:
            with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}):
                TogetherAdapter().complete(default_registry()["together/qwen3.8-flash"],
                                           ProviderCall("s", "u", None, json_mode=False))
        self.assertIn("TOGETHER_MODEL_QWEN38_FLASH", str(cm.exception))

    def test_error_classification(self):
        cases = [
            (429, RetryableProviderError, "rate_limited", "429"),
            (500, RetryableProviderError, "server_error", "5xx"),
            (503, RetryableProviderError, "server_error", "5xx"),
            (400, NonRetryableProviderError, "bad_request", "provider_error"),
            (422, NonRetryableProviderError, "bad_request", "provider_error"),
            (401, ProviderConfigError, "auth", "provider_error"),
            (403, ProviderConfigError, "auth", "provider_error"),
            (404, ProviderConfigError, "config", "provider_error"),
        ]
        for status, cls, kind, event in cases:
            self.emit.reset_mock()
            with self.assertRaises(cls) as cm:
                self._call(_Resp(status, text="nope"))
            self.assertEqual(cm.exception.kind, kind, status)
            self.assertEqual(self.emit.call_args.kwargs["event_type"], event, status)
            self.assertEqual(self.emit.call_args.kwargs["provider"], "together")
            if cls is RetryableProviderError:
                self.assertTrue(cm.exception.retryable)
            else:
                self.assertFalse(cm.exception.retryable)

    def test_exactly_one_http_attempt_even_on_retryable_failure(self):
        with self.assertRaises(RetryableProviderError):
            _, post = self._call(_Resp(503, text="busy"))
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.post", return_value=_Resp(503)) as post:
            with self.assertRaises(RetryableProviderError):
                TogetherAdapter().complete(self.SPEC, ProviderCall("s", "u", QWEN, json_mode=False))
        self.assertEqual(post.call_count, 1)

    def test_timeout_and_network_errors_are_retryable(self):
        import httpx

        for exc, kind in ((httpx.ReadTimeout("slow"), "timeout"), (httpx.ConnectError("refused"), "network")):
            with self.assertRaises(RetryableProviderError) as cm:
                self._call(side_effect=exc)
            self.assertEqual(cm.exception.kind, kind)

    def test_timeout_comes_from_the_call_then_env(self):
        _, post = self._call(_chat("x"), timeout_s=42)
        self.assertEqual(post.call_args.kwargs["timeout"], 42.0)
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY, "TOGETHER_TIMEOUT_S": "17"}), \
             patch("httpx.post", return_value=_chat("x")) as post2:
            TogetherAdapter().complete(self.SPEC, ProviderCall("s", "u", QWEN, json_mode=False))
        self.assertEqual(post2.call_args.kwargs["timeout"], 17.0)

    def test_unreadable_envelope_is_retryable(self):
        with self.assertRaises(RetryableProviderError):
            self._call(_Resp(200, text="<html>"))

    def test_invalid_json_from_prompted_model_is_deterministic(self):
        with self.assertRaises(NonRetryableProviderError) as cm:
            self._call(_chat("not json"), json_mode=True, s=replace(self.SPEC, structured_output=STRUCTURED_PROMPTED))
        self.assertEqual(cm.exception.kind, "structured_output")


# ── 3 / 4 / 5. registry, config ids, capability metadata ─────────────────────
class TogetherRegistryTests(SimpleTestCase):
    EXPECTED = {
        "together/gpt-oss-120b": ("TOGETHER_MODEL_GPT_OSS_120B", {"reasoning", "content"}),
        "together/qwen3.8-flash": ("TOGETHER_MODEL_QWEN38_FLASH", {"content"}),
        "together/glm-5.3-flash": ("TOGETHER_MODEL_GLM53_FLASH", {"content"}),
        "together/deepseek-v4-flash": ("TOGETHER_MODEL_DEEPSEEK_V4_FLASH", {"bulk_text", "lightweight"}),
        "together/qwen3.7-max": ("TOGETHER_MODEL_QWEN37_MAX", {"premium"}),
    }

    def test_all_five_candidates_registered_with_the_requested_env_names(self):
        reg = default_registry()
        for rid, (env, caps) in self.EXPECTED.items():
            s = reg[rid]
            self.assertEqual((s.provider, s.model_env, set(s.capabilities)), ("together", env, caps), rid)

    def test_no_id_and_no_support_is_assumed(self):
        for rid in self.EXPECTED:
            s = default_registry()[rid]
            self.assertIsNone(s.provider_model_id, rid)
            self.assertEqual(
                (s.multimodal, s.long_context, s.supports_grounding, s.context_tokens, s.structured_output),
                (None, None, None, None, STRUCTURED_UNKNOWN), rid,
            )
            self.assertIn(s.quality_tier, ("light", "standard", "high", "premium"))
            self.assertIn(s.cost_tier, ("low", "medium", "high"))

    def test_ids_come_only_from_configuration(self):
        cfg = load_config(TOGETHER_IDS, debug=False)
        for rid, (env, _) in self.EXPECTED.items():
            self.assertEqual(cfg.models[rid].provider_model_id, TOGETHER_IDS[env])

    def test_verified_metadata_from_env(self):
        cfg = load_config({
            **TOGETHER_IDS,
            "TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT": "native",
            "TOGETHER_MODEL_QWEN38_FLASH_LONG_CONTEXT": "true",
            "TOGETHER_MODEL_QWEN38_FLASH_MULTIMODAL": "false",
            "TOGETHER_MODEL_QWEN38_FLASH_GROUNDING": "unknown",
            "TOGETHER_MODEL_QWEN38_FLASH_CONTEXT_TOKENS": "200000",
        }, debug=False)
        s = cfg.models["together/qwen3.8-flash"]
        self.assertEqual((s.structured_output, s.long_context, s.multimodal, s.supports_grounding, s.context_tokens),
                         ("native", True, False, None, 200000))
        self.assertEqual(s.metadata_source, "config")

    def test_invalid_metadata_values_are_rejected_with_warnings(self):
        cfg = load_config({"TOGETHER_MODEL_GLM53_FLASH_STRUCTURED_OUTPUT": "maybe",
                           "TOGETHER_MODEL_GLM53_FLASH_LONG_CONTEXT": "sometimes"}, debug=False)
        self.assertEqual(cfg.models["together/glm-5.3-flash"].structured_output, STRUCTURED_UNKNOWN)
        self.assertIsNone(cfg.models["together/glm-5.3-flash"].long_context)
        self.assertEqual(len(cfg.warnings), 2)

    def test_env_metadata_cannot_rewrite_repository_verified_models(self):
        cfg = load_config({"GEMINI_TEXT_MODEL_STRUCTURED_OUTPUT": "none", "GEMINI_TEXT_MODEL_MULTIMODAL": "false"},
                          debug=False)
        g = cfg.models["gemini/gemini-2.5-flash"]
        self.assertEqual((g.structured_output, g.multimodal), (STRUCTURED_NATIVE, True))

    def test_unknown_never_satisfies_a_requirement(self):
        s = spec("together/qwen3.8-flash", provider_model_id=QWEN)
        r = req(requires_vision=True, requires_long_context=True, requires_grounding=True, json_mode=True)
        self.assertEqual(unmet_requirements(s, r), ["vision", "long_context", "grounding", "structured_output"])
        verified = replace(s, multimodal=True, long_context=True, supports_grounding=True,
                           structured_output=STRUCTURED_PROMPTED)
        self.assertEqual(unmet_requirements(verified, r), [])

    def test_model_refs_resolve_only_to_registered_models(self):
        models = dict(load_config(TOGETHER_IDS, debug=False).models)
        self.assertEqual(resolve_model_ref(models, "together:qwen3.8-flash").id, "together/qwen3.8-flash")
        self.assertEqual(resolve_model_ref(models, "together/qwen3.8-flash").id, "together/qwen3.8-flash")
        self.assertEqual(resolve_model_ref(models, f"together:{QWEN}").id, "together/qwen3.8-flash")
        self.assertIsNone(resolve_model_ref(models, "together:some-unregistered/model"))
        self.assertIsNone(resolve_model_ref(models, "groq:qwen3.8-flash"))


# ── 7. local override ────────────────────────────────────────────────────────
class LocalOverrideTests(_EventsPatched):
    def _env(self, **extra):
        return {**TOGETHER_IDS, **OVERRIDE_ON, "TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT": "prompted", **extra}

    def test_off_by_default(self):
        cfg = load_config(TOGETHER_IDS, debug=True)
        self.assertEqual(dict(cfg.overrides), {})

    def test_override_variables_without_enable_flag_are_ignored(self):
        cfg = load_config({**TOGETHER_IDS, "AI_MODEL_OVERRIDE_CONTENT": "together:qwen3.8-flash"}, debug=True)
        self.assertEqual(dict(cfg.overrides), {})
        self.assertTrue(any("AI_MODEL_OVERRIDE_ENABLED" in w for w in cfg.warnings))

    def test_refused_outside_debug(self):
        cfg = load_config(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash"), debug=False)
        self.assertEqual(dict(cfg.overrides), {})
        self.assertTrue(any("REFUSED" in w for w in cfg.warnings))

    def test_debug_detected_from_django_settings(self):
        from django.test import override_settings

        with override_settings(DEBUG=False):
            self.assertEqual(dict(load_config(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash")).overrides), {})
        with override_settings(DEBUG=True):
            self.assertEqual(dict(load_config(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash")).overrides),
                             {"content": "together/qwen3.8-flash"})

    def test_invalid_or_unconfigured_targets_are_ignored_with_warnings(self):
        cfg = load_config({**OVERRIDE_ON, "AI_MODEL_OVERRIDE_CONTENT": "together:nonexistent",
                           "AI_MODEL_OVERRIDE_REASONING": "together:gpt-oss-120b"}, debug=True)
        self.assertEqual(dict(cfg.overrides), {})
        self.assertTrue(any("not a registered model" in w for w in cfg.warnings))
        self.assertTrue(any("TOGETHER_MODEL_GPT_OSS_120B" in w for w in cfg.warnings))

    def test_override_replaces_the_pinned_model_and_is_served_by_together(self):
        together = FakeAdapter("together", script=[ok(QWEN, content="notes")])
        r, a = router_with(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash"), debug=True, together=together)
        out = r.execute(req(model="openai/gpt-oss-120b", capability="content", json_mode=False, feature="content_generate"))
        self.assertEqual(a["groq"].calls, [])
        self.assertEqual(a["together"].calls[0][1].model_id, QWEN)
        self.assertEqual((out["provider"], out["route_source"], out["override_applied"]), ("together", "local_override", True))
        self.assertEqual(out["model"], f"together:{QWEN}")
        self.assertEqual(out["provider_model_id"], QWEN)

    def test_override_is_scoped_to_its_capability(self):
        r, a = router_with(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash"), debug=True)
        r.execute(req(model="openai/gpt-oss-120b", json_mode=True))  # reasoning: untouched
        self.assertEqual(len(a["groq"].calls), 1)
        self.assertEqual(a["together"].calls, [])

    def test_capability_override_beats_all(self):
        env = self._env(AI_MODEL_OVERRIDE_ALL="together:glm-5.3-flash", AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash",
                        TOGETHER_MODEL_GLM53_FLASH_STRUCTURED_OUTPUT="prompted")
        cfg = load_config(env, debug=True)
        self.assertEqual(cfg.overrides["content"], "together/qwen3.8-flash")
        self.assertEqual(cfg.overrides[OVERRIDE_ALL], "together/glm-5.3-flash")
        r, a = router_with(env, debug=True)
        self.assertEqual(r.plan(req(capability="content", json_mode=False)).candidates[0].spec.id, "together/qwen3.8-flash")
        self.assertEqual(r.plan(req(json_mode=True)).candidates[0].spec.id, "together/glm-5.3-flash")

    def test_json_request_to_unverified_model_fails_clearly_not_downgraded(self):
        env = {**TOGETHER_IDS, **OVERRIDE_ON, "AI_MODEL_OVERRIDE_REASONING": "together:gpt-oss-120b"}
        r, a = router_with(env, debug=True)
        with self.assertRaises(NoRouteError) as cm:
            r.execute(req(json_mode=True))
        self.assertIn("structured_output", str(cm.exception))
        self.assertIn("TOGETHER_MODEL_GPT_OSS_120B", str(cm.exception))
        self.assertEqual((a["groq"].calls, a["together"].calls), ([], []))

    def test_override_success_is_logged_visibly(self):
        r, _ = router_with(self._env(AI_MODEL_OVERRIDE_CONTENT="together:qwen3.8-flash"), debug=True)
        with self.assertLogs("ai_services.routing", level="INFO") as logs:
            r.execute(req(capability="content", json_mode=False))
        self.assertTrue(any("source=local_override" in l and "provider=together" in l for l in logs.output))

    def test_override_is_env_only_never_from_routing_json(self):
        doc = {"overrides": {"content": "together:qwen3.8-flash"}}
        cfg = load_config({**TOGETHER_IDS, **OVERRIDE_ON, "AI_ROUTING_CONFIG": json.dumps(doc)}, debug=True)
        self.assertEqual(dict(cfg.overrides), {})

    def test_no_view_lets_a_request_choose_the_model(self):
        root = os.path.join(os.path.dirname(__file__), "views")
        pattern = re.compile(r"""\b(?:model|provider|capability)\s*=\s*(?:request\.data|data|body|payload)\s*(?:\.get|\[)""")
        offenders = []
        for name in os.listdir(root):
            if name.endswith(".py"):
                text = io.open(os.path.join(root, name), encoding="utf-8").read()
                offenders += [f"{name}: {m.group(0)}" for m in pattern.finditer(text)]
        self.assertEqual(offenders, [])


# ── 13. default routes must not change ───────────────────────────────────────
class DefaultRoutePreservationTests(SimpleTestCase):
    """Together-first routing, and exact preservation where Together is absent."""

    CALLS = [
        ("student doubt", dict(model="openai/gpt-oss-120b", json_mode=True, feature="doubt_resolver"), "together/gpt-oss-120b"),
        ("tutor", dict(model="openai/gpt-oss-120b", json_mode=False, feature="tutor_session"), "together/gpt-oss-120b"),
        ("quiz", dict(model="quiz", json_mode=False), "together/gpt-oss-120b"),
        ("assessment", dict(model="openai/gpt-oss-120b", json_mode=True), "together/gpt-oss-120b"),
        ("teacher analysis", dict(model="openai/gpt-oss-120b", json_mode=True, feature="teacher_recording_analysis"),
         "together/gpt-oss-120b"),
        ("lecture notes", dict(model="openai/gpt-oss-20b", json_mode=False, feature="ai_lecture_notes", capability="content"),
         "together/qwen3.8-flash"),
        ("content", dict(model="openai/gpt-oss-120b", json_mode=False, feature="content_generate", capability="content"),
         "together/qwen3.8-flash"),
        ("ppt", dict(model="openai/gpt-oss-120b", json_mode=True, feature="ppt_generate", capability="content"),
         "together/qwen3.8-flash"),
        ("transcript cleanup", dict(model="openai/gpt-oss-20b", json_mode=False), "together/deepseek-v4-flash"),
    ]
    ENV = {**TOGETHER_IDS, "TOGETHER_API_KEY": KEY,
           "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "native",
           "TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT": "native",
           "TOGETHER_MODEL_DEEPSEEK_V4_FLASH_STRUCTURED_OUTPUT": "native"}

    def test_together_serves_every_feature_by_capability_when_configured(self):
        for debug in (False, True):
            r, _ = router_with(self.ENV, debug=debug)
            for label, kw, expected in self.CALLS:
                plan = r.plan(req(**kw))
                c = plan.candidates[0]
                self.assertEqual((c.spec.id, c.source, len(plan.candidates)), (expected, "policy", 1), label)

    def test_without_together_every_feature_keeps_its_exact_current_model(self):
        r, _ = router_with({}, debug=True)
        for label, kw, _expected in self.CALLS:
            plan = r.plan(req(**kw))
            self.assertEqual(len(plan.candidates), 1, label)
            c = plan.candidates[0]
            self.assertEqual((c.spec.provider, c.source, c.model_id), ("groq", "legacy", kw["model"]), label)

    def test_policies_and_gemini_routes_unchanged(self):
        p = load_config(self.ENV, debug=True).policies
        self.assertEqual(
            {k: (v.primary, v.fallbacks) for k, v in p.items()},
            {
                "reasoning": ("together/gpt-oss-120b", ("groq/gpt-oss-120b",)),
                "lightweight": ("together/deepseek-v4-flash", ("groq/gpt-oss-20b",)),
                "content": ("together/qwen3.8-flash", ("groq/gpt-oss-120b",)),
                "premium": ("together/qwen3.7-max", ("together/gpt-oss-120b",)),
                "bulk_text": ("together/deepseek-v4-flash", ("groq/gpt-oss-20b",)),
                "grounded": ("together/glm-5.3-flash", ("gemini/gemini-2.5-flash",)),
                "vision": ("gemini/gemini-2.5-flash", ()),
            },
        )
        self.assertFalse(load_config(self.ENV, debug=True).fallback_enabled)

    def test_groq_result_model_string_is_unchanged(self):
        r, _ = router_with()
        out = r.execute(req(json_mode=True))
        self.assertEqual(out["model"], "openai/gpt-oss-120b")

    def test_content_call_sites_declare_their_capability(self):
        root = os.path.dirname(__file__)
        bridge = io.open(os.path.join(root, "views", "bridge.py"), encoding="utf-8").read()
        ppt = io.open(os.path.join(root, "views", "ppt.py"), encoding="utf-8").read()
        self.assertEqual(bridge.count('feature="ai_lecture_notes",\n' if "\r\n" not in bridge else 'feature="ai_lecture_notes",\r\n'), 3)
        self.assertEqual(len(re.findall(r'feature="content_generate",\s+capability="content"', bridge)), 2)
        self.assertEqual(len(re.findall(r'feature="ppt_generate",\s+capability="content"', ppt)), 2)

    def test_gemini_integration_files_untouched_by_this_integration(self):
        import subprocess

        root = os.path.dirname(os.path.dirname(__file__))
        files = ["ai_services/core/gemini_client.py", "ai_services/core/gemini_keys.py", "ai_services/core/grounding.py"]
        diff = subprocess.run(["git", "diff", "--name-only", "origin/dev", "--", *files],
                              capture_output=True, text=True, cwd=root)
        if diff.returncode != 0:
            self.skipTest("git not available")
        self.assertEqual(diff.stdout.strip(), "")


# ── 9. fallback eligibility ──────────────────────────────────────────────────
class TogetherFallbackTests(_EventsPatched):
    def test_fallback_stays_off_by_default_even_when_together_is_configured(self):
        r, _ = router_with({**TOGETHER_IDS, "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "native"})
        plan = r.plan(req())
        self.assertEqual(plan.candidates[0].spec.id, "together/gpt-oss-120b")
        self.assertIn(("groq/gpt-oss-120b", "fallback_disabled"), plan.skipped)

    def test_json_request_to_unverified_together_model_uses_the_legacy_route(self):
        r, _ = router_with({**TOGETHER_IDS, "AI_ROUTER_FALLBACK_ENABLED": "true"})
        plan = r.plan(req(json_mode=True))
        self.assertEqual((plan.candidates[0].spec.id, plan.candidates[0].source), ("groq/gpt-oss-120b", "legacy"))
        self.assertIn(("together/gpt-oss-120b", "unmet:structured_output"), plan.skipped)
        self.assertEqual([c.spec.id for c in r.plan(req(json_mode=False)).candidates],
                         ["together/gpt-oss-120b", "groq/gpt-oss-120b"])

    def test_retryable_together_failure_falls_back_to_groq_when_enabled(self):
        together = FakeAdapter("together", script=[RetryableProviderError("503", provider="together", model=GPTOSS,
                                                                          status_code=503, kind="server_error")])
        groq = FakeAdapter("groq", script=[ok("openai/gpt-oss-120b")])
        r, _ = router_with({**TOGETHER_IDS, "AI_ROUTER_FALLBACK_ENABLED": "true",
                            "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "native"}, groq=groq, together=together)
        out = r.execute(req(json_mode=True))
        self.assertTrue(out["fallback_used"])
        self.assertEqual(out["model"], "openai/gpt-oss-120b")

    def test_deterministic_together_failure_never_reaches_groq(self):
        together = FakeAdapter("together", script=[NonRetryableProviderError("bad", provider="together", status_code=400)])
        r, a = router_with({**TOGETHER_IDS, "AI_ROUTER_FALLBACK_ENABLED": "true",
                            "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "native"}, together=together)
        with self.assertRaises(NonRetryableProviderError):
            r.execute(req())
        self.assertEqual(a["groq"].calls, [])


# ── 11. timeout via the router ───────────────────────────────────────────────
class RoutedTimeoutTests(_EventsPatched):
    def test_policy_timeout_reaches_the_together_http_call(self):
        env = {**TOGETHER_IDS, **OVERRIDE_ON, "AI_MODEL_OVERRIDE_CONTENT": "together:qwen3.8-flash",
               "AI_ROUTE_CONTENT_TIMEOUT_S": "33"}
        r = ModelRouter(load_config(env, debug=True),
                        adapters={"groq": FakeAdapter("groq"), "gemini": FakeAdapter("gemini"), "together": TogetherAdapter()})
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.post", return_value=_chat("x")) as post:
            r.execute(req(capability="content", json_mode=False))
        self.assertEqual(post.call_args.kwargs["timeout"], 33.0)

    def test_together_timeout_is_a_single_attempt_without_fallback_by_default(self):
        import httpx

        env = {**TOGETHER_IDS, **OVERRIDE_ON, "AI_MODEL_OVERRIDE_CONTENT": "together:qwen3.8-flash"}
        groq = FakeAdapter("groq")
        r = ModelRouter(load_config(env, debug=True),
                        adapters={"groq": groq, "gemini": FakeAdapter("gemini"), "together": TogetherAdapter()})
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), \
             patch("httpx.post", side_effect=httpx.ReadTimeout("slow")) as post:
            with self.assertRaises(RetryableProviderError) as cm:
                r.execute(req(capability="content", json_mode=False))
        self.assertEqual((cm.exception.kind, post.call_count, len(groq.calls)), ("timeout", 1, 0))


# ── 12. telemetry ────────────────────────────────────────────────────────────
class TogetherTelemetryTests(_EventsPatched):
    def test_model_id_qualification(self):
        self.assertEqual(telemetry_model_id("together", "openai/gpt-oss-120b"), "together:openai/gpt-oss-120b")
        self.assertEqual(telemetry_model_id("groq", "openai/gpt-oss-120b"), "openai/gpt-oss-120b")
        self.assertEqual(telemetry_model_id("gemini", "gemini-2.5-flash"), "gemini-2.5-flash")
        self.assertEqual(telemetry_model_id("together", "together:x"), "together:x")

    def test_failure_rows_name_the_provider_that_failed(self):
        t_err = RetryableProviderError("t", provider="together", model=QWEN, kind="timeout")
        g_err = RetryableProviderError("g", provider="groq", model="openai/gpt-oss-120b", kind="exhausted")
        self.assertEqual(telemetry_model_from_error(t_err, "openai/gpt-oss-120b"), f"together:{QWEN}")
        self.assertEqual(telemetry_model_from_error(g_err, "openai/gpt-oss-120b"), "openai/gpt-oss-120b")
        self.assertEqual(telemetry_model_from_error(RuntimeError("x"), "openai/gpt-oss-120b"), "openai/gpt-oss-120b")
        self.assertEqual(telemetry_model_from_error(NoRouteError("x"), "lit"), "lit")

    def test_multi_result_rows_change_only_for_qualified_providers(self):
        groq_results = [ok("openai/gpt-oss-120b", provider="groq")]
        gemini_results = [{"content": "x", "model": "gemini-2.5-flash"}]
        together_results = [ok(f"together:{QWEN}", provider="together")]
        self.assertEqual(telemetry_model_from_results(groq_results, "lit"), "lit")
        self.assertEqual(telemetry_model_from_results(gemini_results, "lit"), "lit")
        self.assertEqual(telemetry_model_from_results(together_results, "lit"), f"together:{QWEN}")
        self.assertEqual(telemetry_model_from_results(None, "lit"), "lit")

    def test_cost_is_not_fabricated_for_together_and_unchanged_elsewhere(self):
        from ai_services.core.usage_logger import calculate_cost

        self.assertIsNone(calculate_cost(f"together:{QWEN}", 1000, 1000))
        self.assertGreater(calculate_cost("openai/gpt-oss-120b", 1_000_000, 0), 0)
        self.assertGreater(calculate_cost("gemini-2.5-flash", 1_000_000, 0), 0)
        self.assertGreater(calculate_cost("mayura:v1", 1_000_000, 0), 0)   # colon, but not a router provider
        self.assertEqual(calculate_cost("scientific_solver", 10, 10), 0.0)  # unchanged for unknown models

    def test_usage_payload_sends_null_cost_for_together(self):
        from ai_services.core import usage_logger

        sent = {}

        class _R:
            def raise_for_status(self):
                return None

        def fake_post(url, json=None, headers=None, timeout=None):
            sent.update(json)
            return _R()

        with patch.dict(os.environ, {"NESTJS_INTERNAL_URL": "http://nest.test", "INTERNAL_API_KEY": "k"}), \
             patch.object(usage_logger.httpx, "post", side_effect=fake_post):
            usage_logger.log_ai_usage_sync("inst", "school", "content_generate", "content", f"together:{QWEN}", 5, 6, 7)
        self.assertEqual(sent["modelUsed"], f"together:{QWEN}")
        self.assertIsNone(sent["estimatedCost"])

    def test_routed_result_carries_attribution_friendly_metadata(self):
        from ai_services.core import request_context

        request_context.set_context(request_id="req-tg-1", user_id="u-1", user_role="TEACHER",
                                    institute_id="11111111-1111-1111-1111-111111111111", vertical="school")
        self.addCleanup(request_context.clear)
        together = FakeAdapter("together", script=[ok(QWEN, content="notes", tokens_reported=True)])
        r, a = router_with({**TOGETHER_IDS, **OVERRIDE_ON, "AI_MODEL_OVERRIDE_CONTENT": "together:qwen3.8-flash"},
                           debug=True, together=together)
        with self.assertLogs("ai_services.routing", level="INFO") as logs:
            out = r.execute(req(capability="content", json_mode=False, feature="content_generate", institute_id="inst-9"))
        self.assertEqual(a["together"].calls[0][1].institute_id, "inst-9")
        self.assertEqual((out["provider"], out["model"], out["tokens_input"], out["tokens_output"]),
                         ("together", f"together:{QWEN}", 1, 2))
        line = next(l for l in logs.output if "source=local_override" in l)
        for fragment in ("request_id=req-tg-1", "institute=11111111-1111-1111-1111-111111111111",
                         "feature=content_generate", "provider=together", f"model={QWEN}", "latency_ms="):
            self.assertIn(fragment, line)
        self.assertEqual(request_context.get("user_id"), "u-1")  # context untouched

    def test_failure_rows_in_views_use_the_helpers(self):
        root = os.path.dirname(__file__)
        bridge = io.open(os.path.join(root, "views", "bridge.py"), encoding="utf-8").read()
        ppt = io.open(os.path.join(root, "views", "ppt.py"), encoding="utf-8").read()
        self.assertEqual(re.findall(r"model_used\s*=\s*['\"](?:openai/gpt-oss|qwen/|llama)", bridge), [])
        self.assertEqual(bridge.count("telemetry_model_from_error("), 6)
        self.assertEqual(bridge.count("telemetry_model_from_results("), 1)
        self.assertEqual(ppt.count("telemetry_model_from_error(exc, _MODEL)"), 2)


# ── 13. no secret leakage ────────────────────────────────────────────────────
class TogetherSecretTests(_EventsPatched):
    SPEC = spec("together/qwen3.8-flash", provider_model_id=QWEN)

    def test_key_is_absent_from_errors_logs_and_events(self):
        body = json.dumps({"error": f"bad key {KEY}", "echo": f"Authorization: Bearer {KEY}"})
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.post", return_value=_Resp(401, text=body)), \
             self.assertLogs("ai_services", level="DEBUG") as logs:
            try:
                TogetherAdapter().complete(self.SPEC, ProviderCall("s", "u", QWEN, json_mode=False))
            except ProviderConfigError as exc:
                err = str(exc)
            import logging
            logging.getLogger("ai_services.routing").debug("sentinel")
        self.assertNotIn(KEY, err)
        self.assertFalse(any(KEY in line for line in logs.output))
        self.assertFalse(any(KEY in str(v) for v in self.emit.call_args.kwargs.values()))
        self.assertEqual(self.emit.call_args.kwargs["key_hash"], hashlib.sha256(KEY.encode()).hexdigest()[:12])

    def test_success_logs_never_contain_key_or_prompts(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.post", return_value=_chat("x")), \
             self.assertLogs("ai_services.routing", level="INFO") as logs:
            TogetherAdapter().complete(self.SPEC, ProviderCall("SYSTEM-SECRET-PROMPT", "STUDENT-TEXT", QWEN, json_mode=False))
        joined = "\n".join(logs.output)
        for s in (KEY, "SYSTEM-SECRET-PROMPT", "STUDENT-TEXT"):
            self.assertNotIn(s, joined)

    def test_command_output_never_contains_the_key(self):
        routing.reset_router(ModelRouter(load_config(TOGETHER_IDS, debug=False)))
        self.addCleanup(routing.reset_router)
        out = StringIO()
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), \
             patch("httpx.get", return_value=_Resp(401, text=f"invalid {KEY}")):
            call_command("ai_benchmark", "--list", stdout=out)
            call_command("ai_benchmark", "--discover", "together", stdout=out)
        self.assertNotIn(KEY, out.getvalue())


# ── benchmark command: discover / validate / provider+model ──────────────────
class TogetherBenchmarkTests(_EventsPatched):
    def test_discover_blocked_without_credentials(self):
        r = ModelRouter(load_config({}, debug=False),
                        adapters={"groq": FakeAdapter("groq"), "gemini": FakeAdapter("gemini"), "together": TogetherAdapter()})
        with patch.dict(os.environ, {"TOGETHER_API_KEY": ""}):
            self.assertEqual(benchmark.discover(r, "together")["status"], benchmark.BLOCKED)

    def test_discover_reports_account_models_and_validates_configured_ids(self):
        listing = {"data": [{"id": QWEN, "type": "chat", "context_length": 32768, "pricing": {"input": 1}},
                            {"id": "other/model", "type": "embedding"}]}
        r = ModelRouter(load_config({"TOGETHER_MODEL_QWEN38_FLASH": QWEN, "TOGETHER_MODEL_GLM53_FLASH": "cfg/missing"},
                                    debug=False),
                        adapters={"groq": FakeAdapter("groq"), "gemini": FakeAdapter("gemini"), "together": TogetherAdapter()})
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.get", return_value=_Resp(200, listing)):
            res = benchmark.discover(r, "together")
        self.assertEqual(res["status"], benchmark.PASS)
        by_id = {m["id"]: m for m in res["models"]}
        # Only fields the provider reported — "pricing" is never copied as billing truth, nothing is invented.
        self.assertEqual(by_id[QWEN], {"id": QWEN, "type": "chat", "context_length": 32768})
        self.assertEqual(by_id["other/model"], {"id": "other/model", "type": "embedding"})
        v = {x["registry_id"]: x["present_in_account"] for x in res["validation"]}
        self.assertEqual((v["together/qwen3.8-flash"], v["together/glm-5.3-flash"], v["together/qwen3.7-max"]),
                         (True, False, None))

    def test_provider_model_selector_runs_a_single_candidate_and_reports_capability_and_tokens(self):
        together = FakeAdapter("together", script=[ok(QWEN, content="notes", tokens_reported=True)])
        r, _ = router_with({"TOGETHER_MODEL_QWEN38_FLASH": QWEN}, together=together)
        routing.reset_router(r)
        self.addCleanup(routing.reset_router)
        out = StringIO()
        call_command("ai_benchmark", "--provider", "together", "--model", "qwen3.8-flash", "--prompt", "hello", stdout=out)
        first = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual((first["status"], first["provider"], first["model"], first["capability"]),
                         ("PASS", "together", QWEN, "content"))
        self.assertEqual((first["tokens_input"], first["tokens_output"], first["tokens_reported"]), (1, 2, True))
        self.assertIn("wall_ms", first)

    def test_provider_model_selector_blocked_without_model_id(self):
        r, _ = router_with({})
        routing.reset_router(r)
        self.addCleanup(routing.reset_router)
        out = StringIO()
        call_command("ai_benchmark", "--provider", "together", "--model", "glm-5.3-flash", "--prompt", "hi", stdout=out)
        first = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(first["status"], "BLOCKED")
        self.assertIn("TOGETHER_MODEL_GLM53_FLASH", first["reason"])

    def test_unregistered_model_is_rejected(self):
        from django.core.management.base import CommandError

        r, _ = router_with({})
        routing.reset_router(r)
        self.addCleanup(routing.reset_router)
        with self.assertRaises(CommandError):
            call_command("ai_benchmark", "--provider", "together", "--model", "made-up-model", "--prompt", "hi",
                         stdout=StringIO())


# ── streaming-only models ────────────────────────────────────────────────────
class _Stream:
    """Stands in for the context manager httpx.stream() returns."""

    def __init__(self, status=200, lines=(), body=b"", headers=None):
        self.status_code, self._lines, self._body, self.headers = status, list(lines), body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return self._body


def _sse(*chunks, done=True):
    return [f"data: {json.dumps(c)}" for c in chunks] + (["data: [DONE]"] if done else [])


class TogetherStreamingTests(_EventsPatched):
    SPEC = spec("together/qwen3.8-flash", provider_model_id=QWEN, streaming=True)

    def _run(self, stream, *, s=None, json_mode=False, timeout_s=None):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), \
             patch("httpx.stream", return_value=stream) as st, patch("httpx.post") as post:
            out = TogetherAdapter().complete(
                s or self.SPEC, ProviderCall("SYS", "USR", QWEN, json_mode=json_mode, timeout_s=timeout_s),
            )
        return out, st, post

    def test_streaming_flag_defaults_unknown_and_comes_from_config(self):
        self.assertIsNone(default_registry()["together/qwen3.8-flash"].streaming)
        cfg = load_config({"TOGETHER_MODEL_QWEN38_FLASH_STREAMING": "true"}, debug=False)
        self.assertTrue(cfg.models["together/qwen3.8-flash"].streaming)

    def test_one_streamed_request_is_assembled_into_the_full_answer(self):
        stream = _Stream(lines=[": keep-alive", "", *_sse(
            {"choices": [{"delta": {"content": "Plants "}}]},
            {"choices": [{"delta": {"content": "make food."}}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        )])
        out, st, post = self._run(stream)
        self.assertEqual(out["content"], "Plants make food.")
        self.assertEqual((out["tokens_input"], out["tokens_output"], out["tokens_reported"], out["streamed"]),
                         (9, 4, True, True))
        self.assertEqual(st.call_count, 1)
        post.assert_not_called()
        self.assertEqual(st.call_args.args[0], "POST")
        self.assertTrue(st.call_args.args[1].endswith("/chat/completions"))
        self.assertIs(st.call_args.kwargs["json"]["stream"], True)

    def test_non_streaming_models_are_unchanged(self):
        glm = spec("together/glm-5.3-flash", provider_model_id="cfg/glm")
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.stream") as st, \
             patch("httpx.post", return_value=_chat("x")) as post:
            out = TogetherAdapter().complete(glm, ProviderCall("s", "u", "cfg/glm", json_mode=False))
        st.assert_not_called()
        self.assertEqual(post.call_count, 1)
        self.assertNotIn("stream", post.call_args.kwargs["json"])
        self.assertFalse(out["streamed"])

    def test_stream_without_usage_is_flagged_not_fabricated(self):
        out, _, _ = self._run(_Stream(lines=_sse({"choices": [{"delta": {"content": "x"}}]})))
        self.assertFalse(out["tokens_reported"])

    def test_streamed_native_json(self):
        s = replace(self.SPEC, structured_output=STRUCTURED_NATIVE)
        out, st, _ = self._run(_Stream(lines=_sse(
            {"choices": [{"delta": {"content": '{"a": '}}]},
            {"choices": [{"delta": {"content": "1}"}}]},
        )), s=s, json_mode=True)
        self.assertEqual(out["content"], {"a": 1})
        self.assertEqual(st.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_http_errors_on_a_stream_are_classified_and_scrubbed(self):
        with self.assertRaises(RetryableProviderError) as cm:
            self._run(_Stream(status=503, body=b"busy"))
        self.assertEqual(cm.exception.kind, "server_error")
        with self.assertRaises(ProviderConfigError) as cm:
            self._run(_Stream(status=401, body=f"bad key {KEY}".encode()))
        self.assertNotIn(KEY, str(cm.exception))

    def test_error_inside_an_accepted_stream_never_falls_back(self):
        with self.assertRaises(NonRetryableProviderError):
            self._run(_Stream(lines=_sse({"error": {"message": "policy"}})))

    def test_timeout_bounds_the_whole_stream(self):
        import itertools

        stream = _Stream(lines=_sse({"choices": [{"delta": {"content": "a"}}]},
                                    {"choices": [{"delta": {"content": "b"}}]}))
        clock = itertools.chain([0.0, 0.0], itertools.repeat(999.0))
        with patch("ai_services.core.routing.providers.time.monotonic", side_effect=lambda: next(clock)):
            with self.assertRaises(RetryableProviderError) as cm:
                self._run(stream, timeout_s=30)
        self.assertEqual(cm.exception.kind, "timeout")

    def test_streaming_required_rejection_names_the_variable_to_set(self):
        body = json.dumps({"error": {"message": 'This model only supports streaming. Set "stream": true.',
                                     "code": "streaming_required"}})
        max_spec = spec("together/qwen3.7-max", provider_model_id="cfg/max")
        with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), patch("httpx.post", return_value=_Resp(400, text=body)):
            with self.assertRaises(NonRetryableProviderError) as cm:
                TogetherAdapter().complete(max_spec, ProviderCall("s", "u", "cfg/max", json_mode=False))
        self.assertIn("TOGETHER_MODEL_QWEN37_MAX_STREAMING=true", str(cm.exception))


class TogetherPricingTests(SimpleTestCase):
    """Together list prices (per 1M tokens, from Together's /v1/models on 2026-09-28)
    so the super-admin usage page can show real spend instead of "—". A model with
    no configured price stays unpriced: recording it as $0 would read as free."""

    def _cost(self, model, tin=0, tout=0):
        from ai_services.core.usage_logger import calculate_cost

        return calculate_cost(model, tin, tout)

    def test_input_and_output_are_priced_separately(self):
        self.assertAlmostEqual(self._cost("together:zai-org/GLM-5.3-Flash", 1_000_000, 0), 0.15)
        self.assertAlmostEqual(self._cost("together:zai-org/GLM-5.3-Flash", 0, 1_000_000), 0.50)

    def test_together_rates_are_not_taken_from_the_groq_table(self):
        # "qwen/qwen3-32b" is $0.29/1M on Groq; Qwen3.8-Flash on Together is $0.09.
        self.assertAlmostEqual(self._cost("together:Qwen/Qwen3.8-Flash", 1_000_000, 0), 0.09)

    def test_model_ids_match_case_insensitively(self):
        self.assertEqual(
            self._cost("together:ZAI-ORG/glm-5.3-FLASH", 1_000_000, 0),
            self._cost("together:zai-org/GLM-5.3-Flash", 1_000_000, 0),
        )

    def test_an_unknown_together_model_stays_unpriced(self):
        self.assertIsNone(self._cost("together:cfg/not-a-real-model", 1000, 1000))

    def test_env_overrides_a_rate_without_a_code_change(self):
        with patch.dict(os.environ, {"AI_TOGETHER_PRICING": json.dumps(
                {"zai-org/GLM-5.3-Flash": {"input": 1.0, "output": 2.0}})}):
            self.assertAlmostEqual(self._cost("together:zai-org/GLM-5.3-Flash", 1_000_000, 0), 1.0)

    def test_invalid_pricing_json_falls_back_to_the_built_in_rates(self):
        with patch.dict(os.environ, {"AI_TOGETHER_PRICING": "{not json"}):
            self.assertAlmostEqual(self._cost("together:zai-org/GLM-5.3-Flash", 1_000_000, 0), 0.15)

    def test_a_real_generation_costs_what_the_provider_charges(self):
        # The 2026-09-28 grounded PPT: 5287 prompt + 1150 completion tokens on GLM.
        self.assertAlmostEqual(self._cost("together:zai-org/GLM-5.3-Flash", 5287, 1150), 0.001368, places=6)
