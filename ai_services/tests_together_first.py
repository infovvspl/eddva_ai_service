"""
Tests for the Together-first routing migration (feature/ai-model-router).

Two guarantees over the audited call-site catalog:
  * Together configured and verified -> every routed call site is served by the
    Together model for its capability;
  * Together absent -> every routed call site keeps its exact current provider
    and model (source=legacy). This is what keeps deployed environments, which
    carry no Together configuration, unchanged.

Plus the grounded-generation migration contract: retrieval and prompts are
untouched, legacy Gemini receives byte-identical parameters, and the teacher's
PPT badge reason codes stay meaningful. No network.
"""
import io
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core import routing
from ai_services.core.routing import (
    AIRequest,
    ModelRouter,
    NonRetryableProviderError,
    ProviderConfigError,
    RetryableProviderError,
    load_config,
)
from ai_services.core.routing.catalog import BLOCKED, CALL_SITES, ROUTED, SPECIALIZED, routed_sites

FULL_TOGETHER = {
    "TOGETHER_API_KEY": "test-key-not-real-0000",
    "TOGETHER_MODEL_GPT_OSS_120B": "cfg/gpt-oss",
    "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "native",
    "TOGETHER_MODEL_QWEN38_FLASH": "cfg/qwen-flash",
    "TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT": "native",
    "TOGETHER_MODEL_GLM53_FLASH": "cfg/glm-flash",
    "TOGETHER_MODEL_GLM53_FLASH_STRUCTURED_OUTPUT": "native",
    "TOGETHER_MODEL_GLM53_FLASH_GROUNDING": "true",
    "TOGETHER_MODEL_GLM53_FLASH_LONG_CONTEXT": "true",
    "TOGETHER_MODEL_DEEPSEEK_V4_FLASH": "cfg/deepseek-flash",
    "TOGETHER_MODEL_DEEPSEEK_V4_FLASH_STRUCTURED_OUTPUT": "native",
    "TOGETHER_MODEL_QWEN37_MAX": "cfg/qwen-max",
}
EXPECTED_BY_CAPABILITY = {
    "reasoning": "together/gpt-oss-120b",
    "lightweight": "together/deepseek-v4-flash",
    "bulk_text": "together/deepseek-v4-flash",
    "content": "together/qwen3.8-flash",
    "grounded": "together/glm-5.3-flash",
}


class FakeAdapter:
    def __init__(self, name, configured=True, result=None):
        self.name, self.configured, self.calls = name, configured, []
        self.result = result

    def is_configured(self):
        return self.configured

    def complete(self, spec, call):
        self.calls.append((spec, call))
        return dict(self.result or {"content": "ok", "model": call.model_id, "latency_ms": 1,
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def fake_adapters(together_configured=True):
    return {"groq": FakeAdapter("groq"), "gemini": FakeAdapter("gemini"),
            "together": FakeAdapter("together", configured=together_configured)}


def site_request(site):
    return AIRequest(system_prompt="", user_prompt="", model=site.current_model, provider=site.current_provider,
                     feature=site.route_feature, capability=site.capability, json_mode=site.json_mode,
                     requires_grounding=site.grounding)


class CatalogTests(SimpleTestCase):
    def test_catalog_is_well_formed(self):
        ids = [c.id for c in CALL_SITES]
        self.assertEqual(len(ids), len(set(ids)))
        for c in CALL_SITES:
            self.assertIn(c.status, (ROUTED, SPECIALIZED, BLOCKED), c.id)
            if c.status != ROUTED:
                self.assertTrue(c.notes, f"{c.id}: a non-routed site must say why")
        self.assertGreaterEqual(len(routed_sites()), 40)

    def test_every_routed_site_goes_to_together_by_capability_when_configured(self):
        router = ModelRouter(load_config(FULL_TOGETHER, debug=False), adapters=fake_adapters())
        for site in routed_sites():
            plan = router.plan(site_request(site))
            c = plan.candidates[0]
            self.assertEqual(c.spec.provider, "together", site.id)
            self.assertEqual(c.source, "policy", site.id)
            self.assertEqual(c.spec.id, EXPECTED_BY_CAPABILITY[plan.capability], site.id)

    def test_every_routed_site_keeps_its_exact_current_route_without_together(self):
        for together_configured, env in ((False, {}), (False, FULL_TOGETHER), (True, {})):
            router = ModelRouter(load_config(env, debug=False), adapters=fake_adapters(together_configured))
            for site in routed_sites():
                plan = router.plan(site_request(site))
                c = plan.candidates[0]
                self.assertEqual((c.spec.provider, c.model_id, c.source),
                                 (site.current_provider, site.current_model, "legacy"),
                                 f"{site.id} (together_configured={together_configured}, env={bool(env)})")
                self.assertEqual(len(plan.candidates), 1, site.id)

    def test_specialized_sites_are_never_routed_through_llmclient(self):
        non_routed = {c.id for c in CALL_SITES if c.status != ROUTED}
        for prefix in ("odia.", "vision.", "textbook.pdf_ocr", "speech.", "translate.", "image."):
            self.assertTrue(any(i.startswith(prefix) for i in non_routed), prefix)

    def test_grounded_sites_use_the_grounded_capability(self):
        grounded = [c for c in routed_sites() if c.grounding]
        self.assertEqual({c.id for c in grounded}, {"content.grounded", "ppt.grounded_deck"})
        self.assertTrue(all(c.capability == "grounded" and c.current_provider == "gemini" for c in grounded))

    def test_unverified_grounding_keeps_grounded_generation_on_gemini(self):
        env = {k: v for k, v in FULL_TOGETHER.items() if not k.endswith("_GROUNDING")}
        router = ModelRouter(load_config(env, debug=False), adapters=fake_adapters())
        for site in (c for c in routed_sites() if c.grounding):
            c = router.plan(site_request(site)).candidates[0]
            self.assertEqual((c.spec.provider, c.source), ("gemini", "legacy"), site.id)


class LegacyLoggingTests(SimpleTestCase):
    def test_legacy_selection_is_logged_once_per_reason(self):
        from ai_services.core.routing import router as router_mod

        router_mod._LEGACY_WARNED.clear()
        r = ModelRouter(load_config({}, debug=False), adapters=fake_adapters())
        request = AIRequest(system_prompt="", user_prompt="", model="openai/gpt-oss-120b", capability="reasoning")
        with self.assertLogs("ai_services.routing", level="WARNING") as logs:
            for _ in range(5):
                r.plan(request)
        self.assertEqual(sum("AI route legacy" in line for line in logs.output), 1)


class _RouterReset(SimpleTestCase):
    def use(self, env, adapters=None, debug=False):
        routing.reset_router(ModelRouter(load_config(env, debug=debug), adapters=adapters))
        self.addCleanup(routing.reset_router)


class LegacyGeminiExactnessTests(_RouterReset):
    """Without Together, grounded calls reach gemini_client exactly as before."""

    RAW_SYSTEM = "SOURCE RULES — raw grounded system prompt"

    def _call(self, json_mode):
        from ai_services.core import gemini_client as gc
        from ai_services.core.llm_client import LLMClient

        return LLMClient().complete(
            system_prompt=self.RAW_SYSTEM, user_prompt="USER + SOURCE BLOCK", model=gc.DEFAULT_MODEL,
            provider="gemini", capability="grounded", feature="content_generate", json_mode=json_mode,
            temperature=0.3, max_tokens=8000, legacy_prompt_shaping=False, requires_grounding=True,
            min_context_tokens=20000,
        )

    def test_text_grounded_call_is_byte_identical(self):
        self.use({})
        payload = {"content": "grounded text", "model": "gemini-2.5-flash", "latency_ms": 5,
                   "tokens_input": 3, "tokens_output": 4}
        with patch("ai_services.core.gemini_client.complete_text", return_value=payload) as ct:
            out = self._call(json_mode=False)
        self.assertEqual(ct.call_args.kwargs, {
            "system_prompt": self.RAW_SYSTEM, "user_prompt": "USER + SOURCE BLOCK",
            "model": "gemini-2.5-flash", "temperature": 0.3, "max_output_tokens": 8000,
        })
        self.assertEqual((out["content"], out["model"], out["provider"], out["route_source"]),
                         ("grounded text", "gemini-2.5-flash", "gemini", "legacy"))

    def test_json_grounded_call_is_byte_identical(self):
        self.use({})
        payload = {"content": {"slides": []}, "model": "gemini-2.5-flash", "latency_ms": 5}
        with patch("ai_services.core.gemini_client.complete_json", return_value=payload) as cj:
            self._call(json_mode=True)
        self.assertEqual(cj.call_args.kwargs["system_prompt"], self.RAW_SYSTEM)
        self.assertEqual(cj.call_args.kwargs["max_output_tokens"], 8000)

    def test_verified_together_grounded_model_serves_grounded_calls(self):
        adapters = fake_adapters()
        self.use(FULL_TOGETHER, adapters=adapters)
        with patch("ai_services.core.gemini_client.complete_text") as ct:
            out = self._call(json_mode=False)
        ct.assert_not_called()
        spec, call = adapters["together"].calls[0]
        self.assertEqual((spec.id, call.model_id, call.system_prompt, call.legacy_prompt_shaping, call.max_tokens),
                         ("together/glm-5.3-flash", "cfg/glm-flash", self.RAW_SYSTEM, False, 8000))
        self.assertEqual(out["model"], "together:cfg/glm-flash")

    def test_kill_switch_uses_gemini_directly(self):
        self.use({"AI_ROUTER_ENABLED": "false"})
        payload = {"content": "x", "model": "gemini-2.5-flash", "latency_ms": 1}
        with patch("ai_services.core.gemini_client.complete_text", return_value=payload) as ct, \
             patch.object(ModelRouter, "execute") as execute:
            self._call(json_mode=False)
        execute.assert_not_called()
        self.assertEqual(ct.call_args.kwargs["system_prompt"], self.RAW_SYSTEM)


class CanRouteTests(_RouterReset):
    KW = dict(capability="grounded", model="gemini-2.5-flash", provider="gemini", json_mode=False,
              requires_grounding=True)

    def test_legacy_gemini_availability_decides_without_together(self):
        from ai_services.core.llm_client import LLMClient

        self.use({})
        with patch("ai_services.core.gemini_client.is_available", return_value=True):
            self.assertTrue(LLMClient().can_route(**self.KW))
        with patch("ai_services.core.gemini_client.is_available", return_value=False):
            self.assertFalse(LLMClient().can_route(**self.KW))

    def test_verified_together_model_is_routable_even_without_gemini(self):
        from ai_services.core.llm_client import LLMClient

        self.use(FULL_TOGETHER, adapters=fake_adapters())
        with patch("ai_services.core.gemini_client.is_available", return_value=False):
            self.assertTrue(LLMClient().can_route(**self.KW))

    def test_can_route_makes_no_provider_call(self):
        from ai_services.core.llm_client import LLMClient

        adapters = fake_adapters()
        self.use(FULL_TOGETHER, adapters=adapters)
        LLMClient().can_route(**self.KW)
        self.assertEqual([a.calls for a in adapters.values()], [[], [], []])


class GroundedCallSiteContractTests(SimpleTestCase):
    ROOT = os.path.dirname(__file__)

    def _src(self, name):
        return io.open(os.path.join(self.ROOT, "views", name), encoding="utf-8").read()

    def test_grounded_views_route_through_llmclient(self):
        bridge, ppt = self._src("bridge.py"), self._src("ppt.py")
        self.assertNotIn("_gc.complete_text(", bridge)
        self.assertNotIn("_gc.complete_json(", ppt)
        self.assertNotIn("_gc.is_available()", ppt)
        for src in (bridge, ppt):
            self.assertIn('capability="grounded"', src)
            self.assertIn("legacy_prompt_shaping=False", src)
            self.assertIn("requires_grounding=True", src)
            self.assertIn(".can_route(", src)

    def test_grounded_retrieval_and_citation_code_is_unchanged(self):
        bridge = self._src("bridge.py")
        for marker in ("_gr.select_source(", "_gr.format_source_block(", '"citations": grounded_citations',
                       "system_prompt = ungrounded_system_prompt"):
            self.assertIn(marker, bridge)

    def test_ppt_badge_reason_codes(self):
        from ai_services.views.ppt import _classify_grounded_failure

        together_errors = [
            RetryableProviderError("Together 429", provider="together", model="m", status_code=429, kind="rate_limited"),
            ProviderConfigError("Together 401", provider="together", kind="auth"),
            NonRetryableProviderError("bad json", provider="together", kind="structured_output"),
        ]
        for exc in together_errors:
            self.assertEqual(_classify_grounded_failure(exc), "unavailable", exc)
        try:
            try:
                raise RuntimeError("All 3 Gemini key(s) failed for JSON completion: 429 RESOURCE_EXHAUSTED quota")
            except RuntimeError as inner:
                raise RetryableProviderError("Gemini key rotation exhausted", provider="gemini",
                                             kind="exhausted") from inner
        except RetryableProviderError as wrapped:
            self.assertEqual(_classify_grounded_failure(wrapped), "gemini_exhausted")

    def test_frontend_known_reason_codes_still_produced(self):
        ppt = self._src("ppt.py")
        for code in ("gemini_unavailable", "no_relevant_passages", '"unavailable"'):
            self.assertIn(code, ppt)
