"""
No-Gemini policy and pilot cost guard for chapter lesson plans
(core/lesson_plan_route.py and the lesson-plan branch of bridge.generate_topic_content).

Everything here is mocked: no provider is called. Gemini's adapter, the Gemini
client used by the Odia path and the model router are all replaced by spies that
fail the test if they are touched.
"""

import json
import os
import tempfile
from unittest import mock
from django.test import TestCase

from ai_services.core import lesson_plan_route as lpr
from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache

GOOD_ENV = {"LESSON_PLAN_AI_MODEL": "together/glm-5.3-flash", "TOGETHER_MODEL_GLM53_FLASH": "zai-org/GLM-5.3-Flash",
            "LESSON_PLAN_COST_GUARD": "false"}


def _forbidden(name):
    def boom(*a, **k):
        raise AssertionError(f"{name} must never be called for lesson plans")
    return boom


class _Adapter:
    def __init__(self, result=None, exc=None):
        self.calls = []
        self.result = result or {"content": '{"schemaVersion": 2}', "model": "zai-org/GLM-5.3-Flash",
                                 "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
        self.exc = exc

    def complete(self, spec, call):
        self.calls.append((spec, call))
        if self.exc:
            raise self.exc
        return self.result


class ResolveRouteTests(TestCase):
    def test_the_configured_together_model_resolves(self):
        r = lpr.resolve_route(GOOD_ENV)
        self.assertEqual((r.provider, r.model_id, r.registry_id), ("together", "zai-org/GLM-5.3-Flash", "together/glm-5.3-flash"))

    def test_missing_configuration_fails_clearly(self):
        with self.assertRaisesRegex(lpr.LessonPlanRouteError, "LESSON_PLAN_AI_MODEL is not set"):
            lpr.resolve_route({})

    def test_a_gemini_primary_is_rejected_in_any_spelling(self):
        for ref in ("gemini/gemini-2.5-flash", "gemini:gemini-2.5-flash", "GEMINI/whatever"):
            with self.assertRaisesRegex(lpr.LessonPlanRouteError, "Gemini is not allowed"):
                lpr.resolve_route({**GOOD_ENV, "LESSON_PLAN_AI_MODEL": ref})

    def test_a_model_id_pointing_at_gemini_is_rejected(self):
        with self.assertRaises(lpr.LessonPlanRouteError):
            lpr.resolve_route({**GOOD_ENV, "TOGETHER_MODEL_GLM53_FLASH": "google/gemini-2.5-flash"})

    def test_a_reference_without_its_provider_is_ambiguous_and_rejected(self):
        for ref in ("glm-5.3-flash", "zai-org/GLM-5.3-Flash", " "):
            with self.assertRaises(lpr.LessonPlanRouteError):
                lpr.resolve_route({**GOOD_ENV, "LESSON_PLAN_AI_MODEL": ref})

    def test_unknown_or_unconfigured_models_are_rejected(self):
        with self.assertRaisesRegex(lpr.LessonPlanRouteError, "not a registered model"):
            lpr.resolve_route({**GOOD_ENV, "LESSON_PLAN_AI_MODEL": "together/no-such-model"})
        with self.assertRaisesRegex(lpr.LessonPlanRouteError, "no provider model id"):
            lpr.resolve_route({"LESSON_PLAN_AI_MODEL": "together/glm-5.3-flash"})


class CostGuardTests(TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "ledger.jsonl")
        self.env = {**GOOD_ENV, "LESSON_PLAN_COST_GUARD": "true", "LESSON_PLAN_LEDGER_PATH": self.ledger,
                    "LESSON_PLAN_CEILING_INR": "450", "LESSON_PLAN_MAX_ATTEMPTS": "6",
                    "LESSON_PLAN_PRICE_INPUT_INR_PER_M": "17.7", "LESSON_PLAN_PRICE_OUTPUT_INR_PER_M": "59"}
        lpr.init_ledger(self.ledger, note="test")

    def _call(self, adapter, env=None, **kw):
        with mock.patch.object(lpr, "_adapter", return_value=adapter):
            return lpr.complete(system_prompt="s" * 400, user_prompt="u" * 1600, max_tokens=8192, temperature=0.3,
                                institute_id="i", content_type="lesson_plan", env=env or self.env, **kw)

    def test_each_call_is_recorded_with_tokens_and_cost_no_secrets(self):
        self._call(_Adapter())
        summary = lpr.summarise(lpr.read_ledger(self.ledger))
        self.assertEqual(summary["attempts"], 1)
        row = summary["calls"][0]
        self.assertEqual((row["result"], row["input_tokens"], row["output_tokens"]), ("ok", 1000, 500))
        self.assertAlmostEqual(row["cost_inr"], (1000 * 17.7 + 500 * 59) / 1e6, places=6)
        raw = open(self.ledger, encoding="utf-8").read().lower()
        for secret_word in ("api_key", "authorization", "bearer"):
            self.assertNotIn(secret_word, raw)

    def test_a_call_that_would_cross_the_ceiling_is_refused_before_it_is_made(self):
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "exceed the ceiling"):
            self._call(adapter, env={**self.env, "LESSON_PLAN_CEILING_INR": "0.5"})
        self.assertEqual(adapter.calls, [])

    def test_the_attempt_cap_stops_further_calls(self):
        env = {**self.env, "LESSON_PLAN_MAX_ATTEMPTS": "2"}
        self._call(_Adapter(), env=env)
        self._call(_Adapter(), env=env)
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "Attempt cap"):
            self._call(adapter, env=env)
        self.assertEqual(adapter.calls, [])

    def test_failed_and_usage_less_calls_count_at_their_worst_case(self):
        from ai_services.core.routing.errors import ProviderError
        with self.assertRaises(ProviderError):
            self._call(_Adapter(exc=ProviderError("boom", provider="together", model="m", kind="server_error")))
        self._call(_Adapter(result={"content": "{}", "usage": {}}))
        s = lpr.summarise(lpr.read_ledger(self.ledger))
        self.assertEqual(s["attempts"], 2)
        self.assertTrue(all(c["cost_inr"] is None for c in s["calls"]))
        self.assertAlmostEqual(s["spent_inr"], sum(c["worst_case_inr"] for c in s["calls"]))

    # ── accounting (second pilot preparation) ────────────────────────────────

    def test_a_missing_ledger_refuses_instead_of_starting_a_new_count(self):
        adapter = _Adapter()
        env = {**self.env, "LESSON_PLAN_LEDGER_PATH": os.path.join(self.dir, "typo.jsonl")}
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "Cost ledger not found"):
            self._call(adapter, env=env)
        self.assertEqual(adapter.calls, [])
        self.assertFalse(os.path.exists(env["LESSON_PLAN_LEDGER_PATH"]))

    def test_init_ledger_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            lpr.init_ledger(self.ledger, note="again")

    def test_a_corrupt_ledger_refuses(self):
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "unreadable"):
            self._call(adapter)
        self.assertEqual(adapter.calls, [])

    def test_the_count_survives_a_restart(self):
        env = {**self.env, "LESSON_PLAN_MAX_ATTEMPTS": "2"}
        self._call(_Adapter(), env=env)
        self._call(_Adapter(), env=env)
        import importlib
        importlib.reload(lpr)  # a fresh process: nothing in memory, only the file
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "Attempt cap"):
            self._call(adapter, env=env)
        self.assertEqual(adapter.calls, [])

    def test_the_call_is_reserved_in_the_ledger_before_it_is_sent(self):
        seen = []

        class Peeking(_Adapter):
            def complete(inner, spec, call):
                seen.append([r["result"] for r in lpr.read_ledger(self.ledger) if r.get("type") != "init"])
                return super().complete(spec, call)
        self._call(Peeking())
        self.assertEqual(seen, [["started"]])

    def test_a_crash_mid_call_stays_counted_at_its_worst_case(self):
        with self.assertRaises(KeyboardInterrupt):
            self._call(_Adapter(exc=KeyboardInterrupt()))
        s = lpr.summarise(lpr.read_ledger(self.ledger))
        self.assertEqual((s["attempts"], s["calls"][0]["result"]), (1, "started"))
        self.assertGreater(s["estimated_inr"], 0)
        self.assertEqual(s["confirmed_inr"], 0)

    def test_actual_and_estimated_cost_are_kept_apart(self):
        self._call(_Adapter())
        self._call(_Adapter(result={"content": "{}", "usage": {}}))
        rows = [r for r in lpr.read_ledger(self.ledger) if r.get("result") == "ok"]
        self.assertEqual([r["cost_basis"] for r in rows], ["actual", "worst_case_estimate"])
        self.assertEqual([r["cost_inr"] is None for r in rows], [False, True])
        s = lpr.summarise(lpr.read_ledger(self.ledger))
        self.assertAlmostEqual(s["confirmed_inr"], rows[0]["cost_inr"])
        self.assertAlmostEqual(s["estimated_inr"], rows[1]["worst_case_inr"])

    def test_the_provider_reported_model_is_recorded_as_given(self):
        self._call(_Adapter(result={"content": "{}", "model": "zai-org/GLM-5.3-Flash", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        row = [r for r in lpr.read_ledger(self.ledger) if r.get("result") == "ok"][0]
        self.assertIsNone(row["provider_reported_model"])  # not reported: never filled in with the configured id
        self.assertFalse(row["model_mismatch"])

    def test_a_different_reported_model_stops_the_pilot(self):
        self._call(_Adapter(result={"content": "{}", "provider_reported_model": "other/model",
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "different model"):
            self._call(adapter)
        self.assertEqual(adapter.calls, [])

    def test_the_per_case_cap(self):
        env = {**self.env, "LESSON_PLAN_MAX_ATTEMPTS_PER_CASE": "2"}
        self._call(_Adapter(), env=env, label="Chemistry|Solutions")
        self._call(_Adapter(), env=env, label="Chemistry|Solutions")
        self._call(_Adapter(), env=env, label="Science|Chemical Reactions")
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "Case cap reached"):
            self._call(adapter, env=env, label="Chemistry|Solutions")
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "no case label"):
            self._call(adapter, env=env)
        self.assertEqual(adapter.calls, [])

    def test_repairs_can_be_refused_before_any_call_or_ledger_row(self):
        adapter = _Adapter()
        env = {**self.env, "LESSON_PLAN_GUARD_REFUSE_REPAIRS": "true"}
        with mock.patch.object(lpr, "_adapter", return_value=adapter):
            with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "Repair calls are disabled"):
                lpr.complete(system_prompt="s", user_prompt="x YOUR PREVIOUS REPLY WAS NOT USABLE. Problems: ...",
                             max_tokens=100, temperature=0.3, institute_id="i", content_type="lesson_plan", env=env)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(lpr.summarise(lpr.read_ledger(self.ledger))["attempts"], 0)
        self._call(adapter, env=env)  # a first attempt still goes through
        self.assertEqual(len(adapter.calls), 1)

    def test_a_partly_configured_guard_fails_closed(self):
        env = dict(self.env)
        env.pop("LESSON_PLAN_PRICE_OUTPUT_INR_PER_M")
        adapter = _Adapter()
        with self.assertRaisesRegex(lpr.LessonPlanBudgetError, "not fully configured"):
            self._call(adapter, env=env)
        self.assertEqual(adapter.calls, [])


class EndpointNoGeminiTests(TestCase):
    """The real /content/generate path: lesson plans never reach Gemini or the router."""
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="S", slug="s", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, adapter, env=GOOD_ENV, **body):
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(lpr, "_adapter", return_value=adapter), \
                mock.patch("ai_services.views.bridge.get_llm", side_effect=_forbidden("The model router")), \
                mock.patch("ai_services.core.routing.providers.GeminiAdapter.complete", _forbidden("GeminiAdapter")):
            return self.client.post(
                "/content/generate",
                data={"topicName": "Real Numbers", "subjectName": "Mathematics", "chapterName": "Real Numbers",
                      "contentType": "lesson_plan", "extraContext": "Number of periods: 2. Minutes per period: 45.",
                      "sourcePassages": [{"page_no": 3, "content": "Euclid's division lemma. " * 30, "tokens": 200}], **body},
                content_type="application/json", HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school",
            )

    def test_a_grounded_lesson_plan_uses_only_the_configured_model(self):
        adapter = _Adapter()
        resp = self._post(adapter)
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertEqual(len(adapter.calls), 1)
        spec, call = adapter.calls[0]
        self.assertEqual((spec.provider, call.model_id, call.max_tokens), ("together", "zai-org/GLM-5.3-Flash", 8192))
        self.assertEqual(resp.json()["_meta"]["provider"], "together")
        self.assertTrue(resp.json()["source"]["grounded"])

    def test_a_provider_failure_is_an_error_with_no_content_and_no_fallback(self):
        from ai_services.core.routing.errors import ProviderError
        adapter = _Adapter(exc=ProviderError("upstream 500", provider="together", model="m", kind="server_error"))
        resp = self._post(adapter)
        self.assertEqual(resp.status_code, 502)
        self.assertNotIn("content", resp.json())
        self.assertEqual(len(adapter.calls), 1)

    def test_a_repair_call_takes_the_same_route(self):
        adapter = _Adapter()
        self._post(adapter)
        self._post(adapter, extraContext="Number of periods: 2. YOUR PREVIOUS REPLY WAS NOT USABLE. Problems: ...")
        self.assertEqual([c[1].model_id for c in adapter.calls], ["zai-org/GLM-5.3-Flash"] * 2)

    def test_missing_or_gemini_configuration_refuses_without_calling_anything(self):
        adapter = _Adapter()
        env = {k: v for k, v in GOOD_ENV.items() if k != "LESSON_PLAN_AI_MODEL"}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LESSON_PLAN_AI_MODEL", None)
            resp = self._post(adapter, env=env)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("LESSON_PLAN_AI_MODEL", resp.json()["error"])
        resp = self._post(adapter, env={**GOOD_ENV, "LESSON_PLAN_AI_MODEL": "gemini/gemini-2.5-flash"})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(adapter.calls, [])

    def test_router_kill_switch_and_odia_still_never_reach_gemini(self):
        adapter = _Adapter()
        with mock.patch("ai_services.views.bridge.has_gemini_api_key", return_value=True):
            resp = self._post(adapter, env={**GOOD_ENV, "AI_ROUTER_ENABLED": "false"}, language="odia")
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertEqual(len(adapter.calls), 1)

    def test_an_unconfigured_provider_refuses_without_calling_anything(self):
        class Unconfigured:
            def is_configured(self):
                return False
        with mock.patch.dict(os.environ, GOOD_ENV, clear=False), \
                mock.patch("ai_services.core.routing.providers.default_adapters", return_value={"together": Unconfigured()}), \
                mock.patch("ai_services.views.bridge.get_llm", side_effect=_forbidden("The model router")), \
                mock.patch("ai_services.core.routing.providers.GeminiAdapter.complete", _forbidden("GeminiAdapter")):
            resp = self.client.post("/content/generate", data={"topicName": "T", "subjectName": "S", "chapterName": "C",
                                    "contentType": "lesson_plan"}, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("not configured", resp.json()["error"])

    def test_any_other_failure_is_a_502_with_one_call_and_no_fallback(self):
        for exc in (TimeoutError("read timed out"), RuntimeError("boom"), ValueError("bad body")):
            adapter = _Adapter(exc=exc)
            resp = self._post(adapter)
            self.assertEqual(resp.status_code, 502)
            self.assertNotIn("content", resp.json())
            self.assertEqual(len(adapter.calls), 1)

    def test_the_upload_extraction_takes_the_same_route(self):
        adapter = _Adapter()
        resp = self._post(adapter, contentType="lesson_plan_extract", sourceDocument="Day 1: introduce HCF.")
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertEqual(adapter.calls[0][1].model_id, "zai-org/GLM-5.3-Flash")

    def test_each_call_is_labelled_with_its_case_for_the_per_case_cap(self):
        with mock.patch.object(lpr, "complete", wraps=lpr.complete) as spy:
            self._post(_Adapter())
            self._post(_Adapter(), evalLabel="chem12-solutions")
        self.assertEqual([c.kwargs["label"] for c in spy.call_args_list], ["Mathematics|Real Numbers", "chem12-solutions"])

    def test_the_output_budget_grows_with_the_periods(self):
        # 10 periods were cut off at 8,192 tokens (24 Oct 2026): the budget is sized to the plan.
        adapter = _Adapter()
        self._post(adapter, extraContext="Number of periods: 10. Minutes per period: 45.")
        self._post(adapter, extraContext="Number of periods: 2. Minutes per period: 45.")
        self._post(adapter, contentType="lesson_plan_extract", sourceDocument="Day 1: introduce HCF.", extraContext="Minutes per period: 45.")
        self._post(adapter, contentType="lesson_plan_section", extraContext="Part to rewrite: period 2.")
        self.assertEqual([c[1].max_tokens for c in adapter.calls], [15000, 8192, 16384, 8192])

    def test_the_output_budget_is_bounded(self):
        from ai_services.views.bridge import _lesson_plan_max_tokens as budget
        self.assertEqual(budget("lesson_plan", "Number of periods: 12."), 17400)
        self.assertEqual(budget("lesson_plan", "Number of periods: 20."), 24576)   # the ceiling
        self.assertEqual(budget("lesson_plan", "Number of periods: 99."), 24576)
        self.assertEqual(budget("lesson_plan", "Number of periods: 1."), 8192)
        self.assertEqual(budget("lesson_plan", ""), 16384)                         # no count given: the generous default

    def test_the_section_rewrite_takes_the_same_route(self):
        adapter = _Adapter()
        resp = self._post(adapter, contentType="lesson_plan_section", extraContext="Part to rewrite: period 2.")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(adapter.calls[0][1].model_id, "zai-org/GLM-5.3-Flash")


class RegisteredSpecTests(TestCase):
    """The adapter must get the registered spec (recorded request params) and low reasoning effort."""

    def test_recorded_settings_and_low_reasoning_reach_the_adapter(self):
        adapter = _Adapter()
        env = {**GOOD_ENV, "TOGETHER_MODEL_GLM53_FLASH_REASONING_EFFORT": "low", "TOGETHER_MODEL_GLM53_FLASH_STREAMING": "true"}
        with mock.patch.object(lpr, "_adapter", return_value=adapter):
            lpr.complete(system_prompt="s", user_prompt="u", max_tokens=8192, temperature=0.3,
                         institute_id="i", content_type="lesson_plan", env=env)
        spec, call = adapter.calls[0]
        self.assertEqual(spec.id, "together/glm-5.3-flash")
        self.assertEqual((spec.extra or {}).get("request_params", {}).get("reasoning_effort"), "low")
        self.assertTrue(spec.streaming)
        self.assertEqual(call.reasoning_effort, "low")

    def test_an_invalid_reasoning_override_is_not_sent(self):
        adapter = _Adapter()
        with mock.patch.object(lpr, "_adapter", return_value=adapter):
            lpr.complete(system_prompt="s", user_prompt="u", max_tokens=100, temperature=0.3, institute_id="i",
                         content_type="lesson_plan", env={**GOOD_ENV, "LESSON_PLAN_REASONING_EFFORT": "extreme"})
        self.assertIsNone(adapter.calls[0][1].reasoning_effort)
