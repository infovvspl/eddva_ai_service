"""Per-call reasoning effort: the PPT text calls ask for "low" without changing
the model for every other feature.

Measured on a real 21-page chapter (three decks each): output tokens fell from
10.3k-18.2k to 6.6k-6.8k and the text steps from 121-144s to 26-29s, with the
same key-fact coverage. No network: providers and the LLM are fakes.
"""
import json
import os
from dataclasses import replace
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core.routing import AIRequest, ModelRouter, load_config
from ai_services.core.routing.providers import (ProviderCall, TogetherAdapter,
                                                accepts_reasoning_effort)
from ai_services.core.routing.registry import STRUCTURED_PROMPTED, default_registry

TOGETHER_ID = "openai/gpt-oss-120b"


class _Resp:
    status_code = 200
    headers = {}

    def __init__(self):
        self.text = json.dumps(self.json())

    def json(self):
        return {"model": TOGETHER_ID, "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def _spec(model_id=TOGETHER_ID, request_params=None):
    spec = replace(default_registry()["together/gpt-oss-120b"], provider_model_id=model_id,
                   structured_output=STRUCTURED_PROMPTED)
    if request_params is not None:
        spec = replace(spec, extra={**(spec.extra or {}), "request_params": request_params})
    return spec


class TogetherBodyTests(SimpleTestCase):
    def _body(self, call, spec=None):
        with mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "k"}), \
             mock.patch("httpx.post", return_value=_Resp()) as post:
            TogetherAdapter().complete(spec or _spec(call.model_id), call)
        return post.call_args.kwargs["json"]

    def test_a_per_call_effort_is_sent(self):
        body = self._body(ProviderCall("s", "u", TOGETHER_ID, json_mode=False, reasoning_effort="low"))
        self.assertEqual(body["reasoning_effort"], "low")

    def test_no_per_call_effort_leaves_the_body_as_before(self):
        body = self._body(ProviderCall("s", "u", TOGETHER_ID, json_mode=False))
        self.assertNotIn("reasoning_effort", body)

    def test_the_per_call_effort_wins_over_the_configured_one(self):
        spec = _spec(request_params={"reasoning_effort": "medium"})
        body = self._body(ProviderCall("s", "u", TOGETHER_ID, json_mode=False, reasoning_effort="low"), spec)
        self.assertEqual(body["reasoning_effort"], "low")

    def test_without_a_per_call_effort_the_configured_one_still_applies(self):
        spec = _spec(request_params={"reasoning_effort": "medium"})
        body = self._body(ProviderCall("s", "u", TOGETHER_ID, json_mode=False), spec)
        self.assertEqual(body["reasoning_effort"], "medium")

    def test_never_sent_to_a_model_that_may_reject_it(self):
        other = "Qwen/Qwen3.8-Flash"
        body = self._body(ProviderCall("s", "u", other, json_mode=False, reasoning_effort="low"), _spec(other))
        self.assertNotIn("reasoning_effort", body)

    def test_an_unknown_value_is_ignored(self):
        body = self._body(ProviderCall("s", "u", TOGETHER_ID, json_mode=False, reasoning_effort="max"))
        self.assertNotIn("reasoning_effort", body)

    def test_which_models_accept_it(self):
        self.assertTrue(accepts_reasoning_effort("openai/gpt-oss-120b"))
        self.assertTrue(accepts_reasoning_effort("zai-org/GLM-5.3-Flash"))
        self.assertFalse(accepts_reasoning_effort("Qwen/Qwen3.8-Flash"))
        self.assertFalse(accepts_reasoning_effort(None))


class RouterPassThroughTests(SimpleTestCase):
    def test_the_router_hands_the_effort_to_the_provider(self):
        seen = []

        class Fake:
            name = "together"

            def is_configured(self):
                return True

            def complete(self, spec, call):
                seen.append(call.reasoning_effort)
                return {"content": {"ok": True}, "model": call.model_id, "latency_ms": 1,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        env = {"TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID,
               "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "prompted"}
        router = ModelRouter(load_config(env), adapters={"together": Fake(), "groq": Fake(), "gemini": Fake()})
        router.execute(AIRequest(system_prompt="s", user_prompt="u", capability="reasoning",
                                 reasoning_effort="low"))
        self.assertEqual(seen, ["low"])


class LLMClientTests(SimpleTestCase):
    def test_the_client_forwards_the_effort(self):
        from ai_services.core import llm_client, routing

        router = mock.Mock()
        router.execute.return_value = {"content": {}}
        with mock.patch.object(routing, "router_enabled", return_value=True), \
             mock.patch.object(routing, "get_router", return_value=router):
            llm_client.LLMClient.__new__(llm_client.LLMClient).complete(
                system_prompt="s", user_prompt="u", model="m", reasoning_effort="low")
        self.assertEqual(router.execute.call_args.args[0].reasoning_effort, "low")


class GroqFallbackTests(SimpleTestCase):
    """A call that falls back to Groq keeps the effort it asked for."""

    def _kwargs(self, model, effort):
        from ai_services.core import llm_client

        sent = {}

        class _Choice:
            message = mock.Mock(content='{"ok": true}')
            finish_reason = "stop"

        class _Client:
            def __init__(self, api_key=None):
                self.chat = mock.Mock()
                self.chat.completions.create.side_effect = self._create

            def _create(self, **kw):
                sent.update(kw)
                return mock.Mock(choices=[_Choice()], usage=mock.Mock(prompt_tokens=1, completion_tokens=1,
                                                                       total_tokens=2), model=model)

        with mock.patch("groq.Groq", _Client), \
             mock.patch.object(llm_client, "GROQ_API_KEYS", ["gsk_test_key_1234567890"]):
            try:
                llm_client.LLMClient.__new__(llm_client.LLMClient)._complete_groq(
                    system_prompt="s", user_prompt="u", model=model, json_mode=True,
                    reasoning_effort=effort)
            except Exception:
                pass
        return sent

    def test_gpt_oss_on_groq_gets_the_effort(self):
        self.assertEqual(self._kwargs("openai/gpt-oss-120b", "low").get("reasoning_effort"), "low")

    def test_only_requests_that_reach_gpt_oss_get_it(self):
        # This client maps some retired model names onto gpt-oss, so the rule is
        # about the model actually sent, not the one the caller named.
        for named in ("llama-3.3-70b-versatile", "gemma2-9b-it", "openai/gpt-oss-20b"):
            kw = self._kwargs(named, "low")
            self.assertEqual("reasoning_effort" in kw, "gpt-oss" in kw.get("model", "").lower(), named)

    def test_unsupported_values_are_not_sent(self):
        self.assertNotIn("reasoning_effort", self._kwargs("openai/gpt-oss-120b", "none"))
        self.assertNotIn("reasoning_effort", self._kwargs("openai/gpt-oss-120b", None))
