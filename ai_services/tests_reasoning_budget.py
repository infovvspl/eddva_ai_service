"""A reasoning model can spend its whole output budget thinking.

Observed on 2026-09-15: a grounded call to GLM-5.3-Flash with max_tokens=8000
over a ~12k-token source returned HTTP 200 with an empty body after 70s. The
adapter called that an "empty response", which hid the cause. Measured directly
on 2026-09-28: all 8000 completion tokens were reasoning tokens, finish_reason
was "length", and `reasoning_effort=low` cut reasoning to 42 tokens while
producing MORE content (6055 chars) in half the time.

So: the budget case must be reported as itself, and an operator must be able to
record a per-model request parameter without touching code.
"""
import json
import os
from dataclasses import replace
from unittest.mock import patch

from django.test import SimpleTestCase

from ai_services.core.routing import NonRetryableProviderError, load_config
from ai_services.core.routing.providers import ProviderCall, TogetherAdapter
from ai_services.core.routing.registry import STRUCTURED_NATIVE, default_registry

KEY = "tgp-TEST-SECRET-0123456789abcdef"
GLM = "cfg/glm-fixture"
GLM_ENV = {"TOGETHER_MODEL_GLM53_FLASH": GLM}


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload


def reply(content, *, finish=None, reasoning_tokens=None, completion_tokens=7):
    usage = {"prompt_tokens": 11, "completion_tokens": completion_tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return _Resp(200, {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                       "usage": usage})


def glm_spec(**changes):
    base = replace(default_registry()["together/glm-5.3-flash"], provider_model_id=GLM)
    return replace(base, **changes) if changes else base


def run(spec, response, *, json_mode=False, max_tokens=8000):
    with patch.dict(os.environ, {"TOGETHER_API_KEY": KEY}), \
         patch("httpx.post", return_value=response) as post:
        out = TogetherAdapter().complete(
            spec, ProviderCall("SYS", "USR", GLM, json_mode=json_mode, max_tokens=max_tokens))
    return out, post


class ReasoningEffortConfigTests(SimpleTestCase):
    def test_effort_is_recorded_as_a_request_parameter(self):
        cfg = load_config({**GLM_ENV, "TOGETHER_MODEL_GLM53_FLASH_REASONING_EFFORT": "low"})
        spec = cfg.models["together/glm-5.3-flash"]
        self.assertEqual(spec.extra["request_params"], {"reasoning_effort": "low"})
        self.assertEqual(spec.metadata_source, "config")

    def test_effort_is_absent_unless_configured(self):
        spec = load_config(GLM_ENV).models["together/glm-5.3-flash"]
        self.assertEqual((spec.extra or {}).get("request_params"), None)

    def test_an_invalid_effort_is_warned_about_and_ignored(self):
        cfg = load_config({**GLM_ENV, "TOGETHER_MODEL_GLM53_FLASH_REASONING_EFFORT": "maximum"})
        self.assertEqual((cfg.models["together/glm-5.3-flash"].extra or {}).get("request_params"), None)
        self.assertTrue(any("REASONING_EFFORT" in w for w in cfg.warnings), cfg.warnings)

    def test_other_models_are_untouched(self):
        cfg = load_config({**GLM_ENV, "TOGETHER_MODEL_GLM53_FLASH_REASONING_EFFORT": "low"})
        for mid, spec in cfg.models.items():
            if mid != "together/glm-5.3-flash":
                self.assertEqual((spec.extra or {}).get("request_params"), None, mid)


class ReasoningEffortRequestTests(SimpleTestCase):
    def test_the_parameter_is_sent_with_the_request(self):
        spec = glm_spec(extra={"request_params": {"reasoning_effort": "low"}})
        _, post = run(spec, reply("notes", finish="stop"))
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")

    def test_nothing_extra_is_sent_when_unconfigured(self):
        _, post = run(glm_spec(), reply("notes", finish="stop"))
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])

    def test_a_recorded_parameter_cannot_overwrite_what_the_adapter_sets(self):
        spec = glm_spec(extra={"request_params": {"max_tokens": 5, "model": "sneaky/other"}})
        _, post = run(spec, reply("notes", finish="stop"), max_tokens=8000)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["max_tokens"], 8000)
        self.assertEqual(body["model"], GLM)


class BudgetExhaustedTests(SimpleTestCase):
    def test_empty_body_cut_short_reports_the_budget_not_an_empty_response(self):
        with self.assertRaises(NonRetryableProviderError) as ctx:
            run(glm_spec(), reply("", finish="length", reasoning_tokens=8000, completion_tokens=8000))
        exc = ctx.exception
        self.assertEqual(exc.kind, "budget_exhausted")
        self.assertIn("8000", str(exc))
        self.assertIn("reasoning", str(exc).lower())
        self.assertNotIn(KEY, str(exc))

    def test_the_message_points_at_the_fix(self):
        with self.assertRaises(NonRetryableProviderError) as ctx:
            run(glm_spec(), reply("   ", finish="length", reasoning_tokens=8000))
        self.assertIn("REASONING_EFFORT=low", str(ctx.exception))

    def test_json_mode_blames_the_budget_rather_than_malformed_json(self):
        # JSON-capable, as .env declares for this model: the structured-output
        # guard passes, so an empty body must be blamed on the budget and not
        # reported as "did not return valid JSON".
        spec = glm_spec(structured_output=STRUCTURED_NATIVE)
        with self.assertRaises(NonRetryableProviderError) as ctx:
            run(spec, reply("", finish="length", reasoning_tokens=8000), json_mode=True)
        self.assertEqual(ctx.exception.kind, "budget_exhausted")

    def test_an_empty_body_that_was_not_cut_short_is_still_an_empty_response(self):
        with self.assertRaises(NonRetryableProviderError) as ctx:
            run(glm_spec(), reply("", finish="stop"))
        self.assertEqual(ctx.exception.kind, "empty_response")

    def test_a_normal_answer_is_unaffected(self):
        out, _ = run(glm_spec(), reply("real notes", finish="length"))
        self.assertEqual(out["content"], "real notes")
