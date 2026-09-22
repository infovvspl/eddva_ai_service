"""G3 Class-B Phase B1 — Django contract for teacher recording analysis.

Replaces a direct Groq call that lived in the NestJS school-teacher service and
named llama-3.3-70b-versatile, a model Groq decommissioned on 2026-08-16.
Routing through ai_call() resolves the model centrally instead.

The LLM layer is mocked throughout; no test makes a real provider call.
"""
from unittest.mock import patch

from django.conf import settings
from django.test import Client, TestCase

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.models import Institute
from ai_services.views.base import _cache

URL = "/teacher/analyze-recording"

# Comfortably above any per-run usage, and inside the int4 range the
# daily_*_cap columns use.
_CAP_HEADROOM = 2_000_000_000

ANALYSIS = {
    "overallScore": 8,
    "summary": "Clear delivery with good structure.",
    "clarity": {"score": 8, "feedback": "Explanations were well sequenced."},
    "pacing": {"score": 7, "feedback": "Slightly rushed near the end."},
    "contentCoverage": {"score": 9, "feedback": "Covered the syllabus fully."},
    "studentEngagement": {"score": 6, "feedback": "Few questions invited."},
    "languageQuality": {"score": 8, "feedback": "Simple, concrete vocabulary."},
    "suggestions": ["Ask more questions", "Slow the final section", "Add an example"],
    "strengths": ["Clear structure", "Accurate content"],
}

# The shape ai_call() expects back from LLMClient.complete().
LLM_RESULT = {
    "content": ANALYSIS,
    "model": "openai/gpt-oss-120b",
    "latency_ms": 1234.0,
    "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200},
}


class _Base(TestCase):
    def setUp(self):
        # Django validates Host on unsafe methods, and the test client sends
        # "testserver" — without this every POST 400s before reaching a view.
        if "testserver" not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS = list(settings.ALLOWED_HOSTS) + ["testserver"]
        # The usage limiter and the response cache are Redis-backed singletons:
        # their state outlives a test class and even a whole test run, and the
        # DB rollback between tests does not touch it. Whatever budget an
        # earlier suite recorded is therefore still charged against this
        # institute, and ai_call() returns 429 before it ever reaches the LLM
        # layer. Give this tenant headroom well above anything a suite spends —
        # caps are exercised in tests_metering, not here.
        self.institute = Institute.objects.create(
            name="Test School", slug="trc-school", api_key="k-trc-test",
            vertical="school", plan="free",
            daily_soft_cap=_CAP_HEADROOM, daily_hard_cap=_CAP_HEADROOM,
        )
        # Same reasoning for the cache: a hit left by a previous run would
        # short-circuit the call these tests assert on. flush_tenant handles a
        # missing/erroring Redis itself.
        _cache.flush_tenant(self.institute.slug)

        self.client = Client()
        self.headers = {"HTTP_X_API_KEY": "k-trc-test", "HTTP_X_VERTICAL": "school"}

    def post(self, body):
        return self.client.post(
            URL, data=body, content_type="application/json", **self.headers
        )


class ValidationTests(_Base):
    def test_1_missing_transcript_is_rejected(self):
        self.assertEqual(self.post({}).status_code, 400)

    def test_2_empty_and_whitespace_transcript_are_rejected(self):
        for value in ("", "   ", "\n\t "):
            self.assertEqual(
                self.post({"transcript": value}).status_code, 400,
                f"failed for {value!r}",
            )

    def test_2b_non_string_transcript_is_rejected(self):
        """A number or object would otherwise reach str.format and yield a
        confident-looking rubric over nonsense."""
        for value in (123, {"a": 1}, ["x"], None, True):
            self.assertEqual(
                self.post({"transcript": value}).status_code, 400,
                f"failed for {value!r}",
            )

    def test_3_valid_request_is_accepted(self):
        with patch("ai_services.views.base._llm.complete", return_value=LLM_RESULT):
            res = self.post({"transcript": "A" * 200, "title": "Algebra L1"})
        self.assertEqual(res.status_code, 200)


class ContractTests(_Base):
    def test_4_prompt_carries_the_transcript(self):
        transcript = "Photosynthesis converts light into chemical energy. " * 5
        with patch("ai_services.views.base._llm.complete",
                   return_value=LLM_RESULT) as m:
            self.post({"transcript": transcript, "title": "Bio L3"})
        kwargs = m.call_args.kwargs
        self.assertIn("Photosynthesis converts light", kwargs["user_prompt"])
        self.assertIn("Bio L3", kwargs["user_prompt"])

    def test_5_generation_parameters_match_the_previous_direct_call(self):
        with patch("ai_services.views.base._llm.complete",
                   return_value=LLM_RESULT) as m:
            self.post({"transcript": "B" * 200})
        kwargs = m.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_tokens"], 1024)

    def test_5b_transcript_is_capped_at_8000_chars(self):
        with patch("ai_services.views.base._llm.complete",
                   return_value=LLM_RESULT) as m:
            self.post({"transcript": "C" * 20000})
        self.assertEqual(m.call_args.kwargs["user_prompt"].count("C"), 8000)

    def test_5c_braces_in_a_transcript_do_not_break_rendering(self):
        """user_template goes through str.format(); a transcript containing
        { or } must not raise."""
        with patch("ai_services.views.base._llm.complete",
                   return_value=LLM_RESULT) as m:
            res = self.post({"transcript": "Set {x} where y={1,2}. " * 10})
        self.assertEqual(res.status_code, 200)
        self.assertIn("{x}", m.call_args.kwargs["user_prompt"])

    def test_6_response_carries_the_nine_fields(self):
        with patch("ai_services.views.base._llm.complete", return_value=LLM_RESULT):
            body = self.post({"transcript": "D" * 200}).json()
        for field in ("overallScore", "summary", "clarity", "pacing",
                      "contentCoverage", "studentEngagement", "languageQuality",
                      "suggestions", "strengths"):
            self.assertIn(field, body, f"missing {field}")
        # Returned unwrapped, matching every other bridge endpoint.
        self.assertEqual(body["overallScore"], 8)
        self.assertEqual(body["clarity"]["score"], 8)
        self.assertEqual(len(body["suggestions"]), 3)
        self.assertEqual(len(body["strengths"]), 2)

    def test_6b_model_resolves_centrally_not_to_the_decommissioned_llama(self):
        model = get_model_for_task("teacher_recording_analysis")
        self.assertNotIn("llama", model)
        self.assertEqual(model, "openai/gpt-oss-120b")

    def test_6c_system_prompt_defines_the_full_rubric(self):
        system = get_template("teacher_recording_analysis").system
        for field in ("overallScore", "summary", "clarity", "pacing",
                      "contentCoverage", "studentEngagement", "languageQuality",
                      "suggestions", "strengths"):
            self.assertIn(field, system)


class FailureTests(_Base):
    def test_7_provider_failure_is_not_reported_as_success(self):
        """A fake empty {} would be persisted by the caller as a completed
        analysis, so failure must surface as non-2xx."""
        with patch("ai_services.views.base._llm.complete",
                   side_effect=RuntimeError("all keys rate-limited")):
            res = self.post({"transcript": "E" * 200})
        self.assertGreaterEqual(res.status_code, 400)
        self.assertNotIn("overallScore", res.json())


class UsageAndScopeTests(_Base):
    def test_8_usage_is_logged_under_the_feature_name(self):
        with patch("ai_services.views.base._llm.complete", return_value=LLM_RESULT), \
             patch("ai_services.views.base._log_usage_to_db") as logged:
            self.post({"transcript": "F" * 200})
        self.assertTrue(logged.called)
        # (institute, institute_id, feature, result, ...)
        self.assertEqual(logged.call_args.args[2], "teacher_recording_analysis")

    def test_9_no_direct_provider_call_in_the_endpoint(self):
        import inspect

        from ai_services.views import bridge
        src = inspect.getsource(bridge)
        start = src.index("def analyze_teacher_recording(")
        block = src[start:start + 2500]
        for forbidden in ("api.groq.com", "requests.post", "httpx.post",
                          "urllib", "GROQ_API_KEY", "llama-3.3-70b-versatile"):
            self.assertNotIn(forbidden, block, f"found {forbidden}")
        self.assertIn("ai_call(", block)

    def test_10_django_does_not_persist_class_recordings(self):
        """Persistence and the processing/done/failed machine stay in NestJS."""
        import inspect

        from ai_services.views import bridge
        src = inspect.getsource(bridge)
        start = src.index("def analyze_teacher_recording(")
        block = src[start:start + 2500]
        for forbidden in ("class_recordings", "ai_teaching_analysis",
                          "UPDATE ", "INSERT "):
            self.assertNotIn(forbidden, block, f"found {forbidden}")
