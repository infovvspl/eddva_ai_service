"""
Regression tests for the deterministic-400 retry amplification fix.

A DEV student doubt request exposed this: Groq returned 400 json_validate_failed
("max completion tokens reached before generating a valid document") for the doubt
detector, and LLMClient treated it as transient — rotating 3 rounds x 20 keys, up to
60 provider calls, holding one request for ~63.6s.

The failure is REQUEST-shaped, not KEY-shaped: identical prompt -> identical
rejection on every key. These tests pin the fail-fast behaviour and prove the
genuinely transient classes still rotate.

Plain unittest (no Django settings, no DB, no network) so it runs standalone.
"""
import unittest
from unittest.mock import patch


# ── the classifier under test ────────────────────────────────────────────────
# _is_deterministic_request_error is a closure inside LLMClient.complete(), so it
# is reproduced here from the source to test the predicate directly, and the
# rotation behaviour itself is covered end-to-end by the FakeGroq tests below.
def _classify(msg: str) -> bool:
    m = (msg or "").lower()
    return any(
        token in m
        for token in (
            "json_validate_failed",
            "failed_generation",
            "max completion tokens reached before generating a valid document",
        )
    )


class DeterministicErrorClassificationTests(unittest.TestCase):
    """A. / B. the observed Groq error forms must classify as deterministic."""

    def test_json_validate_failed_is_deterministic(self):
        self.assertTrue(_classify(
            "Error code: 400 - {'error': {'code': 'json_validate_failed', "
            "'message': 'Failed to generate JSON. Please adjust your prompt.'}}"
        ))

    def test_failed_generation_is_deterministic(self):
        self.assertTrue(_classify(
            "failed_generation: max completion tokens reached before generating a valid document"
        ))

    def test_max_completion_tokens_phrase_is_deterministic(self):
        self.assertTrue(_classify(
            "max completion tokens reached before generating a valid document"
        ))

    def test_case_insensitive(self):
        self.assertTrue(_classify("JSON_VALIDATE_FAILED"))

    # C./D./E. transient + key-shaped classes must NOT be caught by this predicate
    def test_rate_limit_is_not_deterministic(self):
        self.assertFalse(_classify("Rate limit reached for model. Please try again in 3.5s"))

    def test_5xx_is_not_deterministic(self):
        self.assertFalse(_classify("Error code: 503 - service unavailable"))

    def test_timeout_is_not_deterministic(self):
        self.assertFalse(_classify("Request timed out."))

    def test_invalid_key_is_not_deterministic(self):
        self.assertFalse(_classify("Invalid API Key"))
        self.assertFalse(_classify("organization has been restricted"))

    def test_413_is_not_deterministic(self):
        # 413 has its own dedicated fail-fast branch; the two must not overlap.
        self.assertFalse(_classify("Request too large ... reduce your message size"))

    def test_generic_400_still_rotates(self):
        # Deliberately narrow: not every 400 is request-shaped.
        self.assertFalse(_classify("Error code: 400 - bad request"))


# ── end-to-end rotation behaviour against a fake Groq ────────────────────────
class _Err(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class RotationBehaviourTests(unittest.TestCase):
    """Drives LLMClient.complete() with a fake Groq client and counts attempts."""

    def _run(self, raise_exc, n_keys=20, rate_limit=False):
        """Returns (attempts, outcome). Groq is imported INSIDE complete(), so the
        patch target is the `groq` module itself, not the llm_client namespace."""
        import groq as groq_mod
        import ai_services.core.llm_client as lc

        attempts = {"n": 0}

        class FakeRateLimit(Exception):
            pass

        class FakeCompletions:
            def create(self, **kwargs):
                attempts["n"] += 1
                raise FakeRateLimit("Rate limit reached, try again in 2s") if rate_limit else raise_exc()

        class FakeChat:
            completions = FakeCompletions()

        class FakeGroq:
            def __init__(self, api_key=None, **kw):
                self.chat = FakeChat()

        keys = [f"k{i}" for i in range(n_keys)]
        with patch.object(groq_mod, "Groq", FakeGroq),              patch.object(groq_mod, "RateLimitError", FakeRateLimit),              patch.object(lc, "GROQ_API_KEYS", keys),              patch.object(lc, "_DISABLED_GROQ_KEYS", set()),              patch.object(lc.time, "sleep", lambda *_a, **_k: None):
            client = lc.LLMClient()
            try:
                client.complete(system_prompt="s", user_prompt="u", model="openai/gpt-oss-20b",
                                max_tokens=60, json_mode=True)
                return attempts["n"], "returned"
            except Exception:
                return attempts["n"], "raised"

    def test_A_json_validate_failed_makes_exactly_one_attempt(self):
        n, outcome = self._run(lambda: _Err(
            "Error code: 400 - {'error': {'code': 'json_validate_failed', "
            "'message': 'Failed to generate JSON. Please adjust your prompt.'}}", 400))
        self.assertEqual(n, 1, f"expected 1 provider call, got {n}")
        self.assertEqual(outcome, "raised")

    def test_B_failed_generation_makes_exactly_one_attempt(self):
        n, _ = self._run(lambda: _Err(
            "failed_generation: max completion tokens reached before generating a valid document", 400))
        self.assertEqual(n, 1, f"expected 1 provider call, got {n}")

    def test_C_rate_limit_still_rotates_all_keys_and_rounds(self):
        n, _ = self._run(None, rate_limit=True)
        # 3 rounds x 20 keys — unchanged behaviour for a genuinely transient class.
        self.assertGreater(n, 20, f"429 should rotate widely, only made {n} attempts")

    def test_D_5xx_still_rotates(self):
        n, _ = self._run(lambda: _Err("Error code: 503 - service unavailable", 503))
        self.assertGreater(n, 20, f"5xx should rotate widely, only made {n} attempts")

    def test_E_invalid_key_disables_and_continues_rotating(self):
        n, _ = self._run(lambda: _Err("Invalid API Key", 401))
        # Each key is disabled and rotation continues; it must not stop at the first.
        self.assertGreaterEqual(n, 20, f"invalid-key should try every key, made {n}")

    def test_413_still_fails_fast(self):
        n, _ = self._run(lambda: _Err("Request too large, reduce your message size", 413))
        self.assertEqual(n, 1, f"413 must fail fast, made {n} attempts")


class DoubtDetectorConfigTests(unittest.TestCase):
    """G. the corrected detector request, asserted against the real source."""

    def _detector_src(self) -> str:
        import io, os, re
        path = os.path.join(os.path.dirname(__file__), "views", "bridge.py")
        src = io.open(path, encoding="utf-8").read()
        start = src.index("def _detect_subject_and_type_for_doubt")
        return src[start:start + 4000]

    def test_G_detector_uses_a_realistic_token_budget(self):
        body = self._detector_src()
        self.assertIn("max_tokens=300,", body)
        self.assertNotIn("max_tokens=60,", body)

    def test_G_detector_no_longer_requests_server_side_json_validation(self):
        body = self._detector_src()
        self.assertIn("json_mode=False,", body)

    def test_G_detector_model_unchanged(self):
        # The fix must not switch providers or models.
        self.assertIn('model="openai/gpt-oss-20b"', self._detector_src())

    def test_F_detector_failure_still_falls_back_to_keyword_defaults(self):
        body = self._detector_src()
        # The except branch must survive and still yield usable defaults. The
        # default is general/conceptual, not physics/numerical: an unclassifiable
        # question is not a physics question, and claiming it is made the solver
        # refuse to answer anything outside physics.
        self.assertIn("defaulting to general/conceptual", body)
        self.assertIn("except Exception as exc:", body)

    def test_F_detector_failure_logs_elapsed_time(self):
        body = self._detector_src()
        self.assertIn("_detect_started", body)
        self.assertIn("failed after %.2fs", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
