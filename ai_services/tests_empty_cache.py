"""An empty AI result must never be cached.

Found on 2026-09-30 while testing every feature against Together: a memorization
request returned {"items": []}, that payload was stored in the tenant cache, and
later requests with the same prompt were served it instantly — HTTP 200,
_meta.source="cache", zero items. A silent failure that looks like a healthy fast
response, and it persists for the whole TTL.

Caching is an optimisation. A miss costs one extra call; a cached empty answer
costs the student the answer.
"""
from django.test import SimpleTestCase

from ai_services.views.base import _worth_caching


class WorthCachingTests(SimpleTestCase):
    def test_the_observed_empty_payload_is_rejected(self):
        self.assertFalse(_worth_caching({"items": []}))

    def test_real_content_is_cached(self):
        self.assertTrue(_worth_caching({"items": [{"q": "What is chlorophyll?"}]}))
        self.assertTrue(_worth_caching({"answer": "Photosynthesis makes food."}))
        self.assertTrue(_worth_caching("A plain text answer."))

    def test_blank_and_whitespace_strings_are_rejected(self):
        for payload in ("", "   ", "\n\t", {"answer": ""}, {"answer": "   "}):
            self.assertFalse(_worth_caching(payload), repr(payload))

    def test_empty_containers_are_rejected(self):
        for payload in ({}, [], {"a": [], "b": {}}, {"slides": []}, [[], {}]):
            self.assertFalse(_worth_caching(payload), repr(payload))

    def test_none_is_rejected(self):
        self.assertFalse(_worth_caching(None))

    def test_an_error_payload_is_never_cached(self):
        self.assertFalse(_worth_caching({"error": "Daily token budget exceeded"}))
        self.assertFalse(_worth_caching({"error": "boom", "items": [1]}))

    def test_service_metadata_alone_does_not_count_as_content(self):
        # These keys are attached by the service, not produced by the model.
        self.assertFalse(_worth_caching({"_cache_meta": {"tokens": 0}, "_meta": {"model": "x"}}))
        self.assertFalse(_worth_caching({"items": [], "_meta": {"source": "llm"}}))

    def test_a_deliberate_scalar_is_content(self):
        self.assertTrue(_worth_caching({"score": 0}))
        self.assertTrue(_worth_caching({"passed": False}))

    def test_nested_content_is_found(self):
        self.assertTrue(_worth_caching({"data": {"sections": [{"text": "real"}]}}))
        self.assertFalse(_worth_caching({"data": {"sections": [{"text": ""}]}}))


class CacheWritesAreGuardedTests(SimpleTestCase):
    def test_both_cache_write_sites_check_first(self):
        import io
        import os

        src = io.open(os.path.join(os.path.dirname(__file__), "views", "base.py"),
                      encoding="utf-8").read()
        lines = src.splitlines()
        writes = [i for i, ln in enumerate(lines) if "_cache.set(" in ln]
        self.assertEqual(len(writes), 2, [lines[i].strip() for i in writes])
        for i in writes:
            guard = "\n".join(lines[max(0, i - 2):i + 1])   # the write and its if-condition
            self.assertIn("_worth_caching", guard,
                          f"unguarded cache write at line {i + 1}: {lines[i].strip()}")
