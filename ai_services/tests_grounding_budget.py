"""The source budget must fit real chapters, and a trimmed chapter must say so.

Both halves matter. A budget that silently drops a third of a chapter produces
a worksheet missing topics the teacher expected, and nothing in the response
distinguishes it from a complete one.
"""
import os
from unittest.mock import patch

from django.test import TestCase

from ai_services.core import grounding as gr


def passages(n, tokens_each=600, page_start=1):
    return [
        {"page_no": page_start + i, "chunk_index": i,
         "content": f"passage {i} " + ("word " * 10), "tokens": tokens_each}
        for i in range(n)
    ]


class BudgetTests(TestCase):
    def test_budget_is_sized_for_gemini_not_groq(self):
        """12,000 was Groq's per-request ceiling. Grounded generation runs on
        Gemini, and the old figure was cutting real chapters."""
        self.assertGreaterEqual(gr._DEFAULT_SOURCE_TOKEN_BUDGET, 30000)

    def test_budget_is_env_overridable(self):
        with patch.dict(os.environ, {"GROUNDING_TOKEN_BUDGET": "45000"}):
            import importlib
            importlib.reload(gr)
            self.assertEqual(gr._DEFAULT_SOURCE_TOKEN_BUDGET, 45000)
        importlib.reload(gr)  # restore for the rest of the suite

    def test_the_largest_indexed_chapter_now_fits(self):
        """Metals and Non-metals measured ~12,500 tokens across 22 passages and
        was losing 7 of them on the content path."""
        ps = passages(22, tokens_each=570)          # ≈ 12,540 tokens
        sel = gr.select_source(ps, topic="Metals", chapter="Metals and Non-metals")
        self.assertEqual(len(sel["passages"]), 22)
        self.assertFalse(sel["truncated"])

    def test_a_genuinely_oversized_chapter_still_truncates(self):
        """The budget is larger, not absent — a 40-page chapter must still be cut
        rather than blowing the model's context."""
        ps = passages(100, tokens_each=600)          # 60,000 tokens
        sel = gr.select_source(ps, topic="X", chapter="Y")
        self.assertTrue(sel["truncated"])
        self.assertLess(len(sel["passages"]), 100)

    def test_selection_stays_within_budget(self):
        ps = passages(100, tokens_each=600)
        sel = gr.select_source(ps, topic="X", chapter="Y")
        total = sum(p["tokens"] for p in sel["passages"])
        self.assertLessEqual(total, gr._DEFAULT_SOURCE_TOKEN_BUDGET)

    def test_reading_order_is_restored_after_ranking(self):
        """A selection must still read in teaching sequence, not relevance order."""
        ps = passages(20, tokens_each=600)
        sel = gr.select_source(ps, topic="passage 19", chapter="Y")
        pages = [p["page_no"] for p in sel["passages"]]
        self.assertEqual(pages, sorted(pages))


class TruncationReportingTests(TestCase):
    def test_truncated_is_reported(self):
        sel = gr.select_source(passages(100, 600), topic="X", chapter="Y")
        self.assertTrue(sel["truncated"])

    def test_not_truncated_when_everything_fits(self):
        sel = gr.select_source(passages(5, 600), topic="X", chapter="Y")
        self.assertFalse(sel["truncated"])

    def test_empty_source_is_not_truncated(self):
        sel = gr.select_source([], topic="X", chapter="Y")
        self.assertFalse(sel["truncated"])
        self.assertEqual(sel["passages"], [])

    def test_real_token_counts_are_preferred_over_the_character_estimate(self):
        """Rows indexed before tokens was persisted fall back to len/4. When a
        real count is present it must win, because the estimate drifts on dense
        notation and drift is what causes surprise truncation."""
        dense = [{"page_no": 1, "chunk_index": 0, "content": "x" * 400, "tokens": 5000}]
        sel = gr.select_source(dense, topic="X", chapter="Y", token_budget=1000)
        self.assertEqual(sel["passages"], [], "the real 5000-token count was ignored")


class ContentPathWiringTests(TestCase):
    """The content path computed truncation and discarded it; ppt.py did not."""

    @staticmethod
    def _source():
        from ai_services.views import bridge
        return open(bridge.__file__, encoding="utf-8").read()

    def test_content_path_reports_truncation(self):
        src = self._source()
        self.assertIn('"truncated": grounded_truncated', src)
        self.assertIn('"passagesUsed": grounded_used', src)
        self.assertIn('"passagesAvailable": grounded_available', src)

    def test_content_path_no_longer_hardcodes_a_smaller_budget(self):
        self.assertNotIn("token_budget=9000", self._source(),
                         "the content path is truncating earlier than the slide path")
