"""
Tests for the education-board layer (CBSE / ICSE / State).

The bug this prevents: school prompts hardcoded "NCERT" and "Class 10 Board Exam
(CBSE)". NCERT is CBSE's textbook body, so ICSE schools were served CBSE-flavoured
content — wrong textbooks, wrong paper pattern. The rule enforced here is:

    an ICSE request must never be told to use NCERT.
"""

import re

from django.test import SimpleTestCase, TestCase
from unittest.mock import patch

from ai_services.core.boards import (
    BOARDS, DEFAULT_BOARD, board_instruction, get_board, normalize_board,
)
from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache

NCERT = re.compile(r"(?i)\bncert\b")


class BoardRegistryTests(SimpleTestCase):
    def test_seeded_boards(self):
        self.assertEqual(set(BOARDS), {"cbse", "icse", "state", "ib"})

    def test_every_board_the_admin_ui_offers_is_supported(self):
        """
        The school admin UI (eddva_frontend school/admin/Institutes.jsx) offers
        exactly these options. A board offered there but missing here would be
        silently normalised to CBSE — i.e. an IB school served CBSE content.
        Keep this list in sync when the UI adds a board.
        """
        for ui_value, expected in [
            ("CBSE", "cbse"),
            ("ICSE", "icse"),
            ("State Board", "state"),
            ("IB", "ib"),
        ]:
            self.assertEqual(
                normalize_board(ui_value), expected,
                f"admin UI offers {ui_value!r} but it does not map to {expected!r}",
            )

    def test_model_choices_match_the_registry(self):
        from ai_services.models import Institute
        self.assertEqual({k for k, _ in Institute.BOARD_CHOICES}, set(BOARDS))

    def test_default_is_cbse(self):
        self.assertEqual(DEFAULT_BOARD, "cbse")

    def test_unknown_and_empty_fall_back(self):
        for value in ("", None, "martian-board", "   "):
            self.assertEqual(normalize_board(value), DEFAULT_BOARD)

    def test_real_world_aliases_normalise(self):
        # The backend stores whatever the school typed at registration.
        self.assertEqual(normalize_board("CISCE"), "icse")
        self.assertEqual(normalize_board("ISC"), "icse")
        self.assertEqual(normalize_board("  IcSe "), "icse")
        self.assertEqual(normalize_board("State Board"), "state")
        self.assertEqual(normalize_board("NCERT"), "cbse")


class BoardInstructionContentTests(SimpleTestCase):
    def test_icse_never_mentions_ncert(self):
        text = board_instruction("icse")
        self.assertFalse(NCERT.search(text), "ICSE guidance must not reference NCERT")
        self.assertIn("CISCE", text)

    def test_cbse_does_use_ncert(self):
        self.assertTrue(NCERT.search(board_instruction("cbse")))

    def test_state_is_syllabus_neutral(self):
        text = board_instruction("state")
        self.assertFalse(NCERT.search(text))
        self.assertNotIn("CISCE", text)

    def test_ib_is_international_not_indian_board_framed(self):
        text = board_instruction("ib")
        self.assertFalse(NCERT.search(text))
        self.assertNotIn("CISCE", text)
        self.assertIn("International Baccalaureate", text)

    def test_every_board_forbids_cross_board_references(self):
        for key in BOARDS:
            self.assertIn("another board", board_instruction(key))


class DoubtFramingIsBoardAwareTests(SimpleTestCase):
    """The doubt prompt used to hardcode '(NCERT)' for every school request."""

    def _framing(self, vertical, board):
        from ai_services.views.bridge import _doubt_framing
        return _doubt_framing(vertical, board)

    def test_school_icse_framing_has_no_ncert(self):
        f = self._framing("school", "icse")
        blob = " ".join(f.values())
        self.assertFalse(NCERT.search(blob), f"ICSE doubt framing leaked NCERT: {blob}")
        self.assertIn("ICSE", blob)

    def test_school_cbse_framing_uses_ncert_textbooks(self):
        blob = " ".join(self._framing("school", "cbse").values())
        self.assertTrue(NCERT.search(blob))
        self.assertIn("CBSE", blob)

    def test_placeholders_are_always_substituted(self):
        for board in BOARDS:
            blob = " ".join(self._framing("school", board).values())
            self.assertNotIn("{board}", blob)
            self.assertNotIn("{textbooks}", blob)

    def test_coaching_framing_is_unchanged_by_board(self):
        for board in BOARDS:
            blob = " ".join(self._framing("coaching", board).values())
            self.assertIn("JEE/NEET", blob)


class ExamRulesAreBoardAwareTests(SimpleTestCase):
    """content/generate class rules used to say 'Class 10 Board Exam (CBSE)'."""

    def _rule(self, target, board):
        from ai_services.views.bridge import _resolve_exam_rule
        return _resolve_exam_rule(target, board)

    def test_class10_icse_has_no_ncert_or_cbse(self):
        rule = self._rule("Class 10", "icse")
        self.assertFalse(NCERT.search(rule), f"ICSE Class 10 rule leaked NCERT: {rule}")
        self.assertIn("ICSE", rule)

    def test_class10_cbse_keeps_ncert(self):
        rule = self._rule("Class 10", "cbse")
        self.assertTrue(NCERT.search(rule))
        self.assertIn("CBSE", rule)

    def test_no_unsubstituted_placeholders_for_any_class_or_board(self):
        for target in ("Class 10", "Class 11", "Class 12"):
            for board in BOARDS:
                rule = self._rule(target, board)
                self.assertNotIn("{board}", rule)
                self.assertNotIn("{textbooks}", rule)

    def test_competitive_rules_are_untouched_by_board(self):
        for board in BOARDS:
            self.assertIn("JEE", self._rule("JEE", board))
            self.assertIn("NEET", self._rule("NEET", board))


class _FakeLLM:
    def __init__(self, sink):
        self._sink = sink

    def complete(self, system_prompt, user_prompt, **kw):
        self._sink["system_prompt"] = system_prompt
        self._sink["user_prompt"] = user_prompt
        return {"content": "# Mock", "model": "mock", "latency_ms": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class BoardEndToEndThroughHttpTests(TestCase):
    """Drives the real middleware + view path with X-Board."""

    API_KEY = "board-test-key"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(
            name="ICSE School", slug="icse-school", api_key=self.API_KEY,
            vertical="school", board="icse", is_active=True,
        )

    def _generate(self, board_header=None):
        sink = {}
        headers = {"HTTP_X_API_KEY": self.API_KEY, "HTTP_X_VERTICAL": "school"}
        if board_header:
            headers["HTTP_X_BOARD"] = board_header
        with patch("ai_services.views.bridge.get_llm", return_value=_FakeLLM(sink)):
            resp = self.client.post(
                "/content/generate",
                data={"topicName": "Photosynthesis", "subjectName": "Science",
                      "chapterName": "Life Processes", "contentType": "lesson"},
                content_type="application/json", **headers,
            )
        return resp, sink

    def test_header_board_is_echoed_and_applied(self):
        resp, sink = self._generate(board_header="icse")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Board"], "icse")
        self.assertFalse(NCERT.search(sink["system_prompt"]),
                         "ICSE request produced an NCERT-flavoured prompt")

    def test_falls_back_to_the_tenant_board_when_header_absent(self):
        # Institute.board == "icse", so no header should still yield ICSE.
        resp, _ = self._generate(board_header=None)
        self.assertEqual(resp["X-Board"], "icse")

    def test_unknown_board_value_degrades_to_default(self):
        resp, _ = self._generate(board_header="not-a-real-board")
        self.assertEqual(resp["X-Board"], DEFAULT_BOARD)


class BoardCacheScopingTests(SimpleTestCase):
    """A CBSE answer must never be served from cache to an ICSE school."""

    def test_school_cache_scope_includes_board(self):
        from ai_services.views.base import _cache_scope
        self.assertNotEqual(_cache_scope("school", "cbse"), _cache_scope("school", "icse"))

    def test_coaching_cache_scope_is_unchanged(self):
        from ai_services.views.base import _cache_scope
        self.assertEqual(_cache_scope("coaching", "cbse"), "coaching")
