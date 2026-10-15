"""The book text an assessment paper is written from (core.assessment_source).

A topic test reads its own section; a chapter test reads the chapter digest
plus an even share of every section's pages; anything unclear keeps every
passage. No network: the digest is built with a fake model from a chapter
made here (the same shape as the real NCERT one).
"""
from unittest import mock

from django.test import SimpleTestCase, TestCase

from ai_services.core import assessment_source as asrc
from ai_services.core import chapter_digest as cd
from ai_services.tests_chapter_digest import FakeLLM, chapter


def digest():
    return cd.build(chapter(), chapter="Life Processes", llm=FakeLLM())


def long_chapter():
    """The sample chapter with every passage large, so trimming has something to trim."""
    out = []
    for p in chapter():
        q = dict(p)
        q["content"] = p["content"] + (" Detail sentence about this section." * 120)
        q["tokens"] = len(q["content"]) // 4
        out.append(q)
    return out


class TopicTests(SimpleTestCase):
    def test_a_topic_test_reads_only_its_section(self):
        out = asrc.select(long_chapter(), topic="Respiration", chapter="Life Processes", digest=digest())
        self.assertEqual(out["mode"], "topic_section")
        self.assertEqual(sorted({p["page_no"] for p in out["passages"]}), [9, 10])
        self.assertLess(out["tokens_after"], out["tokens_before"])
        self.assertEqual(out["digest_text"], "")          # a full section needs no digest

    def test_a_thin_section_also_gets_the_digest(self):
        out = asrc.select(chapter(), topic="Respiration", chapter="Life Processes", digest=digest())
        self.assertEqual(out["mode"], "topic_section")
        self.assertIn("FOCUS SECTION 5.3 RESPIRATION", out["digest_text"])


class ChapterTests(SimpleTestCase):
    def test_a_chapter_test_reads_the_digest_and_every_section(self):
        with mock.patch.dict("os.environ", {"ASSESSMENT_CHAPTER_PAGE_TOKENS": "1500"}):
            out = asrc.select(long_chapter(), topic="Science chapter assessment",
                              chapter="Life Processes", digest=digest())
        self.assertEqual(out["mode"], "chapter_digest")
        self.assertIn("SECTION 5.3 RESPIRATION", out["digest_text"])
        pages = {p["page_no"] for p in out["passages"]}
        for section_pages in ({1, 2}, {3}, {9, 10}, {13}):   # every section is represented
            self.assertTrue(pages & section_pages, section_pages)
        self.assertLess(out["tokens_after"], out["tokens_before"])

    def test_even_share_keeps_reading_order_and_one_passage_per_section(self):
        sections = digest()["sections"]
        chosen = asrc.even_share(long_chapter(), sections, budget=10)
        self.assertEqual([p["page_no"] for p in chosen], sorted(p["page_no"] for p in chosen))
        self.assertGreaterEqual(len(chosen), len([s for s in sections if s.get("pages")]) - 1)

    def test_a_small_chapter_is_left_whole(self):
        out = asrc.select(chapter(), topic="", chapter="Life Processes", digest=digest())
        self.assertEqual(out["mode"], "full")
        self.assertEqual(len(out["passages"]), len(chapter()))


class FallbackTests(SimpleTestCase):
    def test_multi_chapter_papers_keep_every_passage(self):
        out = asrc.select(long_chapter(), topic="", chapter="3 chapters", digest=digest())
        self.assertEqual((out["mode"], out["reason"]), ("full", "multi-chapter paper"))

    def test_lecture_transcripts_keep_every_passage(self):
        book = long_chapter() + [{"page_no": 0, "content": "lecture words", "source": "lecture"}]
        self.assertEqual(asrc.select(book, topic="Respiration", digest=digest())["mode"], "full")

    def test_switched_off(self):
        with mock.patch.dict("os.environ", {"ASSESSMENT_SOURCE_OPTIMIZE": "false"}):
            self.assertEqual(asrc.select(long_chapter(), topic="Respiration", digest=digest())["mode"], "full")

    def test_no_digest_keeps_every_passage(self):
        with mock.patch.object(cd, "cached", return_value=None), \
                mock.patch.object(cd, "build_in_background", return_value=False):
            out = asrc.select(long_chapter(), topic="Respiration")
        self.assertEqual((out["mode"], out["reason"]), ("full", "no digest"))

    def test_a_missing_digest_is_built_in_the_background_not_waited_for(self):
        with mock.patch.object(cd, "cached", return_value=None), \
                mock.patch.object(cd, "build_in_background", return_value=True) as bg, \
                mock.patch.object(cd, "get_or_build", side_effect=AssertionError("must not block")):
            out = asrc.select(long_chapter(), topic="Respiration", chapter="Life Processes")
        self.assertEqual((out["mode"], out["reason"]), ("full", "digest building"))
        self.assertEqual(len(out["passages"]), len(long_chapter()))
        bg.assert_called_once()

    def test_a_failure_keeps_every_passage(self):
        with mock.patch.object(cd, "cached", side_effect=RuntimeError("redis down")):
            out = asrc.select(long_chapter(), topic="Respiration")
        self.assertEqual((out["mode"], out["reason"]), ("full", "selection failed"))
        self.assertEqual(len(out["passages"]), len(long_chapter()))


class _GroundedFakeLLM:
    """Takes the grounded route and records the prompt it was given."""

    def __init__(self, sink):
        self.sink = sink

    def can_route(self, **kw):
        return True

    def complete(self, system_prompt, user_prompt, **kw):
        self.sink["user_prompt"] = user_prompt
        return {"content": "## Section A\n1. Q?\n## Answer Key\nQ1. Answer: a", "model": "mock",
                "latency_ms": 1, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class GenerateEndpointTests(TestCase):
    """The real /content/generate path with the model faked."""
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        from ai_services.middleware import invalidate_institute_cache
        from ai_services.models import Institute
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _generate(self, content_type, topic):
        sink = {}
        with mock.patch("ai_services.views.bridge.get_llm", return_value=_GroundedFakeLLM(sink)), \
                mock.patch.object(cd, "cached", return_value=digest()):
            resp = self.client.post(
                "/content/generate",
                data={"topicName": topic, "subjectName": "Science", "chapterName": "Life Processes",
                      "contentType": content_type, "sourcePassages": long_chapter()},
                content_type="application/json", HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school",
            )
        return resp, sink

    def test_a_topic_paper_is_written_from_its_section_only(self):
        resp, sink = self._generate("assessment_paper", "Respiration")
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertIn("pyruvate", sink["user_prompt"])
        self.assertNotIn("Haemoglobin carries oxygen", sink["user_prompt"])   # section 5.4
        self.assertEqual(resp.json()["source"]["assessmentSource"]["mode"], "topic_section")

    def test_other_content_still_reads_the_whole_chapter(self):
        resp, sink = self._generate("notes", "Respiration")
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertIn("Haemoglobin carries oxygen", sink["user_prompt"])
        self.assertNotIn("assessmentSource", resp.json()["source"])


class CoverageTests(SimpleTestCase):
    def test_a_topic_gets_its_sections_key_terms(self):
        d = digest()
        section = cd.focus_section(d, "Respiration")
        section["key_terms"] = ["pyruvate", "fermentation", "lactic acid", "mitochondria"]
        self.assertEqual(asrc.coverage(d, section), ["pyruvate", "fermentation", "lactic acid", "mitochondria"])

    def test_a_chapter_gets_every_titled_section(self):
        self.assertEqual(asrc.coverage(digest(), None),
                         ["What Are Life Processes?", "Nutrition", "Respiration", "Transportation"])

    def test_too_little_gives_none(self):
        d = digest()
        section = cd.focus_section(d, "Respiration")
        section["key_terms"] = ["pyruvate"]
        self.assertIsNone(asrc.coverage(d, section))


class FauxBoldBookTests(SimpleTestCase):
    def test_a_book_indexed_before_the_fix_still_has_sections(self):
        def fb(word):
            return "".join(ch * 5 for ch in word)
        book = [dict(p) for p in chapter()]
        for p in book:   # headings as the old extraction stored them
            p["content"] = p["content"].replace("5.3 RESPIRATION", "%s %s" % (fb("5.3"), fb("RESPIRATION")))
        self.assertNotIn("5.3", [s["number"] for s in cd.detect_sections(book)])
        cleaned = [asrc.clean_passage(p) for p in book]
        self.assertIn("5.3", [s["number"] for s in cd.detect_sections(cleaned)])

    def test_clean_text_is_left_as_it_is(self):
        p = chapter()[0]
        self.assertIs(asrc.clean_passage(p), p)


class PageGroupCoverageTests(SimpleTestCase):
    def test_page_groups_give_their_key_terms_in_turn(self):
        d = {"sections": [
            {"number": "", "title": "Pages 1-5", "pages": [1, 5], "key_terms": ["metals", "nonmetals", "lustre"]},
            {"number": "", "title": "Pages 6-11", "pages": [6, 11], "key_terms": ["oxygen", "Metals", "sulphur"]},
        ]}
        self.assertEqual(asrc.coverage(d, None), ["metals", "oxygen", "nonmetals", "lustre", "sulphur"])
