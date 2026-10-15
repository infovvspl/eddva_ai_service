"""Chapter digest, focus pages, terms to avoid and faux-bold text.

Measured on a real NCERT Class 10 "Life Processes" chapter (21 pages) with a
Respiration deck: tokens per deck 37.6k-59.7k -> 32.8k-33.0k, the off-scope
repair call gone, the same coverage and grounding. No network: the model is a
fake and the chapter is built here.
"""
import json
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core import chapter_digest as cd
from ai_services.core.textbook import clean_page_text, undo_faux_bold


def chapter():
    """A small chapter shaped like the real one: capitalised numbered headings."""
    def p(page, text, i=0):
        return {"page_no": page, "chunk_index": i, "content": text, "tokens": len(text) // 4}
    return [
        p(1, "How do we tell what is alive? 5.1 WHAT ARE LIFE PROCESSES? The maintenance "
             "functions of living organisms must go on even when they are not doing anything."),
        p(2, "Energy for maintenance comes from outside the body, as food."),
        p(3, "Before the meal. 5.2 NUTRITION When we walk we use energy. Autotrophs make "
             "their own food by photosynthesis using sunlight and chlorophyll."),
        p(9, "Absorbed food is used by cells. 5.3 RESPIRATION Pichkari Activity 5.4 Take lime water. "
             "The first step is the break-down of glucose into pyruvate in the cytoplasm."),
        p(10, "In yeast pyruvate is converted into ethanol and carbon dioxide during fermentation. "
              "In muscle cells lack of oxygen converts pyruvate into lactic acid causing cramps."),
        p(13, "5.4 TRANSPORTATION 5.4.1 Transportation in Human Beings Blood carries oxygen "
              "and food. Haemoglobin carries oxygen."),
    ]


class FakeLLM:
    def __init__(self, reply=None, fail_titles=()):
        self.calls = []
        self.reply = reply
        self.fail_titles = fail_titles

    def complete(self, **kw):
        self.calls.append(kw)
        section = kw["user_prompt"].split("Section: ", 1)[1].split("\n", 1)[0]
        if section in self.fail_titles:
            raise RuntimeError("provider down")
        reply = self.reply or {
            "summary": "About %s." % section,
            "key_facts": [{"fact": "The first step is the break-down of glucose into pyruvate in the cytoplasm.", "page": 9},
                          {"fact": "Pyruvate is converted into ethanol and carbon dioxide in yeast.", "page": 9},
                          {"fact": "Mitochondria make ATP from sunlight in plants.", "page": 10}],
            "key_terms": ["pyruvate", "fermentation"]}
        return {"content": reply, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FauxBoldTests(SimpleTestCase):
    def test_repeated_glyphs_are_collapsed(self):
        self.assertEqual(undo_faux_bold("55555.....33333 RRRRREEEEESSSSSPPPPPIIIIIRRRRRAAAAATTTTTIIIIIOOOOONNNNN"),
                         "5.3 RESPIRATION")
        self.assertEqual(undo_faux_bold("FFFFFiiiiiggggguuuuurrrrreeeee 55555.....1111100000"), "Figure 5.10")

    def test_real_words_and_numbers_are_untouched(self):
        for text in ("book 11 ... aaa cooperation 1000 Mississippi ooo", "Glucose -> pyruvate (3 carbons)"):
            self.assertEqual(undo_faux_bold(text), text)

    def test_clean_page_text_applies_it(self):
        self.assertIn("5.3 RESPIRATION", clean_page_text("55555.....33333 RRRRREEEEESSSSSPPPPPIIIIIRRRRRAAAAATTTTTIIIIIOOOOONNNNN text"))


class DetectSectionsTests(SimpleTestCase):
    def test_capitalised_headings_split_the_chapter(self):
        sections = cd.detect_sections(chapter())
        self.assertEqual([s["number"] for s in sections], ["", "5.1", "5.2", "5.3", "5.4"])
        self.assertEqual(sections[3]["title"], "RESPIRATION")      # not "RESPIRATION Pichkari"
        self.assertEqual(sections[3]["pages"], [9, 10])
        self.assertEqual(sections[4]["title"], "TRANSPORTATION")   # not swallowed by 5.4.1

    def test_text_before_a_heading_stays_with_the_previous_section(self):
        sections = cd.detect_sections(chapter())
        nutrition = next(s for s in sections if s["number"] == "5.2")
        self.assertIn("Absorbed food", " ".join(p["content"] for p in nutrition["passages"]))

    def test_figure_and_activity_numbers_are_not_sections(self):
        numbers = [s["number"] for s in cd.detect_sections(chapter())]
        self.assertNotIn("5.4.1", numbers)
        self.assertEqual(numbers.count("5.4"), 1)

    def test_a_book_without_headings_is_split_into_page_groups(self):
        plain = [{"page_no": n, "content": "Plain text with no headings at all. " * 60, "tokens": 500}
                 for n in range(1, 13)]
        sections = cd.detect_sections(plain)
        self.assertGreater(len(sections), 1)
        self.assertEqual(sorted({p["page_no"] for s in sections for p in s["passages"]}), list(range(1, 13)))


class CheckFactsTests(SimpleTestCase):
    def setUp(self):
        self.pages = cd._page_words(chapter())

    def test_a_fact_the_book_states_is_kept(self):
        kept, dropped = cd.check_facts(
            [{"fact": "Haemoglobin carries oxygen.", "page": 13}], self.pages)
        self.assertEqual((len(kept), dropped), (1, 0))

    def test_an_unsupported_fact_is_dropped(self):
        kept, dropped = cd.check_facts(
            [{"fact": "Mitochondria make ATP from sunlight in plants.", "page": 10}], self.pages)
        self.assertEqual((kept, dropped), ([], 1))

    def test_a_fact_cited_on_the_wrong_page_gets_the_right_page(self):
        kept, _ = cd.check_facts(
            [{"fact": "In yeast pyruvate is converted into ethanol and carbon dioxide.", "page": 9}], self.pages)
        self.assertEqual(kept[0]["page"], 10)


class BuildTests(SimpleTestCase):
    def test_one_small_call_per_section_and_unsupported_facts_dropped(self):
        llm = FakeLLM()
        digest = cd.build(chapter(), chapter="Life Processes", llm=llm)
        self.assertEqual(len(llm.calls), 5)
        self.assertTrue(all(c["reasoning_effort"] == "low" for c in llm.calls))
        self.assertTrue(all(c["feature"] == "chapter_digest" for c in llm.calls))
        resp = next(s for s in digest["sections"] if s["number"] == "5.3")
        self.assertEqual([f["fact"][:24] for f in resp["key_facts"]],
                         ["The first step is the br", "Pyruvate is converted in"])
        self.assertGreater(digest["checks"]["facts_dropped"], 0)

    def test_a_missing_section_means_no_digest(self):
        # An incomplete digest would hide that section from planning.
        with mock.patch.object(cd.time, "sleep"):
            self.assertIsNone(cd.build(chapter(), llm=FakeLLM(fail_titles=("RESPIRATION",))))

    def test_cached_by_chapter_text(self):
        cd._memory.clear()
        llm = FakeLLM()
        with mock.patch("ai_services.core.ppt_v2.deck_cache._client", return_value=None):
            first = cd.get_or_build(chapter(), llm=llm)
            second = cd.get_or_build(chapter(), llm=llm)
        self.assertEqual(first, second)
        self.assertEqual(len(llm.calls), 5)          # built once

    def test_switched_off(self):
        with mock.patch.dict("os.environ", {"CHAPTER_DIGEST": "false"}):
            self.assertIsNone(cd.get_or_build(chapter(), llm=FakeLLM()))


class FocusTests(SimpleTestCase):
    def setUp(self):
        self.digest = cd.build(chapter(), llm=FakeLLM())

    def test_the_title_decides_the_section(self):
        self.assertEqual(cd.focus_section(self.digest, "Respiration")["number"], "5.3")

    def test_unclear_topic_gives_no_section(self):
        self.assertIsNone(cd.focus_section(self.digest, "Magnetism"))

    def test_focus_passages_are_the_sections_pages(self):
        pages = sorted({p["page_no"] for p in cd.focus_passages(self.digest, chapter(), "Respiration")})
        self.assertEqual(pages, [9, 10])

    def test_focus_passages_fall_back_to_the_whole_chapter(self):
        self.assertEqual(len(cd.focus_passages(self.digest, chapter(), "Magnetism")), len(chapter()))
        self.assertEqual(len(cd.focus_passages(None, chapter(), "Respiration")), len(chapter()))
        with mock.patch.dict("os.environ", {"PPT_FOCUS_SECTION_PAGES": "false"}):
            self.assertEqual(len(cd.focus_passages(self.digest, chapter(), "Respiration")), len(chapter()))

    def test_prompt_text_gives_the_focus_section_in_full(self):
        text = cd.format_for_prompt(self.digest, "Respiration")
        self.assertIn("FOCUS SECTION 5.3 RESPIRATION (pp. 9-10)", text)
        self.assertIn("glucose into pyruvate", text)
        self.assertEqual(text.count("FOCUS"), 1)


class BackgroundBuildTests(SimpleTestCase):
    def test_cached_never_builds(self):
        cd._memory.clear()
        with mock.patch("ai_services.core.ppt_v2.deck_cache._client", return_value=None), \
                mock.patch.object(cd, "build", side_effect=AssertionError("must not build")):
            self.assertIsNone(cd.cached(chapter()))

    def test_a_chapter_is_built_once_in_the_background(self):
        import threading
        cd._memory.clear()
        release = threading.Event()
        calls = []

        def slow_build(passages, **kw):
            calls.append(1)
            release.wait(2)
            return None

        with mock.patch("ai_services.core.ppt_v2.deck_cache._client", return_value=None), \
                mock.patch.object(cd, "get_or_build", side_effect=slow_build):
            self.assertTrue(cd.build_in_background(chapter()))
            self.assertFalse(cd.build_in_background(chapter()))      # already building
            release.set()
            for t in threading.enumerate():
                if t.name.startswith("chapter-digest-"):
                    t.join(2)
        self.assertEqual(len(calls), 1)
