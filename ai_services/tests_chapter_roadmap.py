"""/learning/chapter-roadmap: AI topics -> subtopics for a chapter, linked to the school's topics. LLM mocked."""

from unittest.mock import patch
from django.test import TestCase

from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache
from ai_services.views.learning import parse_roadmap


class _FakeLLM:
    def __init__(self, sink, content):
        self._sink, self._content = sink, content

    def complete(self, system_prompt, user_prompt, **kwargs):
        self._sink["user_prompt"] = user_prompt
        return {"content": self._content, "model": "mock", "latency_ms": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


GOOD = {"topics": [
    {"topic": "Magnetic field and field lines", "subtopics": ["Bar magnet field", "Properties of field lines", "Compass needle"],
     "covers": ["MAGNETIC FIELD AND FIELD LINES"]},
    {"topic": "Magnetic field due to a current-carrying conductor",
     "subtopics": ["Straight conductor", "Right-hand thumb rule", "Circular loop", "Solenoid"],
     "covers": ["MAGNETIC FIELD DUE TO A CURRENT---CCCARRYING CONDUCTOR"]},
    {"topic": "Force on a current-carrying conductor", "subtopics": ["Fleming's left-hand rule", "Electric motor principle", "Introduction"],
     "covers": ["Not a recorded topic"]},
    {"topic": "Domestic electric circuits", "subtopics": ["Live, neutral and earth wires", "Fuse and overloading", "Short circuit"],
     "covers": ["DOMESTIC ELECTRIC CIRCUITS", "MAGNETIC FIELD AND FIELD LINES"]},
]}


class ChapterRoadmapViewTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, data, content):
        sink = {}
        with patch("ai_services.views.learning.get_llm", return_value=_FakeLLM(sink, content)):
            resp = self.client.post("/learning/chapter-roadmap", data=data, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, sink

    def test_returns_ordered_topics_with_subtopics_and_links(self):
        existing = ["MAGNETIC FIELD AND FIELD LINES", "MAGNETIC FIELD DUE TO A CURRENT---CCCARRYING CONDUCTOR",
                    "DOMESTIC ELECTRIC CIRCUITS"]
        resp, sink = self._post({"subject": "Science", "chapter": "Magnetic Effects of Electric Current",
                                 "class_name": "10", "existing_topics": existing,
                                 "passages": ["A magnetic field exists around a magnet."]}, GOOD)
        self.assertEqual(resp.status_code, 200)
        topics = resp.json()["topics"]
        self.assertEqual(len(topics), 4)
        self.assertEqual(topics[0]["covers"], ["MAGNETIC FIELD AND FIELD LINES"])
        self.assertEqual(topics[2]["covers"], [])               # invented name dropped
        self.assertNotIn("Introduction", topics[2]["subtopics"])
        self.assertEqual(topics[3]["covers"], ["DOMESTIC ELECTRIC CIRCUITS"])  # each recorded topic used once
        self.assertIn("  - DOMESTIC ELECTRIC CIRCUITS", sink["user_prompt"])
        self.assertIn("A magnetic field exists around a magnet.", sink["user_prompt"])

    def test_works_for_a_chapter_with_no_recorded_topics(self):
        resp, sink = self._post({"chapter": "Acids, Bases and Salts", "existing_topics": []}, GOOD)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("(none recorded)", sink["user_prompt"])

    def test_rejects_an_incomplete_roadmap(self):
        resp, _ = self._post({"chapter": "Light"}, {"topics": [{"topic": "Reflection", "subtopics": ["Laws"]}]})
        self.assertEqual(resp.status_code, 502)

    def test_requires_a_chapter(self):
        resp, _ = self._post({"existing_topics": ["x"]}, GOOD)
        self.assertEqual(resp.status_code, 400)


class TopicRoadmapViewTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, data, content):
        sink = {}
        with patch("ai_services.views.learning.get_llm", return_value=_FakeLLM(sink, content)):
            resp = self.client.post("/learning/topic-roadmap", data=data, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, sink

    def test_returns_a_clean_name_and_ordered_subtopics(self):
        resp, sink = self._post(
            {"subject": "Science", "chapter": "Magnetic Effects", "class_name": "10",
             "topic": "MAGNETIC FIELD DUE TO A CURRENT---CCCARRYING CONDUCTOR", "passages": ["Oersted showed..."]},
            {"topic": "Magnetic field due to a current-carrying conductor",
             "subtopics": ["Introduction", "Straight conductor", "Right-hand thumb rule", "Solenoid", "Solenoid"]},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["topic"], "Magnetic field due to a current-carrying conductor")
        self.assertEqual(body["subtopics"], ["Straight conductor", "Right-hand thumb rule", "Solenoid"])
        self.assertIn("THIS TOPIC ONLY", sink["user_prompt"])
        self.assertIn("Oersted showed", sink["user_prompt"])

    def test_keeps_the_recorded_name_when_the_model_gives_none(self):
        resp, _ = self._post({"topic": "Fuse"}, {"subtopics": ["Why fuses melt", "Fuse ratings"]})
        self.assertEqual(resp.json()["topic"], "Fuse")

    def test_rejects_too_few_subtopics_and_a_missing_topic(self):
        self.assertEqual(self._post({"topic": "Fuse"}, {"topic": "Fuse", "subtopics": ["Only one"]})[0].status_code, 502)
        self.assertEqual(self._post({"chapter": "X"}, {})[0].status_code, 400)


class ParseRoadmapTests(TestCase):
    def test_drops_duplicates_generic_topics_and_repeated_subtopics(self):
        out = parse_roadmap({"topics": [
            {"topic": "Reflection", "subtopics": ["Laws of reflection", "Plane mirror"]},
            {"topic": "reflection!", "subtopics": ["x"]},
            {"topic": "Summary", "subtopics": ["y"]},
            {"topic": "Refraction", "subtopics": ["Laws of reflection", "Snell's law"]},
            "junk",
        ]}, [])
        self.assertEqual([t["topic"] for t in out], ["Reflection", "Refraction"])
        self.assertEqual(out[1]["subtopics"], ["Snells law"])

    def test_caps_topics(self):
        out = parse_roadmap([{"topic": f"Topic {i}", "subtopics": [f"S{i}"]} for i in range(20)], [])
        self.assertEqual(len(out), 12)
