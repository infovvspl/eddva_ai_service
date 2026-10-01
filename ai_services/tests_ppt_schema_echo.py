"""A model can echo a schema field name into the bullets list.

Found on 2026-09-29 testing Class 10 CBSE decks for every EDDVA subject: the
grounded English deck ("A Letter to God") ended every content slide with a
literal "pages" bullet, while the slide's own "pages" array was correctly filled
([3, 4, 5]). Citations were fine; the stray bullet renders as a one-word line on
the teacher's slide. Six other subjects were clean, so this is intermittent and
must be handled defensively rather than by prompt wording alone.
"""
from django.test import SimpleTestCase

from ai_services.views.ppt import _drop_schema_echo_bullets


class SchemaEchoBulletTests(SimpleTestCase):
    def test_the_observed_pages_bullet_is_dropped(self):
        slides = [{
            "title": "Lencho's Faith and the Hailstorm",
            "bullets": ["For an hour the hail rained on the whole valley.", "pages"],
            "pages": [3, 4, 5],
        }]
        _drop_schema_echo_bullets(slides)
        self.assertEqual(slides[0]["bullets"], ["For an hour the hail rained on the whole valley."])
        self.assertEqual(slides[0]["pages"], [3, 4, 5])   # real citations untouched

    def test_every_schema_field_name_is_covered(self):
        for echoed in ("pages", "bullets", "title", "subtitle", "type",
                       "slideNumber", "speakerNotes", "imageSearchTerm"):
            slides = [{"bullets": ["A real sentence of slide content here.", echoed]}]
            _drop_schema_echo_bullets(slides)
            self.assertEqual(slides[0]["bullets"], ["A real sentence of slide content here."], echoed)

    def test_matching_ignores_case_padding_and_a_trailing_colon(self):
        slides = [{"bullets": ["  Pages: ", "\tPAGES", "real content that must survive"]}]
        _drop_schema_echo_bullets(slides)
        self.assertEqual(slides[0]["bullets"], ["real content that must survive"])

    def test_a_sentence_that_merely_mentions_a_field_name_is_kept(self):
        keep = [
            "The pages of the textbook describe photosynthesis in detail.",
            "Title deeds were issued to farmers after the reform.",
            "Type II diabetes is discussed on the next slide.",
        ]
        slides = [{"bullets": list(keep)}]
        _drop_schema_echo_bullets(slides)
        self.assertEqual(slides[0]["bullets"], keep)

    def test_slides_without_bullets_are_left_alone(self):
        slides = [{"title": "A Letter to God", "type": "title"}, {"bullets": "not-a-list"}, "junk"]
        _drop_schema_echo_bullets(slides)
        self.assertEqual(slides[0], {"title": "A Letter to God", "type": "title"})
        self.assertEqual(slides[1], {"bullets": "not-a-list"})

    def test_both_generation_paths_sanitise_their_deck(self):
        import io
        import os

        src = io.open(os.path.join(os.path.dirname(__file__), "views", "ppt.py"), encoding="utf-8").read()
        self.assertIn('data["slides"] = _drop_schema_echo_bullets([{**s, **img}', src)   # grounded
        self.assertIn("slides = _drop_schema_echo_bullets([_repair_slide_latex(s)", src)  # ungrounded
