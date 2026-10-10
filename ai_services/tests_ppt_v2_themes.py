"""Studio themes: the teacher's "Choose a Theme" pick colours V2 and image decks.

Until now only the old generator honoured the pick; V2 decks always took the
subject's colours and image slides the subject's painted palette. These tests
pin the new contract:

  * a known studio key wins over the subject, everywhere the deck is coloured;
  * "subject", nothing, or an unknown value keeps the subject palette - never
    an unstyled deck;
  * every palette keeps every text/fill pair the renderers draw readable
    (WCAG), including the dark ones, whose roles are inverted;
  * a dark theme's cover stays dark, and its painted slides ask for a dark
    background rather than the white one subject palettes use.

No network: the planner, providers and painter are stubs.
"""
from unittest import mock

from django.test import SimpleTestCase
from rest_framework.response import Response

from ai_services.core.ppt_v2 import image_slides, qa
from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
from ai_services.core.ppt_v2.design import (
    MIN_BODY_CONTRAST,
    MIN_LARGE_CONTRAST,
    PALETTE_WORDS,
    STUDIO_PALETTES,
    SUBJECT_THEMES,
    TEXT_ON_FILL_PAIRS,
    build_theme,
    contrast_ratio,
    is_dark,
    studio_theme_key,
)
from ai_services.core.ppt_v2.schema import SlideType
from ai_services.core.ppt_v2.visuals import VisualCapabilities

STUDIO_KEYS = ("dark-professional", "ocean-blue", "warm-sunset", "forest-green",
               "royal-purple", "clean-white")
DARK_KEYS = ("dark-professional", "ocean-blue", "warm-sunset", "forest-green",
             "royal-purple")
NO_AI = VisualCapabilities(ai_images=False, image_search=True, diagrams=True, cache=True)


def every_theme(slide_type=SlideType.CONCEPT):
    """(label, ThemeSpec) for every subject palette and every studio palette."""
    for subject in SUBJECT_THEMES:
        yield subject, build_theme(subject=subject, slide_type=slide_type)
    for key in STUDIO_PALETTES:
        yield key, build_theme(subject="Science", slide_type=slide_type, palette=key)


class StudioKeyTests(SimpleTestCase):
    def test_the_six_studio_cards_each_have_a_palette(self):
        # The keys are the studio's data-theme values; a renamed card would
        # silently fall back to the subject palette.
        self.assertEqual(set(STUDIO_PALETTES), set(STUDIO_KEYS))

    def test_only_a_known_key_selects_a_palette(self):
        for key in STUDIO_KEYS:
            self.assertEqual(studio_theme_key(key), key)
            self.assertEqual(studio_theme_key("  %s " % key.upper()), key)
        for other in ("subject", "", None, "neon", "dark professional", 7, {"x": 1}):
            self.assertIsNone(studio_theme_key(other), repr(other))

    def test_a_studio_palette_wins_over_the_subject(self):
        theme = build_theme(subject="Chemistry", palette="ocean-blue")
        self.assertEqual(theme.name, "ocean-blue")
        self.assertEqual(theme.color("primary"), STUDIO_PALETTES["ocean-blue"]["primary"])

    def test_match_the_subject_and_unknown_keys_keep_the_subject_palette(self):
        expected = build_theme(subject="Chemistry")
        for palette in (None, "", "subject", "not-a-theme"):
            theme = build_theme(subject="Chemistry", palette=palette)
            self.assertEqual(theme.name, "chemistry", repr(palette))
            self.assertEqual(theme.colors, expected.colors, repr(palette))

    def test_five_themes_are_dark_and_clean_white_is_light(self):
        for key in STUDIO_KEYS:
            self.assertEqual(is_dark(STUDIO_PALETTES[key]), key in DARK_KEYS, key)
        for key, colors in SUBJECT_THEMES.items():
            self.assertFalse(is_dark(build_theme(subject=key).colors), key)


class ContrastTests(SimpleTestCase):
    def test_every_pair_the_renderers_draw_is_readable(self):
        for name, theme in every_theme():
            for text, fill, minimum in TEXT_ON_FILL_PAIRS:
                ratio = contrast_ratio(theme.color(text), theme.color(fill))
                self.assertGreaterEqual(
                    ratio, minimum, "%s: %s on %s is only %.2f:1" % (name, text, fill, ratio))

    def test_text_is_readable_on_the_background_of_every_slide_type(self):
        for slide_type in SlideType:
            for name, theme in every_theme(slide_type):
                background = theme.color("surface")
                for token, minimum in (("ink", MIN_BODY_CONTRAST),
                                       ("ink_muted", MIN_BODY_CONTRAST),
                                       ("heading", MIN_LARGE_CONTRAST)):
                    ratio = contrast_ratio(theme.color(token), background)
                    self.assertGreaterEqual(
                        ratio, minimum, "%s/%s: %s is only %.2f:1"
                        % (name, slide_type.value, token, ratio))

    def test_cover_text_is_readable_across_the_whole_gradient(self):
        # The title sits on a gradient; reading well at one end is not enough.
        for name, theme in every_theme(SlideType.TITLE):
            if not theme.background.startswith("gradient:"):
                continue
            for stop in theme.background[len("gradient:"):].split("->"):
                ratio = contrast_ratio(theme.color("ink"), theme.color(stop.strip()))
                self.assertGreaterEqual(ratio, MIN_BODY_CONTRAST,
                                        "%s: title text on %s is %.2f:1" % (name, stop, ratio))


class DarkThemeTests(SimpleTestCase):
    def test_a_dark_cover_stays_dark(self):
        for key in DARK_KEYS:
            theme = build_theme(palette=key, slide_type=SlideType.TITLE)
            self.assertEqual(theme.background, "gradient:surface->surface_alt", key)
            self.assertTrue(is_dark(theme.colors), key)
            self.assertEqual(theme.color("heading"), theme.color("ink"), key)

    def test_pictures_are_mounted_on_white_on_a_dark_theme(self):
        # Diagrams and textbook figures are drawn for a white page; their dark
        # labels vanish on a dark panel.
        for slide_type in SlideType:
            for key in DARK_KEYS:
                theme = build_theme(palette=key, slide_type=slide_type)
                self.assertEqual(theme.color("image_mat"), "#FFFFFF", key)

    def test_the_mat_follows_the_slides_own_background(self):
        # White behind a picture whenever the slide itself is dark - a dark
        # theme, or a light theme's cover, which sits on its primary colour.
        # A light slide keeps its own panel colour, exactly as before.
        for slide_type in SlideType:
            for name, theme in every_theme(slide_type):
                expected = "#FFFFFF" if is_dark(theme.colors) else theme.color("surface_alt")
                self.assertEqual(theme.color("image_mat"), expected,
                                 "%s/%s" % (name, slide_type.value))
        self.assertEqual(build_theme(subject="Biology").color("image_mat"),
                         build_theme(subject="Biology").color("surface_alt"))

    def test_a_light_cover_still_inverts_onto_its_colours(self):
        for name in ("clean-white", "biology"):
            theme = build_theme(subject=name, palette=name, slide_type=SlideType.TITLE)
            self.assertEqual(theme.background, "gradient:primary->accent", name)


class RendererContractTests(SimpleTestCase):
    def test_every_palette_carries_every_token_the_renderer_needs(self):
        for slide_type in SlideType:
            for name, theme in every_theme(slide_type):
                missing = [t for t in qa.RENDERER_REQUIRED_TOKENS if not theme.colors.get(t)]
                self.assertEqual(missing, [], "%s/%s" % (name, slide_type.value))

    def test_colours_are_hex_the_renderers_can_parse(self):
        for key, colors in STUDIO_PALETTES.items():
            for token, value in colors.items():
                self.assertRegex(value, r"^#[0-9A-F]{6}$", "%s.%s" % (key, token))

    def test_every_palette_has_words_for_the_illustration_prompt(self):
        for key in list(SUBJECT_THEMES) + list(STUDIO_PALETTES):
            self.assertIn(key, PALETTE_WORDS)


PLANNED = [
    {"type": "title", "title": "Respiration", "subtitle": "Life Processes"},
    {"type": "concept", "title": "Glucose", "bullets": ["Glucose is broken down in the cytoplasm."]},
    {"type": "summary", "title": "Summary", "bullets": ["Energy is stored as ATP."]},
]


class ComposeTests(SimpleTestCase):
    def test_the_deck_takes_the_teachers_theme_on_every_slide(self):
        deck = compose_deck(PLANNED, DeckContext(subject="Science", chapter="Life Processes",
                                                 theme_key="royal-purple"), caps=NO_AI)
        self.assertEqual({s.theme.name for s in deck.slides}, {"royal-purple"})
        codes = {i.code for i in qa.check_deck(deck.slides)}
        self.assertNotIn("missing_token", codes)

    def test_without_a_theme_the_subject_still_decides(self):
        deck = compose_deck(PLANNED, DeckContext(subject="Science", chapter="Life Processes"),
                            caps=NO_AI)
        self.assertEqual({s.theme.name for s in deck.slides}, {"biology"})

    def test_generated_pictures_are_asked_for_in_the_themes_colours(self):
        slide = compose_deck(PLANNED, DeckContext(subject="Science", theme_key="forest-green"),
                             caps=NO_AI).slides[1]
        self.assertEqual(slide.theme.name, "forest-green")
        self.assertTrue(PALETTE_WORDS["forest-green"])


class ServiceTests(SimpleTestCase):
    def test_generate_v2_hands_the_theme_to_every_slide(self):
        from ai_services.tests_ppt_v2_service import run
        payload = run(theme_key="warm-sunset").to_payload()
        names = {s["v2"]["theme"]["name"] for s in payload["slides"]}
        self.assertEqual(names, {"warm-sunset"})

    def test_an_unknown_theme_is_the_subject_palette_not_an_error(self):
        from ai_services.tests_ppt_v2_service import run
        payload = run(theme_key="neon").to_payload()
        self.assertNotIn("neon", {s["v2"]["theme"]["name"] for s in payload["slides"]})


PROCESS = {
    "slideType": "process", "title": "How Plants Make Food",
    "blocks": [{"kind": "step", "label": "Absorb light", "text": "Chlorophyll traps sunlight."}],
}


class ImageSlideTests(SimpleTestCase):
    def test_every_studio_card_has_a_painted_palette(self):
        self.assertEqual(set(image_slides._STUDIO_PALETTES), set(STUDIO_PALETTES))

    def test_a_dark_theme_asks_for_a_dark_slide_with_light_text(self):
        for key in DARK_KEYS:
            prompt = image_slides.slide_prompt(PROCESS, number=1, total=3,
                                               subject="Science", theme_key=key)
            self.assertNotIn("white background", prompt, key)
            self.assertIn(image_slides._STUDIO_PALETTES[key], prompt, key)
            # Where each text colour goes: a live deck put white text on pale panels.
            self.assertIn("crisp white titles", prompt, key)
            self.assertIn("white rounded cards with dark slate text", prompt, key)

    def test_every_slide_is_painted_edge_to_edge_not_as_a_mockup(self):
        # A live dark-theme slide came back drawn on a pale page with a shadow.
        for key in ("", "dark-professional", "clean-white"):
            prompt = image_slides.slide_prompt(PROCESS, number=1, total=3,
                                               subject="Science", theme_key=key)
            self.assertIn("fills the whole image edge to edge", prompt, repr(key))

    def test_the_subject_palette_is_unchanged(self):
        prompt = image_slides.slide_prompt(PROCESS, number=1, total=3, subject="Science")
        self.assertIn("soft pastel greens with fresh leaf-green accents on a white background",
                      prompt)
        for other in ("", "subject", "neon"):
            self.assertEqual(image_slides.palette_for("Science", other),
                             image_slides.palette_for("Science"), repr(other))

    def test_the_deck_paints_every_slide_in_the_chosen_theme(self):
        from ai_services.core.ppt_v2.planner import PlannerResult
        prompts = []

        def render(prompt, **kw):
            prompts.append(prompt)
            return {"ok": True, "url": "http://x/1.png"}
        plan = PlannerResult(title="Photosynthesis", slides=[PROCESS, PROCESS], model="stub")
        with mock.patch("ai_services.core.ppt_v2.planner.plan_deck", return_value=plan):
            image_slides.generate_image_deck(
                topic="Photosynthesis", slide_count=2, render=render,
                ctx={"className": "Class 8", "subjectName": "Science"},
                theme_key="ocean-blue")
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("deep ocean-blue background", prompt)


class EndpointTests(SimpleTestCase):
    def _run(self, version, data):
        from ai_services.views import ppt
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return Response({"success": True, "data": {}})
        request = mock.Mock(data=dict({"topic": "Respiration"}, **data), headers={},
                            institute_id="i", vertical="school", board="CBSE")
        target = "_try_generate_image_deck" if version == "image" else "_try_generate_v2"
        with mock.patch.object(ppt, "ppt_version",
                               return_value=ppt.IMAGE if version == "image" else ppt.V2), \
             mock.patch.object(ppt, target, side_effect=fake):
            ppt._generate_presentation(request)
        return seen

    def test_the_studio_theme_reaches_both_generators(self):
        for version in ("v2", "image"):
            self.assertEqual(self._run(version, {"deckTheme": "forest-green"})["theme_key"],
                             "forest-green", version)

    def test_match_the_subject_missing_or_unknown_sends_no_theme(self):
        for data in ({}, {"deckTheme": "subject"}, {"deckTheme": "<script>"},
                     {"deckTheme": None}):
            self.assertEqual(self._run("v2", data)["theme_key"], "", repr(data))
