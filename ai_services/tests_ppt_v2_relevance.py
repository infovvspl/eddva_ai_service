"""Asset relevance: a picture that is merely on-topic is not good enough.

The defect behind this file: an introductory "What is Photosynthesis?" slide
was given a diagram of *photosynthesis and respiration*. Tracing it, neither
the word nor that diagram was the problem. The problem was structural:

  * relevance was a one-sided boolean - "does the caption share a word with
    my query?" - so a figure carrying a second, competing process scored the
    same as one carrying only the subject;
  * the search loop returned the FIRST candidate that passed, so a passable
    result that happened to rank highly beat a better one below it.

Any topic with a well-known counterpart has the same failure: mitosis pulls
meiosis, arteries pull veins, reflection pulls refraction. So the fix is
scored both ways and comparative, and these tests pin that down rather than
pinning down the word "respiration".

Everything here is deterministic string work on captions the search provider
already returns: no model call, no per-image cost.
"""
from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import relevance as R


def spec(**kw):
    kw.setdefault("topic", "Photosynthesis")
    kw.setdefault("required", ["green leaf", "sunlight", "chlorophyll"])
    kw.setdefault("visual_kind", "illustration")
    kw.setdefault("class_name", "Class 8")
    return R.build_spec(**kw)


class ConflictDerivationTests(SimpleTestCase):
    def test_a_topic_brings_its_competing_concepts_with_it(self):
        self.assertIn("respiration", spec().avoid)

    def test_the_relationship_works_in_both_directions(self):
        self.assertIn("photosynthesis", R.build_spec(topic="Respiration").avoid)

    def test_it_is_not_specific_to_photosynthesis(self):
        # The mechanism has to generalise, or the next chapter hits the same
        # defect with different words.
        for topic, expected in (("Mitosis", "meiosis"),
                                ("Meiosis", "mitosis"),
                                ("Reflection of light", "refraction"),
                                ("Series circuit", "parallel circuit"),
                                ("Arteries", "vein")):
            self.assertIn(expected, R.build_spec(topic=topic).avoid, topic)

    def test_a_comparison_slide_may_ask_for_the_counterpart(self):
        # "Photosynthesis vs Respiration" is a legitimate slide. Declaring
        # respiration as required must cancel the penalty, which is what
        # makes this a preference rather than a blocklist.
        s = R.build_spec(topic="Photosynthesis", required=["respiration"])
        self.assertNotIn("respiration", s.avoid)

    def test_an_unknown_topic_simply_has_nothing_to_avoid(self):
        self.assertEqual(R.build_spec(topic="Sanskrit grammar").avoid, [])


class ScoringTests(SimpleTestCase):
    def test_the_caption_that_caused_the_defect_is_rejected(self):
        verdict = R.score_candidate(
            "Photosynthesis and Respiration in Plants diagram", spec())
        self.assertFalse(verdict.accepted)
        self.assertLess(verdict.score, 0)
        self.assertTrue(any("conflict" in r for r in verdict.reasons))

    def test_a_plainly_on_topic_caption_is_accepted(self):
        # The topic match must actually fire. It silently did not at first:
        # matches_any lowercases the haystack but not the needle, so a
        # capitalised topic scored zero and every generic figure was rejected.
        self.assertTrue(R.score_candidate("Photosynthesis diagram", spec()).accepted)

    def test_required_concepts_raise_the_score(self):
        bare = R.score_candidate("Photosynthesis diagram", spec()).score
        rich = R.score_candidate(
            "Photosynthesis: green leaf in sunlight showing chlorophyll", spec()).score
        self.assertGreater(rich, bare)

    def test_forbidden_concepts_lower_the_score(self):
        clean = R.score_candidate("Photosynthesis in a green leaf", spec()).score
        dirty = R.score_candidate(
            "Photosynthesis and cellular respiration in a green leaf", spec()).score
        self.assertLess(dirty, clean)

    def test_the_right_kind_of_figure_scores_higher(self):
        s = R.build_spec(topic="Photosynthesis", required=[], visual_kind="diagram")
        with_kind = R.score_candidate("Photosynthesis labelled diagram", s).score
        without = R.score_candidate("Photosynthesis", s).score
        self.assertGreater(with_kind, without)

    def test_out_of_band_terminology_is_penalised_for_a_junior_class(self):
        verdict = R.score_candidate(
            "Calvin cycle and thylakoid electron transport chain", spec())
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("advanced" in r for r in verdict.reasons))

    def test_the_same_terminology_is_not_penalised_for_a_senior_class(self):
        senior = R.build_spec(topic="Photosynthesis", class_name="Class 12")
        self.assertFalse(any("advanced" in r for r in R.score_candidate(
            "Calvin cycle in the stroma", senior).reasons))

    def test_the_wrong_kind_of_object_entirely_is_rejected(self):
        for caption in ("Photosynthesis worksheet answer key",
                        "Photosynthesis PPT template free download",
                        "Photosynthesis quiz question paper"):
            self.assertFalse(R.score_candidate(caption, spec()).accepted, caption)

    def test_an_empty_caption_is_not_accepted_on_faith(self):
        self.assertFalse(R.score_candidate("", spec()).accepted)

    def test_a_verdict_explains_itself(self):
        # The reasons end up in a log line, so a bad pick can be explained
        # after the fact rather than re-derived.
        self.assertIn("conflict:respiration",
                      str(R.score_candidate("Photosynthesis and respiration", spec())))


class HostTests(SimpleTestCase):
    """Where a picture lives, not only what it is called.

    The four-topic run put an Etsy product photo on a Force and Pressure
    slide and a watermarked sciencephoto.com preview on a Sound slide. Both
    captions were genuinely on topic, so caption scoring could not see the
    problem; the host could.
    """

    def test_a_product_listing_is_rejected_however_good_the_caption(self):
        s = R.build_spec(topic="Force and Pressure", class_name="Class 8")
        verdict = R.score_candidate("Force and pressure educational poster", s,
                                    "https://i.etsystatic.com/639/il/x.jpg")
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("unsuitable" in r for r in verdict.reasons))

    def test_stock_agencies_are_rejected(self):
        s = R.build_spec(topic="Sound", class_name="Class 8")
        for url in ("https://media.sciencephoto.com/image/x.jpg",
                    "https://www.shutterstock.com/image-photo/x",
                    "https://as2.ftcdn.net/gettyimages/x.jpg"):
            self.assertFalse(
                R.score_candidate("Sound waves from a tuning fork", s, url).accepted,
                url)

    def test_a_video_thumbnail_is_rejected(self):
        # On topic by caption, but built to be clicked: overlaid title text,
        # an arrow, often a presenter's face.
        s = R.build_spec(topic="Light", required=["mirror"], class_name="Class 8")
        self.assertFalse(R.score_candidate(
            "Reflection of light explained", s,
            "https://i.ytimg.com/vi/abc/maxresdefault.jpg").accepted)

    def test_a_resource_marketplace_listing_is_rejected(self):
        # Educational in subject but commercial in form: the image is a
        # product preview, not a figure. A live run put a Twinkl listing on a
        # Photosynthesis slide - two poster thumbnails and an "ink saving"
        # badge - and both the caption and the host looked educational.
        s = R.build_spec(topic="Photosynthesis", required=["leaf"],
                         class_name="Class 8")
        for url in ("https://images.twinkl.co.uk/tw1n/image/private/t_630_eco/image_repo/x.png",
                    "https://cdn.numerade.com/ask_images/x.jpg",
                    "https://www.coursehero.com/thumb/x.jpg"):
            self.assertFalse(R.score_candidate(
                "Photosynthesis leaf diagram", s, url).accepted, url)

    def test_a_teaching_source_is_rewarded(self):
        s = R.build_spec(topic="Force and Pressure", class_name="Class 8")
        plain = R.score_candidate("Pressure diagram", s, "https://example.test/x.png")
        trusted = R.score_candidate("Pressure diagram", s, "https://ncert.nic.in/x.png")
        self.assertGreater(trusted.score, plain.score)

    def test_an_unknown_host_is_neither_rewarded_nor_punished(self):
        s = R.build_spec(topic="Sound", class_name="Class 8")
        with_url = R.score_candidate("Sound diagram", s, "https://example.test/x.png")
        without = R.score_candidate("Sound diagram", s)
        self.assertEqual(with_url.score, without.score)


class TopicWordTests(SimpleTestCase):
    def test_a_multi_word_topic_scores_on_its_words(self):
        # "Force and Pressure" never appears verbatim in a caption, so
        # phrase matching scored a perfectly good "Pressure diagram" at zero
        # and rejected it.
        s = R.build_spec(topic="Force and Pressure", class_name="Class 8")
        verdict = R.score_candidate("Pressure diagram", s,
                                    "https://hyperphysics.phy-astr.gsu.edu/x.gif")
        self.assertTrue(verdict.accepted)
        self.assertIn("topic:1/2", verdict.reasons)

    def test_naming_the_whole_topic_scores_higher_than_half_of_it(self):
        s = R.build_spec(topic="Force and Pressure", class_name="Class 8")
        half = R.score_candidate("Pressure diagram", s).score
        whole = R.score_candidate("Force and pressure diagram", s).score
        self.assertGreater(whole, half)

    def test_filler_words_in_a_chapter_title_are_ignored(self):
        s = R.build_spec(topic="Introduction to the Basics of Sound",
                         class_name="Class 8")
        # Only "sound" carries meaning, so naming it is a full topic match.
        self.assertIn("topic:1/1", R.score_candidate("Sound diagram", s).reasons)


class BestCandidateTests(SimpleTestCase):
    CANDIDATES = [
        ("Photosynthesis and Respiration in Plants diagram", "both"),
        ("Photosynthesis diagram", "generic"),
        ("Green leaf in sunlight showing chlorophyll - educational diagram", "best"),
        ("Photosynthesis worksheet answer key", "junk"),
    ]

    def test_the_best_candidate_wins_not_the_first_acceptable_one(self):
        # This is the other half of the original defect: the loop returned
        # the first passable result, so position beat relevance.
        best, verdict = R.best_candidate(self.CANDIDATES, spec())
        self.assertEqual(best, "best")
        self.assertTrue(verdict.accepted)

    def test_nothing_is_returned_when_nothing_is_good_enough(self):
        # Returning the least-bad option is how an unrelated picture reaches
        # a slide. The caller must be told to try another strategy instead.
        best, verdict = R.best_candidate(
            [("Photosynthesis and respiration", "both"),
             ("Photosynthesis worksheet", "junk")], spec())
        self.assertIsNone(best)
        self.assertIsNone(verdict)

    def test_an_empty_candidate_list_is_handled(self):
        self.assertEqual(R.best_candidate([], spec()), (None, None))


class SearchQueryTests(SimpleTestCase):
    def test_queries_describe_the_visual_not_just_the_chapter(self):
        from ai_services.core.ppt_v2.prompts import build_search_terms
        terms = build_search_terms("Photosynthesis", chapter="Photosynthesis",
                                   subject="Science",
                                   objects=["green leaf", "sunlight"],
                                   visual_kind="illustration")
        # The most specific query leads with what must be visible, so the
        # result set is biased towards the right figure before ranking runs.
        self.assertIn("green leaf", terms[0])
        self.assertIn("sunlight", terms[0])

    def test_queries_stay_short_enough_to_match_real_captions(self):
        from ai_services.core.ppt_v2.prompts import build_search_terms
        terms = build_search_terms(
            "Photosynthesis", subject="Science",
            objects=["green leaf", "sunlight", "water", "carbon dioxide",
                     "oxygen", "glucose", "chloroplast"],
            visual_kind="illustration")
        for t in terms:
            self.assertLessEqual(len(t.split()), 9, t)

    def test_queries_broaden_rather_than_repeat(self):
        from ai_services.core.ppt_v2.prompts import build_search_terms
        terms = build_search_terms("Photosynthesis", subject="Science",
                                   objects=["green leaf"])
        self.assertEqual(len(terms), len(set(t.lower() for t in terms)))
        self.assertIn("Photosynthesis", terms[-1] + terms[0])


class AiImageBranchTests(SimpleTestCase):
    """The generated-image path, exercised without a provider.

    No HuggingFace credential is configured in this environment, so the real
    call cannot be made here. What can be pinned down is everything around
    it: that the prompt carries the constraints that make a usable teaching
    figure, that a generated image travels through the same asset pipeline as
    any other, and that the absence of a provider costs the picture and never
    the deck.
    """

    @staticmethod
    def _photosynthesis_prompt():
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.visuals import VisualCapabilities
        deck = compose_deck(
            [{"type": "anatomy", "title": "Inside a Leaf",
              "learningObjective": "Name the parts of a leaf that make food",
              "bullets": ["a", "b"],
              "visual": {"kind": "diagram",
                         "description": "a cross section of a leaf showing "
                                        "chloroplasts inside the cells",
                         "objects": ["leaf cross section", "chloroplast"],
                         "mustShow": ["chloroplast"]}}],
            DeckContext(topic="Photosynthesis", subject="Science",
                        chapter="Photosynthesis", class_name="Class 8",
                        board="CBSE"),
            caps=VisualCapabilities(ai_images=True, ai_image_budget=4))
        return deck.slides[0].visual.image_prompt

    def test_the_prompt_carries_the_teaching_context(self):
        prompt = self._photosynthesis_prompt().lower()
        for needed in ("class 8", "science", "leaf", "chloroplast"):
            self.assertIn(needed, prompt, needed)

    def test_the_prompt_states_what_must_be_visible(self):
        self.assertIn("showing", self._photosynthesis_prompt().lower())

    def test_the_prompt_forbids_text_in_the_image(self):
        # Diffusion models cannot spell; labels are added afterwards as real
        # text by the renderer, where they stay legible and editable.
        prompt = self._photosynthesis_prompt().lower()
        for forbidden in ("no text", "no letters", "no captions"):
            self.assertIn(forbidden, prompt, forbidden)

    def test_the_prompt_specifies_composition_and_style(self):
        prompt = self._photosynthesis_prompt().lower()
        self.assertTrue(any(w in prompt for w in ("landscape", "portrait", "square")))
        self.assertIn("flat vector", prompt)

    def test_the_prompt_is_pitched_at_the_class(self):
        self.assertIn("textbook", self._photosynthesis_prompt().lower())

    def test_a_generated_image_travels_the_normal_asset_pipeline(self):
        from unittest.mock import patch
        from ai_services.core.ppt_v2.producers import ProducerSet, cache_key
        from ai_services.core.ppt_v2.schema import VisualSpec

        store = {}
        producers = ProducerSet(cache_get=store.get, cache_put=store.__setitem__)
        spec = VisualSpec(image_prompt="a leaf cross section")
        with patch("ai_services.core.image_generation.generate_note_image",
                   return_value={"ok": True, "url": "https://img.test/a.png",
                                 "provider": "huggingface", "model": "flux"}):
            out = producers.ai_image(spec)
        self.assertEqual(out["url"], "https://img.test/a.png")
        # Same cache the searched and drawn assets use, tenant-scoped by the
        # caller — not a separate store for generated images.
        self.assertIn(cache_key(spec), store)
        self.assertTrue(producers.cached(spec)["ok"])

    def test_an_unconfigured_provider_costs_the_picture_not_the_deck(self):
        from unittest.mock import patch
        from ai_services.core.ppt_v2 import qa
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.visuals import detect_capabilities

        with patch("ai_services.core.image_generation.can_generate_note_images",
                   return_value=False):
            caps = detect_capabilities()
        self.assertFalse(caps.ai_images)
        deck = compose_deck(
            [{"type": "anatomy", "title": "Inside a Leaf", "bullets": ["a", "b"],
              "visual": {"kind": "diagram", "description": "a leaf cross section"}}],
            DeckContext(topic="Photosynthesis", subject="Science",
                        class_name="Class 8"), caps=caps)
        self.assertEqual(len(deck.slides), 1)
        self.assertEqual(qa.errors(qa.check_deck(deck.slides)), [])
