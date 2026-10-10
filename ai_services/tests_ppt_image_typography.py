"""Painted (image) slides keep one text size across the whole deck.

A live 10-slide painted deck had big text on some slides and small on others.
Each slide is painted separately, so nothing tied the sizes together; the
model scaled type to the amount of text (43 to 86 words a slide), and process
slides were sent as `1) "Cause": "..."`, which it painted literally - quote
marks, numbering and all. These tests pin what keeps a deck consistent:

  * every slide carries the same typography rule, word for word;
  * every card is described the same way (bold heading, one short line), with
    no `"label": "text"` or `1)` text for the model to copy;
  * card text is its first sentence, trimmed at a clause break when long, and
    a slide carries at most four cards;
  * content titles sit top-left; the cover carries no other text.

No network: prompts only.
"""
import re

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import image_slides
from ai_services.core.ppt_v2.image_slides import CARD_WORDS, MAX_BLOCKS, TYPOGRAPHY, _q, _short, slide_prompt

LONG = ("When there is a lack of oxygen in our muscle cells, pyruvate is converted into lactic acid, "
        "which builds up and causes cramps during vigorous exercise. It goes away after rest.")


def block(label, text, kind="bullet"):
    return {"kind": kind, "label": label, "text": text}


SLIDES = [
    {"slideType": "title", "title": "Respiration", "subtitle": "How cells release energy",
     "visual": {"description": "a cell releasing energy"}},
    {"slideType": "concept", "title": "What is Respiration?", "blocks": [
        block("Glucose", "Glucose is a six-carbon molecule."), block("Where", LONG)]},
    {"slideType": "process", "title": "Lactic Acid Formation", "blocks": [
        block("Cause", LONG), block("Effect", "Pyruvate becomes lactic acid."),
        block("Cramps", "The build-up causes cramps.")]},
    {"slideType": "summary", "title": "Summary", "blocks": [
        block("A%d" % i, LONG) for i in range(6)]},
    {"slideType": "comparison", "title": "Aerobic vs Anaerobic", "blocks": [
        block("Aerobic", "Uses oxygen.", "compare_left"), block("Anaerobic", LONG, "compare_right")]},
    {"slideType": "quiz", "title": "Quick Check", "blocks": [
        block("", "Where does glycolysis happen?", "question"), block("", "Cytoplasm"),
        block("", "Nucleus")]},
]


def prompts():
    return [slide_prompt(s, number=i + 1, total=len(SLIDES), class_name="Class 10",
                         subject="Science", chapter="Life Processes") for i, s in enumerate(SLIDES)]


class SameTypeEverywhereTests(SimpleTestCase):
    def test_every_slide_carries_the_same_typography_rule(self):
        for i, p in enumerate(prompts()):
            self.assertEqual(p.count(TYPOGRAPHY), 1, "slide %d" % (i + 1))
        self.assertIn("never shrink text to fit", TYPOGRAPHY)

    def test_no_label_text_pairs_or_numbering_for_the_model_to_copy(self):
        for i, p in enumerate(prompts()):
            self.assertIsNone(re.search(r'"\s*:\s*"', p), "slide %d has a quoted label: text pair" % (i + 1))
            self.assertIsNone(re.search(r'\d\)\s*"', p), "slide %d has 1) numbering" % (i + 1))
            self.assertIn("never draw quote marks, asterisks", p)

    def test_every_card_has_the_same_shape(self):
        process = prompts()[2]
        self.assertEqual(process.count("a card: heading "), 3)
        self.assertIn('a card: heading "Cause", line ', process)


class CardTextTests(SimpleTestCase):
    def test_a_long_card_keeps_its_first_sentence_trimmed_at_a_clause_break(self):
        short = _short(LONG)
        self.assertEqual(short, "When there is a lack of oxygen in our muscle cells, pyruvate is "
                                "converted into lactic acid.")
        # A clause break a few words past the limit is used rather than a worse cut.
        self.assertLessEqual(len(short.split()), CARD_WORDS + image_slides._CUT_SLACK)

    def test_a_short_card_is_unchanged(self):
        self.assertEqual(_short("Glucose is a six-carbon molecule."), "Glucose is a six-carbon molecule.")

    def test_a_long_sentence_with_no_break_is_kept_whole_not_cut_mid_clause(self):
        whole = " ".join(["word"] * (CARD_WORDS + 6)) + "."
        self.assertEqual(_short(whole), whole)

    def test_a_slide_carries_at_most_four_cards_and_short_headings(self):
        summary = prompts()[3]
        self.assertEqual(MAX_BLOCKS, 4)
        self.assertEqual(summary.count("a card: heading "), 4)
        slide = {"slideType": "concept", "title": "T", "blocks": [
            block("A very long heading of many many words", "Short line.")]}
        p = slide_prompt(slide, number=1, total=1)
        self.assertIn('"A very long heading of"', p)

    def test_the_speaker_notes_keep_the_full_text(self):
        notes = image_slides._bullets(SLIDES[2])
        self.assertIn(LONG, notes[0])


class LayoutTests(SimpleTestCase):
    def test_content_titles_sit_top_left_and_the_cover_is_the_big_one(self):
        ps = prompts()
        self.assertIn('Large bold title "Respiration"', ps[0])
        for p in ps[1:]:
            self.assertIn("At the top-left, the slide title", p)

    def test_the_cover_carries_no_other_text(self):
        self.assertIn("No other text on this slide: no cards, captions or bullet points.", prompts()[0])


class PromptLimitTests(SimpleTestCase):
    """Qwen-Image-2.0 on Together rejects a prompt over 2,000 characters (a misleading
    400: "Positive prompt must be a non-empty ... between 2 and 2000"). Five of six live
    slides were lost to it once the typography rules made the prompts longer."""

    def test_every_slide_fits_even_at_its_fullest(self):
        long_words = " ".join(["photosynthesis"] * 30) + "."
        worst = {"slideType": "process", "title": "A Very Long Slide Title About Many Things " * 2,
                 "emphasisText": "Very Long", "subtitle": "A subtitle that runs on and on " * 3,
                 "blocks": [block("Heading %d with words" % i, long_words) for i in range(6)],
                 "visual": {"description": "a detailed illustration of a leaf " * 6,
                            "annotations": [{"text": "Label number %d" % i} for i in range(5)]}}
        for kind in ("process", "concept", "summary", "comparison", "formula", "title", "quiz"):
            p = slide_prompt(dict(worst, slideType=kind), number=10, total=10, class_name="Class 10",
                             subject="Science", chapter="Life Processes and More",
                             theme_key="dark-professional")
            self.assertLessEqual(len(p), image_slides.PROMPT_LIMIT, kind)
            self.assertIn(TYPOGRAPHY, p, kind)        # the consistency rule survives the fitting

    def test_text_is_trimmed_before_anything_is_dropped(self):
        normal = prompts()[2]
        self.assertLessEqual(len(normal), image_slides.PROMPT_LIMIT)
        self.assertIn("When there is a lack of oxygen in our muscle cells, pyruvate is converted "
                      "into lactic acid.", normal)    # an ordinary slide keeps its full first clause


class ConsistentGridTests(SimpleTestCase):
    """Second round, from a live 8-slide "Human Eye" deck: titles 7.9-11.1% of the
    slide height (long titles shrank to fit one line), cards of 10-30 words."""

    def test_a_trimmed_card_never_leaves_a_fragment_or_a_cut_list(self):
        cases = {
            # a leading subordinate clause alone states nothing
            LONG: "When there is a lack of oxygen in our muscle cells, pyruvate is converted "
                  "into lactic acid.",
            # cutting inside the list would say carbon dioxide is the only product
            "In aerobic respiration, pyruvate is broken down using oxygen in the mitochondria, "
            "giving carbon dioxide, water and much more energy.":
                "In aerobic respiration, pyruvate is broken down using oxygen in the mitochondria.",
            # a list with no clause break is kept whole
            "The products of respiration are carbon dioxide, water, heat, energy and a few "
            "by-products in small amounts.":
                "The products of respiration are carbon dioxide, water, heat, energy and a few "
                "by-products in small amounts.",
        }
        for text, expected in cases.items():
            self.assertEqual(_short(text), expected)

    def test_a_long_title_is_split_at_its_natural_break(self):
        split = image_slides._split_title
        self.assertEqual(split("Presbyopia – Age-related Loss of Accommodation", "Ageing"),
                         ("Presbyopia", "Age-related Loss of Accommodation · Ageing"))
        self.assertEqual(split("Hypermetropia: Long-sighted Eyes and their Correction", ""),
                         ("Hypermetropia", "Long-sighted Eyes and their Correction"))
        self.assertEqual(split("Myopia (Nearsightedness)", "s"), ("Myopia (Nearsightedness)", "s"))
        self.assertEqual(split("A Very Long Title With No Break At All", "s"),
                         ("A Very Long Title With No Break At All", "s"))
        p = slide_prompt({"slideType": "concept", "title": "Presbyopia – Age-related Loss of "
                          "Accommodation", "blocks": [block("A", "B is C.")]}, number=2, total=8)
        self.assertIn('the slide title "Presbyopia"', p)
        self.assertIn('Subtitle below the title: "Age-related Loss of Accommodation"', p)

    def test_column_slides_carry_three_cards_and_the_same_grid(self):
        many = [block("H%d" % i, "Line %d is short." % i) for i in range(5)]
        for kind in ("concept", "process"):
            p = slide_prompt({"slideType": kind, "title": "T", "blocks": many}, number=2, total=8)
            self.assertEqual(p.count("a card: heading "), 3, kind)
            self.assertIn(image_slides.COLUMN_GRID, p, kind)
        summary = slide_prompt({"slideType": "summary", "title": "T", "blocks": many}, number=8, total=8)
        self.assertEqual(summary.count("a card: heading "), 4)


class CardSpaceTests(SimpleTestCase):
    """Third round, from a live 8-slide Respiration deck read back with OCR: card text
    4.4-6.5% of the slide height. The two big slides were a comparison (two short
    lines in half-slide panels) and a quiz (four questions in full-width cards), and
    a process slide had every card cut mid-sentence to make room for its picture."""

    GLUCOSE = {"slideType": "process", "title": "Break-down of Glucose – First Step", "blocks": [
        block("Glucose entry", "Glucose, a six-carbon sugar, enters the cytoplasm of the cell "
                               "where respiration begins.", "step"),
        block("Split into three-carbon units", "Inside the cytoplasm, glucose is broken into a "
              "three-carbon molecule as the first stage of respiration.", "step"),
        block("Pyruvate formed", "The resulting three-carbon molecule is called pyruvate, which "
              "will be used in later respiration steps.", "step")],
        "visual": {"description": "A cell with a glucose molecule entering the cytoplasm and "
                                  "splitting into two pyruvate molecules, arrows showing the "
                                  "direction of the reaction inside the cytoplasm of the cell.",
                   "annotations": [{"text": "Glucose"}, {"text": "Cytoplasm"},
                                   {"text": "Pyruvate"}, {"text": "Cell membrane"}]}}

    def live(self, slide):
        return slide_prompt(slide, number=3, total=8, class_name="Class 10", subject="Science",
                            chapter="Life Processes", theme_key="clean-white")

    def test_the_picture_is_described_more_briefly_before_any_card_is_cut(self):
        p = self.live(self.GLUCOSE)
        self.assertLessEqual(len(p), image_slides.PROMPT_LIMIT)
        self.assertNotIn("…", p)
        for b in self.GLUCOSE["blocks"]:
            self.assertIn(_q(b["text"]), p)
        self.assertIn("illustration of A cell with a glucose molecule", p)

    def test_every_slide_asks_for_the_same_card_size(self):
        self.assertIn("every card 44% of the slide width and 17% of its height", TYPOGRAPHY)

    def test_a_comparison_uses_ordinary_cards_with_the_picture_below(self):
        p = self.live({"slideType": "comparison", "title": "Anaerobic vs Aerobic", "blocks": [
            block("Anaerobic", "In yeast, pyruvate becomes ethanol.", "compare_left"),
            block("Aerobic", "In mitochondria, pyruvate is broken down.", "compare_right")],
            "visual": {"description": "yeast cells beside a mitochondrion"}})
        self.assertNotIn("panel", p)
        self.assertIn("Below the cards, across the slide: a clear, accurate educational "
                      "illustration of yeast cells beside a mitochondrion", p)

    def test_four_question_or_key_point_cards_sit_in_a_two_by_two_grid(self):
        questions = [block("Q%d" % i, "What is question number %d about?" % i, "question")
                     for i in range(1, 5)]
        quiz = self.live({"slideType": "quiz", "title": "Quick Quiz", "blocks": questions})
        self.assertEqual(quiz.count("a card: "), 4)
        self.assertIn(image_slides.SQUARE_GRID, quiz)
        summary = self.live({"slideType": "summary", "title": "Summary", "blocks": questions})
        self.assertIn(image_slides.SQUARE_GRID, summary)

    def test_a_bare_number_label_is_left_to_the_number_circle(self):
        p = self.live({"slideType": "quiz", "title": "Quick Quiz", "blocks": [
            block("Q1", "Where is glucose split?", "question"),
            block("Step 2", "Where does pyruvate go?", "question"),
            block("Glucose", "What is glucose?", "question")]})
        self.assertNotIn('heading "Q1"', p)
        self.assertNotIn('heading "Step 2"', p)
        self.assertIn('a card: "Where is glucose split?"', p)
        self.assertIn('a card: heading "Glucose"', p)      # a real heading stays
