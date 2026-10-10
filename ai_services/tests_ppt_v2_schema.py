"""The V2 spec has to reject slides a renderer cannot draw.

Validation here is not pedantry: the renderer is JavaScript in the browser,
where a region of w=1.4 does not raise, it silently draws a box off the edge
of the slide and the teacher finds out in front of a class. Every check below
corresponds to something that fails invisibly downstream.
"""
from django.test import SimpleTestCase

from ai_services.core.ppt_v2.schema import (
    MAX_CONTENT_BLOCKS,
    Annotation,
    BlockKind,
    ContentBlock,
    LayoutSpec,
    Region,
    SlideSpec,
    SlideType,
    TitleSpec,
    VisualSpec,
    VisualStrategy,
    validate_deck,
)


def slide(number=1, **kw):
    kw.setdefault("title", TitleSpec("Photosynthesis"))
    kw.setdefault("slide_type", SlideType.CONCEPT)
    return SlideSpec(slide_number=number, **kw)


class RegionTests(SimpleTestCase):
    def test_a_region_inside_the_slide_is_clean(self):
        self.assertEqual(Region(0.1, 0.1, 0.5, 0.5).errors("r"), [])

    def test_a_region_past_the_edge_is_caught(self):
        self.assertTrue(any("past the slide edge" in e
                            for e in Region(0.8, 0.1, 0.5, 0.2).errors("r")))

    def test_a_negative_coordinate_is_caught(self):
        self.assertTrue(Region(-0.1, 0.1, 0.5, 0.2).errors("r"))

    def test_a_zero_area_region_is_caught(self):
        self.assertTrue(any("zero or negative area" in e
                            for e in Region(0.1, 0.1, 0.0, 0.2).errors("r")))

    def test_overlap_detection(self):
        a = Region(0.0, 0.0, 0.5, 0.5)
        self.assertTrue(a.overlaps(Region(0.3, 0.3, 0.5, 0.5)))
        self.assertFalse(a.overlaps(Region(0.6, 0.0, 0.3, 0.5)))

    def test_touching_edges_do_not_count_as_overlap(self):
        # Two columns sharing a boundary is the normal case, not a defect.
        self.assertFalse(Region(0.0, 0.0, 0.5, 1.0).overlaps(Region(0.5, 0.0, 0.5, 1.0)))


class SlideValidationTests(SimpleTestCase):
    def test_a_minimal_slide_is_valid(self):
        self.assertTrue(slide().is_valid)

    def test_an_empty_title_is_rejected(self):
        self.assertIn("title is empty", " ".join(slide(title=TitleSpec("   ")).errors()))

    def test_slide_numbers_start_at_one(self):
        self.assertTrue(slide(number=0).errors())

    def test_emphasis_must_appear_in_the_title(self):
        bad = slide(title=TitleSpec("Photosynthesis", emphasis_text="Respiration"))
        self.assertTrue(any("emphasisText" in e for e in bad.errors()))
        good = slide(title=TitleSpec("Photosynthesis in plants", emphasis_text="plants"))
        self.assertTrue(good.is_valid)

    def test_too_many_content_blocks_is_rejected(self):
        blocks = [ContentBlock(BlockKind.BULLET, "point %d" % i)
                  for i in range(MAX_CONTENT_BLOCKS + 1)]
        self.assertTrue(any("exceeds" in e for e in slide(content_blocks=blocks).errors()))

    def test_an_empty_content_block_is_rejected(self):
        bad = slide(content_blocks=[ContentBlock(BlockKind.BULLET, "  ", label="")])
        self.assertTrue(any("is empty" in e for e in bad.errors()))

    def test_an_annotation_anchor_must_be_normalised(self):
        visual = VisualSpec(annotations=[Annotation("stoma", anchor=(1.4, 0.2))])
        self.assertTrue(any("outside 0..1" in e for e in slide(visual=visual).errors()))

    def test_wanting_a_visual_without_a_region_for_it_is_rejected(self):
        bad = slide(
            visual=VisualSpec(strategy=VisualStrategy.SEARCHED_IMAGE),
            layout=LayoutSpec("text_only", {"title": Region(0.05, 0.05, 0.9, 0.1)}),
        )
        self.assertTrue(any("no 'visual' region" in e for e in bad.errors()))

    def test_a_slide_that_wants_no_visual_needs_no_visual_region(self):
        ok = slide(layout=LayoutSpec("text_only", {"title": Region(0.05, 0.05, 0.9, 0.1)}))
        self.assertTrue(ok.is_valid)


class DeckValidationTests(SimpleTestCase):
    def test_an_empty_deck_is_rejected(self):
        self.assertEqual(validate_deck([]), ["deck has no slides"])

    def test_duplicate_slide_numbers_are_caught(self):
        self.assertTrue(any("duplicate" in e
                            for e in validate_deck([slide(1), slide(1)])))

    def test_out_of_order_slides_are_caught(self):
        self.assertTrue(any("out of order" in e
                            for e in validate_deck([slide(2), slide(1)])))

    def test_a_deck_where_every_slide_looks_the_same_is_flagged(self):
        # This is the V1 failure mode V2 exists to prevent, so it is an error
        # rather than something to notice later in review.
        same = LayoutSpec("concept_hero_split", {"title": Region(0.05, 0.05, 0.9, 0.1)})
        deck = [slide(i + 1, layout=same) for i in range(4)]
        self.assertTrue(any("visual variety" in e for e in validate_deck(deck)))

    def test_a_short_deck_is_allowed_to_repeat_a_layout(self):
        # Two slides on one topic legitimately share a shape.
        same = LayoutSpec("concept_hero_split", {"title": Region(0.05, 0.05, 0.9, 0.1)})
        self.assertEqual(validate_deck([slide(1, layout=same), slide(2, layout=same)]), [])


class BackCompatTests(SimpleTestCase):
    """A renderer that has not learned V2 must still produce a usable slide."""

    def test_v1_keys_are_present_and_populated(self):
        spec = slide(content_blocks=[ContentBlock(BlockKind.BULLET, "Leaves absorb light")],
                     visual=VisualSpec(search_terms=["photosynthesis diagram"],
                                       url="https://example.test/a.png"),
                     speaker_notes="say this")
        out = spec.to_v1_dict()
        self.assertEqual(out["title"], "Photosynthesis")
        self.assertEqual(out["bullets"], ["Leaves absorb light"])
        self.assertEqual(out["imageSearchTerm"], "photosynthesis diagram")
        self.assertEqual(out["imageUrl"], "https://example.test/a.png")
        self.assertEqual(out["speakerNotes"], "say this")
        self.assertEqual(out["type"], "content")

    def test_the_v2_block_rides_alongside(self):
        out = slide().to_v1_dict()
        self.assertEqual(out["v2"]["version"], "2.0")
        self.assertEqual(out["v2"]["slideType"], "concept")

    def test_v2_only_slide_types_degrade_to_a_v1_type(self):
        for slide_type, expected in ((SlideType.PROCESS, "content"),
                                     (SlideType.TITLE, "title"),
                                     (SlideType.SUMMARY, "summary"),
                                     (SlideType.QUIZ, "content")):
            self.assertEqual(slide(slide_type=slide_type).to_v1_dict()["type"], expected)

    def test_a_block_with_only_a_label_still_becomes_a_bullet(self):
        spec = slide(content_blocks=[ContentBlock(BlockKind.STEP, "", label="Absorb light")])
        self.assertEqual(spec.to_v1_dict()["bullets"], ["Absorb light"])

    def test_everything_serialises_to_json(self):
        import json
        spec = slide(content_blocks=[ContentBlock(BlockKind.STEP, "x")],
                     visual=VisualSpec(annotations=[Annotation("stoma", (0.2, 0.3))],
                                       strategy=VisualStrategy.AI_GENERATED_IMAGE),
                     layout=LayoutSpec("anatomy_labeled",
                                       {"visual": Region(0.2, 0.2, 0.5, 0.5)}))
        json.dumps(spec.to_v1_dict())  # must not raise
