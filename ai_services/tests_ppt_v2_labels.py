"""Labels must not land on each other, and must not be drawn twice.

Phase 7 rasterised a Cell Structure slide whose labels read
"VacuoCldoroplast" - two of them printed on top of one another - with
"Chloroplast" appearing three times over. The labels came out of free-form
matplotlib code, which has no idea where it has already put text.

Placement is therefore done once, deterministically, before either renderer
draws anything, and these tests pin down the properties that were violated:
deduplication, no overlap, inside the picture, and a label dropped rather
than shipped on top of its neighbour.

No model is involved. This is geometry.
"""
from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import labels as L


def items(*pairs):
    return [{"text": t, "anchor": a} for t, a in pairs]


class NormalisationTests(SimpleTestCase):
    def test_case_and_padding_do_not_make_a_new_label(self):
        for variant in ("Chloroplast", "chloroplast", " Chloroplast ",
                        "CHLOROPLAST", "Chloroplast."):
            self.assertEqual(L.normalise(variant), "chloroplast", variant)

    def test_deduplication_keeps_the_first_spelling(self):
        kept = L.dedupe(["Chloroplast", "chloroplast", " Chloroplast "])
        self.assertEqual(kept, ["Chloroplast"])

    def test_different_labels_survive(self):
        self.assertEqual(len(L.dedupe(["Nucleus", "Vacuole", "Cell wall"])), 3)

    def test_blank_labels_are_dropped(self):
        self.assertEqual(L.dedupe(["", "   ", "Nucleus"]), ["Nucleus"])


class PlacementTests(SimpleTestCase):
    def test_labels_never_overlap(self):
        # Six anchors crowded into one corner, which is the arrangement that
        # produced the overlap.
        placed, _ = L.place(items(
            ("Chloroplast", (0.60, 0.40)), ("Vacuole", (0.61, 0.41)),
            ("Nucleus", (0.59, 0.42)), ("Cell wall", (0.62, 0.39)),
            ("Cell membrane", (0.58, 0.43)), ("Cytoplasm", (0.63, 0.38)),
        ))
        self.assertEqual(L.problems(placed), [])

    def test_every_label_stays_inside_the_picture(self):
        placed, _ = L.place(items(
            ("Top left", (0.01, 0.01)), ("Top right", (0.99, 0.02)),
            ("Bottom left", (0.02, 0.99)), ("Bottom right", (0.99, 0.98)),
        ))
        self.assertTrue(placed)
        for p in placed:
            self.assertTrue(p.box.inside(), "%s at %s" % (p.text, p.box.to_dict()))

    def test_duplicates_are_placed_once(self):
        placed, _ = L.place(items(
            ("Chloroplast", (0.6, 0.4)), ("chloroplast", (0.62, 0.42)),
            (" Chloroplast ", (0.58, 0.38)),
        ))
        self.assertEqual(len(placed), 1)
        self.assertEqual(L.problems(placed), [])

    def test_every_label_keeps_the_anchor_it_points_at(self):
        placed, _ = L.place(items(("Nucleus", (0.3, 0.7))))
        self.assertEqual(placed[0].anchor, (0.3, 0.7))

    def test_a_label_is_dropped_rather_than_drawn_over_its_neighbour(self):
        # Twenty labels on one anchor cannot all be placed clear.
        placed, dropped = L.place(items(*[("Label %d" % i, (0.5, 0.5))
                                          for i in range(20)]))
        self.assertTrue(dropped)
        self.assertEqual(L.problems(placed), [])

    def test_shrinking_is_tried_before_dropping(self):
        placed, _ = L.place(items(*[("L%d" % i, (0.5, 0.1 * i)) for i in range(9)]))
        self.assertTrue(any(p.scale < 1.0 for p in placed))
        self.assertEqual(L.problems(placed), [])

    def test_a_preferred_side_is_honoured_when_it_is_free(self):
        placed, _ = L.place([{"text": "Stoma", "anchor": (0.5, 0.5),
                              "side": "top"}])
        self.assertEqual(placed[0].side, "top")

    def test_an_anchor_on_the_left_gets_a_label_on_its_right(self):
        # The opposite would run off the picture.
        placed, _ = L.place(items(("Xylem", (0.05, 0.5))))
        self.assertEqual(placed[0].side, "right")

    def test_a_malformed_anchor_does_not_break_placement(self):
        placed, _ = L.place([{"text": "Odd", "anchor": None},
                             {"text": "Odder", "anchor": ("x", "y")}])
        self.assertEqual(len(placed), 2)
        self.assertEqual(L.problems(placed), [])

    def test_placement_is_deterministic(self):
        spec = items(("A", (0.3, 0.3)), ("B", (0.31, 0.31)), ("C", (0.7, 0.7)))
        runs = {tuple((p.text, p.side, p.box.x, p.box.y)
                      for p in L.place(spec)[0]) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_earlier_labels_win_a_contested_position(self):
        placed, _ = L.place(items(("First", (0.5, 0.5)), ("Second", (0.5, 0.5))))
        self.assertEqual(placed[0].text, "First")
        self.assertEqual(L.problems(placed), [])


class ProblemDetectionTests(SimpleTestCase):
    def test_overlap_is_reported(self):
        a = L.Placed("A", (0.5, 0.5), L.Box(0.1, 0.1, 0.3, 0.1), "right")
        b = L.Placed("B", (0.5, 0.5), L.Box(0.2, 0.12, 0.3, 0.1), "right")
        self.assertTrue(any(c == "label_overlap" for c, _ in L.problems([a, b])))

    def test_a_duplicate_is_reported(self):
        a = L.Placed("Chloroplast", (0.2, 0.2), L.Box(0.05, 0.05, 0.2, 0.08), "right")
        b = L.Placed("chloroplast", (0.8, 0.8), L.Box(0.6, 0.6, 0.2, 0.08), "right")
        self.assertTrue(any(c == "duplicate_label" for c, _ in L.problems([a, b])))

    def test_an_out_of_bounds_label_is_reported(self):
        a = L.Placed("A", (0.9, 0.9), L.Box(0.9, 0.9, 0.3, 0.2), "right")
        self.assertTrue(any(c == "label_out_of_bounds" for c, _ in L.problems([a])))

    def test_a_zero_size_label_is_reported(self):
        a = L.Placed("A", (0.5, 0.5), L.Box(0.3, 0.3, 0.0, 0.1), "right")
        self.assertTrue(any(c == "zero_size_label" for c, _ in L.problems([a])))

    def test_a_clean_arrangement_reports_nothing(self):
        a = L.Placed("A", (0.2, 0.2), L.Box(0.05, 0.05, 0.25, 0.09), "right")
        b = L.Placed("B", (0.8, 0.8), L.Box(0.6, 0.6, 0.25, 0.09), "right")
        self.assertEqual(L.problems([a, b]), [])


class CellStructureRegressionTests(SimpleTestCase):
    """The exact slide that failed, as a permanent fixture."""

    def _compose(self):
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.visuals import VisualCapabilities
        plan = [{
            "type": "anatomy", "title": "Plant Cell: Main Parts",
            "learningObjective": "Name the main parts of a plant cell",
            "blocks": [{"label": "Cell Wall", "text": "A rigid outer layer."},
                       {"label": "Nucleus", "text": "The control centre."}],
            "visual": {
                "kind": "diagram", "description": "a plant cell",
                "objects": ["plant cell"],
                # Three spellings of one label plus a crowded neighbour: what
                # the planner actually produced.
                "annotations": [{"text": "Chloroplast", "anchor": [0.60, 0.40]},
                                {"text": "chloroplast", "anchor": [0.62, 0.42]},
                                {"text": " Chloroplast ", "anchor": [0.58, 0.38]},
                                {"text": "Vacuole", "anchor": [0.61, 0.41]},
                                {"text": "Nucleus", "anchor": [0.30, 0.50]},
                                {"text": "Cell wall", "anchor": [0.50, 0.90]}],
            }}]
        return compose_deck(plan, DeckContext(topic="Cell Structure and Functions",
                                              subject="Science",
                                              chapter="Cell Structure and Functions",
                                              class_name="Class 8"),
                            caps=VisualCapabilities(ai_images=True, image_search=True,
                                                    diagrams=True, cache=False,
                                                    ai_image_budget=4))

    def test_chloroplast_is_drawn_once(self):
        annotations = self._compose().slides[0].visual.annotations
        names = [L.normalise(a.text) for a in annotations]
        self.assertEqual(names.count("chloroplast"), 1)

    def test_no_label_overlaps_another(self):
        annotations = self._compose().slides[0].visual.annotations
        placed = [L.Placed(a.text, a.anchor, L.Box(*a.box), a.side, a.scale)
                  for a in annotations]
        self.assertEqual(L.problems(placed), [])

    def test_every_label_has_a_box_inside_the_picture(self):
        for a in self._compose().slides[0].visual.annotations:
            self.assertIsNotNone(a.box, a.text)
            self.assertTrue(L.Box(*a.box).inside(), a.text)

    def test_the_deck_passes_structural_qa(self):
        from ai_services.core.ppt_v2 import qa
        deck = self._compose()
        issues = qa.check_deck(deck.slides)
        self.assertEqual(qa.errors(issues), [], qa.report(issues))

    def test_label_problems_are_structural_errors_not_a_new_category(self):
        from ai_services.core.ppt_v2 import gate
        for code in ("label_overlap", "duplicate_label", "label_out_of_bounds",
                     "zero_size_label"):
            self.assertEqual(gate.classify_issue(code), gate.LAYOUT_ERROR, code)


class AxesStayOffTests(SimpleTestCase):
    """Phase 6 switched the matplotlib axes off; they must stay off."""

    def test_a_drawing_still_has_no_axes(self):
        from ai_services.core import diagram as D
        preamble = D._preamble(4.2, 3.46, 13.0, axes=False)
        self.assertIn("'axes.spines.left': False", preamble)
        self.assertIn("'xtick.labelbottom': False", preamble)

    def test_a_figure_we_label_ourselves_is_asked_for_no_text(self):
        from ai_services.core import diagram as D
        prompt = D._build_user_prompt("a plant cell", "Science", "Class 8",
                                      "CBSE", width_in=4.2, height_in=3.46,
                                      label_text=False)
        self.assertIn("draw NO text at all", prompt)

    def test_a_figure_that_draws_text_when_asked_not_to_is_rejected(self):
        from ai_services.core import diagram as D
        problem = D._legibility_problem('ax.text(0, 0, "Nucleus")', 4.2, 3.46,
                                        allow_text=False)
        self.assertIsNotNone(problem)
        self.assertIn("no text at all", problem)
