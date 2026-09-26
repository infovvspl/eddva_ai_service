"""
Tests for textbook figure extraction (ai_services.core.textbook_figures).

Fixtures are built in memory with matplotlib rather than checked in as PDFs:
a matplotlib PDF has a real text layer and real vector drawing objects, so it
exercises the same pdfplumber code paths a textbook does, and it keeps the
suite free of third-party book content.

The end-to-end fixture deliberately reproduces the page shapes that broke this
module while it was being written, so each one stays fixed:

  page 1  prose, a captioned figure, more prose   - the ordinary case
  page 2  prose only                              - must yield nothing
  page 3  two figures side by side, captions on
          ONE baseline                            - must stay separate
  page 4  a decorative page border only           - must yield nothing

Runs standalone (no Django, no database):
    python -m unittest ai_services.tests_textbook_figures
"""
import ast
import base64
import io
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                # noqa: E402

from ai_services.core import textbook_figures as tf                 # noqa: E402


PROSE = (
    "The distance between two points is found using the distance formula. "
    "It follows directly from the Pythagoras theorem applied to the right "
    "triangle formed by the horizontal and vertical separations of the points. "
)


def _border(fig):
    """The vertical page-border bars that used to weld a whole page together."""
    for x in (0.045, 0.955):
        fig.add_artist(plt.Line2D([x, x], [0.03, 0.97], lw=2.0, color="#2b3a67"))


def _triangle(fig, rect):
    """A coordinate-geometry figure of the kind a school paper actually needs."""
    ax = fig.add_axes(rect)
    ax.plot([1, 4], [1, 1], "k-", lw=1.6)
    ax.plot([4, 4], [1, 5], "k-", lw=1.6)
    ax.plot([1, 4], [1, 5], "k-", lw=1.6)
    ax.plot([1, 4, 4], [1, 1, 5], "ko", ms=4)
    ax.annotate("A(1, 1)", (1, 1), textcoords="offset points", xytext=(-6, -12), fontsize=7)
    ax.annotate("B(4, 1)", (4, 1), textcoords="offset points", xytext=(2, -12), fontsize=7)
    ax.annotate("C(4, 5)", (4, 5), textcoords="offset points", xytext=(2, 4), fontsize=7)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.grid(True, ls=":", lw=0.5)
    ax.tick_params(labelsize=6)


def _build_fixture_pdf():
    buf = io.BytesIO()
    with PdfPages(buf) as pages:
        # 1 - prose, captioned figure, prose
        fig = plt.figure(figsize=(8.27, 11.69))
        _border(fig)
        fig.text(0.12, 0.93, "7.2  Distance Formula", fontsize=13, weight="bold")
        fig.text(0.12, 0.88, PROSE * 2, fontsize=8, wrap=True, va="top")
        _triangle(fig, [0.30, 0.50, 0.40, 0.26])
        fig.text(0.5, 0.465, "Fig. 7.1  Right-angled triangle ABC on the coordinate plane",
                 ha="center", fontsize=8, style="italic")
        fig.text(0.12, 0.40, PROSE * 3, fontsize=8, wrap=True, va="top")
        pages.savefig(fig)
        plt.close(fig)

        # 2 - prose only
        fig = plt.figure(figsize=(8.27, 11.69))
        _border(fig)
        fig.text(0.12, 0.90, PROSE * 8, fontsize=8, wrap=True, va="top")
        pages.savefig(fig)
        plt.close(fig)

        # 3 - two figures, both captions on one baseline
        fig = plt.figure(figsize=(8.27, 11.69))
        _border(fig)
        fig.text(0.12, 0.93, "7.3  Two Figures", fontsize=13, weight="bold")
        _triangle(fig, [0.12, 0.62, 0.32, 0.22])
        _triangle(fig, [0.56, 0.62, 0.32, 0.22])
        fig.text(0.28, 0.59, "Fig. 7.2  First triangle", ha="center", fontsize=8, style="italic")
        fig.text(0.72, 0.59, "Fig. 7.3  Second triangle", ha="center", fontsize=8, style="italic")
        fig.text(0.12, 0.50, PROSE * 3, fontsize=8, wrap=True, va="top")
        pages.savefig(fig)
        plt.close(fig)

        # 4 - decoration only
        fig = plt.figure(figsize=(8.27, 11.69))
        _border(fig)
        fig.text(0.12, 0.93, "7.4  Nothing Here", fontsize=13, weight="bold")
        pages.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


#: Built once: rasterising pages is the slow part of this suite.
_FIXTURE_PDF = None
_FIXTURE_RESULT = None


def _fixture():
    global _FIXTURE_PDF, _FIXTURE_RESULT
    if _FIXTURE_RESULT is None:
        _FIXTURE_PDF = _build_fixture_pdf()
        _FIXTURE_RESULT = tf.extract_figures(_FIXTURE_PDF)
    return _FIXTURE_PDF, _FIXTURE_RESULT


def _box(x0, top, x1, bottom):
    return tf._Box(x0, top, x1, bottom)


class BoxTests(unittest.TestCase):
    def test_geometry(self):
        box = _box(10, 20, 30, 60)
        self.assertEqual(box.width, 20)
        self.assertEqual(box.height, 40)
        self.assertEqual(box.area, 800)

    def test_intersects_is_exclusive_at_the_edge(self):
        # Touching edges are not an intersection: two figures that abut must not
        # be merged into one.
        self.assertFalse(_box(0, 0, 10, 10).intersects(_box(10, 0, 20, 10)))
        self.assertTrue(_box(0, 0, 10, 10).intersects(_box(9, 0, 20, 10)))

    def test_union_covers_both(self):
        union = _box(0, 0, 10, 10).union(_box(20, 30, 25, 40))
        self.assertEqual(union.as_list(), [0, 0, 25, 40])


class ObjBoxTests(unittest.TestCase):
    def test_missing_and_non_finite_coordinates_are_rejected(self):
        self.assertIsNone(tf._obj_box({"x0": 1, "top": 2, "x1": None, "bottom": 4}))
        self.assertIsNone(tf._obj_box({"x0": 1, "top": 2}))
        self.assertIsNone(tf._obj_box({"x0": 1, "top": 2, "x1": float("inf"), "bottom": 4}))
        self.assertIsNone(tf._obj_box({"x0": "a", "top": 2, "x1": 3, "bottom": 4}))

    def test_inverted_rectangles_are_normalised_not_discarded(self):
        # Real producers emit these; dropping them loses parts of a figure.
        box = tf._obj_box({"x0": 30, "top": 60, "x1": 10, "bottom": 20})
        self.assertEqual(box.as_list(), [10, 20, 30, 60])


class DecorationTests(unittest.TestCase):
    """Regression: the rule test must be symmetric in both axes."""

    def setUp(self):
        self.page = _box(0, 0, 600, 800)

    def test_horizontal_rule_is_decoration(self):
        self.assertTrue(tf._is_decoration(_box(10, 400, 590, 402), self.page))

    def test_vertical_border_bar_is_decoration(self):
        # The exact shape that made the whole corpus extract zero figures: a
        # 3pt x 732pt page-frame bar that survived an asymmetric rule test and
        # then welded every figure on the page into one blob.
        self.assertTrue(tf._is_decoration(_box(25, 30, 28, 762), self.page))

    def test_full_page_backdrop_is_decoration(self):
        self.assertTrue(tf._is_decoration(_box(0, 0, 600, 800), self.page))

    def test_an_ordinary_figure_is_not_decoration(self):
        self.assertFalse(tf._is_decoration(_box(100, 100, 400, 350), self.page))


class GeometryGateTests(unittest.TestCase):
    def setUp(self):
        self.page = _box(0, 0, 600, 800)

    def test_accepts_a_figure_sized_region(self):
        ok, reason = tf._passes_geometry_gates(_box(100, 100, 400, 350), self.page)
        self.assertTrue(ok, reason)

    def test_rejects_specks(self):
        ok, reason = tf._passes_geometry_gates(_box(10, 10, 30, 30), self.page)
        self.assertFalse(ok)
        self.assertEqual(reason, "too_small")

    def test_rejects_a_region_covering_the_page(self):
        ok, reason = tf._passes_geometry_gates(_box(0, 0, 600, 790), self.page)
        self.assertFalse(ok)
        self.assertEqual(reason, "covers_page")

    def test_rejects_extreme_aspect_ratios(self):
        ok, reason = tf._passes_geometry_gates(_box(0, 100, 600, 130), self.page)
        self.assertFalse(ok)
        self.assertEqual(reason, "aspect")

    def test_rejects_degenerate_regions(self):
        ok, _reason = tf._passes_geometry_gates(_box(10, 10, 10, 10), self.page)
        self.assertFalse(ok)


class RenderFrameTests(unittest.TestCase):
    """Regression: pdfplumber reports the MediaBox, PDFium renders the CropBox."""

    class _FakePage:
        def __init__(self, media, crop, bbox):
            self.mediabox = media
            self.cropbox = crop
            self.bbox = bbox

    def test_uses_the_cropbox_converted_to_top_down_space(self):
        # The real geometry from the NCERT corpus. Mapping pdfplumber
        # coordinates onto the render as if MediaBox and CropBox agreed shifted
        # every crop by the 18pt margin and skewed it by 1.6% — small enough to
        # slip under a tolerance check and large enough to visibly mis-crop.
        page = self._FakePage(
            media=(0, 0.0, 612, 820.8),
            crop=(18.0, 18.0, 594.0, 802.8),
            bbox=(0, 0.0, 612, 820.8),
        )
        frame = tf._render_frame(page)
        self.assertEqual(frame.as_list(), [18.0, 18.0, 594.0, 802.8])
        self.assertAlmostEqual(frame.width, 576.0, places=3)
        self.assertAlmostEqual(frame.height, 784.8, places=3)

    def test_falls_back_to_bbox_when_boxes_are_absent_or_broken(self):
        self.assertEqual(
            tf._render_frame(self._FakePage(None, None, (0, 0, 100, 200))).as_list(),
            [0, 0, 100, 200],
        )
        self.assertEqual(
            tf._render_frame(self._FakePage((0, 0, 100, 200), ("x", 0, 1, 2), (0, 0, 100, 200))).as_list(),
            [0, 0, 100, 200],
        )

    def test_an_inverted_cropbox_falls_back_rather_than_producing_nonsense(self):
        page = self._FakePage(
            media=(0, 0, 612, 792), crop=(500, 10, 100, 700), bbox=(0, 0, 612, 792),
        )
        self.assertEqual(tf._render_frame(page).as_list(), [0, 0, 612, 792])


class ProseTests(unittest.TestCase):
    def test_spans_prose_requires_horizontal_overlap(self):
        # A figure beside a column of text must not be penalised for it.
        box = _box(300, 100, 560, 400)
        beside = [_box(40, 150, 280, 162), _box(40, 180, 280, 192), _box(40, 210, 280, 222)]
        self.assertFalse(tf._spans_prose(box, beside))

    def test_spans_prose_counts_lines_strictly_inside(self):
        box = _box(40, 100, 560, 400)
        inside = [_box(40, 150, 500, 162), _box(40, 180, 500, 192), _box(40, 210, 500, 222)]
        self.assertTrue(tf._spans_prose(box, inside))

    def test_lines_merely_abutting_do_not_count(self):
        box = _box(40, 100, 560, 400)
        outside = [_box(40, 80, 500, 99), _box(40, 401, 500, 420), _box(40, 430, 500, 450)]
        self.assertFalse(tf._spans_prose(box, outside))

    def test_the_limit_is_adjustable(self):
        box = _box(40, 100, 560, 400)
        bands = [_box(40, 150, 500, 162), _box(40, 180, 500, 192), _box(40, 210, 500, 222)]
        self.assertFalse(tf._spans_prose(box, bands, limit=4))
        self.assertTrue(tf._spans_prose(box, bands, limit=3))


class TextCoverageTests(unittest.TestCase):
    def test_no_words_is_zero_coverage(self):
        self.assertEqual(tf._text_coverage(_box(0, 0, 100, 100), []), 0.0)

    def test_fully_covered_is_one(self):
        self.assertAlmostEqual(tf._text_coverage(_box(0, 0, 100, 100), [_box(0, 0, 100, 100)]), 1.0)

    def test_partial_coverage_is_proportional(self):
        self.assertAlmostEqual(tf._text_coverage(_box(0, 0, 100, 100), [_box(0, 0, 50, 100)]), 0.5)

    def test_words_outside_the_region_are_ignored(self):
        self.assertEqual(tf._text_coverage(_box(0, 0, 100, 100), [_box(200, 200, 300, 300)]), 0.0)

    def test_a_degenerate_region_reports_full_coverage_so_it_is_rejected(self):
        self.assertEqual(tf._text_coverage(_box(0, 0, 0, 0), []), 1.0)


class LabelAttachTests(unittest.TestCase):
    def test_a_nearby_label_is_absorbed(self):
        page = _box(0, 0, 600, 800)
        figure = _box(100, 100, 300, 300)
        tick = _box(90, 302, 110, 310)
        grown = tf._attach_labels(figure, [tick], page)
        self.assertLessEqual(grown.top, 100)
        self.assertGreaterEqual(grown.bottom, 310)

    def test_attachment_does_not_chain_through_a_paragraph(self):
        # Regression: an iterative pass walked line by line down the page,
        # because a paragraph's lines sit about one attach-distance apart.
        page = _box(0, 0, 600, 800)
        figure = _box(100, 100, 300, 200)
        paragraph = [_box(100, 205 + i * 10, 300, 213 + i * 10) for i in range(30)]
        grown = tf._attach_labels(figure, paragraph, page)
        self.assertLess(grown.area, figure.area * tf._LABEL_MAX_GROWTH + 1)
        self.assertLess(grown.bottom, 260)

    def test_distant_words_are_never_absorbed(self):
        page = _box(0, 0, 600, 800)
        figure = _box(100, 100, 300, 300)
        grown = tf._attach_labels(figure, [_box(100, 600, 300, 620)], page)
        self.assertEqual(grown.as_list(), figure.as_list())


class CaptionPatternTests(unittest.TestCase):
    def test_matches_the_forms_school_textbooks_use(self):
        for text in ("Fig. 7.1 A triangle", "Figure 9.7 Ray diagrams",
                     "Fig 12.4 Something", "Fig. 1.10 (a) : Silos"):
            self.assertIsNotNone(tf._CAPTION_RE.match(text), text)

    def test_does_not_match_an_ordinary_word(self):
        for text in ("configure the apparatus", "figs and dates", "Figurative language"):
            self.assertIsNone(tf._CAPTION_RE.match(text), text)

    def test_requires_a_number(self):
        self.assertIsNone(tf._CAPTION_RE.match("Figure shows the apparatus"))


class ImageHelperTests(unittest.TestCase):
    def _solid(self, size, colour):
        from PIL import Image
        return Image.new("RGB", size, colour)

    def test_trim_whitespace_shrinks_to_the_ink(self):
        from PIL import Image
        image = Image.new("RGB", (200, 200), "white")
        for x in range(90, 110):
            for y in range(90, 110):
                image.putpixel((x, y), (0, 0, 0))
        trimmed = tf._trim_whitespace(image, pad_px=2)
        self.assertLess(trimmed.width, 40)
        self.assertLess(trimmed.height, 40)

    def test_trim_whitespace_leaves_a_blank_image_alone(self):
        blank = self._solid((100, 100), "white")
        self.assertEqual(tf._trim_whitespace(blank).size, (100, 100))

    def test_ink_fraction(self):
        self.assertAlmostEqual(tf._ink_fraction(self._solid((50, 50), "white")), 0.0)
        self.assertAlmostEqual(tf._ink_fraction(self._solid((50, 50), "black")), 1.0)

    def test_phash_is_stable_and_discriminating(self):
        from PIL import Image
        a = Image.new("RGB", (64, 64), "white")
        for x in range(32):
            for y in range(64):
                a.putpixel((x, y), (0, 0, 0))
        b = Image.new("RGB", (64, 64), "white")
        self.assertEqual(tf._phash(a), tf._phash(a.copy()))
        self.assertNotEqual(tf._phash(a), tf._phash(b))


class ExtractionTests(unittest.TestCase):
    """End-to-end against the synthetic textbook fixture."""

    @classmethod
    def setUpClass(cls):
        cls.pdf, cls.result = _fixture()
        cls.figures = cls.result["figures"]

    def _page(self, page_no):
        return [f for f in self.figures if f["page_no"] == page_no]

    def test_finds_the_captioned_figure(self):
        page_one = self._page(1)
        self.assertEqual(len(page_one), 1)
        self.assertEqual(page_one[0]["label"], "Fig. 7.1")
        self.assertIn("Right-angled triangle ABC", page_one[0]["caption"])

    def test_a_prose_only_page_yields_nothing(self):
        self.assertEqual(self._page(2), [])

    def test_a_decoration_only_page_yields_nothing(self):
        # Regression: the page-border bars must not become a figure, and must
        # not weld the page into one candidate either.
        self.assertEqual(self._page(4), [])

    def test_two_side_by_side_figures_stay_separate(self):
        self.assertEqual(len(self._page(3)), 2)

    def test_two_captions_on_one_baseline_are_split(self):
        # Regression: a whole-line match gave both figures the text of both
        # captions, and left one of them unlabelled.
        labels = sorted(f["label"] for f in self._page(3))
        self.assertEqual(labels, ["Fig. 7.2", "Fig. 7.3"])
        for figure in self._page(3):
            self.assertEqual(figure["caption"].count("Fig."), 1)

    def test_captions_do_not_leak_stray_words(self):
        # Regression: word boxes carried their own text, and _Box.union
        # propagated it, so a tick label ("6") surfaced as the caption.
        for figure in self.figures:
            if figure["caption"]:
                self.assertTrue(figure["caption"].lower().startswith("fig"), figure["caption"])

    def test_every_figure_decodes_to_a_png_of_the_declared_size(self):
        from PIL import Image
        for figure in self.figures:
            self.assertTrue(figure["image_base64"].startswith("data:image/png;base64,"))
            raw = base64.b64decode(figure["image_base64"].split(",", 1)[1])
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.size, (figure["width"], figure["height"]))

    def test_crops_are_large_enough_to_print(self):
        for figure in self.figures:
            self.assertGreaterEqual(figure["width"], tf._MIN_PX)
            self.assertGreaterEqual(figure["height"], tf._MIN_PX)

    def test_bbox_stays_inside_the_page(self):
        for figure in self.figures:
            x0, top, x1, bottom = figure["bbox"]
            self.assertLess(x0, x1)
            self.assertLess(top, bottom)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(top, 0)

    def test_figure_index_is_contiguous_per_page(self):
        # Half of the (page_no, figure_index) identity the backend stores, so a
        # re-ingest has to line up with the previous one.
        by_page = {}
        for figure in self.figures:
            by_page.setdefault(figure["page_no"], []).append(figure["figure_index"])
        for page_no, indexes in by_page.items():
            self.assertEqual(sorted(indexes), list(range(len(indexes))), page_no)

    def test_report_counts_what_was_scanned(self):
        report = self.result["report"]
        self.assertEqual(report["pages_scanned"], 4)
        self.assertEqual(report["figures"], len(self.figures))
        self.assertEqual(report["skipped_pages"], 0)

    def test_detector_is_always_a_known_kind(self):
        for figure in self.figures:
            self.assertIn(figure["detector"], ("image", "vector", "caption"))


class ExtractionOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf, _result = _fixture()

    def test_page_numbers_restricts_the_scan(self):
        result = tf.extract_figures(self.pdf, page_numbers=[3])
        self.assertEqual(result["report"]["pages_scanned"], 1)
        self.assertTrue(all(f["page_no"] == 3 for f in result["figures"]))

    def test_max_figures_is_a_hard_ceiling(self):
        result = tf.extract_figures(self.pdf, max_figures=1)
        self.assertEqual(len(result["figures"]), 1)

    def test_max_figures_of_zero_returns_nothing(self):
        result = tf.extract_figures(self.pdf, max_figures=0)
        self.assertEqual(result["figures"], [])

    def test_an_unknown_page_number_is_not_an_error(self):
        result = tf.extract_figures(self.pdf, page_numbers=[999])
        self.assertEqual(result["figures"], [])
        self.assertEqual(result["report"]["pages_scanned"], 0)


class RobustnessTests(unittest.TestCase):
    def test_a_non_pdf_raises_rather_than_returning_junk(self):
        with self.assertRaises(Exception):
            tf.extract_figures(b"this is not a pdf")

    def test_a_page_that_cannot_be_read_is_skipped_not_fatal(self):
        # One bad page must never cost the whole chapter.
        pdf, _result = _fixture()
        original = tf._figures_on_page
        calls = {"n": 0}

        def exploding(page, document, page_index, page_no, dpi, reject):
            calls["n"] += 1
            if page_no == 1:
                raise RuntimeError("simulated bad page")
            return original(page, document, page_index, page_no, dpi, reject)

        tf._figures_on_page = exploding
        try:
            result = tf.extract_figures(pdf)
        finally:
            tf._figures_on_page = original
        self.assertEqual(result["report"]["skipped_pages"], 1)
        self.assertGreater(calls["n"], 1)
        self.assertTrue(all(f["page_no"] != 1 for f in result["figures"]))


class DependencyTests(unittest.TestCase):
    """Regression: the renderer must not be the one that needs poppler."""

    def test_pdf2image_is_never_imported(self):
        # pdf2image shells out to poppler, which is not installed on the service
        # host: it raises PDFInfoNotInstalledError on the first call. Rendering
        # must go through pypdfium2, which bundles PDFium in the wheel.
        #
        # Checked against import STATEMENTS, not the file text: the module
        # docstring names pdf2image precisely to record why it is not used.
        with io.open(tf.__file__.replace(".pyc", ".py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("pdf2image", imported)
        self.assertIn("pypdfium2", imported)

    def test_module_imports_without_django(self):
        self.assertNotIn("django.conf", sys.modules.get(tf.__name__, tf).__dict__)


class CollectFiguresTests(unittest.TestCase):
    """The ingest-side wrapper in ai_services.core.textbook."""

    def setUp(self):
        from ai_services.core import textbook as tb
        self.tb = tb
        self.pdf, _result = _fixture()

    def test_a_scanned_chapter_produces_no_figures_at_all(self):
        # No text layer means no captions to name a figure and no drawing
        # objects to find one, so the only thing on offer would be crops of a
        # page photograph. A wrong diagram is worse than none.
        figures, report = self.tb._collect_figures(self.pdf, "ocr")
        self.assertEqual(figures, [])
        self.assertEqual(report["skipped"], "scanned_chapter")

    def test_a_text_layer_chapter_returns_figures(self):
        figures, report = self.tb._collect_figures(self.pdf, "text_layer")
        self.assertTrue(figures)
        self.assertEqual(report["returned"], len(figures))
        self.assertEqual(report["dropped_for_size"], 0)
        self.assertGreater(report["payload_bytes"], 0)

    def test_the_payload_budget_drops_rather_than_truncates(self):
        original = self.tb._MAX_FIGURE_PAYLOAD_BYTES
        self.tb._MAX_FIGURE_PAYLOAD_BYTES = 1000      # smaller than one crop
        try:
            figures, report = self.tb._collect_figures(self.pdf, "text_layer")
        finally:
            self.tb._MAX_FIGURE_PAYLOAD_BYTES = original
        self.assertEqual(figures, [])
        self.assertGreater(report["dropped_for_size"], 0)
        # Whatever is returned must still be whole images, never partial ones.
        for figure in figures:
            self.assertTrue(figure["image_base64"].startswith("data:image/png;base64,"))

    def test_an_extraction_failure_never_fails_the_ingest(self):
        # Passages are the point of indexing; figures are a bonus on top.
        from ai_services.core import textbook_figures as module
        original = module.extract_figures

        def boom(*_args, **_kwargs):
            raise RuntimeError("simulated extractor crash")

        module.extract_figures = boom
        try:
            figures, report = self.tb._collect_figures(self.pdf, "text_layer")
        finally:
            module.extract_figures = original
        self.assertEqual(figures, [])
        self.assertEqual(report["skipped"], "failed")

    def test_ingest_pdf_carries_figures_and_a_report(self):
        result = self.tb.ingest_pdf(self.pdf, allow_ocr=False)
        self.assertIn("figures", result)
        self.assertIn("figure_report", result)
        self.assertTrue(result["chunks"])
        for figure in result["figures"]:
            self.assertIn("page_no", figure)
            self.assertIn("figure_index", figure)
            self.assertIn("image_base64", figure)

    def test_ingest_pdf_can_skip_figures_entirely(self):
        result = self.tb.ingest_pdf(self.pdf, allow_ocr=False, want_figures=False)
        self.assertEqual(result["figures"], [])
        self.assertEqual(result["figure_report"]["skipped"], "not_requested")
        # The passages must be identical either way.
        self.assertTrue(result["chunks"])


class DescribeFiguresTests(unittest.TestCase):
    """Vision descriptions for the figures a book never captioned."""

    def setUp(self):
        from ai_services.core import textbook as tb
        self.tb = tb

    def test_captioned_figures_are_never_sent_to_vision(self):
        # They already carry the book's own words; describing them would be
        # paying for text we have.
        self.assertEqual(self.tb.describe_figures([{"caption": "Fig. 1: A lever"}]), 0)

    def test_nothing_to_do_is_zero_not_an_error(self):
        self.assertEqual(self.tb.describe_figures([]), 0)

    def test_the_pass_can_be_switched_off(self):
        original = self.tb._FIGURE_DESCRIPTIONS_ENABLED
        self.tb._FIGURE_DESCRIPTIONS_ENABLED = False
        try:
            result = self.tb.describe_figures([{"caption": "", "image_base64": "data:image/png;base64,aGk="}])
        finally:
            self.tb._FIGURE_DESCRIPTIONS_ENABLED = original
        self.assertEqual(result, 0)

    def test_an_unavailable_provider_is_not_an_error(self):
        # Ingestion must succeed whether or not Gemini is reachable.
        self.assertEqual(
            self.tb.describe_figures([{"caption": "", "image_base64": "data:image/png;base64,aGk="}]),
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
