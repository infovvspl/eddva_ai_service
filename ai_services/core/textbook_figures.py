"""
Textbook figure extraction — crop the diagrams out of a chapter PDF.

WHY THIS EXISTS
A generated question paper is text-only today, so any question that needs a
diagram (a coordinate grid, a ray diagram, a circuit, a labelled cell) either
cannot be set or is set badly as prose. The school's own book already contains
exactly the right figure, drawn for exactly this syllabus, and the PDF is kept
in study_materials long after indexing — so the figure is recoverable at any
time, including retroactively for books already indexed.

This module finds those figures and returns them as cropped PNGs. Like the rest
of ai_services.core.textbook it is deliberately STATELESS: it returns images and
metadata, and the NestJS backend owns persisting them. Nothing here writes to a
database or to object storage.

HOW A FIGURE IS FOUND
Three detectors, because no single one covers a real textbook:

  image   - an embedded raster XObject (page.images). Exact bbox, free. Covers
            scanned/photographic figures, and misses vector line art entirely.
  vector  - a dense cluster of curves/lines/rects. This is what catches the
            line diagrams that most school geometry and physics figures are
            drawn as, which carry no raster image at all.
  caption - the region directly above a "Fig. 7.1"-style caption. Textbooks
            label their figures, so this both locates a figure the other two
            detectors missed AND names it, which is what makes the result
            useful downstream rather than an anonymous crop.

Clustering is done on a coarse occupancy grid with OpenCV connected components
rather than by comparing drawing objects pairwise. One page of a formula book in
the test corpus carries over 12,000 curves; the pairwise version of this is
~10^8 comparisons per page, and the grid version is linear in the object count.

RENDERING
Pages are rasterised with pypdfium2, NOT pdf2image. pdf2image shells out to
poppler, which is absent from the service host - it raises
PDFInfoNotInstalledError on the first call. pypdfium2 ships PDFium inside the
wheel and needs nothing installed, and it is already present as a pdfplumber
dependency.

THE GOVERNING RULE
A wrong diagram on an exam paper is far worse than no diagram. Every ambiguous
case here resolves to "emit nothing" — an unreadable scan, a page whose geometry
does not match its render, a crop that is mostly blank or mostly text. Precision
over recall, deliberately.
"""
import base64
import io
import logging
import math
import re

logger = logging.getLogger("ai_services.textbook")

# ── Tunables ────────────────────────────────────────────────────────────────
# Chosen against the mixed test corpus (vector formula sheets, raster physics
# sheets, text-only notes). They are deliberately conservative: every one of
# them can only cause a figure to be DROPPED, never a wrong crop to be emitted.

#: Raster resolution for page rendering. 170 DPI keeps a half-page figure around
#: 700x500 px — sharp enough to print on a question paper without making the
#: stored PNG large.
DEFAULT_DPI = 170

#: Occupancy-grid cell, in PDF points. 4pt (~1.4mm) is fine enough to separate
#: two diagrams sitting side by side and coarse enough that a 12k-curve page
#: still costs one cheap pass.
_GRID_CELL_PT = 4.0

#: Dilation applied to the drawing grid before connected components, in cells.
#: Bridges the gaps inside a single figure (an axis and its tick labels, a
#: dashed line) without welding two neighbouring figures into one blob.
_GRID_DILATE_CELLS = 3

#: A figure must occupy at least this fraction of the page. Below it we are
#: looking at a bullet glyph, a rule, or a stray mark.
_MIN_AREA_FRAC = 0.015

#: ...and at most this fraction. Above it the "cluster" is the page border or a
#: full-page background box, not a figure.
_MAX_AREA_FRAC = 0.80

#: Minimum rendered size in pixels on each side, at DEFAULT_DPI.
_MIN_PX = 110

#: Aspect-ratio band. Outside it the region is a rule, a sidebar, or a column
#: separator rather than a diagram.
_MIN_ASPECT = 0.12
_MAX_ASPECT = 9.0

#: A region whose area is more than this fraction covered by word boxes is a
#: paragraph or a table, not a figure. Kept generous because a real diagram
#: carries labels inside it (axis names, vertex labels like "A(1, 1)").
_MAX_TEXT_COVERAGE = 0.45

#: A crop must have at least this fraction of non-background pixels. Rejects the
#: blank regions that caption-anchoring can produce when a figure is actually on
#: the previous page.
_MIN_INK_FRAC = 0.004

#: Page rules and border bars are decoration. Anything thin in one dimension and
#: long in the other is dropped before clustering, so it cannot glue unrelated
#: regions together.
#:
#: This test MUST be symmetric. When it only caught horizontal rules, the 3pt x
#: 732pt vertical border bars that designed textbook pages are framed with
#: survived, and after dilation a single border welded every figure on the page
#: into one page-covering blob that was then rejected wholesale — the whole
#: corpus extracted zero figures.
_RULE_MAX_THICKNESS_PT = 3.5
_RULE_MIN_LENGTH_FRAC = 0.55

#: A single drawing object this large is a background panel or a page frame, not
#: part of a figure. Dropped before clustering for the same reason as the rules.
_BACKDROP_MIN_AREA_FRAC = 0.55

#: Repeated-artwork guard. A crop whose perceptual hash recurs on at least this
#: many pages AND on at least this fraction of pages is a running header, logo
#: or watermark rather than a figure.
_REPEAT_MIN_PAGES = 3
_REPEAT_PAGE_FRAC = 0.30

#: Hard ceilings so one pathological book cannot exhaust the worker.
_MAX_FIGURES_PER_PAGE = 6
_DEFAULT_MAX_FIGURES = 60

#: Padding applied around a detected bbox before cropping, in PDF points, so a
#: figure is never shaved at the edge.
_CROP_PAD_PT = 5.0

#: A figure's own text — axis tick labels, vertex labels like "A(1, 1)", units —
#: lives in the text layer, not among the drawing objects, so a cluster bbox
#: stops at the artwork and shaves those labels off. Words within this many
#: points of the cluster are pulled into it.
_LABEL_ATTACH_PT = 11.0

#: ...but only while the region stays within this multiple of its original area.
#: Without the cap, one adjacent paragraph drags the crop across the whole page.
#:
#: Attachment is a SINGLE pass for the same reason. When it iterated, each newly
#: absorbed word extended the reach by another _LABEL_ATTACH_PT, and since the
#: lines of a paragraph sit about that far apart the region chained down through
#: the body text until it hit the area cap.
_LABEL_MAX_GROWTH = 1.25

#: A region with at least this many lines of body prose running through it is
#: page furniture — a background panel or a watermark — not a figure.
#:
#: This is what catches the NCERT "do not republish" watermark, a raster image
#: covering 48% of every page. It sits under the area threshold for a backdrop,
#: so it survived as a candidate and then merged with every real figure on the
#: page. Area alone cannot separate it from a legitimate half-page composite
#: figure; the prose running across it can.
_MAX_PROSE_LINES_INSIDE = 3

#: The same test applied to a finished candidate is a safety net, not a
#: classifier, so it is deliberately more permissive. At the panel threshold it
#: was rejecting genuine wide figures whose internal label rows read as prose —
#: one chapter went from 35 figures to 13. Page furniture is already removed at
#: the object level by then; this only has to catch what slipped through.
_FIGURE_MAX_PROSE_LINES = 6

#: A line of body prose is a barrier: textbook figures sit between paragraphs,
#: they do not straddle a line of running text. Lines at least this wide (as a
#: fraction of the page) and this many words long cut the occupancy grid, so a
#: cluster can never span one.
#:
#: The width threshold is well below half the page because school books set text
#: in two columns and inside sidebar panels; at 0.55 a narrow column of prose
#: was not recognised as prose at all, and panels containing both text and a
#: figure were emitted whole. The word count is what keeps a figure's own
#: internal labels from being mistaken for a line of prose.
_PROSE_MIN_WIDTH_FRAC = 0.32
_PROSE_MIN_WORDS = 6

#: Rendered-vs-declared page geometry must agree within this relative tolerance.
#: A mismatch means the page is rotated or has an offset MediaBox that the
#: coordinate mapping below would silently get wrong, so the page is skipped.
_GEOMETRY_TOLERANCE = 0.02

#: "Fig. 7.1", "Figure 3", "Fig 12.4(a)" — the caption forms school textbooks
#: actually use. Anchored at a word start so "configure" never matches.
_CAPTION_RE = re.compile(r"^(fig(?:ure)?)\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)

#: How far above a caption to look for its figure, as a fraction of page height.
_CAPTION_LOOKUP_FRAC = 0.45

#: A caption's own line, plus this much slack, is excluded from the crop so the
#: caption text is not baked into the image.
_CAPTION_GAP_PT = 2.0


class _Box:
    """A rectangle in pdfplumber's top-down page space, in PDF points."""

    __slots__ = ("x0", "top", "x1", "bottom", "detector", "caption", "label")

    def __init__(self, x0, top, x1, bottom, detector="vector", caption="", label=""):
        self.x0 = float(x0)
        self.top = float(top)
        self.x1 = float(x1)
        self.bottom = float(bottom)
        self.detector = detector
        self.caption = caption
        self.label = label

    @property
    def width(self):
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self):
        return max(0.0, self.bottom - self.top)

    @property
    def area(self):
        return self.width * self.height

    def as_list(self):
        return [round(self.x0, 2), round(self.top, 2), round(self.x1, 2), round(self.bottom, 2)]

    def intersects(self, other):
        return not (
            self.x1 <= other.x0 or other.x1 <= self.x0
            or self.bottom <= other.top or other.bottom <= self.top
        )

    def union(self, other):
        return _Box(
            min(self.x0, other.x0), min(self.top, other.top),
            max(self.x1, other.x1), max(self.bottom, other.bottom),
            self.detector, self.caption or other.caption, self.label or other.label,
        )


def _obj_box(obj):
    """A pdfplumber object's bbox, or None when any coordinate is missing.

    Malformed objects do appear in real books — a curve with a null bound is
    enough to abort a whole page if it is not filtered here.
    """
    try:
        x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    except (KeyError, TypeError):
        return None
    if None in (x0, top, x1, bottom):
        return None
    try:
        x0, top, x1, bottom = float(x0), float(top), float(x1), float(bottom)
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x0, top, x1, bottom))):
        return None
    # Some producers emit inverted rectangles; normalise rather than discard.
    if x1 < x0:
        x0, x1 = x1, x0
    if bottom < top:
        top, bottom = bottom, top
    return _Box(x0, top, x1, bottom)


def _clamp_to_page(box, page_box):
    """Clip a bbox to the page, returning None when nothing is left inside it."""
    x0 = max(box.x0, page_box.x0)
    top = max(box.top, page_box.top)
    x1 = min(box.x1, page_box.x1)
    bottom = min(box.bottom, page_box.bottom)
    if x1 - x0 <= 0 or bottom - top <= 0:
        return None
    return _Box(x0, top, x1, bottom, box.detector, box.caption, box.label)


def _is_decoration(box, page_box):
    """True for a page rule, a border bar, or a full-page background panel.

    Symmetric in both axes on purpose — see _RULE_MAX_THICKNESS_PT.
    """
    if page_box.width <= 0 or page_box.height <= 0:
        return False
    horizontal_rule = (
        box.height <= _RULE_MAX_THICKNESS_PT
        and box.width >= page_box.width * _RULE_MIN_LENGTH_FRAC
    )
    vertical_rule = (
        box.width <= _RULE_MAX_THICKNESS_PT
        and box.height >= page_box.height * _RULE_MIN_LENGTH_FRAC
    )
    backdrop = box.area >= page_box.area * _BACKDROP_MIN_AREA_FRAC
    return horizontal_rule or vertical_rule or backdrop


def _drawing_boxes(page, page_box, prose_bands=()):
    """Every vector drawing object on the page, minus obvious decoration.

    A sidebar panel ("Activity 9.4", "Ready to Go Beyond") is a single filled
    rounded rect whose BOUNDING BOX covers the whole panel — text and figure
    alike. Marked into the occupancy grid it fills the panel solid, so cutting
    the prose out of the grid no longer separates anything and the panel is
    emitted whole. Dropping the panel object itself leaves only the artwork
    inside it, which the prose cuts then separate correctly.

    Only sizeable objects are tested, since _spans_prose is pointless for a
    glyph-sized curve and this runs over every drawing object on the page.
    """
    out = []
    panel_min_area = page_box.area * _MIN_AREA_FRAC
    for kind in ("curves", "lines", "rects"):
        for obj in getattr(page, kind, None) or []:
            box = _obj_box(obj)
            if box is None:
                continue
            box = _clamp_to_page(box, page_box)
            if box is None:
                continue
            if _is_decoration(box, page_box):
                continue
            if box.area >= panel_min_area and _spans_prose(box, prose_bands):
                continue
            out.append(box)
    return out


def _word_boxes(page, page_box):
    """Word bboxes, used to measure how text-heavy a candidate region is."""
    out = []
    try:
        words = page.extract_words() or []
    except Exception as exc:                     # a bad page must not lose the book
        logger.warning("Word extraction failed on page %s: %s", getattr(page, "page_number", "?"), exc)
        return out
    for word in words:
        box = _obj_box(word)
        if box is None:
            continue
        box = _clamp_to_page(box, page_box)
        if box is not None:
            # Deliberately NOT carrying the word's text on the box. _Box.union
            # propagates `caption`, so a word box that carried its own text
            # leaked a stray tick label ("6") into the figure's caption when
            # _attach_labels absorbed it.
            out.append(box)
    return out


def _vector_clusters(boxes, page_box, prose_bands=()):
    """Group drawing objects into figure-sized regions.

    Marks a coarse occupancy grid, dilates it so the parts of one drawing join
    up, then takes connected components. Linear in the number of objects, which
    matters: pages with five figures of dense vector art in the test corpus
    carry over 12,000 curve objects each.
    """
    if not boxes:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError as exc:                   # pragma: no cover - env guard
        logger.warning("OpenCV/numpy unavailable, vector figure detection off: %s", exc)
        return []

    cols = max(1, int(math.ceil(page_box.width / _GRID_CELL_PT)))
    rows = max(1, int(math.ceil(page_box.height / _GRID_CELL_PT)))
    grid = np.zeros((rows, cols), dtype="uint8")

    for box in boxes:
        c0 = int((box.x0 - page_box.x0) / _GRID_CELL_PT)
        c1 = int(math.ceil((box.x1 - page_box.x0) / _GRID_CELL_PT))
        r0 = int((box.top - page_box.top) / _GRID_CELL_PT)
        r1 = int(math.ceil((box.bottom - page_box.top) / _GRID_CELL_PT))
        # A zero-extent object (a dot, a vertical hairline) still occupies one
        # cell; without the +1 it would mark nothing and vanish from clustering.
        c0 = min(max(c0, 0), cols - 1)
        r0 = min(max(r0, 0), rows - 1)
        c1 = min(max(c1, c0 + 1), cols)
        r1 = min(max(r1, r0 + 1), rows)
        grid[r0:r1, c0:c1] = 255

    if _GRID_DILATE_CELLS > 0:
        k = 2 * _GRID_DILATE_CELLS + 1
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # Cut the grid along every line of body prose, AFTER dilating — dilation is
    # what bridges a figure's own parts, and clearing the barriers afterwards is
    # what stops that bridge from continuing through the text below it.
    for band in prose_bands:
        r0 = int((band.top - page_box.top) / _GRID_CELL_PT)
        r1 = int(math.ceil((band.bottom - page_box.top) / _GRID_CELL_PT))
        r0 = min(max(r0, 0), rows)
        r1 = min(max(r1, r0 + 1), rows)
        # Only the columns the line actually occupies. Clearing the whole row
        # would cut a figure standing beside the text in the other column.
        c0 = int((band.x0 - page_box.x0) / _GRID_CELL_PT)
        c1 = int(math.ceil((band.x1 - page_box.x0) / _GRID_CELL_PT))
        c0 = min(max(c0, 0), cols)
        c1 = min(max(c1, c0 + 1), cols)
        grid[r0:r1, c0:c1] = 0

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(grid, connectivity=8)
    clusters = []
    for i in range(1, count):                    # 0 is the background component
        x, y, w, h, _area = stats[i]
        # Undo the dilation so the bbox hugs the drawing rather than its halo.
        x0 = page_box.x0 + (x + _GRID_DILATE_CELLS) * _GRID_CELL_PT
        top = page_box.top + (y + _GRID_DILATE_CELLS) * _GRID_CELL_PT
        x1 = page_box.x0 + (x + w - _GRID_DILATE_CELLS) * _GRID_CELL_PT
        bottom = page_box.top + (y + h - _GRID_DILATE_CELLS) * _GRID_CELL_PT
        if x1 <= x0 or bottom <= top:
            continue
        box = _clamp_to_page(_Box(x0, top, x1, bottom, "vector"), page_box)
        if box is not None:
            clusters.append(box)
    return clusters


def _image_regions(page, page_box):
    """Embedded raster images, merged when they tile one picture.

    Books produced from scans routinely slice a single figure into several
    XObjects laid edge to edge; emitting each slice as its own figure would be
    wrong, so touching images are unioned.

    Page-sized images are dropped BEFORE that merge, not gated afterwards. Every
    NCERT chapter in the test corpus paints a full-page background image; when
    it was left in, the merge welded it to every real figure on the page and the
    whole page became one candidate that was then rejected for covering the
    page — the entire corpus yielded zero figures.
    """
    boxes = []
    for obj in getattr(page, "images", None) or []:
        box = _obj_box(obj)
        if box is None:
            continue
        box = _clamp_to_page(box, page_box)
        if box is None:
            continue
        if _is_decoration(box, page_box):
            continue
        box.detector = "image"
        boxes.append(box)
    return _merge_overlapping(boxes)


def _spans_prose(box, prose_bands, limit=_MAX_PROSE_LINES_INSIDE):
    """True when lines of running text cross the region — see _MAX_PROSE_LINES_INSIDE."""
    crossing = 0
    for band in prose_bands:
        # Strictly inside vertically, and actually overlapping horizontally, so
        # a figure that merely abuts the paragraph above or below it — or sits
        # in the neighbouring column — is not penalised.
        if band.top < box.top or band.bottom > box.bottom:
            continue
        if min(band.x1, box.x1) - max(band.x0, box.x0) <= 0:
            continue
        crossing += 1
        if crossing >= limit:
            return True
    return False


def _render_frame(page):
    """The page region pypdfium2 will actually rasterise, in pdfplumber space.

    pdfplumber reports the MediaBox while PDFium renders the CropBox, and real
    textbooks set them differently — every NCERT chapter here crops 18pt of
    margin away. Mapping pdfplumber coordinates onto that render as if the two
    agreed shifts every crop by the margin and skews it by the size ratio. The
    skew in the corpus was 1.6%, which is small enough to slip under a
    sanity-check tolerance and large enough to visibly mis-crop a figure, so the
    frame is computed explicitly instead of being assumed and spot-checked.

    Returns a _Box in pdfplumber's top-down coordinate space.
    """
    media = getattr(page, "mediabox", None)
    crop = getattr(page, "cropbox", None) or media
    if not media or not crop or len(media) < 4 or len(crop) < 4:
        return _Box(*page.bbox)
    try:
        media_top = float(media[1])
        media_bottom = float(media[3])
        crop_x0, crop_y0, crop_x1, crop_y1 = (float(v) for v in crop[:4])
    except (TypeError, ValueError):
        return _Box(*page.bbox)

    # MediaBox/CropBox are PDF rects (bottom-up); pdfplumber object coordinates
    # measure `top` downward from the top of the MediaBox. Convert the crop's
    # vertical extent into that same downward-measured space.
    media_height = media_bottom - media_top
    top = media_height - crop_y1
    bottom = media_height - crop_y0
    if crop_x1 <= crop_x0 or bottom <= top:
        return _Box(*page.bbox)
    return _Box(crop_x0, top, crop_x1, bottom)


def _merge_overlapping(boxes, gap_pt=6.0):
    """Union boxes that touch or nearly touch, repeatedly until stable."""
    items = list(boxes)
    merged = True
    while merged and len(items) > 1:
        merged = False
        out = []
        while items:
            current = items.pop()
            grown = _Box(
                current.x0 - gap_pt, current.top - gap_pt,
                current.x1 + gap_pt, current.bottom + gap_pt,
            )
            keep = []
            for other in items:
                if grown.intersects(other):
                    current = current.union(other)
                    merged = True
                else:
                    keep.append(other)
            items = keep
            out.append(current)
        items = out
    return items


def _captions(page, page_box):
    """Find "Fig. 7.1"-style captions and return (label, text, line_box).

    Words are grouped into lines by their vertical position first, so the
    returned caption is the whole sentence the book prints under its figure,
    not just the two words that matched the pattern.
    """
    words = []
    try:
        raw = page.extract_words() or []
    except Exception:
        return []
    for word in raw:
        box = _obj_box(word)
        if box is None:
            continue
        words.append((box, str(word.get("text") or "")))
    if not words:
        return []

    words.sort(key=lambda w: (round(w[0].top, 1), w[0].x0))
    lines = []
    for box, text in words:
        if lines and abs(lines[-1]["top"] - box.top) <= 3.0:
            lines[-1]["words"].append((box, text))
            lines[-1]["box"] = lines[-1]["box"].union(box)
        else:
            lines.append({"top": box.top, "box": box, "words": [(box, text)]})

    found = []
    for line in lines:
        # Two figures printed side by side have their captions on ONE baseline,
        # so a line can hold more than one caption. Each "Fig. N" on the line
        # starts a new segment, and each segment keeps only its own words — a
        # whole-line match gave both figures the text of both captions.
        #
        # The probe joins a short run of words because a PDF tokenises
        # "Fig. 7.1" as the separate words "Fig." and "7.1"; matching a single
        # word finds the prefix but never the number.
        words = line["words"]
        starts = [
            i for i in range(len(words))
            if _CAPTION_RE.match(" ".join(t for _b, t in words[i:i + 3]).strip())
        ]
        if not starts:
            continue
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(words)
            segment = words[start:end]
            if not segment:
                continue
            text = " ".join(t for _b, t in segment).strip()
            match = _CAPTION_RE.match(text)
            if not match:
                continue
            box = segment[0][0]
            for word_box, _text in segment[1:]:
                box = box.union(word_box)
            box = _clamp_to_page(box, page_box)
            if box is not None:
                found.append({
                    "label": f"{match.group(1).title()}. {match.group(2)}",
                    "text": text,
                    "box": box,
                })
    return found


def _region_above_caption(caption_box, page_box, blockers):
    """The whitespace-bounded band directly above a caption.

    Used only when neither the image nor the vector detector produced anything
    overlapping the caption — a figure drawn in a way this module cannot see
    (an embedded form XObject, say) is still worth cropping blind, because the
    caption is strong evidence that something is printed there.

    `blockers` are the text lines that must not be swallowed: the band stops at
    the lowest one that sits above the caption.
    """
    ceiling = page_box.top
    limit = caption_box.top - page_box.height * _CAPTION_LOOKUP_FRAC
    for box in blockers:
        if box.bottom <= caption_box.top - _CAPTION_GAP_PT and box.bottom > ceiling:
            if box.bottom >= limit:
                ceiling = box.bottom
    top = max(ceiling, limit, page_box.top)
    bottom = caption_box.top - _CAPTION_GAP_PT
    if bottom - top <= 0:
        return None
    return _clamp_to_page(_Box(page_box.x0, top, page_box.x1, bottom, "caption"), page_box)


def _attach_labels(box, word_boxes, page_box):
    """Grow a region to include the figure's own labels.

    Tick numbers and vertex labels are text, so a cluster built from drawing
    objects alone stops at the artwork and crops them in half. Growth is
    iterative (a newly attached label can bring its neighbour into reach) and
    capped by area, so a paragraph sitting next to the figure cannot drag the
    crop across the page.
    """
    original_area = box.area
    if original_area <= 0:
        return box
    grown = _Box(box.x0, box.top, box.x1, box.bottom, box.detector, box.caption, box.label)
    # Reach is measured from the ORIGINAL box and the pass runs once, so an
    # absorbed word can never extend the reach that finds the next one.
    reach = _Box(
        box.x0 - _LABEL_ATTACH_PT, box.top - _LABEL_ATTACH_PT,
        box.x1 + _LABEL_ATTACH_PT, box.bottom + _LABEL_ATTACH_PT,
    )
    for word in word_boxes:
        if not reach.intersects(word):
            continue
        candidate = grown.union(word)
        if candidate.area > original_area * _LABEL_MAX_GROWTH:
            continue
        grown = candidate
    clamped = _clamp_to_page(grown, page_box)
    return clamped if clamped is not None else box


def _prose_bands(page, page_box):
    """Vertical bands occupied by lines of running text.

    A textbook figure sits between paragraphs; it never straddles a line of body
    prose. Treating those lines as barriers is what stops a cluster from running
    from a diagram, through a caption and an activity box, into the paragraphs
    below — which is exactly what an NCERT page produced without this.

    Returns the line boxes themselves, not just their vertical extent: a page
    set in two columns must have its left column cut without also cutting the
    figure that sits beside it in the right one.
    """
    try:
        words = page.extract_words() or []
    except Exception:
        return []

    boxes = []
    for word in words:
        box = _obj_box(word)
        if box is not None:
            clamped = _clamp_to_page(box, page_box)
            if clamped is not None:
                boxes.append(clamped)
    if not boxes:
        return []

    boxes.sort(key=lambda b: (round(b.top, 1), b.x0))
    lines = []
    for box in boxes:
        if lines and abs(lines[-1]["box"].top - box.top) <= 3.0:
            lines[-1]["box"] = lines[-1]["box"].union(box)
            lines[-1]["count"] += 1
        else:
            lines.append({"box": box, "count": 1})

    bands = []
    for line in lines:
        if (
            line["count"] >= _PROSE_MIN_WORDS
            and line["box"].width >= page_box.width * _PROSE_MIN_WIDTH_FRAC
        ):
            bands.append(line["box"])
    return bands


def _text_coverage(box, word_boxes):
    """Fraction of a region's area covered by word boxes.

    Approximated by summing intersections rather than rasterising a mask: words
    on a page do not overlap each other meaningfully, so the sum is close, and
    it is far cheaper than a per-region bitmap.
    """
    if box.area <= 0:
        return 1.0
    covered = 0.0
    for word in word_boxes:
        ix0 = max(box.x0, word.x0)
        ix1 = min(box.x1, word.x1)
        itop = max(box.top, word.top)
        ibottom = min(box.bottom, word.bottom)
        if ix1 > ix0 and ibottom > itop:
            covered += (ix1 - ix0) * (ibottom - itop)
    return min(1.0, covered / box.area)


def _passes_geometry_gates(box, page_box):
    """Size and shape gates, in PDF space, before anything is rendered."""
    page_area = page_box.area
    if page_area <= 0 or box.area <= 0:
        return False, "degenerate"
    frac = box.area / page_area
    if frac < _MIN_AREA_FRAC:
        return False, "too_small"
    if frac > _MAX_AREA_FRAC:
        return False, "covers_page"
    if box.height <= 0:
        return False, "degenerate"
    aspect = box.width / box.height
    if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
        return False, "aspect"
    return True, ""


def _phash(pil_image):
    """64-bit average hash. Cheap, and enough to spot a repeated logo."""
    small = pil_image.convert("L").resize((8, 8))
    pixels = list(small.getdata())
    mean = sum(pixels) / float(len(pixels))
    bits = 0
    for i, value in enumerate(pixels):
        if value >= mean:
            bits |= (1 << i)
    return bits


def _trim_whitespace(pil_image, pad_px=6):
    """Shrink a crop to its ink, so figures are tight without tuned margins.

    NumPy only. This used to import cv2 as well and never use it, which meant
    that anywhere OpenCV was absent the function quietly returned the crop
    untrimmed — a figure with a page-width margin around it, and no error to
    say why.
    """
    try:
        import numpy as np
    except ImportError:                          # pragma: no cover - env guard
        return pil_image
    arr = np.array(pil_image.convert("L"))
    if arr.size == 0:
        return pil_image
    # Anything meaningfully darker than paper counts as ink. A fixed threshold
    # beats Otsu here: Otsu on an almost-blank crop invents a split and "finds"
    # structure in paper grain.
    mask = (arr < 245).astype("uint8")
    if not mask.any():
        return pil_image
    ys, xs = mask.nonzero()
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    height, width = arr.shape[:2]
    left = max(0, left - pad_px)
    top = max(0, top - pad_px)
    right = min(width - 1, right + pad_px)
    bottom = min(height - 1, bottom + pad_px)
    if right <= left or bottom <= top:
        return pil_image
    return pil_image.crop((left, top, right + 1, bottom + 1))


def _ink_fraction(pil_image):
    """Fraction of non-paper pixels — the blank-crop guard."""
    try:
        import numpy as np
    except ImportError:                          # pragma: no cover - env guard
        return 1.0
    arr = np.array(pil_image.convert("L"))
    if arr.size == 0:
        return 0.0
    return float((arr < 245).sum()) / float(arr.size)


def _render_page(pdf_page, dpi):
    """Rasterise one page with pypdfium2, returning a PIL image.

    pypdfium2 rather than pdf2image on purpose: pdf2image shells out to poppler,
    which is not installed on the service host and fails at the first call with
    PDFInfoNotInstalledError. pypdfium2 bundles PDFium in the wheel.
    """
    bitmap = pdf_page.render(scale=dpi / 72.0)
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        close = getattr(bitmap, "close", None)
        if callable(close):
            close()


def _to_png_data_uri(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _candidates_for_page(page, page_box):
    """Every candidate region on one page, captioned where possible."""
    word_boxes = _word_boxes(page, page_box)
    captions = _captions(page, page_box)
    prose_bands = _prose_bands(page, page_box)

    # Watermarks and background panels are dropped before the merge, for the
    # same reason page-sized images are: left in, one of them unions with every
    # real figure on the page and the whole page becomes a single candidate.
    images = [b for b in _image_regions(page, page_box) if not _spans_prose(b, prose_bands)]

    candidates = []
    candidates.extend(images)
    candidates.extend(_vector_clusters(
        _drawing_boxes(page, page_box, prose_bands), page_box, prose_bands,
    ))
    # Merged with no slack: the grid has already bridged the parts of one
    # figure, so any gap left here is a real separation between two of them.
    candidates = _merge_overlapping(candidates, gap_pt=0.0)

    # Attach each caption to the candidate directly above it. A caption whose
    # figure no detector saw becomes its own candidate instead.
    for caption in captions:
        cbox = caption["box"]
        best, best_gap = None, None
        for cand in candidates:
            if cand.bottom > cbox.top + _CAPTION_GAP_PT:
                continue                          # sits below the caption
            horizontal_overlap = min(cand.x1, cbox.x1) - max(cand.x0, cbox.x0)
            if horizontal_overlap <= 0:
                continue                          # different column
            gap = cbox.top - cand.bottom
            if gap > page_box.height * _CAPTION_LOOKUP_FRAC:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = cand, gap
        if best is not None:
            if not best.label:
                best.label = caption["label"]
                best.caption = caption["text"]
            continue
        # The caption's own words are not filtered out of the blockers: they sit
        # on the caption line, and _region_above_caption only considers words
        # that end strictly above it, so they exclude themselves.
        region = _region_above_caption(cbox, page_box, word_boxes)
        if region is not None:
            region.label = caption["label"]
            region.caption = caption["text"]
            candidates.append(region)

    return candidates, word_boxes, prose_bands


def extract_figures(pdf_bytes, dpi=DEFAULT_DPI, max_figures=_DEFAULT_MAX_FIGURES, page_numbers=None):
    """Find and crop the figures in a chapter PDF.

    Args:
        pdf_bytes:    the PDF, already downloaded (the caller owns the size
                      ceiling — see textbook.fetch_pdf).
        dpi:          raster resolution for crops.
        max_figures:  hard ceiling across the whole document.
        page_numbers: 1-based pages to restrict to, or None for all.

    Returns:
        {"figures": [...], "report": {...}} — never raises for a readable PDF.
        Each figure is
        {page_no, figure_index, label, caption, bbox, width, height,
         detector, image_base64}.

    Figures are returned, never persisted: this service does not own the school
    database or its object storage.
    """
    report = {
        "pages_scanned": 0,
        "candidates": 0,
        "figures": 0,
        "rejected": {},
        "skipped_pages": 0,
    }

    def reject(reason):
        report["rejected"][reason] = report["rejected"].get(reason, 0) + 1

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required to read a chapter PDF") from exc
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is required to rasterise chapter pages but is not installed"
        ) from exc

    wanted = set(page_numbers) if page_numbers else None
    figures = []
    hashes = {}

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for index, page in enumerate(pdf.pages):
                page_no = index + 1
                if wanted is not None and page_no not in wanted:
                    continue
                if len(figures) >= max_figures:
                    break
                report["pages_scanned"] += 1
                try:
                    page_figures = _figures_on_page(
                        page, document, index, page_no, dpi, reject,
                    )
                except Exception as exc:
                    # One unreadable page must never cost the whole chapter.
                    logger.warning("Figure extraction failed on page %d: %s", page_no, exc)
                    report["skipped_pages"] += 1
                    continue
                report["candidates"] += len(page_figures)
                for figure in page_figures:
                    if len(figures) >= max_figures:
                        break
                    hashes.setdefault(figure["_hash"], []).append(figure)
                    figures.append(figure)
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    # Repeated artwork: a logo or running header appears on many pages in the
    # same place. Dropped last, because it can only be recognised across pages.
    repeat_threshold = max(_REPEAT_MIN_PAGES, int(total_pages * _REPEAT_PAGE_FRAC))
    kept = []
    for figure in figures:
        group = hashes.get(figure["_hash"], [])
        distinct_pages = len({f["page_no"] for f in group})
        if distinct_pages >= _REPEAT_MIN_PAGES and distinct_pages >= repeat_threshold:
            reject("repeated_artwork")
            continue
        figure.pop("_hash", None)
        kept.append(figure)

    # figure_index is per page and must stay contiguous after the drops above,
    # because it is half of the (page_no, figure_index) identity the backend
    # stores and re-ingestion must line up with.
    per_page = {}
    for figure in kept:
        seq = per_page.get(figure["page_no"], 0)
        figure["figure_index"] = seq
        per_page[figure["page_no"]] = seq + 1

    report["figures"] = len(kept)
    logger.info(
        "Figure extraction: %d figures from %d pages (candidates=%d rejected=%s skipped=%d)",
        len(kept), report["pages_scanned"], report["candidates"],
        report["rejected"], report["skipped_pages"],
    )
    return {"figures": kept, "report": report}


def _figures_on_page(page, document, page_index, page_no, dpi, reject):
    """Detect, gate, render and crop every figure on one page."""
    # The render frame, not page.bbox — see _render_frame. Using it for both
    # clamping and pixel mapping keeps detection and cropping in one space, and
    # clips away margin content that is not rendered at all.
    page_box = _render_frame(page)
    if page_box.width <= 0 or page_box.height <= 0:
        reject("bad_page_box")
        return []

    candidates, word_boxes, prose_bands = _candidates_for_page(page, page_box)
    if not candidates:
        return []

    # Geometry gates first: they are free, and they usually remove most of the
    # candidates before a single page is rasterised.
    survivors = []
    for box in candidates:
        # A caption-anchored region is already a full-width band bounded by the
        # surrounding text, so growing it toward nearby words would only pull in
        # the paragraph it was bounded away from.
        if box.detector != "caption":
            box = _attach_labels(box, word_boxes, page_box)
        ok, reason = _passes_geometry_gates(box, page_box)
        if not ok:
            reject(reason)
            continue
        if _text_coverage(box, word_boxes) > _MAX_TEXT_COVERAGE:
            reject("text_block")
            continue
        # Last line of defence, whichever detector or merge produced the region:
        # a figure does not have paragraphs of running text inside it.
        if _spans_prose(box, prose_bands, _FIGURE_MAX_PROSE_LINES):
            reject("spans_prose")
            continue
        survivors.append(box)
    if not survivors:
        return []

    # Biggest first, so the per-page ceiling keeps the figures that matter.
    survivors.sort(key=lambda b: b.area, reverse=True)
    survivors = survivors[:_MAX_FIGURES_PER_PAGE]

    pdf_page = document[page_index]
    try:
        rendered = _render_page(pdf_page, dpi)
    finally:
        close = getattr(pdf_page, "close", None)
        if callable(close):
            close()

    # Map PDF points to pixels from the render itself rather than from the
    # nominal DPI. If the two disagree the page is rotated or has an offset
    # MediaBox, and every crop from it would be silently misplaced — so the
    # page is dropped rather than guessed at.
    scale_x = rendered.width / page_box.width
    scale_y = rendered.height / page_box.height
    if scale_x <= 0 or scale_y <= 0:
        reject("bad_render")
        return []
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > _GEOMETRY_TOLERANCE:
        logger.warning(
            "Page %d render geometry mismatch (sx=%.4f sy=%.4f) - skipping its figures",
            page_no, scale_x, scale_y,
        )
        reject("geometry_mismatch")
        return []

    out = []
    for box in survivors:
        left = int(max(0, math.floor((box.x0 - page_box.x0 - _CROP_PAD_PT) * scale_x)))
        top = int(max(0, math.floor((box.top - page_box.top - _CROP_PAD_PT) * scale_y)))
        right = int(min(rendered.width, math.ceil((box.x1 - page_box.x0 + _CROP_PAD_PT) * scale_x)))
        bottom = int(min(rendered.height, math.ceil((box.bottom - page_box.top + _CROP_PAD_PT) * scale_y)))
        if right - left < _MIN_PX or bottom - top < _MIN_PX:
            reject("too_small_px")
            continue

        crop = _trim_whitespace(rendered.crop((left, top, right, bottom)))
        if crop.width < _MIN_PX or crop.height < _MIN_PX:
            reject("too_small_px")
            continue
        if _ink_fraction(crop) < _MIN_INK_FRAC:
            reject("blank")
            continue

        out.append({
            "page_no": page_no,
            "figure_index": 0,                   # assigned after cross-page drops
            "label": box.label or "",
            "caption": box.caption or "",
            "bbox": box.as_list(),
            "width": crop.width,
            "height": crop.height,
            "detector": box.detector,
            "image_base64": _to_png_data_uri(crop),
            "_hash": _phash(crop),
        })
    return out
