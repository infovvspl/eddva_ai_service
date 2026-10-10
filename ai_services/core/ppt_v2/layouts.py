"""The layout registry.

A layout is a named function from *how much content there is* to *where it
goes*. It returns normalised rectangles keyed by slot name, so the renderer
never computes geometry and the planner never names a coordinate.

Why a registry rather than one flexible layout: V1 has exactly one arrangement
(title, bullets, picture on the right) and that is the single biggest reason
its decks read as filler. Ten arrangements chosen by slide type give a deck a
shape - a process looks like a process, a comparison looks like a comparison -
without anyone hand-placing a box.

Slot names are a contract. A renderer looks up the slots it understands and
ignores the rest, so a new slot is always backwards-compatible:

    title, subtitle, footer      every layout
    content                      a single stack of blocks
    block_0 .. block_n           individually placed blocks
    visual                       the image / diagram
    connector_0 .. connector_n   arrow or line between two blocks
    divider, axis                a rule the renderer draws
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ai_services.core.ppt_v2.schema import Region, SlideType

# The standard frame. Everything below is expressed against these so that
# changing the margin moves every layout together.
MX = 0.055          # left / right margin
TOP = 0.070         # top margin
GUT = 0.030         # gap between columns
FULL_W = 1.0 - 2 * MX

TITLE_BAND = Region(MX, TOP, FULL_W, 0.125)
SUBTITLE_BAND = Region(MX, 0.202, FULL_W, 0.055)
FOOTER_BAND = Region(MX, 0.918, FULL_W, 0.042)

BODY_TOP = 0.285
BODY_BOTTOM = 0.900
BODY_H = BODY_BOTTOM - BODY_TOP


def _frame(include_subtitle: bool = True) -> Dict[str, Region]:
    out = {"title": TITLE_BAND, "footer": FOOTER_BAND}
    if include_subtitle:
        out["subtitle"] = SUBTITLE_BAND
    return out


def _row(n: int, x: float, y: float, w: float, h: float,
         gap: float = 0.022) -> List[Region]:
    """Split a band into n equal columns."""
    if n <= 0:
        return []
    cw = (w - gap * (n - 1)) / n
    return [Region(x + i * (cw + gap), y, cw, h) for i in range(n)]


def _column(n: int, x: float, y: float, w: float, h: float,
            gap: float = 0.022) -> List[Region]:
    """Split a band into n equal rows."""
    if n <= 0:
        return []
    ch = (h - gap * (n - 1)) / n
    return [Region(x, y + i * (ch + gap), w, ch) for i in range(n)]


def _connectors(boxes: Sequence[Region], horizontal: bool = True,
                thickness: float = 0.05) -> Dict[str, Region]:
    """The gaps between consecutive boxes, for arrows."""
    out: Dict[str, Region] = {}
    for i in range(len(boxes) - 1):
        a, b = boxes[i], boxes[i + 1]
        if horizontal:
            x = a.x + a.w
            w = max(b.x - x, 0.004)
            out["connector_%d" % i] = Region(x, a.y + a.h / 2 - thickness / 2,
                                             w, thickness)
        else:
            y = a.y + a.h
            h = max(b.y - y, 0.004)
            out["connector_%d" % i] = Region(a.x + a.w / 2 - thickness / 2, y,
                                             thickness, h)
    return out


def _numbered(boxes: Sequence[Region], prefix: str = "block") -> Dict[str, Region]:
    return {"%s_%d" % (prefix, i): r for i, r in enumerate(boxes)}


# --- the layouts ------------------------------------------------------------
# Each takes (block_count, has_visual) and returns the slot map.

def _title_hero(n: int, has_visual: bool) -> Dict[str, Region]:
    """Cover slide: the title owns the slide, the visual is a side panel."""
    if has_visual:
        return {
            "title": Region(MX, 0.300, 0.50, 0.200),
            "subtitle": Region(MX, 0.520, 0.50, 0.110),
            "footer": Region(MX, 0.860, 0.50, 0.050),
            "visual": Region(0.560, 0.0, 0.440, 1.0),   # full-bleed right
        }
    return {
        "title": Region(MX, 0.330, FULL_W, 0.200),
        "subtitle": Region(MX, 0.550, FULL_W, 0.110),
        "footer": Region(MX, 0.860, FULL_W, 0.050),
    }


def _concept_hero_split(n: int, has_visual: bool) -> Dict[str, Region]:
    """The workhorse: points on the left, one strong image on the right."""
    out = _frame()
    if not has_visual:
        out["content"] = Region(MX, BODY_TOP, FULL_W, BODY_H)
        return out
    content_w = 0.44
    out["content"] = Region(MX, BODY_TOP, content_w, BODY_H)
    out.update(_numbered(_column(max(n, 1), MX, BODY_TOP, content_w, BODY_H)))
    out["visual"] = Region(MX + content_w + GUT, BODY_TOP,
                           FULL_W - content_w - GUT, BODY_H)
    return out


def _process_steps(n: int, has_visual: bool) -> Dict[str, Region]:
    """Ordered stages left to right, with arrows in the gaps.

    Beyond four steps a horizontal row makes each box too narrow to read, so
    the arrangement flips to a vertical stack.
    """
    out = _frame()
    n = max(n, 2)
    top, height = BODY_TOP, BODY_H
    if has_visual:
        out["visual"] = Region(MX, BODY_TOP, FULL_W, 0.300)
        top, height = BODY_TOP + 0.330, BODY_H - 0.330

    if n <= 4:
        boxes = _row(n, MX, top, FULL_W, height)
        out.update(_numbered(boxes))
        out.update(_connectors(boxes, horizontal=True))
    else:
        boxes = _column(n, MX, top, FULL_W, height, gap=0.014)
        out.update(_numbered(boxes))
        out.update(_connectors(boxes, horizontal=False, thickness=0.03))
    return out


def _comparison(n: int, has_visual: bool) -> Dict[str, Region]:
    """Two sides, a rule down the middle."""
    out = _frame()
    top, height = BODY_TOP, BODY_H
    if has_visual:
        out["visual"] = Region(MX, BODY_TOP, FULL_W, 0.260)
        top, height = BODY_TOP + 0.290, BODY_H - 0.290
    col_w = (FULL_W - GUT) / 2
    out["block_0"] = Region(MX, top, col_w, height)
    out["block_1"] = Region(MX + col_w + GUT, top, col_w, height)
    out["divider"] = Region(MX + col_w + GUT / 2 - 0.0015, top, 0.003, height)
    return out


def _cause_effect(n: int, has_visual: bool) -> Dict[str, Region]:
    """Cause on the left, effect on the right, an arrow between them."""
    out = _frame()
    top, height = BODY_TOP, BODY_H
    if has_visual:
        out["visual"] = Region(MX, 0.640, FULL_W, 0.260)
        top, height = BODY_TOP, 0.320
    arrow_w = 0.075
    box_w = (FULL_W - arrow_w - 2 * GUT) / 2
    out["block_0"] = Region(MX, top, box_w, height)
    out["connector_0"] = Region(MX + box_w + GUT, top + height / 2 - 0.035,
                                arrow_w, 0.070)
    out["block_1"] = Region(MX + box_w + GUT + arrow_w + GUT, top, box_w, height)
    return out


def _anatomy_labeled(n: int, has_visual: bool) -> Dict[str, Region]:
    """One big diagram with label gutters either side.

    This is the layout a labelled biology or physics figure wants: the image
    is the content, and the text exists only to point at parts of it.
    """
    out = _frame()
    if not has_visual:
        out["content"] = Region(MX, BODY_TOP, FULL_W, BODY_H)
        return out
    gutter_w = 0.195
    out["visual"] = Region(MX + gutter_w + GUT, BODY_TOP,
                           FULL_W - 2 * (gutter_w + GUT), BODY_H)
    left = _column(max((n + 1) // 2, 1), MX, BODY_TOP, gutter_w, BODY_H)
    right = _column(max(n // 2, 1), 1.0 - MX - gutter_w, BODY_TOP, gutter_w, BODY_H)
    for i, r in enumerate(left):
        out["label_l_%d" % i] = r
    for i, r in enumerate(right):
        out["label_r_%d" % i] = r
    return out


def _magnification(n: int, has_visual: bool) -> Dict[str, Region]:
    """Whole on the left, zoomed detail on the right - the "and inside it" slide."""
    out = _frame()
    if not has_visual:
        return _concept_hero_split(n, has_visual)
    out["visual"] = Region(MX, BODY_TOP, 0.50, BODY_H)
    out["visual_detail"] = Region(MX + 0.50 + GUT, BODY_TOP,
                                  FULL_W - 0.50 - GUT, 0.380)
    out["connector_0"] = Region(MX + 0.50, BODY_TOP + 0.150, GUT, 0.080)
    out["content"] = Region(MX + 0.50 + GUT, BODY_TOP + 0.410,
                            FULL_W - 0.50 - GUT, BODY_H - 0.410)
    return out


def _timeline(n: int, has_visual: bool) -> Dict[str, Region]:
    """Events pinned along a horizontal axis."""
    out = _frame()
    n = max(n, 2)
    axis_y = 0.560 if not has_visual else 0.760
    if has_visual:
        # Must clear the event cards below it: they start at
        # axis_y - card_h - 0.030 = 0.620.
        out["visual"] = Region(MX, BODY_TOP, FULL_W, 0.300)
    out["axis"] = Region(MX, axis_y, FULL_W, 0.006)
    card_h = 0.200 if not has_visual else 0.110
    boxes = _row(n, MX, axis_y - card_h - 0.030, FULL_W, card_h, gap=0.018)
    out.update(_numbered(boxes))
    labels = _row(n, MX, axis_y + 0.030, FULL_W, 0.080, gap=0.018)
    for i, r in enumerate(labels):
        out["label_%d" % i] = r
    return out


def _formula_equation(n: int, has_visual: bool) -> Dict[str, Region]:
    """The equation gets the stage; the terms are explained underneath."""
    out = _frame()
    out["equation"] = Region(MX, BODY_TOP, FULL_W, 0.180)
    rest_top = BODY_TOP + 0.210
    rest_h = BODY_BOTTOM - rest_top
    if has_visual:
        out["visual"] = Region(1.0 - MX - 0.330, rest_top, 0.330, rest_h)
        terms_w = FULL_W - 0.330 - GUT
    else:
        terms_w = FULL_W
    out.update(_numbered(_column(max(n, 1), MX, rest_top, terms_w, rest_h,
                                 gap=0.016)))
    out["content"] = Region(MX, rest_top, terms_w, rest_h)
    return out


def _summary_grid(n: int, has_visual: bool) -> Dict[str, Region]:
    """Takeaways as cards. Two columns, because three stops being scannable."""
    out = _frame()
    top, height = BODY_TOP, BODY_H
    if has_visual:
        out["visual"] = Region(1.0 - MX - 0.280, BODY_TOP, 0.280, BODY_H)
        width = FULL_W - 0.280 - GUT
    else:
        width = FULL_W
    n = max(n, 1)
    cols = 1 if n <= 2 else 2
    rows = (n + cols - 1) // cols
    col_w = (width - GUT * (cols - 1)) / cols
    row_h = (height - 0.020 * (rows - 1)) / rows
    for i in range(n):
        r, c = divmod(i, cols)
        out["block_%d" % i] = Region(MX + c * (col_w + GUT),
                                     top + r * (row_h + 0.020), col_w, row_h)
    return out


def _quiz(n: int, has_visual: bool) -> Dict[str, Region]:
    """A question and its options, stacked."""
    out = _frame(include_subtitle=False)
    q_h = 0.150
    out["question"] = Region(MX, 0.215, FULL_W, q_h)
    top = 0.215 + q_h + 0.030
    height = BODY_BOTTOM - top
    width = FULL_W
    if has_visual:
        out["visual"] = Region(1.0 - MX - 0.300, top, 0.300, height)
        width = FULL_W - 0.300 - GUT
    out.update(_numbered(_column(max(n, 1), MX, top, width, height, gap=0.016)))
    return out


@dataclass(frozen=True)
class Layout:
    """One arrangement, plus the rules for when it applies."""

    name: str
    build: Callable[[int, bool], Dict[str, Region]]
    serves: Tuple[SlideType, ...]
    min_blocks: int = 0
    max_blocks: int = 6
    wants_visual: bool = True
    flow: str = "ltr"
    density: str = "normal"

    def fits(self, slide_type: SlideType, block_count: int) -> bool:
        return (slide_type in self.serves
                and self.min_blocks <= block_count <= self.max_blocks)

    def regions(self, block_count: int, has_visual: bool) -> Dict[str, Region]:
        return self.build(block_count, has_visual)


LAYOUTS: Dict[str, Layout] = {
    layout.name: layout for layout in (
        Layout("title_hero", _title_hero, (SlideType.TITLE,),
               min_blocks=0, max_blocks=2, density="sparse"),
        Layout("concept_hero_split", _concept_hero_split,
               (SlideType.CONCEPT, SlideType.EXAMPLE, SlideType.DATA),
               min_blocks=1, max_blocks=5),
        Layout("process_steps", _process_steps,
               (SlideType.PROCESS,), min_blocks=2, max_blocks=6, flow="ltr"),
        Layout("comparison", _comparison,
               (SlideType.COMPARISON,), min_blocks=2, max_blocks=2),
        Layout("cause_effect", _cause_effect,
               (SlideType.CAUSE_EFFECT,), min_blocks=2, max_blocks=2),
        Layout("anatomy_labeled", _anatomy_labeled,
               (SlideType.ANATOMY, SlideType.CONCEPT),
               min_blocks=2, max_blocks=6, density="sparse"),
        Layout("magnification", _magnification,
               (SlideType.ANATOMY, SlideType.CONCEPT),
               min_blocks=1, max_blocks=4),
        Layout("timeline", _timeline,
               (SlideType.TIMELINE,), min_blocks=2, max_blocks=5),
        Layout("formula_equation", _formula_equation,
               (SlideType.FORMULA,), min_blocks=0, max_blocks=5),
        Layout("summary_grid", _summary_grid,
               (SlideType.SUMMARY,), min_blocks=1, max_blocks=6),
        Layout("quiz", _quiz, (SlideType.QUIZ,), min_blocks=2, max_blocks=5),
    )
}

# Used when nothing else fits. Chosen because it degrades to a plain
# title-and-bullets slide when there is no visual, which is always drawable.
FALLBACK_LAYOUT = "concept_hero_split"


def select_layout(slide_type: SlideType, block_count: int,
                  has_visual: bool = True,
                  avoid: Optional[Sequence[str]] = None) -> Layout:
    """Pick a layout for one slide.

    ``avoid`` carries the layouts used on the preceding slides. When more than
    one candidate fits, a name not in ``avoid`` wins - that is the whole
    mechanism that stops a V2 deck collapsing back into V1's one-shape deck.
    Correctness beats variety: if only one layout fits, it is used regardless.
    """
    avoid = list(avoid or [])
    candidates = [lay for lay in LAYOUTS.values() if lay.fits(slide_type, block_count)]
    if not candidates:
        # Block count out of range for every layout that serves this type:
        # keep the type's own layout and let the composer trim the overflow.
        by_type = [lay for lay in LAYOUTS.values() if slide_type in lay.serves]
        candidates = by_type or [LAYOUTS[FALLBACK_LAYOUT]]

    if has_visual:
        with_visual = [lay for lay in candidates if lay.wants_visual]
        candidates = with_visual or candidates

    fresh = [lay for lay in candidates if lay.name not in avoid]
    pool = fresh or candidates
    # Deterministic: same inputs must give the same deck, so no randomness.
    return pool[0]
