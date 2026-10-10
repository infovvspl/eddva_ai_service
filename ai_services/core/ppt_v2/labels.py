"""Placing labels on a figure without them landing on each other.

Phase 7 rasterised a Cell Structure slide whose labels read
"VacuoCldoroplast" - two labels drawn on top of one another - with
"Chloroplast" appearing twice. The labels came out of free-form matplotlib
code, which has no idea where it has already put text.

Rather than ask a model to be careful, placement is done here, deterministically,
before either renderer draws anything:

    1. anatomy geometry   the picture, and the region it occupies
    2. label candidates   deduplicated, in reading order
    3. label placement    four candidate positions per anchor
    4. collision detection against placed labels and the region bounds
    5. leader rendering    each label keeps the anchor it points at

Every label gets a bounding box in normalised 0..1 coordinates of the
*picture*, so both the PPTX renderer and the browser preview draw the same
arrangement and neither has to solve the problem again. When no position is
free, the label shrinks within safe bounds, and if it still does not fit it
is dropped rather than shipped overlapping - a figure with four clear labels
teaches more than one with six illegible ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Normalised against the picture, not the slide.
DEFAULT_W = 0.30
DEFAULT_H = 0.11
GAP = 0.05                 # between the anchor and its label
MIN_SCALE = 0.72           # a label may shrink to this before being dropped
EDGE = 0.005               # keep this much clear of the picture border

# Order matters: the first position that is free wins, so this is a
# preference. Right and left first because a wide figure has more room
# horizontally, and because a label beside the anchor reads better than one
# stacked above it.
SIDES: Tuple[str, ...] = ("right", "left", "top", "bottom")


def normalise(text: str) -> str:
    """The comparison key for a label.

    "Chloroplast", "chloroplast" and " Chloroplast " are one label. Phase 7
    drew all three.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def dedupe(items: Iterable, key=None) -> List:
    """Keep the first of each semantically identical label."""
    seen = set()
    out = []
    for item in items:
        text = key(item) if key else item
        k = normalise(text)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def overlaps(self, other: "Box", pad: float = 0.004) -> bool:
        dx = min(self.right, other.right) - max(self.x, other.x)
        dy = min(self.bottom, other.bottom) - max(self.y, other.y)
        return dx > pad and dy > pad

    def inside(self, pad: float = EDGE) -> bool:
        return (self.x >= -1e-9 + pad and self.y >= -1e-9 + pad
                and self.right <= 1.0 - pad + 1e-9
                and self.bottom <= 1.0 - pad + 1e-9)

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "w": round(self.w, 4), "h": round(self.h, 4)}


@dataclass
class Placed:
    text: str
    anchor: Tuple[float, float]
    box: Box
    side: str
    scale: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {"text": self.text,
                "anchor": [round(self.anchor[0], 4), round(self.anchor[1], 4)],
                "box": self.box.to_dict(), "side": self.side,
                "scale": round(self.scale, 3)}


def _candidate(anchor: Tuple[float, float], side: str,
               w: float, h: float) -> Box:
    ax, ay = anchor
    if side == "right":
        return Box(ax + GAP, ay - h / 2, w, h)
    if side == "left":
        return Box(ax - GAP - w, ay - h / 2, w, h)
    if side == "top":
        return Box(ax - w / 2, ay - GAP - h, w, h)
    return Box(ax - w / 2, ay + GAP, w, h)


def _clamp(box: Box) -> Box:
    """Pull a box back inside the picture without resizing it."""
    w = min(box.w, 1.0 - 2 * EDGE)
    h = min(box.h, 1.0 - 2 * EDGE)
    return Box(min(max(box.x, EDGE), 1.0 - EDGE - w),
               min(max(box.y, EDGE), 1.0 - EDGE - h), w, h)


def place(items: Sequence[dict], width: float = DEFAULT_W,
          height: float = DEFAULT_H,
          preferred_sides: Optional[Sequence[str]] = None) -> Tuple[List[Placed], List[str]]:
    """Place labels so none overlaps another. Returns (placed, dropped).

    `items` are dicts with `text` and `anchor`; a `side` is honoured as a
    first preference. Earlier items win a contested position, so the planner's
    order is the priority order.
    """
    placed: List[Placed] = []
    dropped: List[str] = []

    for item in dedupe(items, key=lambda i: (i or {}).get("text", "")):
        text = str((item or {}).get("text") or "").strip()
        if not text:
            continue
        anchor = (item or {}).get("anchor") or (0.5, 0.5)
        try:
            ax = min(max(float(anchor[0]), 0.0), 1.0)
            ay = min(max(float(anchor[1]), 0.0), 1.0)
        except (TypeError, ValueError, IndexError):
            ax, ay = 0.5, 0.5

        wanted = str((item or {}).get("side") or "").strip().lower()
        order = ([wanted] if wanted in SIDES else []) + [s for s in SIDES if s != wanted]
        # Prefer the side with more room: a label on the left of a left-hand
        # anchor runs off the picture.
        if not wanted:
            order = (["right", "top", "bottom", "left"] if ax < 0.5
                     else ["left", "top", "bottom", "right"])
        if preferred_sides:
            order = [s for s in preferred_sides if s in SIDES] + \
                    [s for s in order if s not in preferred_sides]

        chosen: Optional[Placed] = None
        # Try every side at full size, then shrink and try again. Shrinking
        # late keeps labels a consistent size whenever that is possible.
        for scale in (1.0, 0.88, MIN_SCALE):
            w, h = width * scale, height * scale
            for side in order:
                box = _candidate((ax, ay), side, w, h)
                if not box.inside():
                    box = _clamp(box)
                if not box.inside():
                    continue
                if any(box.overlaps(p.box) for p in placed):
                    continue
                chosen = Placed(text=text, anchor=(ax, ay), box=box,
                                side=side, scale=scale)
                break
            if chosen:
                break

        if chosen:
            placed.append(chosen)
        else:
            # Nowhere clear at any permitted size. A dropped label is better
            # than one drawn across its neighbour.
            dropped.append(text)

    return placed, dropped


def problems(placed: Sequence[Placed]) -> List[Tuple[str, str]]:
    """Deterministic checks on a finished arrangement: (code, detail)."""
    out: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}

    for p in placed:
        key = normalise(p.text)
        if key in seen:
            out.append(("duplicate_label",
                        "%r appears more than once" % p.text))
        seen[key] = p.text

        if p.box.w <= 0 or p.box.h <= 0:
            out.append(("zero_size_label", "%r has no area" % p.text))
        if not p.box.inside():
            out.append(("label_out_of_bounds",
                        "%r is outside the picture" % p.text))

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if a.box.overlaps(b.box):
                out.append(("label_overlap",
                            "%r overlaps %r" % (a.text, b.text)))
    return out


# --- label policy ---------------------------------------------------------------
#
# How a slide's required labels reach the final slide depends on whether we
# know where things are in the picture, and that depends on who made it:
#
#   anchored  chips with leader arrows to planned anchors. Only for a picture
#             composed to the plan - a generated image prompted with the scene.
#   legend    the same required terms as a key along the bottom of the picture,
#             NO leaders. For a picture nobody composed to our anchors (a
#             searched photo, a cached asset): an arrow to the planner's
#             imagined coordinates would point at the wrong thing, which is
#             worse than no arrow. The terms are present; the pointers are not
#             invented.
#   native    the asset labels itself (a drawn diagram told which labels to
#             draw, an icon row). Overlaying would print every label twice.
#   none      no labels - acceptable only when none are required.

ANCHORED, LEGEND, NATIVE, NO_LABELS = "anchored", "legend", "native", "none"
DRAWN_BY_RENDERER = frozenset({ANCHORED, LEGEND})

LEGEND_COLS = 2
LEGEND_GAP = 0.015


def legend_boxes(n: int) -> List[Tuple[float, float, float, float]]:
    """Normalised boxes for an n-term key along the bottom of the picture."""
    if n <= 0:
        return []
    cols = 1 if n == 1 else LEGEND_COLS
    rows = (n + cols - 1) // cols
    w = (1.0 - 2 * EDGE - (cols - 1) * LEGEND_GAP) / cols
    h = DEFAULT_H
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = EDGE + c * (w + LEGEND_GAP)
        y = 1.0 - EDGE - (rows - r) * h - (rows - r - 1) * LEGEND_GAP
        out.append((round(x, 4), round(y, 4), round(w, 4), round(h, 4)))
    return out


def apply_policy(visual) -> None:
    """Shape a resolved visual's annotations to its label policy. In place.

    Only LEGEND changes anything: the planner's anchored boxes become a key
    with no leaders. Every other policy leaves the annotations as composed -
    kept on the spec as the record of what the slide had to teach, and drawn
    or not by the renderer according to the policy.
    """
    if getattr(visual, "label_policy", "") != LEGEND or not visual.annotations:
        return
    for annotation, box in zip(visual.annotations,
                               legend_boxes(len(visual.annotations))):
        annotation.box = box
        annotation.leader = False
        annotation.scale = 1.0
