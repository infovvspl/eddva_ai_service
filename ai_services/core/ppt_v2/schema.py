"""The V2 slide specification.

This is the contract between the planner (what to teach, what to show) and the
renderer (where to draw it). It is deliberately renderer-agnostic:

  * geometry is normalised - every rectangle is (x, y, w, h) in 0..1 of the
    slide, so a 13.33x7.5in deck and a 1920x1080 canvas read the same spec;
  * colour is a token name ("primary", "accent", "ink_muted"), resolved against
    a theme at render time, never a hex literal on the slide;
  * no field names a drawing API. "emphasis" is a role, not a bold run.

Everything serialises to plain JSON via :func:`slide_to_dict`, because the
renderer that consumes it today is JavaScript in the browser.

Back-compatibility: :meth:`SlideSpec.to_v1_dict` emits the V1 keys
(``title``/``bullets``/``imageUrl``/...) alongside the V2 ``v2`` block, so a
renderer that has not learned V2 still produces a usable - if plain - slide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bump when a change would make an older renderer mis-draw a newer spec.
SPEC_VERSION = "2.0"

# A slide holding more than this reads as a document, not a visual aid.
MAX_CONTENT_BLOCKS = 6
MAX_ANNOTATIONS = 8
MAX_HIGHLIGHTED_TERMS = 6


class SlideType(str, Enum):
    """What job the slide does in the deck. Drives default layout + visual."""

    TITLE = "title"
    CONCEPT = "concept"
    PROCESS = "process"
    COMPARISON = "comparison"
    CAUSE_EFFECT = "cause_effect"
    ANATOMY = "anatomy"
    DATA = "data"
    TIMELINE = "timeline"
    FORMULA = "formula"
    EXAMPLE = "example"
    SUMMARY = "summary"
    QUIZ = "quiz"


class BlockKind(str, Enum):
    """Shape of one piece of slide content."""

    BULLET = "bullet"            # a single teaching point
    STEP = "step"                # ordered stage of a process
    DEFINITION = "definition"    # term + meaning
    STAT = "stat"                # a number worth its own emphasis
    EQUATION = "equation"        # formula / chemical equation
    CALLOUT = "callout"          # "remember this" box
    COMPARE_LEFT = "compare_left"
    COMPARE_RIGHT = "compare_right"
    QUESTION = "question"
    ANSWER = "answer"


class VisualStrategy(str, Enum):
    """How the slide's visual gets made.

    Ordered loosely by fidelity-for-this-purpose, not by cost. The decision
    engine in :mod:`ai_services.core.ppt_v2.visuals` picks one and records the
    fallbacks it is allowed to degrade to.
    """

    PROGRAMMATIC_DIAGRAM = "programmatic_diagram"  # core.diagram, matplotlib
    AI_GENERATED_IMAGE = "ai_generated_image"      # core.image_generation, FLUX
    SEARCHED_IMAGE = "searched_image"              # existing V1 image search
    CACHED_ASSET = "cached_asset"                  # previously generated, reused
    ICON_COMPOSITION = "icon_composition"          # shapes + glyphs, always works
    NONE = "none"                                  # typographic slide, by choice


class Emphasis(str, Enum):
    NORMAL = "normal"
    STRONG = "strong"
    KEY_TERM = "key_term"


@dataclass(frozen=True)
class Region:
    """A normalised rectangle on the slide: 0..1 in both axes."""

    x: float
    y: float
    w: float
    h: float

    def errors(self, label: str) -> List[str]:
        out: List[str] = []
        for name, value in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            if not isinstance(value, (int, float)):
                out.append(f"{label}.{name} must be a number")
            elif not (0.0 <= float(value) <= 1.0):
                out.append(f"{label}.{name}={value} is outside 0..1")
        if not out:
            if self.w <= 0 or self.h <= 0:
                out.append(f"{label} has zero or negative area")
            if self.x + self.w > 1.0001 or self.y + self.h > 1.0001:
                out.append(f"{label} extends past the slide edge")
        return out

    def overlaps(self, other: "Region", tolerance: float = 0.01) -> bool:
        """True when the two rectangles share more than a hairline of space."""
        dx = min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        dy = min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        return dx > tolerance and dy > tolerance

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "w": round(self.w, 4), "h": round(self.h, 4)}


@dataclass
class TitleSpec:
    """Slide title, with the part worth emphasising called out separately.

    The renderer decides how to show emphasis (weight, colour, underline). The
    planner only says which words carry the idea.
    """

    text: str
    emphasis_text: str = ""
    emphasis: Emphasis = Emphasis.STRONG

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "emphasisText": self.emphasis_text,
                "emphasis": self.emphasis.value}


@dataclass
class ContentBlock:
    """One teaching point.

    ``text`` is the sentence a learner reads. ``label`` is the short handle a
    layout may show instead when space is tight (a step number's caption, a
    comparison column header). Keeping both means a dense layout can degrade
    without the planner re-running.
    """

    kind: BlockKind
    text: str
    label: str = ""
    detail: str = ""
    icon: str = ""          # semantic icon name, e.g. "sun", "water", "leaf"
    order: int = 0
    emphasis: Emphasis = Emphasis.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "text": self.text, "label": self.label,
                "detail": self.detail, "icon": self.icon, "order": self.order,
                "emphasis": self.emphasis.value}


@dataclass
class Annotation:
    """A label pinned to a point on the visual (the arrows into a diagram).

    ``anchor`` is normalised against the *visual's* region, not the slide, so a
    layout can move the image without the planner recomputing anchors.
    """

    text: str
    anchor: Tuple[float, float] = (0.5, 0.5)
    leader: bool = True      # draw a line from label to anchor
    side: str = "auto"       # auto | left | right | top | bottom
    # Where the label actually goes, as a normalised box inside the PICTURE,
    # computed once by core.ppt_v2.labels. Both renderers draw what they are
    # given rather than each solving collision avoidance again - and a
    # rasterised slide showed what happens when nobody solves it: two labels
    # printed on top of each other as "VacuoCldoroplast".
    box: Optional[Tuple[float, float, float, float]] = None
    scale: float = 1.0

    def errors(self, label: str) -> List[str]:
        out: List[str] = []
        if not (self.text or "").strip():
            out.append(f"{label} has no text")
        try:
            ax, ay = float(self.anchor[0]), float(self.anchor[1])
        except (TypeError, ValueError, IndexError):
            return out + [f"{label}.anchor must be a (x, y) pair"]
        if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0):
            out.append(f"{label}.anchor {self.anchor} is outside 0..1")
        if self.side not in ("auto", "left", "right", "top", "bottom"):
            out.append(f"{label}.side={self.side!r} is not a known side")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text,
                "anchor": [round(float(self.anchor[0]), 4), round(float(self.anchor[1]), 4)],
                "leader": self.leader, "side": self.side,
                "box": ([round(v, 4) for v in self.box] if self.box else None),
                "scale": round(self.scale, 3)}


@dataclass
class VisualSpec:
    """What the learner should see, and how it will be obtained.

    The planner fills ``description`` and ``must_show``; the decision engine
    fills ``strategy``, ``fallbacks`` and the prompt; the asset step fills
    ``url``/``base64``/``source``. A spec whose strategy resolved to NONE is
    valid - some slides are better typographic.
    """

    description: str = ""
    must_show: List[str] = field(default_factory=list)
    # Concepts that would make the picture wrong for this slide. Consumed by
    # core.ppt_v2.relevance when scoring search candidates; never rendered.
    avoid_concepts: List[str] = field(default_factory=list)
    # What the picture must SHOW happening, not merely contain. This is the
    # difference between topic relevance and teaching relevance: "a mirror"
    # is on topic, "an incident ray striking a mirror and a reflected ray
    # leaving it" is the lesson. Consumed by the image prompt and the vision
    # contract; never rendered.
    relationships: List[str] = field(default_factory=list)
    # The slide's learning objective, copied here so the asset producers can
    # build a teaching contract without reaching back to the slide. Assets
    # resolve concurrently, so anything per-slide has to travel with the
    # spec rather than live on the shared producer set.
    objective: str = ""
    strategy: VisualStrategy = VisualStrategy.NONE
    fallbacks: List[VisualStrategy] = field(default_factory=list)

    # filled by whichever strategy actually produced the asset
    url: str = ""
    base64: str = ""
    source: str = ""              # provider / page the asset came from
    resolved_strategy: Optional[VisualStrategy] = None
    attribution: str = ""

    # inputs the strategies need
    image_prompt: str = ""        # for AI_GENERATED_IMAGE
    search_terms: List[str] = field(default_factory=list)  # for SEARCHED_IMAGE
    diagram_spec: str = ""        # for PROGRAMMATIC_DIAGRAM
    # Size in inches of the region this visual will occupy. A drawn diagram
    # is rendered at exactly this size so its type scale survives placement;
    # before this it was drawn at matplotlib's default and shrunk ~0.35x,
    # which put every label below 4pt on the slide.
    region_in: Tuple[float, float] = (0.0, 0.0)
    icons: List[str] = field(default_factory=list)         # for ICON_COMPOSITION

    annotations: List[Annotation] = field(default_factory=list)
    alt_text: str = ""

    # The semantic label contract. `labels_required` is set from the content
    # type and the plan: this picture's named parts must be readable on the
    # final slide. `label_policy` is set by whichever producer actually
    # supplied the asset, and tells the renderer how the labels get there:
    #   overlay  the renderer draws the annotations as real text on top
    #   native   the asset carries its own labels; overlaying would double them
    #   none     no labels (allowed only when none are required)
    # Before this, the renderer decided from the provider name alone, so a
    # diagram fallback that had been told "the renderer will label you" was
    # never labelled by anyone.
    labels_required: bool = False
    label_policy: str = ""

    @property
    def has_asset(self) -> bool:
        return bool(self.url or self.base64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "mustShow": list(self.must_show),
            "avoidConcepts": list(self.avoid_concepts),
            "relationships": list(self.relationships),
            "strategy": self.strategy.value,
            "fallbacks": [s.value for s in self.fallbacks],
            "resolvedStrategy": self.resolved_strategy.value if self.resolved_strategy else "",
            "url": self.url,
            "base64": self.base64,
            "source": self.source,
            "attribution": self.attribution,
            "imagePrompt": self.image_prompt,
            "searchTerms": list(self.search_terms),
            "diagramSpec": self.diagram_spec,
            "regionIn": [round(self.region_in[0], 3), round(self.region_in[1], 3)],
            "icons": list(self.icons),
            "annotations": [a.to_dict() for a in self.annotations],
            "altText": self.alt_text,
            "labelsRequired": self.labels_required,
            "labelPolicy": self.label_policy,
        }


@dataclass
class LayoutSpec:
    """The chosen layout and the regions it resolved to.

    ``regions`` maps a slot name the layout defines ("title", "visual",
    "content", "step_1") to the rectangle it occupies. The renderer looks up
    slots it understands and ignores the rest, so adding a slot never breaks
    an older renderer.
    """

    name: str
    regions: Dict[str, Region] = field(default_factory=dict)
    flow: str = "ltr"        # ltr | rtl | ttb - reading order hint
    density: str = "normal"  # sparse | normal | dense

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "flow": self.flow, "density": self.density,
                "regions": {k: v.to_dict() for k, v in self.regions.items()}}


@dataclass
class ThemeSpec:
    """Resolved colour + type tokens for one slide.

    Produced by :mod:`ai_services.core.ppt_v2.design`. Carried on the slide
    rather than the deck so a summary or quiz slide can legitimately differ.
    """

    name: str
    colors: Dict[str, str] = field(default_factory=dict)
    fonts: Dict[str, str] = field(default_factory=dict)
    sizes: Dict[str, float] = field(default_factory=dict)
    background: str = "surface"   # token name or "gradient:primary->accent"

    def color(self, token: str, default: str = "#111827") -> str:
        return self.colors.get(token, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "colors": dict(self.colors),
                "fonts": dict(self.fonts), "sizes": dict(self.sizes),
                "background": self.background}


@dataclass
class SlideSpec:
    """One V2 slide."""

    slide_number: int
    slide_type: SlideType
    title: TitleSpec
    subtitle: str = ""
    learning_objective: str = ""
    content_blocks: List[ContentBlock] = field(default_factory=list)
    highlighted_terms: List[str] = field(default_factory=list)
    visual: VisualSpec = field(default_factory=VisualSpec)
    layout: Optional[LayoutSpec] = None
    theme: Optional[ThemeSpec] = None
    speaker_notes: str = ""
    footer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    # What the slide teaches and how the engine chose to teach it. Recorded,
    # not rendered: the renderer reads layout and visual, never these.
    content_type: str = ""
    pedagogy: Dict[str, Any] = field(default_factory=dict)

    # ---- validation -------------------------------------------------------

    def errors(self) -> List[str]:
        """Structural problems. Empty list means the slide can be rendered."""
        where = f"slide {self.slide_number}"
        out: List[str] = []

        if self.slide_number < 1:
            out.append(f"{where}: slideNumber must be >= 1")
        if not (self.title.text or "").strip():
            out.append(f"{where}: title is empty")
        if self.title.emphasis_text and self.title.emphasis_text not in self.title.text:
            out.append(f"{where}: emphasisText is not a substring of the title")

        if len(self.content_blocks) > MAX_CONTENT_BLOCKS:
            out.append(f"{where}: {len(self.content_blocks)} content blocks "
                       f"exceeds the {MAX_CONTENT_BLOCKS} a slide can carry")
        for i, block in enumerate(self.content_blocks):
            if not (block.text or "").strip() and not (block.label or "").strip():
                out.append(f"{where}: content block {i} is empty")

        if len(self.highlighted_terms) > MAX_HIGHLIGHTED_TERMS:
            out.append(f"{where}: too many highlighted terms "
                       f"({len(self.highlighted_terms)} > {MAX_HIGHLIGHTED_TERMS})")

        if len(self.visual.annotations) > MAX_ANNOTATIONS:
            out.append(f"{where}: {len(self.visual.annotations)} annotations "
                       f"exceeds {MAX_ANNOTATIONS}")
        for i, ann in enumerate(self.visual.annotations):
            out.extend(ann.errors(f"{where}: annotation {i}"))

        if self.layout is not None:
            for name, region in self.layout.regions.items():
                out.extend(region.errors(f"{where}: region {name!r}"))
            if self.visual.strategy is not VisualStrategy.NONE \
                    and "visual" not in self.layout.regions:
                out.append(f"{where}: layout {self.layout.name!r} has no 'visual' "
                           f"region but the slide wants a {self.visual.strategy.value}")

        return out

    @property
    def is_valid(self) -> bool:
        return not self.errors()

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The V2 block. See :meth:`to_v1_dict` for the wire format."""
        return {
            "version": SPEC_VERSION,
            "slideNumber": self.slide_number,
            "slideType": self.slide_type.value,
            "title": self.title.to_dict(),
            "subtitle": self.subtitle,
            "learningObjective": self.learning_objective,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "highlightedTerms": list(self.highlighted_terms),
            "visual": self.visual.to_dict(),
            "layout": self.layout.to_dict() if self.layout else None,
            "theme": self.theme.to_dict() if self.theme else None,
            "speakerNotes": self.speaker_notes,
            "footer": self.footer,
            "citations": list(self.citations),
            "contentType": self.content_type,
            "pedagogy": dict(self.pedagogy),
        }

    def to_v1_dict(self) -> Dict[str, Any]:
        """V1 keys plus a nested ``v2`` block.

        A renderer that has not learned V2 reads the flat keys and draws the
        old bullets-beside-a-photo slide. One that has reads ``v2`` and ignores
        the flat keys. Neither has to know about the other.
        """
        bullets = [b.text or b.label for b in self.content_blocks if (b.text or b.label)]
        v1_type = {
            SlideType.TITLE: "title",
            SlideType.SUMMARY: "summary",
        }.get(self.slide_type, "content")
        return {
            "slideNumber": self.slide_number,
            "type": v1_type,
            "title": self.title.text,
            "subtitle": self.subtitle,
            "bullets": bullets,
            "speakerNotes": self.speaker_notes,
            "imageSearchTerm": (self.visual.search_terms or [""])[0],
            "imageUrl": self.visual.url,
            "imageBase64": self.visual.base64,
            "v2": self.to_dict(),
        }


def validate_deck(slides: Sequence[SlideSpec]) -> List[str]:
    """Deck-level checks that no single slide can make on its own."""
    out: List[str] = []
    if not slides:
        return ["deck has no slides"]

    numbers = [s.slide_number for s in slides]
    if len(set(numbers)) != len(numbers):
        out.append(f"duplicate slide numbers: {sorted(numbers)}")
    if numbers != sorted(numbers):
        out.append(f"slides are out of order: {numbers}")

    for slide in slides:
        out.extend(slide.errors())

    # A deck where every slide looks identical is the V1 failure V2 exists to
    # fix, so it is worth failing loudly on rather than shipping quietly.
    layouts = {s.layout.name for s in slides if s.layout}
    if len(slides) >= 4 and len(layouts) == 1:
        out.append(f"all {len(slides)} slides use layout {layouts.pop()!r}; "
                   f"a deck needs visual variety to read as a lesson")

    return out
