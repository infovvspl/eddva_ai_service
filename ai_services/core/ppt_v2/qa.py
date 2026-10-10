"""Structural quality checks on a composed deck.

**This is not visual QA.** Nothing here renders a pixel, so nothing here can
tell you a slide is ugly, that an image is wrong for the topic, or that a
generated figure has nonsense baked into it. It catches the failures that are
decidable from the specification alone:

  * geometry that leaves the slide or collides with another slot;
  * more text than the region it is assigned can physically hold;
  * body text that will not have enough contrast against its background;
  * a visual strategy that resolved to nothing with no region to show for it;
  * a deck where every slide came out the same shape.

Those are precisely the regressions a planner change introduces silently, so
they are worth a gate. Judging whether the deck actually teaches well still
requires a human looking at the rendered file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ai_services.core.ppt_v2.design import (
    MIN_BODY_CONTRAST,
    MIN_LARGE_CONTRAST,
    contrast_ratio,
)
from ai_services.core.ppt_v2.schema import (
    BlockKind,
    Region,
    SlideSpec,
    VisualStrategy,
    validate_deck,
)

ERROR = "error"
WARNING = "warning"

# How much text a region holds, derived rather than guessed. The first version
# of this used 900 and flagged ordinary one-sentence bullets as overflowing,
# which is worse than no check: a warning that fires on correct content gets
# ignored, and then it is not there when something really does overflow.
#
# For a 13.33 x 7.5in slide at point size p, in normalised units:
#     chars per line = w * 13.33 * 72 / (p * 0.5)      ~0.5em average glyph
#     lines          = h *  7.5 * 72 / (p * 1.25)      1.25 line height
# which multiplies out to w * h * 2868 at 17pt body, and w * h * 810 at 32pt
# title. Rounded down slightly, because a long word wraps early.
CHARS_PER_UNIT_AREA = 2800.0
TITLE_CHARS_PER_UNIT_AREA = 800.0

# Slots that hold text and must therefore not sit on top of one another.
_TEXT_SLOT_PREFIXES = ("title", "subtitle", "content", "block_", "label", "equation",
                       "question", "equation", "footer")

# The renderer contract. The planner could pick a layout, or the theme builder
# a token set, that the renderer has no case for; the slide would then come
# out silently plain with no error anywhere. These two lists are the backend's
# half of that contract and must match pptV2.js `_layouts` and `_FALLBACK_COLORS`.
RENDERABLE_LAYOUTS = frozenset({
    "title_hero", "concept_hero_split", "process_steps", "comparison",
    "cause_effect", "anatomy_labeled", "magnification", "timeline",
    "formula_equation", "summary_grid", "quiz",
})

RENDERER_REQUIRED_TOKENS = ("ink", "ink_muted", "heading", "surface", "border",
                            "primary", "primary_soft", "accent", "accent_soft",
                            "on_primary", "on_accent")


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    slide_number: Optional[int] = None

    def __str__(self) -> str:
        where = "slide %d" % self.slide_number if self.slide_number else "deck"
        return "[%s] %s: %s - %s" % (self.severity, where, self.code, self.message)


def _is_text_slot(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _TEXT_SLOT_PREFIXES)


def _slot_text(slide: SlideSpec, name: str) -> str:
    if name == "title":
        return slide.title.text
    if name == "subtitle":
        return slide.subtitle
    if name == "footer":
        return slide.footer
    if name == "content":
        return " ".join(b.text or b.label for b in slide.content_blocks)
    if name == "equation":
        eq = next((b for b in slide.content_blocks if b.kind is BlockKind.EQUATION), None)
        return eq.text if eq else ""
    if name.startswith("block_"):
        try:
            block = card_blocks(slide)[int(name.rsplit("_", 1)[1])]
        except (ValueError, IndexError):
            return ""
        return " ".join(x for x in (block.label, block.text, block.detail) if x)
    return ""


def card_blocks(slide: SlideSpec) -> List:
    """The blocks the renderer draws as cards, in slot order.

    Where the layout gives the equation its own band, equation blocks are
    drawn there and skipped by the cards. Indexing slots by raw block
    position measured every formula slot against the block BEFORE the one
    actually drawn in it.
    """
    regions = slide.layout.regions if slide.layout else {}
    if "equation" in regions:
        return [b for b in slide.content_blocks if b.kind is not BlockKind.EQUATION]
    return list(slide.content_blocks)


def check_geometry(slide: SlideSpec) -> List[Issue]:
    out: List[Issue] = []
    if slide.layout is None:
        return [Issue(ERROR, "no_layout", "slide has no layout", slide.slide_number)]

    regions: Dict[str, Region] = slide.layout.regions
    for name, region in regions.items():
        for problem in region.errors("region %r" % name):
            out.append(Issue(ERROR, "out_of_bounds", problem, slide.slide_number))

    names = sorted(n for n in regions if _is_text_slot(n))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # "content" is the single-stack alias for the individually placed
            # block_N slots, so those two overlapping is by construction.
            if {a, b} & {"content"} and (a.startswith("block_") or b.startswith("block_")):
                continue
            if regions[a].overlaps(regions[b]):
                out.append(Issue(ERROR, "overlap",
                                 "regions %r and %r overlap" % (a, b),
                                 slide.slide_number))
    return out


def check_text_fit(slide: SlideSpec) -> List[Issue]:
    out: List[Issue] = []
    if slide.layout is None:
        return out
    for name, region in slide.layout.regions.items():
        if not _is_text_slot(name):
            continue
        text = _slot_text(slide, name)
        if not text:
            continue
        capacity = (TITLE_CHARS_PER_UNIT_AREA if name == "title"
                    else CHARS_PER_UNIT_AREA) * region.w * region.h
        if len(text) > capacity:
            out.append(Issue(
                WARNING, "text_overflow",
                "%r holds %d characters but the region fits about %d"
                % (name, len(text), int(capacity)),
                slide.slide_number))
    return out


def check_contrast(slide: SlideSpec) -> List[Issue]:
    out: List[Issue] = []
    theme = slide.theme
    if theme is None:
        return out

    background = theme.color("surface", "#FFFFFF")
    pairs = (
        ("ink", MIN_BODY_CONTRAST, "body text"),
        ("ink_muted", MIN_BODY_CONTRAST, "secondary text"),
        ("heading", MIN_LARGE_CONTRAST, "headings"),
    )
    for token, minimum, label in pairs:
        try:
            ratio = contrast_ratio(theme.color(token), background)
        except ValueError as exc:
            out.append(Issue(ERROR, "bad_colour", str(exc), slide.slide_number))
            continue
        if ratio < minimum:
            out.append(Issue(
                ERROR, "low_contrast",
                "%s (%s on %s) is %.2f:1, below the %.1f:1 needed"
                % (label, theme.color(token), background, ratio, minimum),
                slide.slide_number))

    for token, fill in (("on_primary", "primary"), ("on_accent", "accent")):
        try:
            ratio = contrast_ratio(theme.color(token), theme.color(fill))
        except ValueError:
            continue
        if ratio < MIN_LARGE_CONTRAST:
            out.append(Issue(
                ERROR, "low_contrast",
                "%s on %s is %.2f:1, below %.1f:1"
                % (token, fill, ratio, MIN_LARGE_CONTRAST),
                slide.slide_number))
    return out


def check_visual(slide: SlideSpec) -> List[Issue]:
    out: List[Issue] = []
    visual = slide.visual
    if visual.strategy is VisualStrategy.NONE:
        return out

    if slide.layout and "visual" not in slide.layout.regions:
        out.append(Issue(ERROR, "no_visual_region",
                         "strategy is %s but the layout has nowhere to put it"
                         % visual.strategy.value, slide.slide_number))

    if visual.strategy is VisualStrategy.AI_GENERATED_IMAGE and not visual.image_prompt:
        out.append(Issue(ERROR, "no_prompt",
                         "AI image strategy with no generation prompt",
                         slide.slide_number))
    if visual.strategy is VisualStrategy.SEARCHED_IMAGE and not visual.search_terms:
        out.append(Issue(ERROR, "no_search_terms",
                         "search strategy with no query", slide.slide_number))

    if not visual.fallbacks:
        out.append(Issue(WARNING, "no_fallback",
                         "strategy %s has no fallback; a failure here leaves "
                         "the slide empty" % visual.strategy.value,
                         slide.slide_number))

    if visual.resolved_strategy is not None:
        if (visual.resolved_strategy not in (VisualStrategy.NONE,
                                             VisualStrategy.ICON_COMPOSITION)
                and not visual.has_asset):
            out.append(Issue(ERROR, "missing_asset",
                             "resolved to %s but produced no image"
                             % visual.resolved_strategy.value, slide.slide_number))
        if not visual.alt_text and visual.has_asset:
            out.append(Issue(WARNING, "no_alt_text",
                             "image has no alt text", slide.slide_number))
    return out


def check_renderable(slide: SlideSpec) -> List[Issue]:
    """Can the renderer actually draw what the backend chose?"""
    out: List[Issue] = []
    if slide.layout is not None and slide.layout.name not in RENDERABLE_LAYOUTS:
        out.append(Issue(ERROR, "unrenderable_layout",
                         "layout %r has no renderer case; the slide would fall "
                         "back to a generic one" % slide.layout.name,
                         slide.slide_number))
    if slide.theme is not None:
        missing = [t for t in RENDERER_REQUIRED_TOKENS if not slide.theme.colors.get(t)]
        if missing:
            out.append(Issue(ERROR, "missing_token",
                             "theme %r is missing colour token(s) the renderer "
                             "needs: %s" % (slide.theme.name, ", ".join(missing)),
                             slide.slide_number))
    return out


def check_labels(slide: SlideSpec) -> List[Issue]:
    """Deterministic checks on the placed labels of one figure.

    Placement happens in core.ppt_v2.labels before either renderer draws, so
    these are a guard on that output rather than a second attempt at it. The
    Phase 7 Cell Structure slide would have failed every one of them.
    """
    from ai_services.core.ppt_v2 import labels as label_engine

    out: List[Issue] = []
    annotations = (slide.visual.annotations or [])
    if not annotations:
        return out

    placed = []
    for a in annotations:
        if not a.box:
            # Unplaced: the renderer would have to guess, which is what
            # produced overlapping labels in the first place.
            out.append(Issue(WARNING, "label_unplaced",
                             "%r has no placed box" % a.text, slide.slide_number))
            continue
        placed.append(label_engine.Placed(
            text=a.text, anchor=a.anchor,
            box=label_engine.Box(*a.box), side=a.side, scale=a.scale))

    for code, detail in label_engine.problems(placed):
        out.append(Issue(ERROR, code, detail, slide.slide_number))
    return out


_RASTER_STRATEGIES = frozenset({
    VisualStrategy.AI_GENERATED_IMAGE, VisualStrategy.PROGRAMMATIC_DIAGRAM,
    VisualStrategy.SEARCHED_IMAGE, VisualStrategy.CACHED_ASSET,
})
_LABELLED_POLICIES = frozenset({"anchored", "legend", "native"})


def check_label_contract(slide: SlideSpec) -> List[Issue]:
    """Required labels reach the final slide, whichever producer won.

    The fallback contract: an AI image rejected in favour of a diagram, or a
    diagram in favour of a searched photo, must not quietly drop the names
    the slide exists to teach. Only meaningful after assets are resolved;
    before that there is nothing to check.
    """
    v = slide.visual
    resolved = v.resolved_strategy
    if not v.labels_required or resolved not in _RASTER_STRATEGIES:
        return []
    n = slide.slide_number
    if v.annotations and v.label_policy not in _LABELLED_POLICIES:
        return [Issue(ERROR, "required_labels_lost",
                      "%d required label(s) will not appear: %s supplied the "
                      "picture with label policy %r"
                      % (len(v.annotations), resolved.value, v.label_policy or "unset"), n)]
    if not v.annotations and resolved is VisualStrategy.AI_GENERATED_IMAGE:
        # Embedded text is stripped from generated images and the plan gave
        # the renderer nothing to put back.
        return [Issue(WARNING, "labels_unspecified",
                      "labels are required but the plan named none, and the "
                      "generated image's own text was removed", n)]
    return []


def check_deck(slides: Sequence[SlideSpec]) -> List[Issue]:
    """Every check, deck-wide. Empty list means nothing structural is wrong."""
    out: List[Issue] = [
        Issue(ERROR, "schema", message) for message in validate_deck(slides)
    ]
    for slide in slides:
        out.extend(check_geometry(slide))
        out.extend(check_text_fit(slide))
        out.extend(check_contrast(slide))
        out.extend(check_visual(slide))
        out.extend(check_renderable(slide))
        out.extend(check_labels(slide))
        out.extend(check_label_contract(slide))

    if len(slides) >= 3:
        strategies = {s.visual.strategy for s in slides}
        if strategies == {VisualStrategy.NONE}:
            out.append(Issue(WARNING, "no_visuals",
                             "no slide in the deck has a visual"))
    return out


def errors(issues: Sequence[Issue]) -> List[Issue]:
    return [i for i in issues if i.severity == ERROR]


def report(issues: Sequence[Issue]) -> str:
    if not issues:
        return "structural QA: no issues (note: no visual QA was performed)"
    lines = ["structural QA: %d error(s), %d warning(s)"
             % (len(errors(issues)), len(issues) - len(errors(issues)))]
    lines.extend("  " + str(i) for i in issues)
    lines.append("  (structural only - no renderer ran, so visual quality is unverified)")
    return "\n".join(lines)
