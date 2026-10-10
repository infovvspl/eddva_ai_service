"""Composition: planner JSON in, fully resolved slides out.

This is where the separate decisions meet. For each slide it:

  1. coerces the planner's JSON into a :class:`SlideSpec` (tolerantly - the
     model will get field names slightly wrong and that must not 500);
  2. picks a layout, avoiding the one used on the previous slides;
  3. resolves the theme: the teacher's studio pick, or else the subject's;
  4. decides the visual strategy and its fallback chain, inside the deck's
     AI-image budget;
  5. writes the generation prompt and the search terms, shaped by the region
     the image will actually occupy.

It does **not** fetch anything. Asset production is a separate, I/O-bound pass
(:func:`ai_services.core.ppt_v2.visuals.resolve_visual`) so composition stays
pure and fully testable, and so the fetches can be parallelised the way V1
already parallelises its image downloads.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ai_services.core.ppt_v2 import content, labels, pedagogy, prompts
from ai_services.core.ppt_v2.design import PALETTE_WORDS, build_theme
from ai_services.core.ppt_v2.layouts import select_layout
from ai_services.core.ppt_v2.schema import (
    Annotation,
    BlockKind,
    ContentBlock,
    Emphasis,
    LayoutSpec,
    MAX_ANNOTATIONS,
    MAX_CONTENT_BLOCKS,
    SlideSpec,
    SlideType,
    TitleSpec,
    VisualSpec,
    VisualStrategy,
)
from ai_services.core.ppt_v2.visuals import (
    DeckVisualPlan,
    VisualCapabilities,
    detect_capabilities,
)

logger = logging.getLogger(__name__)

# Slide types that teach better with a picture than without one. A summary
# or a quiz does not; everything else does.
_WANTS_A_VISUAL = frozenset({
    SlideType.CONCEPT, SlideType.ANATOMY, SlideType.EXAMPLE, SlideType.DATA,
    SlideType.PROCESS, SlideType.COMPARISON, SlideType.CAUSE_EFFECT,
    SlideType.TITLE,
})

# How far back to look when avoiding a repeated layout. Two is enough to break
# up runs without forcing an unsuitable layout onto slide three.
_VARIETY_WINDOW = 2


@dataclass
class DeckContext:
    """What the whole deck is about. Supplied by the caller, not the model."""

    topic: str = ""
    subject: str = ""
    chapter: str = ""
    class_name: str = ""
    board: str = ""
    slide_aspect: float = 16.0 / 9.0
    # A studio palette key (design.STUDIO_PALETTES). Empty, or anything
    # unknown, means the subject decides the colours.
    theme_key: str = ""


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _pick(data: Dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present key among several spellings the model might emit."""
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return default


def _slide_type(value: Any, fallback: SlideType = SlideType.CONCEPT) -> SlideType:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    try:
        return SlideType(raw)
    except ValueError:
        pass
    # The model reaches for near-synonyms; map the ones we have actually seen
    # rather than silently flattening everything to CONCEPT.
    alias = {
        "cover": SlideType.TITLE, "intro": SlideType.TITLE,
        "introduction": SlideType.CONCEPT, "content": SlideType.CONCEPT,
        "bullets": SlideType.CONCEPT, "definition": SlideType.CONCEPT,
        "steps": SlideType.PROCESS, "procedure": SlideType.PROCESS,
        "sequence": SlideType.PROCESS, "cycle": SlideType.PROCESS,
        "compare": SlideType.COMPARISON, "vs": SlideType.COMPARISON,
        "contrast": SlideType.COMPARISON, "difference": SlideType.COMPARISON,
        "causes": SlideType.CAUSE_EFFECT, "effects": SlideType.CAUSE_EFFECT,
        "structure": SlideType.ANATOMY, "parts": SlideType.ANATOMY,
        "labelled": SlideType.ANATOMY, "labeled": SlideType.ANATOMY,
        "diagram": SlideType.ANATOMY,
        "chart": SlideType.DATA, "graph": SlideType.DATA, "table": SlideType.DATA,
        "history": SlideType.TIMELINE, "chronology": SlideType.TIMELINE,
        "equation": SlideType.FORMULA, "formulae": SlideType.FORMULA,
        "recap": SlideType.SUMMARY, "conclusion": SlideType.SUMMARY,
        "takeaways": SlideType.SUMMARY, "key_points": SlideType.SUMMARY,
        "question": SlideType.QUIZ, "mcq": SlideType.QUIZ,
        "check": SlideType.QUIZ, "practice": SlideType.QUIZ,
    }
    return alias.get(raw, fallback)


def _block_kind(value: Any, slide_type: SlideType) -> BlockKind:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    try:
        return BlockKind(raw)
    except ValueError:
        pass
    by_slide = {
        SlideType.PROCESS: BlockKind.STEP,
        SlideType.FORMULA: BlockKind.DEFINITION,
        SlideType.QUIZ: BlockKind.ANSWER,
        SlideType.DATA: BlockKind.STAT,
    }
    return by_slide.get(slide_type, BlockKind.BULLET)


def _blocks_from(data: Dict[str, Any], slide_type: SlideType) -> List[ContentBlock]:
    raw_blocks = _as_list(_pick(data, "blocks", "contentBlocks", "content_blocks",
                                "points", "bullets", default=[]))
    out: List[ContentBlock] = []
    for i, item in enumerate(raw_blocks):
        if isinstance(item, str):
            out.append(ContentBlock(kind=_block_kind(None, slide_type),
                                    text=item.strip(), order=i))
            continue
        if not isinstance(item, dict):
            continue
        text = _text(_pick(item, "text", "content", "description", "value"))
        label = _text(_pick(item, "label", "heading", "term", "title", "name"))
        if not text and not label:
            continue
        out.append(ContentBlock(
            kind=_block_kind(_pick(item, "kind", "type"), slide_type),
            text=text,
            label=label,
            detail=_text(_pick(item, "detail", "note", "meaning", "subtext")),
            icon=_text(_pick(item, "icon", "symbol")),
            order=i,
            emphasis=(Emphasis.STRONG
                      if _pick(item, "emphasis", "highlight") in (True, "strong", "key")
                      else Emphasis.NORMAL),
        ))
    # Hard cap. A slide with nine bullets is not a slide, and the layouts
    # divide their space by the block count, so the extras would come out
    # too small to read rather than merely crowded. The planner prompt asks
    # for at most four; this is the backstop for when it does not comply.
    return out[:MAX_CONTENT_BLOCKS]


def _annotations_from(data: Dict[str, Any]) -> List[Annotation]:
    out: List[Annotation] = []
    for item in _as_list(_pick(data, "annotations", "labels", "callouts", default=[])):
        if isinstance(item, str):
            out.append(Annotation(text=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        text = _text(_pick(item, "text", "label", "name"))
        if not text:
            continue
        anchor = _pick(item, "anchor", "at", "position", default=[0.5, 0.5])
        try:
            ax, ay = float(anchor[0]), float(anchor[1])
            ax, ay = min(max(ax, 0.0), 1.0), min(max(ay, 0.0), 1.0)
        except (TypeError, ValueError, IndexError, KeyError):
            ax, ay = 0.5, 0.5
        out.append(Annotation(text=text, anchor=(ax, ay),
                              side=_text(_pick(item, "side", default="auto")) or "auto"))
    return out[:MAX_ANNOTATIONS]


def slide_from_plan(data: Dict[str, Any], slide_number: int) -> SlideSpec:
    """Coerce one planner slide object into a SlideSpec.

    Tolerant by design: a field the model spelled differently costs a default,
    never an exception. Validation happens later, against the assembled spec,
    where the error message can name the slide.
    """
    data = data if isinstance(data, dict) else {}
    slide_type = _slide_type(_pick(data, "slideType", "slide_type", "type"))

    title_raw = _pick(data, "title", "heading", default="")
    if isinstance(title_raw, dict):
        title_text = _text(_pick(title_raw, "text", "value"))
        emphasis_text = _text(_pick(title_raw, "emphasisText", "emphasis_text"))
    else:
        title_text = _text(title_raw)
        emphasis_text = _text(_pick(data, "emphasisText", "emphasis_text",
                                    "titleEmphasis"))
    # An emphasis that is not actually in the title would fail validation; the
    # emphasis is cosmetic, so drop it rather than reject the slide.
    if emphasis_text and emphasis_text not in title_text:
        emphasis_text = ""

    visual_raw = _pick(data, "visual", "image", default={}) or {}
    if isinstance(visual_raw, str):
        visual_raw = {"description": visual_raw}
    if not isinstance(visual_raw, dict):
        visual_raw = {}

    visual = VisualSpec(
        description=_text(_pick(visual_raw, "description", "desc", "prompt",
                                "whatToShow", "what_to_show")),
        must_show=[_text(x) for x in _as_list(_pick(visual_raw, "mustShow",
                                                    "must_show", "required",
                                                    default=[])) if _text(x)],
        relationships=[_text(x) for x in _as_list(_pick(visual_raw, "relationships",
                                                        "requiredRelationships",
                                                        "spatialRelations",
                                                        default=[])) if _text(x)],
        avoid_concepts=[_text(x) for x in _as_list(_pick(visual_raw, "avoid",
                                                         "avoidConcepts",
                                                         "forbidden",
                                                         default=[])) if _text(x)],
        annotations=_annotations_from(visual_raw) or _annotations_from(data),
        alt_text=_text(_pick(visual_raw, "altText", "alt_text", "alt")),
    )
    # Carried through composition and consumed when the prompt is built.
    visual.icons = [_text(x) for x in _as_list(_pick(visual_raw, "objects",
                                                     "items", default=[])) if _text(x)]
    # The planner's advisory kind. Stored on `source` until the engine has
    # read it, because it is an input to the decision rather than a result.
    visual.source = _text(_pick(visual_raw, "kind", "visualKind", default=""))

    return SlideSpec(
        slide_number=int(_pick(data, "slideNumber", "slide_number",
                               default=slide_number) or slide_number),
        slide_type=slide_type,
        title=TitleSpec(text=title_text, emphasis_text=emphasis_text),
        subtitle=_text(_pick(data, "subtitle", "subheading")),
        learning_objective=_text(_pick(data, "learningObjective",
                                       "learning_objective", "objective")),
        content_blocks=_blocks_from(data, slide_type),
        highlighted_terms=[_text(x) for x in _as_list(
            _pick(data, "highlightedTerms", "highlighted_terms", "keyTerms",
                  default=[])) if _text(x)][:6],
        visual=visual,
        speaker_notes=_text(_pick(data, "speakerNotes", "speaker_notes", "notes")),
    )


@dataclass
class ComposedDeck:
    slides: List[SlideSpec] = field(default_factory=list)
    plan: Optional[DeckVisualPlan] = None
    notes: List[str] = field(default_factory=list)

    def to_v1_slides(self) -> List[Dict[str, Any]]:
        return [s.to_v1_dict() for s in self.slides]


def _state_the_equation(slide: SlideSpec, decision, analysis) -> None:
    """Give a formula-shaped slide the equation its band displays.

    The formula layout shows the slide's ``equation`` block in a band above
    the cards and keeps it out of them. With no such block it fell back to
    the first card - which was then drawn a second time below. A rasterised
    worked example showed "Step 1" printed in the band and again as the
    first step. The band should state the problem; the steps solve it.
    """
    if decision.slide_type is not SlideType.FORMULA:
        return
    if any(b.kind is BlockKind.EQUATION for b in slide.content_blocks):
        return
    if len(slide.content_blocks) >= MAX_CONTENT_BLOCKS:
        return            # no room; the renderer's fallback is the lesser evil
    from ai_services.core.ppt_v2.content import extract_formulas

    statement = (extract_formulas([slide.title.text]) or list(analysis.formulas) or [""])[0]
    statement = statement.strip().rstrip(".")
    if not statement:
        return
    slide.content_blocks.insert(0, ContentBlock(kind=BlockKind.EQUATION, text=statement))


# Who gets the deck's AI images first (lower is sooner). Anything unlisted
# sits in the middle, in slide order.
_AI_BUDGET_PRIORITY: Dict[SlideType, int] = {
    SlideType.ANATOMY: 0,
    SlideType.TITLE: 2,
}


def compose_deck(planned_slides: Sequence[Dict[str, Any]], context: DeckContext,
                 caps: Optional[VisualCapabilities] = None) -> ComposedDeck:
    """Resolve a planned deck into renderable slides."""
    caps = caps or detect_capabilities()
    visual_plan = DeckVisualPlan(caps=caps)
    deck = ComposedDeck(plan=visual_plan)
    recent: List[str] = []

    # Pass 1: what every slide teaches and what it wants to show.
    staged = []
    for index, raw in enumerate(planned_slides):
        slide = slide_from_plan(raw, index + 1)
        slide.slide_number = index + 1

        # What does this slide teach, and what is the best way to teach it?
        # Answered before any layout or visual is chosen; the slide type that
        # selects the layout is the strategy's bridge into the layout system,
        # not the planner's raw label.
        analysis = content.analyse(raw, subject=context.subject,
                                   class_level=context.class_name,
                                   chapter=context.chapter, topic=context.topic)
        decision = pedagogy.decide(analysis)
        if decision.slide_type is not slide.slide_type:
            deck.notes.append("slide %d: %s -> %s (%s)" % (
                index + 1, slide.slide_type.value, decision.content_type.value,
                analysis.source))
        slide.slide_type = decision.slide_type
        slide.content_type = decision.content_type.value
        slide.pedagogy = decision.to_dict()
        slide.visual.labels_required = decision.labels_required
        _state_the_equation(slide, decision, analysis)

        # The planner sometimes returns an empty visual block, and an empty
        # description means "no picture" to the decision engine - so a
        # concept slide silently lost its hero image. The slide type knows
        # whether it wants one, so derive a description rather than let the
        # omission decide.
        if not slide.visual.description and slide.slide_type in _WANTS_A_VISUAL:
            slide.visual.description = " ".join(
                x for x in (slide.title.text, slide.learning_objective) if x)[:200]
            if not slide.visual.icons:
                slide.visual.icons = [t for t in slide.highlighted_terms[:3]] or [
                    b.label for b in slide.content_blocks[:3] if b.label]
            deck.notes.append("slide %d: planner left the visual block empty; "
                              "described it from the title" % (index + 1))

        if slide.slide_type is SlideType.TITLE:
            # A title picture sets the scene; it teaches nothing a vision
            # check could verify. With a checklist it went through search,
            # two looks and a fallback - 81s on a live deck - for decoration.
            slide.visual.must_show = []
            slide.visual.relationships = []
            slide.visual.icons = []

        hint, slide.visual.source = slide.visual.source, ""
        staged.append((slide, decision, hint))

    # Pass 2: visual decisions, in order of need rather than slide order.
    # The AI-image budget is spent by whoever asks first, and slide order put
    # the title first: a live 10-slide deck spent an image on a decorative
    # title picture while the mitochondrion on slide 8 got none. A figure
    # that has to show named parts gets first claim; decoration gets last.
    decided_for = {}
    for slide, decision, hint in sorted(
            staged, key=lambda st: (_AI_BUDGET_PRIORITY.get(st[0].slide_type, 1),
                                    st[0].slide_number)):
        decided_for[slide.slide_number] = visual_plan.decide(
            slide_number=slide.slide_number,
            slide_type=slide.slide_type,
            description=slide.visual.description,
            must_show=slide.visual.must_show,
            block_count=len(slide.content_blocks),
            hint=hint,
            preferred=decision.visual,
            excluded=decision.excluded,
        )

    # Pass 3: layout, theme and inputs, back in slide order.
    for slide, decision, hint in staged:
        decided = decided_for[slide.slide_number]
        # Keep what the planner wrote; take only the decisions from the engine.
        slide.visual.strategy = decided.strategy
        slide.visual.fallbacks = decided.fallbacks

        wants_visual = slide.visual.strategy is not VisualStrategy.NONE
        # The formula layout shows the equation block in its own band and
        # never as a card, so its column holds one slot per CARD. Sizing it
        # for every block left a dead slot and squeezed the real cards until
        # a rasterised worked example had its text sitting on the border.
        cards = (sum(1 for b in slide.content_blocks if b.kind is not BlockKind.EQUATION)
                 if slide.slide_type is SlideType.FORMULA else len(slide.content_blocks))
        layout = select_layout(slide.slide_type, cards,
                               has_visual=wants_visual, avoid=recent[-_VARIETY_WINDOW:])
        regions = layout.regions(cards, wants_visual)
        slide.layout = LayoutSpec(name=layout.name, regions=regions,
                                  flow=layout.flow, density=layout.density)
        recent.append(layout.name)

        slide.theme = build_theme(subject=context.subject, chapter=context.chapter,
                                  slide_type=slide.slide_type,
                                  palette=context.theme_key or None)

        # The curriculum scope, for the small category label the renderer puts
        # above the title. The schema has always had `footer`; nothing filled
        # it, so every V2 slide carried an empty one and the renderer had
        # nothing real to show. Built here rather than in the renderer because
        # the slide should be self-describing wherever it is drawn.
        if not slide.footer:
            slide.footer = " • ".join(
                x for x in (context.class_name, context.subject, context.board) if x)

        _attach_visual_inputs(slide, context, caps, hint)

        if wants_visual and "visual" not in regions:
            # The chosen layout cannot hold an image after all. Rather than
            # emit a spec that fails validation, drop to a typographic slide.
            deck.notes.append(
                "slide %d: layout %r has no visual region; visual dropped"
                % (slide.slide_number, layout.name))
            slide.visual.strategy = VisualStrategy.NONE
            slide.visual.fallbacks = []

        deck.slides.append(slide)

    return deck


def _attach_visual_inputs(slide: SlideSpec, context: DeckContext,
                          caps: VisualCapabilities, hint_kind: str = "") -> None:
    """Write the prompt / search terms / diagram spec the strategy will need."""
    visual = slide.visual
    if visual.strategy is VisualStrategy.NONE:
        return

    # The slide title is a teaching headline ("How Plants Make Food"), which
    # is the wrong string to search an image index with. Searches and prompt
    # fallbacks use the real topic; the title is only a last resort.
    topic = context.topic or context.chapter or slide.title.text
    layout_name = slide.layout.name if slide.layout else ""
    region = (slide.layout.regions.get("visual") if slide.layout else None)
    # The prompt describes the canvas the generator will produce; the region
    # only tells us whether that canvas will have to be cropped to fit.
    orientation = prompts.orientation_for_aspect(caps.image_aspect)
    region_shape = (prompts.orientation_for_region(region.w, region.h,
                                                   context.slide_aspect)
                    if region else orientation)
    crop_safe = region_shape != orientation
    palette = ""
    if slide.theme:
        palette = "%s colour palette" % PALETTE_WORDS.get(slide.theme.name,
                                                          PALETTE_WORDS["default"])

    # The planner does not always fill `objects`. Falling back to the
    # distinctive words of the description keeps each slide's query its own;
    # without it, two slides on one topic search for the same string and the
    # deck ends up showing the same picture twice.
    objects = (list(visual.icons) or list(visual.must_show)
               or prompts.key_nouns(visual.description))

    visual.image_prompt = prompts.build_image_prompt(prompts.ImagePromptRequest(
        subject=context.subject,
        chapter=context.chapter,
        topic=topic,
        class_name=context.class_name,
        learning_objective=slide.learning_objective,
        description=visual.description,
        objects=objects,
        spatial_relations=(list(visual.relationships)
                           or [a.text for a in visual.annotations])[:5],
        orientation=orientation,
        negative_space=prompts.negative_space_for_layout(layout_name),
        palette_hint=palette,
        crop_safe=crop_safe,
    ))
    visual.search_terms = prompts.build_search_terms(
        topic=topic, chapter=context.chapter, subject=context.subject,
        objects=objects, visual_kind=hint_kind, class_name=context.class_name)
    visual.objective = slide.learning_objective or ""

    # Deduplicate and place the labels once, here, so both renderers draw
    # the same arrangement. Phase 7 drew "Chloroplast" twice and printed two
    # labels on top of each other, because placement was left to whatever
    # drew the picture.
    if visual.annotations:
        placed, dropped = labels.place(
            [{"text": a.text, "anchor": a.anchor, "side": a.side}
             for a in visual.annotations])
        by_text = {labels.normalise(p.text): p for p in placed}
        kept = []
        for annotation in visual.annotations:
            spot = by_text.pop(labels.normalise(annotation.text), None)
            if spot is None:
                continue                      # duplicate, or nowhere clear
            annotation.anchor = spot.anchor
            annotation.side = spot.side
            annotation.box = (spot.box.x, spot.box.y, spot.box.w, spot.box.h)
            annotation.scale = spot.scale
            kept.append(annotation)
        visual.annotations = kept
        if dropped:
            logger.info("ppt_v2: %d label(s) had nowhere clear to go: %s",
                        len(dropped), ", ".join(dropped[:3]))
    if region:
        # Inches on the 10x5.625in canvas both renderers draw.
        visual.region_in = (round(region.w * 10.0, 3), round(region.h * 5.625, 3))
    if not visual.diagram_spec:
        visual.diagram_spec = visual.description or topic
    if not visual.alt_text:
        visual.alt_text = visual.description or ("Illustration for %s" % topic)
