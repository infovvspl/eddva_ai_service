"""The visual decision engine: what kind of picture does this slide need?

V1 asks one question - "what should I search for?" - and takes the first usable
result. That is the right question for perhaps a third of slides. A chemical
equilibrium, a cell organelle callout, a 3-step water cycle and a photograph of
a volcano all want different *kinds* of image, and only the last one is a
search result.

This module chooses between the kinds and, critically, says what to do when the
chosen kind is unavailable. Availability is not hypothetical: AI image
generation is off by default and needs a HuggingFace token that is not
configured in every environment, so a planner that assumes it will work
produces blank slides. Every decision therefore carries an ordered fallback
chain ending at something that cannot fail.

Nothing here performs I/O. It decides; :func:`resolve_visual` is the one
function that calls out, and it takes its producers as arguments so tests never
touch the network.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

from ai_services.core.ppt_v2.schema import SlideType, VisualSpec, VisualStrategy
from ai_services.core.ppt_v2.textmatch import matches_any

logger = logging.getLogger(__name__)

# Cost and latency guard. A 10-slide deck generating 10 FLUX images would cost
# more and take longer than the whole rest of the pipeline, and most slides do
# not need one. The planner spends this budget on the slides that benefit most.
DEFAULT_AI_IMAGE_BUDGET = 3

# Preferred order per slide type: what KIND of visual that teaching purpose
# wants, before anything is known about the wording.
#
# This exists because strategy selection was too sensitive to the planner's
# phrasing. The same process slide came out as a generated illustration on
# one run and a shape composition on the next, purely because one
# description happened to contain the word "photosynthesis" and the other
# did not. The teaching purpose did not change between those runs, so the
# visual kind should not have either.
#
# The content heuristics still apply - they can promote a strategy within
# this list - but they can no longer reorder it. Anything not listed for a
# type falls through to CANONICAL_FALLBACK.
SLIDE_TYPE_PREFERENCE: Dict[SlideType, Sequence[VisualStrategy]] = {
    # A sequence is carried by its own numbered steps, so the visual above
    # them only has to show the chain of things involved - which the icon
    # row does well and legibly.
    #
    # The diagram sat first here briefly, on the theory that a drawn figure
    # of the mechanism beats everything else. Rasterising the result settled
    # it: render_diagram produced a sparse figure with text too small to read
    # from a desk, let alone a projector, while the icon row was clear. Until
    # diagram output improves, the icon row is the better default. Still
    # slide-type driven, so still stable run to run.
    SlideType.PROCESS: (VisualStrategy.ICON_COMPOSITION,
                        VisualStrategy.PROGRAMMATIC_DIAGRAM,
                        VisualStrategy.AI_GENERATED_IMAGE,
                        VisualStrategy.SEARCHED_IMAGE),
    # A generated figure first: it is drawn to the slide's own labels. Past
    # the AI budget, a searched textbook figure beats a drawn one. The drawn
    # diagram sat second until a live Respiration deck put a mitochondrion
    # on screen as two bare ellipses with "Matrix" and "Cristae" printed over
    # the lines - the same sparse, illegible output the PROCESS entry above
    # records. A searched figure is shown with a legend rather than anchored
    # labels (labels.LEGEND) and passes the same vision check, so it no
    # longer has to match label positions it never promised.
    SlideType.ANATOMY: (VisualStrategy.AI_GENERATED_IMAGE,
                        VisualStrategy.SEARCHED_IMAGE,
                        VisualStrategy.PROGRAMMATIC_DIAGRAM),
    SlideType.CONCEPT: (VisualStrategy.AI_GENERATED_IMAGE,
                        VisualStrategy.SEARCHED_IMAGE,
                        VisualStrategy.PROGRAMMATIC_DIAGRAM,
                        VisualStrategy.ICON_COMPOSITION),
    SlideType.EXAMPLE: (VisualStrategy.SEARCHED_IMAGE,
                        VisualStrategy.AI_GENERATED_IMAGE,
                        VisualStrategy.ICON_COMPOSITION),
    # Quantitative content is plotted, never photographed.
    SlideType.DATA: (VisualStrategy.PROGRAMMATIC_DIAGRAM,
                     VisualStrategy.ICON_COMPOSITION),
    SlideType.COMPARISON: (VisualStrategy.ICON_COMPOSITION,
                           VisualStrategy.SEARCHED_IMAGE,
                           VisualStrategy.AI_GENERATED_IMAGE),
    SlideType.CAUSE_EFFECT: (VisualStrategy.ICON_COMPOSITION,
                             VisualStrategy.AI_GENERATED_IMAGE,
                             VisualStrategy.SEARCHED_IMAGE),
    # The equation band is drawn natively, so the visual beside it only has
    # to show the quantities involved.
    SlideType.FORMULA: (VisualStrategy.ICON_COMPOSITION,
                        VisualStrategy.AI_GENERATED_IMAGE),
    SlideType.TITLE: (VisualStrategy.AI_GENERATED_IMAGE,
                      VisualStrategy.SEARCHED_IMAGE),
}

# Slide types whose structure BOTH renderers already draw natively, as
# vector shapes: numbered step cards with arrows, two compared columns, a
# cause-and-effect pair, a timeline axis, an equation band.
#
# For these, a generated raster diagram is strictly worse than what we
# already have. It cannot be edited in PowerPoint, it cannot be re-themed, it
# cannot reflow, and - measured on a real reflection figure drawn at the
# right size - free-form matplotlib still clustered the drawing into the
# middle third of the band, rotated a label vertically and overlapped two
# others. Our own layout code gets even spacing and label placement right by
# construction because it is not guessing.
#
# So a programmatic diagram is reserved for what the native primitives
# genuinely cannot express: a labelled figure of a real structure or a plotted
# quantity. This is the "prefer structured diagram types" rule, applied by
# routing to the structured representation that already exists rather than by
# building a second diagram engine beside it.
STRUCTURED_NATIVELY = frozenset({
    SlideType.PROCESS, SlideType.COMPARISON, SlideType.CAUSE_EFFECT,
    SlideType.TIMELINE, SlideType.FORMULA, SlideType.SUMMARY, SlideType.QUIZ,
})

# The chain every decision ultimately degrades along. Order is fidelity for
# teaching purposes, not cost: a correct diagram beats a pretty stock photo.
CANONICAL_FALLBACK: Sequence[VisualStrategy] = (
    VisualStrategy.AI_GENERATED_IMAGE,
    VisualStrategy.SEARCHED_IMAGE,
    VisualStrategy.CACHED_ASSET,
    VisualStrategy.PROGRAMMATIC_DIAGRAM,
    VisualStrategy.ICON_COMPOSITION,
    VisualStrategy.NONE,
)

# Things a camera can actually point at. For these a real photograph teaches
# better than any generated image, and search is both cheaper and more honest.
_PHOTOGRAPHABLE = (
    "leaf", "flower", "plant", "tree", "root", "stem", "seed", "fruit",
    "animal", "bird", "insect", "fish", "microscope", "beaker", "test tube",
    "apparatus", "laboratory", "volcano", "mountain", "river", "glacier",
    "desert", "soil", "rock", "mineral", "crystal", "cloud", "rain",
    "monument", "fort", "temple", "building", "map", "machine", "engine",
    "bridge", "crop", "field", "factory", "telescope", "satellite",
)

# Things no camera can see: too small, too fast, too abstract, or a process
# rather than an object. These want a generated or drawn figure.
_NOT_PHOTOGRAPHABLE = (
    "atom", "electron", "proton", "neutron", "molecule", "bond", "ion",
    "orbital", "nucleus of an atom", "magnetic field", "electric field",
    "force", "energy transfer", "wave", "frequency", "refraction",
    "reflection of light", "circuit", "current flow", "photosynthesis",
    "respiration at", "diffusion", "osmosis", "mitosis", "meiosis", "dna",
    "chromosome", "enzyme", "metabolism", "food chain", "water cycle",
    "nitrogen cycle", "carbon cycle", "gravitation", "orbit", "evolution",
    # Microscopic structures. A search returns a microscope photograph that
    # will not compose with the rest of the slide; a drawn figure will.
    "chlorophyll", "chloroplast", "stomata", "stoma", "organelle",
    "mitochondri", "nucleus", "membrane", "bacteria", "virus", "neuron",
)

# Quantitative content a plotting library draws better than any image source.
_CHARTABLE = (
    "graph", "chart", "plot", "axis", "versus", " vs ", "trend", "percentage",
    "proportion", "distribution", "data table", "bar", "pie", "histogram",
    "coordinate", "slope", "linear equation", "quadratic",
)


@dataclass(frozen=True)
class VisualCapabilities:
    """What this environment can actually produce, right now.

    Built from live configuration by :func:`detect_capabilities` so a decision
    never promises an asset the deployment cannot make.
    """

    ai_images: bool = False
    image_search: bool = True
    diagrams: bool = True
    cache: bool = True
    ai_image_budget: int = DEFAULT_AI_IMAGE_BUDGET

    # The canvas the generator actually produces, which is set by
    # HF_IMAGE_WIDTH/HEIGHT and is *not* the shape of the slide region. The
    # prompt has to describe the canvas; the region only decides how the
    # result is cropped into place.
    image_width: int = 768
    image_height: int = 512

    @property
    def image_aspect(self) -> float:
        return (self.image_width / self.image_height) if self.image_height else 1.5

    def supports(self, strategy: VisualStrategy) -> bool:
        return {
            VisualStrategy.AI_GENERATED_IMAGE: self.ai_images,
            VisualStrategy.SEARCHED_IMAGE: self.image_search,
            VisualStrategy.PROGRAMMATIC_DIAGRAM: self.diagrams,
            VisualStrategy.CACHED_ASSET: self.cache,
            VisualStrategy.ICON_COMPOSITION: True,   # shapes always render
            VisualStrategy.NONE: True,
        }[strategy]


def detect_capabilities(ai_image_budget: int = DEFAULT_AI_IMAGE_BUDGET) -> VisualCapabilities:
    """Read live configuration. Failures degrade to "unavailable", never raise."""
    try:
        from ai_services.core.image_generation import can_generate_note_images
        ai_images = bool(can_generate_note_images())
    except Exception:
        logger.debug("ppt_v2: AI image generation unavailable", exc_info=True)
        ai_images = False

    try:
        from ai_services.core.diagram import render_diagram  # noqa: F401
        diagrams = True
    except Exception:
        logger.debug("ppt_v2: programmatic diagrams unavailable", exc_info=True)
        diagrams = False

    width, height = 768, 512
    try:
        from ai_services.core.image_generation import image_generation_config
        cfg = image_generation_config()
        width, height = int(cfg["width"]), int(cfg["height"])
    except Exception:
        logger.debug("ppt_v2: falling back to default image canvas", exc_info=True)

    return VisualCapabilities(ai_images=ai_images, image_search=True,
                              diagrams=diagrams, cache=True,
                              ai_image_budget=max(0, int(ai_image_budget)),
                              image_width=width, image_height=height)


def _mentions(text: str, needles: Sequence[str]) -> bool:
    # Word-boundary matching, not substring: "atom" must not fire on
    # "anatomy", which would send every labelled-figure slide down the
    # not-photographable branch.
    return matches_any(text, needles)


# What the planner's advisory ``visual.kind`` maps onto. The planner knows the
# pedagogy; the engine knows what the deployment can produce and what the
# content actually is. So the hint breaks ties - it does not give orders.
_HINT_STRATEGY = {
    "photo": VisualStrategy.SEARCHED_IMAGE,
    "illustration": VisualStrategy.AI_GENERATED_IMAGE,
    "diagram": VisualStrategy.AI_GENERATED_IMAGE,
    "chart": VisualStrategy.PROGRAMMATIC_DIAGRAM,
    "none": VisualStrategy.NONE,
}


def preferred_strategy(slide_type: SlideType, description: str = "",
                       must_show: Optional[Sequence[str]] = None,
                       block_count: int = 0, hint: str = "") -> VisualStrategy:
    """The strategy this slide would use if everything were available.

    Separated from availability so the reasoning is testable on its own, and
    so a log line can say "wanted X, got Y" rather than only reporting Y.

    ``hint`` is the planner's ``visual.kind``. It is honoured in exactly two
    cases: when the planner says the slide needs no picture (it knows the
    pedagogy and a decorative image costs attention), and when the content
    itself gives the engine nothing to go on. Everywhere else the engine's
    reading of the content wins - a planner asking for a photograph of an
    electron does not get one.
    """
    text = " ".join([description or ""] + list(must_show or []))
    hint = (hint or "").strip().lower()

    if hint == "none":
        return VisualStrategy.NONE

    # Slide types whose content is already the visual. Adding a photograph to
    # a timeline or a summary grid is decoration, and decoration on a teaching
    # slide costs attention without adding meaning.
    if slide_type in (SlideType.TITLE,):
        return VisualStrategy.AI_GENERATED_IMAGE if text else VisualStrategy.NONE
    if slide_type in (SlideType.SUMMARY, SlideType.QUIZ, SlideType.TIMELINE):
        return VisualStrategy.NONE
    if slide_type is SlideType.FORMULA:
        # The equation itself is drawn natively in its own band, so the
        # visual beside it only has to show the quantities involved.
        return VisualStrategy.ICON_COMPOSITION

    if _mentions(text, _CHARTABLE) or slide_type is SlideType.DATA:
        return VisualStrategy.PROGRAMMATIC_DIAGRAM

    if slide_type is SlideType.PROCESS:
        # The numbered boxes carry the sequence. Which visual accompanies them
        # is decided by the slide type, not by whether the planner happened to
        # use a particular noun — that sensitivity is what made the same slide
        # come out differently on consecutive runs.
        return SLIDE_TYPE_PREFERENCE[SlideType.PROCESS][0]

    if slide_type is SlideType.ANATOMY:
        # A labelled figure must match its labels, which rules out a search
        # result that happens to look right.
        return VisualStrategy.AI_GENERATED_IMAGE

    if slide_type is SlideType.COMPARISON:
        return (VisualStrategy.SEARCHED_IMAGE
                if _mentions(text, _PHOTOGRAPHABLE)
                else VisualStrategy.ICON_COMPOSITION)

    if not text:
        return VisualStrategy.NONE

    if _mentions(text, _NOT_PHOTOGRAPHABLE):
        return VisualStrategy.AI_GENERATED_IMAGE
    if _mentions(text, _PHOTOGRAPHABLE):
        return VisualStrategy.SEARCHED_IMAGE
    # The content says nothing either way: this is the tie the hint breaks.
    return _HINT_STRATEGY.get(hint, VisualStrategy.SEARCHED_IMAGE)


def fallback_chain(primary: VisualStrategy, caps: VisualCapabilities,
                   slide_type: Optional[SlideType] = None) -> List[VisualStrategy]:
    """The ordered list of strategies to try, primary first.

    With a `slide_type`, the order after the primary follows that type's
    preference, so the chain is stable for a teaching purpose rather than
    varying with the planner's wording. Always ends at a strategy that cannot
    fail, so a slide can never end up with an empty visual region and no plan
    for it.
    """
    # Choosing NONE is a decision, not a failure to find an image. A summary
    # slide that deliberately has no visual must not fall back into
    # generating one.
    if primary is VisualStrategy.NONE:
        return [VisualStrategy.NONE]

    native = bool(slide_type and slide_type in STRUCTURED_NATIVELY)

    def allowed(strategy, as_fallback=True):
        if strategy is not VisualStrategy.PROGRAMMATIC_DIAGRAM:
            return True
        # A natively-structured slide does not fall back to a generated
        # raster of the structure it already draws itself.
        if native:
            return False
        # Nor does anything else, except where plotting is the point. As a
        # fallback the drawn diagram cost a live deck 26s and two LLM attempts
        # to put a sparse figure on a TITLE slide; the icon row behind it is
        # instant and clearer. Asked for directly (a graph, a geometric
        # figure) it is still drawn.
        return not (as_fallback and slide_type is not None
                    and slide_type is not SlideType.DATA)

    chain: List[VisualStrategy] = []
    # The primary is filtered too. It was not at first, so a formula slide -
    # natively drawn as an equation band - still led with a generated raster
    # because the content heuristic had asked for one.
    if caps.supports(primary) and allowed(primary, as_fallback=False):
        chain.append(primary)

    preference = list(SLIDE_TYPE_PREFERENCE.get(slide_type, ())) if slide_type else []
    for strategy in preference:
        if strategy is primary or strategy in chain:
            continue
        if strategy is VisualStrategy.NONE or not allowed(strategy):
            continue
        if caps.supports(strategy):
            chain.append(strategy)
    for strategy in CANONICAL_FALLBACK:
        if not allowed(strategy):
            continue
        if strategy is primary or strategy in chain:
            continue
        if strategy is VisualStrategy.NONE:
            continue          # always appended last, below
        if caps.supports(strategy):
            chain.append(strategy)
    if VisualStrategy.ICON_COMPOSITION not in chain:
        chain.append(VisualStrategy.ICON_COMPOSITION)
    chain.append(VisualStrategy.NONE)
    return chain


@dataclass
class DeckVisualPlan:
    """Decisions for a whole deck, so budget is spent across slides, not per slide."""

    caps: VisualCapabilities
    ai_images_used: int = 0
    _wanted: Dict[int, VisualStrategy] = field(default_factory=dict)

    def decide(self, slide_number: int, slide_type: SlideType,
               description: str = "", must_show: Optional[Sequence[str]] = None,
               block_count: int = 0, hint: str = "",
               preferred: Optional[VisualStrategy] = None,
               excluded: Sequence[VisualStrategy] = ()) -> VisualSpec:
        """Choose a slide's visual and its fallback chain.

        ``preferred`` comes from the pedagogical strategy layer when the
        content type names a visual of its own; otherwise the existing
        content-sensitive engine decides, exactly as before. ``excluded``
        removes strategies that would make the representation wrong - a
        generated map, a generated flowchart - from the chain WITHOUT
        reordering what remains: the order is still fallback_chain's.
        """
        wanted = preferred if preferred is not None else preferred_strategy(
            slide_type, description, must_show, block_count, hint)
        excluded = tuple(excluded or ())
        self._wanted[slide_number] = wanted

        caps = self.caps
        if (wanted is VisualStrategy.AI_GENERATED_IMAGE
                and VisualStrategy.AI_GENERATED_IMAGE not in excluded):
            if self.ai_images_used >= caps.ai_image_budget:
                # Out of budget is not a failure - it is the budget working.
                # Degrade rather than overspending on slide 7 of 10.
                caps = replace(caps, ai_images=False)
            elif caps.ai_images:
                self.ai_images_used += 1

        chain = fallback_chain(wanted, caps, slide_type)
        if excluded:
            # NONE is never excluded, so the chain still ends somewhere.
            chain = [s for s in chain if s not in excluded] or [VisualStrategy.NONE]
        return VisualSpec(
            description=description or "",
            must_show=list(must_show or []),
            strategy=chain[0],
            fallbacks=chain[1:],
        )

    def wanted_for(self, slide_number: int) -> Optional[VisualStrategy]:
        """What the slide asked for before availability and budget were applied."""
        return self._wanted.get(slide_number)


# --- execution --------------------------------------------------------------

Producer = Callable[[VisualSpec], Dict[str, object]]
"""A producer takes the spec and returns ``{"ok": bool, "url"/"base64", ...}``.

Deliberately the same shape :func:`generate_note_image` already returns, so the
existing function can be passed through a thin adapter rather than rewritten.
A producer that raises is treated as a failure of that strategy alone.
"""


def resolve_visual(spec: VisualSpec, producers: Dict[VisualStrategy, Producer],
                   on_attempt: Optional[Callable[[VisualStrategy, bool, str], None]] = None,
                   ) -> VisualSpec:
    """Walk the fallback chain until one producer succeeds.

    Each strategy is attempted **once**. There is no retry loop here: the
    producers own their own bounded retries (HuggingFace already retries a
    cold model once and reports ``fatal`` when retrying is pointless), and
    layering a second loop on top is how a slow deck becomes a hung one.
    """
    attempts = [spec.strategy] + list(spec.fallbacks)
    seen: set = set()

    for strategy in attempts:
        if strategy in seen:
            continue
        seen.add(strategy)

        if strategy is VisualStrategy.NONE:
            spec.resolved_strategy = VisualStrategy.NONE
            if on_attempt:
                on_attempt(strategy, True, "")
            return spec

        producer = producers.get(strategy)
        if producer is None:
            if on_attempt:
                on_attempt(strategy, False, "no producer")
            continue

        try:
            result = producer(spec) or {}
        except Exception as exc:
            logger.warning("ppt_v2: %s producer raised: %s", strategy.value, exc)
            if on_attempt:
                on_attempt(strategy, False, str(exc)[:200])
            continue

        if result.get("ok"):
            spec.url = str(result.get("url") or "")
            spec.base64 = str(result.get("base64") or "")
            spec.source = str(result.get("source") or result.get("provider") or "")
            spec.attribution = str(result.get("attribution") or "")
            # How the required labels reach the slide for THIS asset. Set by
            # the producer that made it, because only it knows whether the
            # pixels already carry them.
            spec.label_policy = str(result.get("labelPolicy") or "")
            spec.resolved_strategy = strategy
            if on_attempt:
                on_attempt(strategy, True, "")
            if spec.has_asset or strategy is VisualStrategy.ICON_COMPOSITION:
                return spec
            # Reported success with nothing to draw: treat as a failure so the
            # chain continues rather than leaving an empty region.
            spec.resolved_strategy = None

        reason = str(result.get("error") or "no asset returned")[:200]
        if on_attempt:
            on_attempt(strategy, False, reason)

    # The chain always contains NONE, so this is only reachable if a caller
    # hand-built a spec. Fail to the one outcome that always renders.
    spec.resolved_strategy = VisualStrategy.NONE
    return spec
