"""Structured prompts for generated slide visuals.

A one-line prompt ("photosynthesis diagram") gets you a one-line result: a
green blob with garbled text baked into the pixels. Diffusion models cannot
spell, so every label they attempt is noise a teacher then has to apologise
for. The fix is structural, not stylistic:

  * **say what is in frame, and where** - objects and spatial relations, so the
    composition is specified rather than hoped for;
  * **ask for no text at all** - labels are added afterwards as real text in
    the ``annotations`` of the slide, where they are legible, translatable and
    editable;
  * **leave room for them** - the prompt reserves negative space on the side
    the layout will put labels on.

:func:`build_image_prompt` returns a plain string because that is what
``generate_note_image(prompt)`` takes. The structure lives in how the string is
assembled, so changing the house style is one function, not a prompt rewrite
scattered through the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# generate_note_image() already appends its own grounding clause forbidding
# extra labels, logos and watermarks. These are the constraints it does *not*
# cover, kept short: every token spent on prohibitions is a token not spent
# describing the subject, and over-long negative lists measurably degrade
# diffusion output.
_FORBIDDEN = (
    "no text, no letters, no numbers, no captions, no arrows with words, "
    "no signature, no border frame, no collage, no split panels"
)

# Style per class band. A Class 6 diagram and a Class 12 diagram are different
# documents, and "educational illustration" alone collapses them into one.
_STYLE_BANDS = (
    (1, 5, "simple flat vector illustration, bold rounded shapes, bright "
           "friendly primary colours, very low detail, cartoon-adjacent but "
           "scientifically accurate"),
    (6, 8, "clean flat vector infographic illustration, clear separated "
           "shapes, moderate detail, bright but not childish palette, "
           "textbook diagram style"),
    (9, 10, "precise flat vector scientific diagram, accurate proportions, "
            "restrained palette, clear separation between parts, "
            "NCERT textbook figure style"),
    (11, 12, "detailed technical scientific illustration, accurate structure "
             "and proportion, subdued professional palette, cross-section "
             "where it aids understanding, reference-figure quality"),
)
_DEFAULT_STYLE = _STYLE_BANDS[2][2]

_ORIENTATION_HINTS = {
    "landscape": "wide landscape composition, 3:2",
    "portrait": "tall portrait composition, 2:3",
    "square": "square composition, 1:1",
}

# Where the layout will place labels, so the image leaves that area quiet.
_NEGATIVE_SPACE = {
    "left": "keep the left third visually calm and uncluttered",
    "right": "keep the right third visually calm and uncluttered",
    "top": "keep the upper quarter visually calm and uncluttered",
    "bottom": "keep the lower quarter visually calm and uncluttered",
    "edges": "keep the outer margins calm, subject centred with clear space around it",
    "none": "",
}


def _class_number(class_name: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})", str(class_name or ""))
    if not m:
        return None
    value = int(m.group(1))
    return value if 1 <= value <= 12 else None


def style_for_class(class_name: str = "") -> str:
    n = _class_number(class_name)
    if n is None:
        return _DEFAULT_STYLE
    for low, high, style in _STYLE_BANDS:
        if low <= n <= high:
            return style
    return _DEFAULT_STYLE


@dataclass
class ImagePromptRequest:
    """Everything the prompt builder needs, named rather than positional."""

    subject: str = ""
    chapter: str = ""
    topic: str = ""
    class_name: str = ""
    learning_objective: str = ""
    description: str = ""
    objects: List[str] = field(default_factory=list)
    spatial_relations: List[str] = field(default_factory=list)
    orientation: str = "landscape"
    negative_space: str = "none"
    palette_hint: str = ""
    crop_safe: bool = False


def build_image_prompt(req: ImagePromptRequest) -> str:
    """Assemble one generation prompt.

    The order matters. Diffusion models weight earlier tokens more heavily, so
    the subject of the image leads, the composition follows, and the style and
    prohibitions come last where they shade the result without competing with
    it.
    """
    parts: List[str] = []

    subject_line = (req.description or "").strip()
    if not subject_line:
        subject_line = " ".join(x for x in (req.topic, req.chapter) if x).strip()
    if not subject_line:
        subject_line = "educational science illustration"
    parts.append(subject_line.rstrip(". "))

    objects = [o.strip() for o in req.objects if (o or "").strip()]
    if objects:
        parts.append("showing " + ", ".join(objects[:8]))

    relations = [r.strip() for r in req.spatial_relations if (r or "").strip()]
    if relations:
        parts.append("; ".join(relations[:5]))

    context = " ".join(x for x in (req.class_name, req.subject, req.chapter) if x).strip()
    if context:
        parts.append("for a %s lesson" % context)

    if req.learning_objective:
        # Phrased as what must be visible, not as a teaching goal: the model
        # renders nouns, not pedagogy.
        parts.append("the image must make clear: %s"
                     % req.learning_objective.rstrip(". "))

    parts.append(style_for_class(req.class_name))

    if req.palette_hint:
        parts.append(req.palette_hint)

    parts.append(_ORIENTATION_HINTS.get(req.orientation, _ORIENTATION_HINTS["landscape"]))

    space = _NEGATIVE_SPACE.get(req.negative_space, "")
    if space:
        parts.append(space)

    if req.crop_safe:
        # The generated canvas and the slide region are different shapes, so
        # the image will be cropped to fit. Anything near an edge is at risk.
        parts.append("keep the whole subject well inside the frame with "
                     "generous margins, nothing important near the edges")

    parts.append("plain white or very light background")
    parts.append(_FORBIDDEN)

    prompt = ", ".join(p for p in parts if p)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    # Long prompts get truncated by the provider at an arbitrary point, which
    # silently drops the prohibitions at the end. Trim the middle instead.
    return _trim(prompt, 1200)


def _trim(prompt: str, limit: int) -> str:
    if len(prompt) <= limit:
        return prompt
    tail = ", " + _FORBIDDEN
    head = prompt[: max(limit - len(tail), 0)].rsplit(",", 1)[0]
    return head + tail


def negative_space_for_layout(layout_name: str) -> str:
    """Where the layout will put text over or beside the image."""
    return {
        "concept_hero_split": "left",
        "anatomy_labeled": "edges",
        "magnification": "right",
        "title_hero": "left",
        "process_steps": "bottom",
        "comparison": "bottom",
        "cause_effect": "top",
        "formula_equation": "none",
        "summary_grid": "none",
        "quiz": "none",
    }.get(layout_name, "none")


def orientation_for_aspect(aspect: float) -> str:
    """Name the shape of a canvas from its width/height ratio."""
    if aspect >= 1.25:
        return "landscape"
    if aspect <= 0.8:
        return "portrait"
    return "square"


def orientation_for_region(width: float, height: float,
                           slide_aspect: float = 16.0 / 9.0) -> str:
    """Pick an orientation from the region the image has to fill.

    Regions are normalised, so a 0.4x0.6 region on a 16:9 slide is actually
    wider than it looks: the ratio has to be rescaled by the slide aspect
    before it means anything.
    """
    if width <= 0 or height <= 0:
        return "landscape"
    return orientation_for_aspect((width * slide_aspect) / height)


# Words that say nothing about what is in a picture. Without this filter the
# "distinctive" word pulled out of "Side view of a leaf showing sunlight" is
# "side", and every slide searches for the same thing.
_STOPWORDS = frozenset("""
a an the of in on at to for with and or but from by as is are was were be been
being this that these those it its their there here showing shows show showed
view diagram image picture illustration figure drawing photo photograph
left right top bottom above below inside outside near around between
small large big tiny simple clean clear single one two three four five
indicating labelled labeled label labels arrow arrows part parts process
""".split())


def key_nouns(text: str, limit: int = 3) -> List[str]:
    """The distinctive words of a visual description.

    Used when the planner gives no explicit object list: without it two slides
    on one topic produce the same query and land on the same photograph, which
    the first live run did.
    """
    out: List[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z-]{2,}", text or ""):
        low = word.lower()
        if low in _STOPWORDS or low in {o.lower() for o in out}:
            continue
        out.append(low)
        if len(out) >= limit:
            break
    return out


# What kind of figure each visual kind should be asking for.
_KIND_QUERY = {
    "diagram": "labelled diagram",
    "illustration": "educational diagram",
    "chart": "graph",
    "photo": "photograph",
}


def build_search_terms(topic: str, chapter: str = "", subject: str = "",
                       objects: Optional[Sequence[str]] = None,
                       visual_kind: str = "", class_name: str = "") -> List[str]:
    """Ordered search queries for the SEARCHED_IMAGE strategy.

    Queries describe the VISUAL, not just the chapter. The first version
    opened with a bare "Photosynthesis diagram", which matches every
    photosynthesis figure ever drawn - including the combined
    photosynthesis-and-respiration one that ended up on an introductory
    slide. Leading with the objects the picture must contain biases the
    result set towards the right *kind* of figure before ranking even runs.

    Specific first, broad last, and capped: concatenating every keyword into
    one long query returns fewer and worse results, not better ones. Each
    query stays short enough to match real captions.
    """
    objects = [o.strip() for o in (objects or []) if (o or "").strip()]
    kind = _KIND_QUERY.get((visual_kind or "").strip().lower(), "educational diagram")
    terms: List[str] = []

    def add(*parts: str) -> None:
        value = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
        if value and value.lower() not in {t.lower() for t in terms}:
            terms.append(value)

    # Most specific: topic plus the two things that must be visible.
    if topic and objects:
        add(topic, " ".join(objects[:2]), kind)
        add(topic, objects[0], kind)
    if topic:
        add(topic, kind)
        add(topic, "labelled diagram")
        add(topic)
    for obj in objects[:2]:
        add(obj, subject, kind)
    if chapter and chapter.strip().lower() != (topic or "").strip().lower():
        add(chapter, subject, kind)
    return terms[:6]
