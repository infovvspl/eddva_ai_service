"""Image slides: every slide painted whole by an image model.

The alternative to the composed V2 deck, offered side by side so the two can
be compared on real chapters. The content is the same - the V2 planner
writes it, so both styles teach the same thing - but instead of composing
cards, icons and a picture, each slide's text and layout are described to an
image model that paints the whole slide in one go.

What this buys: the look of a finished infographic. What it costs, and why it
is an option rather than the default:

  * the text is pixels - a teacher cannot edit it in PowerPoint;
  * spelling and notation are only as good as the model's rendering; nothing
    here can verify them (the slide text travels in the speaker notes, so
    the deck stays usable and searchable);
  * one paid image per slide.

A slide whose picture fails keeps its text and renders the plain way, so a
provider hiccup never costs the teacher the deck.
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ai_services.core.ppt_v2 import slide_check

logger = logging.getLogger(__name__)

# 16:9 at a size the slide models accept. Qwen-Image-2.0 was chosen on a live
# comparison: the only one of four candidates that returned a correctly
# spelled slide (subscripts included) inside the request budget.
DEFAULT_MODEL = "Qwen/Qwen-Image-2.0"
WIDTH, HEIGHT = 1536, 864
MAX_BLOCKS = 4          # cards a painted slide carries; more and the model shrinks them
CARD_WORDS = 16         # a card's text is trimmed toward this (see _short)
HEADING_WORDS = 5
# Cards a slide shows: three down the left column, four in a two-by-two grid.
# A slide with more cards than the rest gets smaller text than the rest.
CARDS_BY_KIND = {"summary": 4, "quiz": 4}
COLUMN_CARDS = 3
TITLE_WORDS = 5         # a longer title is split at its natural break (see _split_title)
# The grids, named the same way on every slide that uses them.
COLUMN_GRID = "Cards in one column in the left half, evenly spaced; the title on one line."
SQUARE_GRID = "Cards in a two-by-two grid below the title, evenly spaced; the title on one line."

# Every slide is painted on its own, so nothing ties one slide's type sizes to
# the next unless the prompt does - word for word the same on every slide.
# The model sizes text to the space it is given: a live comparison slide put
# two 15-word lines in half-slide panels and painted them a third larger than
# the deck. So the card size is fixed here too.
TYPOGRAPHY = (
    "Same type and card size on every slide: one sans-serif font; title 7% of the slide height, "
    "card headings bold 3.5%, card text 2.8%; every card 44% of the slide width and 17% of its "
    "height; never shrink text to fit. Each card: a bold heading over one line of regular text.")
# A live slide title came back as **Aerobic vs Anaerobic Respiration** - the
# model added markdown asterisks to a clean title.
QUOTE_RULE = ("Spell quoted text exactly and legibly; never draw quote marks, asterisks, "
              "other text, numbers, watermark or logo.")

# Qwen-Image-2.0 on Together rejects a prompt over 2,000 characters, with a
# misleading 400 ("Positive prompt must be a non-empty ... between 2 and 2000").
PROMPT_LIMIT = 1950
# Rungs tried in order until the prompt fits (see slide_prompt): card words,
# illustration detail (2 description and labels, 1 short description, 0 none),
# cards kept, cut mid-sentence if needed. Picture detail goes before card
# text: a live process slide lost the end of every card ("enters the
# cytoplasm of…") to keep a 28-word picture description.
_LADDER = ((CARD_WORDS, 2, MAX_BLOCKS, False), (CARD_WORDS, 1, MAX_BLOCKS, False),
           (CARD_WORDS, 0, MAX_BLOCKS, False), (12, 0, MAX_BLOCKS, False),
           (10, 0, MAX_BLOCKS, False), (10, 0, 3, False), (8, 0, 3, True))
# A deck is a background job; the studio polls it for up to 8 minutes
# (app.js). A repaint (one slide, usually under 25s) is only started while the
# deck can still finish well inside that. A live 8-slide deck with two
# text-check repaints took 240s in all.
REPAINT_WINDOW_S = 420.0
# Rounds of retries for slides still unpainted after the first pass, and the
# pause before each round after the first (a provider that refused a burst
# usually takes a few seconds to recover).
REPAINT_ATTEMPTS = 3
REPAINT_PAUSE_S = 8.0
# Indirection so tests never really sleep.
_sleep = time.sleep


def repaint_window() -> float:
    """Seconds into a deck a failed slide may still be repainted (SLIDE_REPAINT_WINDOW_S).

    It was 150s while the backend held the request open for 240s, then 240s.
    A deck is a background job now (the studio follows it for 8 minutes, and
    the teacher's background list for a day), and a 10-slide deck often
    reaches 200s: a slide that failed late got no second try and stayed text.
    """
    try:
        return float(os.getenv("SLIDE_REPAINT_WINDOW_S", "") or REPAINT_WINDOW_S)
    except ValueError:
        return REPAINT_WINDOW_S


def repaint_attempts() -> int:
    """Rounds of retries for slides still unpainted (SLIDE_REPAINT_ATTEMPTS)."""
    try:
        return max(0, int(os.getenv("SLIDE_REPAINT_ATTEMPTS", "") or REPAINT_ATTEMPTS))
    except ValueError:
        return REPAINT_ATTEMPTS


def paint_early() -> bool:
    """Start painting before the chapter and fact checks finish (PPT_IMAGE_PAINT_EARLY)."""
    return (os.getenv("PPT_IMAGE_PAINT_EARLY", "true") or "").strip().lower() in ("1", "true", "yes", "on")


class _Restart(Exception):
    """The checks changed the number of slides; paint the checked plan from scratch."""

    def __init__(self, slides: List[Dict[str, Any]]):
        super().__init__("slide count changed")
        self.slides = slides


CHECK_WINDOW_S = 210.0


# The provider's moderation sometimes refuses a harmless slide; a second
# painting of the same prompt usually passes.
_MODERATION_MARKERS = ("content moderation", "invalid content detected", "flagged")


def _moderation_refusal(result: Dict[str, Any]) -> bool:
    text = str(result.get("error") or "").lower()
    return any(m in text for m in _MODERATION_MARKERS)


def check_window() -> float:
    """Seconds into a deck a text-check repaint may still start (PPT_IMAGE_CHECK_UNTIL_S).

    A live 8-slide deck spent 29s of its 194s on these repaints. A deck that is
    already slow (a long one, or a rate-limited provider) goes out as painted
    rather than keep the teacher waiting for a closer match.
    """
    try:
        return float(os.getenv("PPT_IMAGE_CHECK_UNTIL_S", "") or CHECK_WINDOW_S)
    except ValueError:
        return CHECK_WINDOW_S


def max_check_repaints(total: int) -> int:
    """At most this many slides are repainted for their text (PPT_IMAGE_CHECK_MAX_REPAINTS).

    The worst two by default: three at once drew "429 too many requests" from
    Together on a live deck and cost 44s; the worst slides are the ones a
    teacher notices.
    """
    default = min(2, max(1, total // 2))
    try:
        return max(0, int(os.getenv("PPT_IMAGE_CHECK_MAX_REPAINTS", "") or default))
    except ValueError:
        return default

# A palette per subject keeps a deck consistent from slide to slide; the model
# otherwise picks a new colour scheme every time. Most specific first: "Social
# Science", "Computer Science" and "Political Science" all contain "science"
# and were all painted green as if they were biology. Each palette names its
# background too, because a studio theme can be dark (see _STUDIO_PALETTES).
_ON_WHITE = " on a white background"
_PALETTES = (
    (("computer", "informatics", "coding", "information technology"),
     "soft pastel slate blues with cyan accents" + _ON_WHITE),
    (("social", "history", "geography", "civics", "economics", "political"),
     "warm pastel oranges and sand tones with brown accents" + _ON_WHITE),
    (("english", "hindi", "language", "literature", "odia", "sanskrit"),
     "soft pastel purples with rose accents" + _ON_WHITE),
    (("math", "mathematics", "statistics"),
     "soft pastel blues with indigo accents" + _ON_WHITE),
    (("physics", "chemistry"),
     "soft pastel blues with teal and amber accents" + _ON_WHITE),
    (("biology", "science", "botany", "zoology", "environment", "evs"),
     "soft pastel greens with fresh leaf-green accents" + _ON_WHITE),
)
_DEFAULT_PALETTE = "soft pastel blues and greens" + _ON_WHITE

# The studio themes a teacher can pick instead (keys match
# design.STUDIO_PALETTES and the studio's theme cards). Text colours are named
# because on a dark background the model otherwise keeps dark text - and, left
# to itself, put white text on pale panels in a live Dark Professional deck.
# So a dark theme says where each goes: white on the background, dark on the
# white cards.
_DARK_CARDS = "crisp white titles, white rounded cards with dark slate text"
_STUDIO_PALETTES = {
    "dark-professional": "a deep navy background with %s, coral-red accents "
                         "and sky-blue highlights" % _DARK_CARDS,
    "ocean-blue": "a deep ocean-blue background with %s, bright sky-blue "
                  "accents and aqua highlights" % _DARK_CARDS,
    "warm-sunset": "a deep plum background with %s, warm coral accents "
                   "and golden-orange highlights" % _DARK_CARDS,
    "forest-green": "a deep forest-green background with %s, fresh mint-green "
                    "accents and golden highlights" % _DARK_CARDS,
    "royal-purple": "a deep indigo-purple background with %s, vivid violet "
                    "accents and soft pink highlights" % _DARK_CARDS,
    "clean-white": "a clean white background with dark slate text, royal-blue accents "
                   "and teal highlights",
}


def slide_model() -> str:
    return (os.getenv("SLIDE_IMAGE_MODEL", "") or "").strip() or DEFAULT_MODEL


def palette_for(subject: str, theme_key: str = "") -> str:
    """Colours and background for every slide of the deck, in words.

    The teacher's studio theme wins when it is a known one; otherwise the
    subject decides, as it always has.
    """
    studio = _STUDIO_PALETTES.get(str(theme_key or "").strip().lower())
    if studio:
        return studio
    s = " %s " % (subject or "").lower()
    for keys, palette in _PALETTES:
        if any(k in s for k in keys):
            return palette
    return _DEFAULT_PALETTE


# Markdown the planner sometimes writes into slide text ("**Aerobic respiration
# occurs ...**"). Quoted text is painted exactly, so a live slide showed the
# asterisks on its title and every card, whatever QUOTE_RULE said.
_MARKUP = re.compile(r"\*+|__+|`+|^#+\s*")


def _q(text: Any) -> str:
    """Text the model must reproduce, quoted. Inner quotes would end the quote."""
    plain = _MARKUP.sub("", " ".join(str(text or "").split()))
    return '"%s"' % " ".join(plain.replace('"', "'").split())


def _split_title(title: str, subtitle: str) -> Tuple[str, str]:
    """A long title split at its natural break, the rest leading the subtitle.

    A painted title is drawn on one line, so a long one comes out smaller
    than its neighbours ("Presbyopia - Age-related Loss of Accommodation" next
    to "Myopia"). The part before " - ", ": " or " (" stays the title.
    """
    title, subtitle = (title or "").strip(), (subtitle or "").strip()
    if len(title.split()) <= TITLE_WORDS:
        return title, subtitle
    for sep in (" – ", " — ", " - ", ": ", " ("):
        if sep in title:
            head, rest = title.split(sep, 1)
            rest = rest.strip().rstrip(")").strip()
            if head.strip() and rest:
                return head.strip(), ("%s · %s" % (rest, subtitle)) if subtitle else rest
    return title, subtitle


def _blocks(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [b for b in (slide.get("blocks") or []) if isinstance(b, dict)
            and (b.get("text") or b.get("label"))][:MAX_BLOCKS]


# A clause break may sit this many words past the limit and still be used.
_CUT_SLACK = 4
# Words after a comma that start a new clause, so the comma is a safe stop.
_CLAUSE_OPENERS = frozenset((
    "which", "who", "whom", "whose", "where", "while", "whereas", "so", "but", "yet",
    "giving", "making", "causing", "allowing", "forming", "producing", "releasing",
    "leading", "resulting", "thus", "hence", "therefore", "although", "though"))
# Words that open a subordinate clause: what follows them up to the first
# comma is not a statement on its own.
_SUBORDINATORS = frozenset((
    "when", "if", "because", "although", "though", "since", "while", "as", "after",
    "before", "unless", "until", "once", "whereas", "whenever", "in", "during", "for"))


def _short(text: str, limit: int = CARD_WORDS, hard: bool = False) -> str:
    """A card's text at a size that paints at the deck's one text size.

    The first sentence; if that is still long, cut at the last clause break
    (comma, semicolon, dash) within a few words of the limit. Never mid-clause
    - a sentence with no usable break is kept whole rather than cut into a
    wrong statement - and never after only a leading subordinate clause:
    "When there is a lack of oxygen in our muscle cells." states nothing. The
    full text stays in the speaker notes.
    """
    text = " ".join(str(text or "").split())
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    words = first.split()
    if len(words) <= limit:
        return first
    head = " ".join(words[:limit + _CUT_SLACK])
    breaks = [m.start() for m in re.finditer(r", |; | - | \u2013 ", head)]
    for cut in reversed(breaks):
        kept = head[:cut]
        if len(kept.split()) < 8:
            break
        # A comma also separates list items ("carbon dioxide, water and
        # energy"); only a comma that opens a new clause is a place to stop.
        after = head[cut:].lstrip(",;-– ").split(" ", 1)[0].lower()
        if head[cut] == "," and after not in _CLAUSE_OPENERS:
            continue
        # A part that opens with a subordinate clause must reach its main clause.
        if kept.split()[0].lower() in _SUBORDINATORS and not re.search(r"[,;]", kept):
            continue
        return kept.rstrip(" ,;-\u2013") + "."
    # Only on the last rung of slide_prompt's ladder, for a prompt that would
    # otherwise be refused: the words that fit, marked as cut.
    return " ".join(words[:limit]).rstrip(" ,;-\u2013") + "\u2026" if hard else first


_NUMBERING = re.compile(r"^(?:q|question|step|point|no\.?)?\s*\d+[.:)]?$", re.IGNORECASE)


def _point(b: Dict[str, Any], card_words: int = CARD_WORDS, hard: bool = False) -> str:
    """One card, described the same way on every slide: bold heading, one short line.

    Not `"Label": "text"` - a live process slide painted that literally, quote
    marks and numbering included, as one big run of text.
    """
    # A label the planner already ended with ":" became "Carbon Dioxide::" on
    # a live slide once the prompt added its own separator.
    label = " ".join((b.get("label") or "").strip().rstrip(":").strip().split()[:HEADING_WORDS])
    # A bare number ("Q1", "Step 2") is what the card's number circle shows; as
    # a heading too, a live quiz painted "Q1" twice on every card.
    if _NUMBERING.match(label):
        label = ""
    text = _short(b.get("text") or "", card_words, hard)
    if label and text:
        return "a card: heading %s, line %s" % (_q(label), _q(text))
    return "a card: %s" % _q(label or text)


def _illustration(slide: Dict[str, Any], detail: int = 2) -> str:
    """The picture: 2 = description and labels, 1 = a short description, 0 = none
    (each layout then names a plain illustration of the slide's idea)."""
    visual = slide.get("visual") or {}
    desc = (visual.get("description") or "").strip()
    if not desc or detail <= 0:
        return ""
    labels = [a.get("text") for a in (visual.get("annotations") or [])
              if isinstance(a, dict) and a.get("text")][:5] if detail >= 2 else []
    # A description is an instruction, not slide text: it can be cut freely.
    desc = _short(desc, 28 if detail >= 2 else 14, hard=True)
    out = "a clear, accurate educational illustration of %s" % desc.rstrip(".…")
    if labels:
        out += ", with short labels and pointer arrows reading " + ", ".join(_q(t) for t in labels)
    return out


def _body(slide: Dict[str, Any], card_words: int = CARD_WORDS, detail: int = 2,
          max_cards: int = MAX_BLOCKS, hard: bool = False) -> str:
    """How this slide's content is laid out, by what kind of slide it is."""
    kind = (slide.get("slideType") or "concept").lower()
    blocks = _blocks(slide)[:min(max_cards, CARDS_BY_KIND.get(kind, COLUMN_CARDS))]
    picture = _illustration(slide, detail)

    def card(b: Dict[str, Any]) -> str:
        return _point(b, card_words, hard)    # every card on this slide at one length

    if kind == "title":
        # A live cover, given no card text, drew three cards of gibberish.
        return ("A large, bold title centred-left with the subtitle below it. On the right, "
                + (picture or "a large friendly illustration of the topic")
                + ". No other text on this slide: no cards, captions or bullet points.")

    if kind == "comparison":
        # Cards of the deck's one size, not half-slide panels: in a panel the
        # model scaled two short lines up to fill it.
        left = [b for b in blocks if b.get("kind") == "compare_left"] or blocks[:1]
        right = [b for b in blocks if b.get("kind") == "compare_right"] or blocks[1:2]
        side = lambda bs: "; ".join(card(b) for b in bs)  # noqa: E731
        return ("Two columns of cards side by side below the title, the left in one soft colour "
                "and the right in a contrasting one, with a small round \"VS\" badge between "
                "them. Left: %s. Right: %s. Below the cards, across the slide: %s."
                % (side(left), side(right), picture or "a simple illustration contrasting the two"))

    if kind == "quiz":
        # A quiz is either one question with answer options, or - as the
        # planner usually writes it - a list of questions. A live list of four
        # questions was painted as one question with the other three as
        # "options", and the model invented "Mitochondria" four times.
        options = [b for b in blocks if b.get("kind") == "answer"][:4]
        if not options:
            return ("Question cards, each with a coloured number circle: %s. %s Show no answers."
                    % ("; then ".join(card(b) for b in blocks), SQUARE_GRID))
        question = next((b.get("text") for b in blocks if b.get("kind") == "question"),
                        slide.get("subtitle") or "")
        return ("A highlighted question box reading %s. Below it, answer options as "
                "rounded cards lettered A, B, C, D, all in the same neutral style (do not "
                "mark any option as correct): %s."
                % (_q(_short(question, card_words + 6, hard)),
                   "; ".join("%s %s" % (chr(65 + i), _q(_short(b.get("text") or b.get("label") or "",
                                                               card_words, hard)))
                             for i, b in enumerate(options))))

    if kind == "summary":
        return ("Key-point cards, each with a small flat icon: %s. %s"
                % ("; ".join(card(b) for b in blocks), SQUARE_GRID))

    # The numbers come from the circles; the text carries none.
    steps = "; then ".join(card(b) for b in blocks)
    equation = next((b.get("text") for b in blocks if b.get("kind") == "equation"), "")
    if kind == "formula" or equation:
        others = [b for b in blocks if b.get("kind") != "equation"]
        out = ""
        if equation:
            out += ("A prominent rounded box showing the equation %s, written with correct "
                    "chemical/mathematical notation and subscripts. " % _q(equation))
        if others:
            out += "Below it, explanation cards: %s. " % "; ".join(card(b) for b in others)
        return out + (("On the side, " + picture + ".") if picture else "")

    if kind == "process":
        return ("Left half, top to bottom: %s. %s Each card: a step-number circle and a small icon; "
                "thin arrows show the order. "
                "Right half: %s." % (steps, COLUMN_GRID, picture or "an illustration of the process"))

    return ("Left half, top to bottom: %s. %s Each card: a number circle and a small icon. "
            "Right half: %s."
            % (steps, COLUMN_GRID, picture or "an illustration of the idea"))


def slide_prompt(slide: Dict[str, Any], *, number: int, total: int,
                 class_name: str = "", subject: str = "", chapter: str = "",
                 theme_key: str = "") -> str:
    """The full instruction for one slide image, always inside PROMPT_LIMIT.

    The illustration is described more briefly, then not at all, before card
    text is trimmed further; the full text stays in the speaker notes.
    """
    args = dict(number=number, total=total, class_name=class_name, subject=subject,
                chapter=chapter, theme_key=theme_key)
    prompt = ""
    for words, detail, max_cards, hard in _LADDER:
        prompt = _slide_prompt(slide, card_words=words, detail=detail,
                               max_cards=max_cards, hard=hard, **args)
        if len(prompt) <= PROMPT_LIMIT:
            return prompt
    logger.warning("image slides: slide %d prompt still %d chars; cutting to %d",
                   number, len(prompt), PROMPT_LIMIT)
    return prompt[:PROMPT_LIMIT]


def _slide_prompt(slide: Dict[str, Any], *, number: int, total: int,
                  class_name: str = "", subject: str = "", chapter: str = "",
                  theme_key: str = "", card_words: int = CARD_WORDS,
                  detail: int = 2, max_cards: int = MAX_BLOCKS,
                  hard: bool = False) -> str:
    """One slide's instruction at a given card length."""
    header = " | ".join(x for x in (class_name, subject, chapter) if x)
    title, subtitle = _split_title(slide.get("title") or "", slide.get("subtitle") or "")
    emphasis = (slide.get("emphasisText") or "").strip()
    parts = [
        "A 16:9 school lesson slide: a clean, modern infographic in %s, rounded cards with "
        "soft shadows, flat vector icons, generous spacing." % palette_for(subject, theme_key),
        # A live dark-theme slide came back as a picture of a slide: the navy
        # slide drawn floating on a pale page with a drop shadow.
        "The slide fills the whole image edge to edge: no border, page or mock-up frame.",
    ]
    if header:
        parts.append("Top-left: a small pill reading %s with a tiny subject icon." % _q(header))
    is_cover = (slide.get("slideType") or "").lower() == "title"
    parts.append("%s %s%s, with a short coloured underline." % (
        "Large bold title" if is_cover else "At the top-left, the slide title",
        _q(title),
        (" with the words %s in the accent colour" % _q(emphasis)) if emphasis and emphasis in title else ""))
    if subtitle and (slide.get("slideType") or "").lower() != "quiz":
        parts.append("Subtitle below the title: %s." % _q(_short(subtitle, 16, hard)))
    parts.append(_body(slide, card_words, detail, max_cards, hard))
    parts.append(TYPOGRAPHY)
    parts.append("Bottom-right: a small pill reading %s." % _q("%d / %d" % (number, total)))
    # A live grounded slide on glucose -> pyruvate was illustrated with
    # invented structural formulas: beyond the Class 10 book, and wrong.
    parts.append("Illustrations simple and accurate at %s textbook level: no structural "
                 "formulas, molecular models, equations or facts beyond the quoted text."
                 % (class_name or "school"))
    parts.append(QUOTE_RULE)
    return " ".join(p for p in parts if p)


def _expected_text(slide: Dict[str, Any], header: str = "") -> Dict[str, Any]:
    """What a painted slide was asked to show, for slide_check to read it against."""
    kind = (slide.get("slideType") or "concept").lower()
    title, subtitle = _split_title(slide.get("title") or "", slide.get("subtitle") or "")
    blocks = _blocks(slide)[:CARDS_BY_KIND.get(kind, COLUMN_CARDS)]
    cards = [" ".join(x for x in ((b.get("label") or "").rstrip(":"), _short(b.get("text") or "")) if x)
             for b in blocks]
    visual = slide.get("visual") or {}
    other = [subtitle, header, slide.get("emphasisText") or "", "VS"]
    other += [a.get("text") or "" for a in (visual.get("annotations") or []) if isinstance(a, dict)]
    headings = [(b.get("label") or "").rstrip(":") for b in blocks if b.get("label")]
    return {"title": title, "cards": cards, "headings": headings, "other": [o for o in other if o]}


def _bullets(slide: Dict[str, Any]) -> List[str]:
    out = []
    for b in _blocks(slide):
        label = (b.get("label") or "").strip().rstrip(":").strip()
        text = (b.get("text") or "").strip()
        out.append("%s: %s" % (label, text) if label and text else (text or label))
    return out


def generate_image_deck(*, topic: str, slide_count: int, language: str = "English",
                        ctx: Optional[Dict[str, str]] = None, board: str = "",
                        board_rules: str = "", passages: Optional[Sequence[Any]] = None,
                        institute_id: Any = None,
                        on_planner_usage: Optional[Callable[..., None]] = None,
                        on_image: Optional[Callable[..., None]] = None,
                        render: Optional[Callable[..., Dict[str, Any]]] = None,
                        on_progress: Optional[Callable[..., None]] = None,
                        on_event: Optional[Callable[..., None]] = None,
                        auto_range: Optional[Tuple[int, int]] = None,
                        theme_key: str = "") -> Dict[str, Any]:
    """Plan the deck, paint each slide, return the API ``data`` block.

    ``theme_key`` is the teacher's studio theme; empty or unknown keeps the
    subject palette. Raises ``PlannerError`` when there is nothing to paint;
    the caller falls back to V1 exactly as it does for a failed V2 deck.
    """
    from ai_services.core.image_generation import generate_slide_image
    from ai_services.core.ppt_v2.planner import focus_topic_for, plan_deck

    from ai_services.core.ppt_v2 import scope
    from ai_services.core.ppt_v2.planner import revise_args as _revise_args, revise_slides

    ctx = ctx or {}
    render = render or generate_slide_image
    started = time.perf_counter()

    def emit(stage, **kw):
        if on_progress:
            try:
                on_progress(stage, **kw)
            except Exception:
                logger.exception("image slides: progress callback failed")

    emit("planning")
    plan_args = dict(
        topic=topic, slide_count=slide_count, language=language,
        class_name=ctx.get("className", ""), subject=ctx.get("subjectName", ""),
        chapter=ctx.get("chapterName", "") or ctx.get("topicName", ""),
        focus_topic=focus_topic_for(ctx),
        board=board, board_rules=board_rules, passages=passages,
        institute_id=institute_id, on_usage=on_planner_usage, on_event=on_event,
        auto_range=auto_range,
    )
    model = slide_model()
    header = " | ".join(x for x in (ctx.get("className", ""), ctx.get("subjectName", ""),
                                    ctx.get("chapterName", "") or topic) if x)

    def prompt_for(s: Dict[str, Any], i: int, n: int) -> str:
        return slide_prompt(s, number=i + 1, total=n,
                            class_name=ctx.get("className", ""),
                            subject=ctx.get("subjectName", ""),
                            chapter=ctx.get("chapterName", "") or topic,
                            theme_key=theme_key)

    # Filled as slides are written (on_batch), then kept in step with the plan
    # as the checks change it (sync). Every painting records the prompt it was
    # made from, so a painting of text that later changed is never used.
    planned: List[Dict[str, Any]] = []
    prompts: List[str] = []
    expected: List[Any] = []
    state = {"total": 0, "title": topic}
    painted: Dict[int, Dict[str, Any]] = {}
    readings: Dict[int, Any] = {}
    futures: Dict[int, Any] = {}
    lock = __import__("threading").Lock()
    checking = slide_check.enabled()
    workers = max(1, int(os.getenv("SLIDE_IMAGE_CONCURRENCY", "3") or 3))
    early = paint_early()
    pool = ThreadPoolExecutor(max_workers=workers)

    def snapshot() -> Dict[str, Any]:
        with lock:
            n = state["total"]
            slides = []
            for i, s in enumerate(planned):
                out = {"slideNumber": i + 1,
                       "type": "title" if i == 0 else ("summary" if i == n - 1 else "content"),
                       "title": s.get("title") or "", "subtitle": s.get("subtitle") or "",
                       "bullets": _bullets(s)}
                done = painted.get(i)
                if done and done.get("ok"):
                    out["slideImage"] = {"url": done.get("url") or "", "model": model}
                elif done is None:
                    out["slideImagePending"] = True
                slides.append(out)
            return {"title": state["title"], "slides": slides,
                    "generation": {"version": "image", "partial": True}}

    def say(i: int, text: str, level: str = "active") -> None:
        if on_event:
            try:
                on_event("paint-%d" % (i + 1), "Slide %d: %s" % (i + 1, text), level)
            except Exception:
                logger.exception("image slides: activity event failed")

    def current(i: int, result: Dict[str, Any]) -> bool:
        """The painting shows the slide's text as it stands now."""
        return bool(result.get("ok")) and result.get("prompt") == prompts[i]

    def render_once(prompt: str) -> Dict[str, Any]:
        try:
            return dict(render(prompt, model=model, width=WIDTH, height=HEIGHT,
                               tenant=str(institute_id or "")) or {})
        except Exception as exc:  # one slide never costs the deck
            logger.exception("image slides: render raised")
            return {"ok": False, "error": str(exc)[:200]}

    def paint(i: int) -> Dict[str, Any]:
        say(i, "painting the slide" if i not in painted else "repainting the slide")
        with lock:
            prompt, expect = prompts[i], expected[i]
        result = render_once(prompt)
        if not result.get("ok") and _moderation_refusal(result):
            # The provider's filter refuses the odd harmless slide (a live
            # 15-slide deck lost one); a second painting usually passes, and
            # waiting for the end-of-deck retry left no time to make it.
            logger.info("image slides: slide %d refused by the image filter; painting again", i + 1)
            if on_image:
                on_image(i + 1, False, result.get("error"))
            result = render_once(prompt)
        result["prompt"] = prompt
        if on_image:
            on_image(i + 1, bool(result.get("ok")), result.get("error"))
        if result.get("ok") and result.get("path") and slide_check.unframe_enabled():
            # About 1 slide in 12 came back as a picture of a slide on a page;
            # in the studio that is a band of padding round the slide.
            result["unframed"] = slide_check.unframe(result["path"])
        if checking and i > 0 and result.get("ok") and result.get("path") and expect:
            # Read it now, while the other slides are still being painted.
            reading = slide_check.read_slide(result["path"], **expect)
            if reading is not None:
                with lock:
                    if prompts[i] == prompt:      # not a painting that has gone stale
                        readings[i] = reading
        say(i, "painted" if result.get("ok") else "could not be painted yet; kept as text",
            "done" if result.get("ok") else "warn")
        with lock:
            painted[i] = result
            done = sum(1 for r in painted.values() if r.get("ok"))
        emit("pictures", done=done, total=state["total"], partial=snapshot())
        return result

    def submit(indices) -> None:
        with lock:
            for i in indices:
                if i not in futures:
                    futures[i] = pool.submit(paint, i)

    def sync(slides: Sequence[Dict[str, Any]]) -> bool:
        """Make the shared state show ``slides``; False when the slide count changed."""
        slides = list(slides)
        with lock:
            if planned and len(slides) != len(planned):
                return False
            if not planned:
                state["total"] = len(slides)
                planned.extend({} for _ in slides)
                prompts.extend("" for _ in slides)
                expected.extend(None for _ in slides)
            n = len(slides)
            for i, s in enumerate(slides):
                p = prompt_for(s, i, n)
                planned[i] = s
                if p != prompts[i]:
                    prompts[i] = p
                    expected[i] = _expected_text(s, header)
                    readings.pop(i, None)
            return True

    def on_batch(start: int, slides: List[Dict[str, Any]], deck_size: int) -> None:
        """Paint slides as soon as their batch is written (see planner.plan_deck)."""
        with lock:
            if not planned:
                state["total"] = deck_size
                planned.extend({} for _ in range(deck_size))
                prompts.extend("" for _ in range(deck_size))
                expected.extend(None for _ in range(deck_size))
            if deck_size != state["total"]:
                return
            mine = []
            for j, s in enumerate(slides):
                i = start + j
                if i < deck_size and i not in futures:
                    planned[i] = s
                    prompts[i] = prompt_for(s, i, deck_size)
                    expected[i] = _expected_text(s, header)
                    mine.append(i)
        submit(mine)

    try:
        plan = plan_deck(**plan_args, **({"on_batch": on_batch} if early else {}))
        state["title"] = plan.title or topic
        if not sync(plan.slides):
            raise _Restart(list(plan.slides))
        emit("pictures", done=sum(1 for r in painted.values() if r.get("ok")),
             total=state["total"], partial=snapshot())
        if early:
            # Painting starts as soon as the slides are written - batch by batch
            # where the plan is written in batches. The chapter and fact checks
            # then run alongside it: on a live deck they took 43s and changed 2
            # slides of 8, and painting used to wait for both.
            submit(range(state["total"]))
        # The chapter check, then the facts: painted text cannot be edited
        # afterwards, so a slide either check changes is painted again below.
        plan = scope.replan_if_off_scope(
            plan, revise=lambda note, bad: revise_slides(
                slides=bad, note=note, **_revise_args(plan_args)),
            passages=passages, class_name=ctx.get("className", ""),
            started=started, label="image slides", on_event=on_event)
        from ai_services.core.ppt_v2.service import _fact_check
        _fact_check(plan, passages=passages, topic=topic,
                    chapter=ctx.get("chapterName", "") or ctx.get("topicName", ""),
                    institute_id=institute_id, started=started,
                    on_event=on_event, on_usage=on_planner_usage)
        if not sync(plan.slides):
            # A check that changed the number of slides: nothing painted so far
            # lines up any more. Start the pictures again from the checked plan.
            logger.info("image slides: checks changed the deck from %d to %d slides; "
                        "painting from scratch", state["total"], len(plan.slides))
            raise _Restart(list(plan.slides))
        state["title"] = plan.title or topic
        submit(range(state["total"]))
        # A slide the checks changed is painted again the moment its painting
        # of the old text is done - beside the paintings still in progress: a
        # live deck waited 29s for one such repaint that only began once every
        # other slide was done.
        redo: List[int] = []

        def repaint_if_stale(i: int, fut) -> None:
            try:
                r = fut.result()
            except Exception:
                return
            with lock:
                if (futures.get(i) is fut and r.get("ok") and r.get("prompt") != prompts[i]):
                    futures[i] = pool.submit(paint, i)
                    redo.append(i)
                    logger.info("image slides: repainting slide %d with the checked text", i + 1)
        with lock:
            pending = list(futures.items())
        for i, fut in pending:      # runs at once for a painting that is already done
            fut.add_done_callback(lambda f, i=i: repaint_if_stale(i, f))
        # Wait for every painting, including repaints started while waiting.
        while True:
            with lock:
                waiting = dict(futures)
            for fut in waiting.values():
                fut.result()
            with lock:
                if all(futures[i] is waiting[i] for i in futures):
                    break
        results = [futures[i].result() for i in range(state["total"])]
        # Any still painting the old text when the checks finished.
        late = [i for i, r in enumerate(results) if r.get("ok") and not current(i, r)]
        if late:
            logger.info("image slides: repainting slide(s) %s with the checked text",
                        [i + 1 for i in late])
            for i, r in zip(late, pool.map(paint, late)):
                results[i] = r
        stale = sorted(set(redo) | set(late))
    except _Restart as restart:
        for f in futures.values():
            f.cancel()
        pool.shutdown(wait=True)
        pool = ThreadPoolExecutor(max_workers=workers)
        with lock:
            planned[:] = []
            prompts[:] = []
            expected[:] = []
            painted.clear()
            readings.clear()
            futures.clear()
        sync(restart.slides)
        stale = []
        results = list(pool.map(paint, range(state["total"])))
    finally:
        pool.shutdown(wait=True)
    total = state["total"]
    checking = checking and total >= 3

    # More tries for slides the first pass lost (rate limiting, a provider
    # hiccup, a timeout, a filter that refused twice): a text-only slide in a
    # painted deck is what sends a teacher back to generate the whole deck
    # again. Up to REPAINT_ATTEMPTS rounds, the slides of a round in parallel,
    # while the deck is inside repaint_window(). A "fatal" failure (a rejected
    # key, a model that is not served) would fail the same way again.
    attempts = repaint_attempts()
    if results and not any(r.get("ok") for r in results):
        # Not one slide painted: the image provider is down, not unlucky.
        # Rounds of retries would only delay the standard deck the caller
        # builds instead (see views/ppt._try_generate_image_deck).
        logger.warning("image slides: no slide could be painted; not retrying")
        attempts = 0
    for attempt in range(1, attempts + 1):
        lost = [i for i, r in enumerate(results) if not r.get("ok") and not r.get("fatal")]
        if not lost:
            break
        if time.perf_counter() - started > repaint_window():
            logger.info("image slides: no time left to repaint slide(s) %s", [i + 1 for i in lost])
            break
        if attempt > 1:
            _sleep(REPAINT_PAUSE_S * (attempt - 1))
        logger.info("image slides: repainting slide(s) %s (round %d of %d)",
                    [i + 1 for i in lost], attempt, attempts)
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(lost)))) as retry_pool:
            for i, r in zip(lost, retry_pool.map(paint, lost)):
                results[i] = r

    # Text check: no prompt makes the model keep one type size from slide to
    # slide, so each painted slide was read back (above) and the ones off the
    # deck's own sizes, garbled or missing text get one repaint. The better of
    # the two paintings is kept.
    text_check = {"checked": len(readings), "repainted": [], "kept": [], "skipped": []}
    if checking and len(readings) >= 3 and time.perf_counter() - started < check_window():
        body_t, title_t = slide_check.targets(readings)
        bad = slide_check.judge(readings)
        order = sorted(bad, key=lambda k: -slide_check.score(readings[k], body_t, title_t))
        order = order[:max_check_repaints(total)]
        if on_event:
            try:
                on_event("textcheck", ("Checked the text on every slide; repainting %d for consistency"
                                   % len(order)) if order else
                         "Checked the text on every slide: consistent", "active" if order else "done")
            except Exception:
                logger.exception("image slides: activity event failed")

        def redo(k: int) -> Dict[str, Any]:
            old_result, old_reading = results[k], readings.get(k)
            if time.perf_counter() - started > check_window():
                # Queued behind the others and the deck has run long: as painted.
                logger.info("image slides: slide %d: no time left for a text-check repaint", k + 1)
                text_check["skipped"].append(k + 1)
                return old_result
            logger.info("image slides: repainting slide %d for %s", k + 1, ", ".join(bad[k]))
            new = paint(k)
            new_reading = readings.get(k) if new.get("ok") else None
            if new.get("ok") and new_reading is not old_reading and (
                    slide_check.score(new_reading, body_t, title_t)
                    <= slide_check.score(old_reading, body_t, title_t)):
                logger.info("image slides: slide %d: kept the repaint", k + 1)
                text_check["repainted"].append(k + 1)
                return new
            logger.info("image slides: slide %d: the repaint was no better; kept the first", k + 1)
            with lock:                     # the first painting was better: keep it
                painted[k] = old_result
                readings[k] = old_reading
            text_check["kept"].append(k + 1)
            say(k, "kept the first painting", "done")
            emit("pictures", done=sum(1 for r in painted.values() if r.get("ok")),
                 total=total, partial=snapshot())
            return old_result

        if order:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for k, kept in zip(order, pool.map(redo, order)):
                    results[k] = kept
            if on_event:
                try:
                    on_event("textcheck", "Checked the text on every slide; %d repainted for consistency"
                             % len(text_check["repainted"]), "done")
                except Exception:
                    logger.exception("image slides: activity event failed")

    slides, failed = [], []
    for i, (s, result) in enumerate(zip(planned, results)):
        out = {
            "slideNumber": i + 1,
            "type": "title" if i == 0 else ("summary" if i == total - 1 else "content"),
            "title": s.get("title") or "",
            "subtitle": s.get("subtitle") or "",
            "bullets": _bullets(s),
            # The words are pixels on the slide; the notes keep them editable,
            # searchable and readable by a screen reader.
            "speakerNotes": "\n".join(x for x in (s.get("speakerNotes") or "",
                                                  "\n".join(_bullets(s))) if x),
        }
        if current(i, result):
            out["slideImage"] = {"url": result.get("url") or "", "model": model}
        else:
            # Includes a painting of text a check changed that could not be
            # painted again: the slide stays text, never the unchecked words.
            failed.append(i + 1)
            out["slideImageError"] = str(
                result.get("error") if not result.get("ok")
                else "the checked text could not be painted")[:200]
        slides.append(out)

    seconds = time.perf_counter() - started
    logger.info("image slides: complete | slides=%d painted=%d failed=%s model=%s %.1fs",
                total, total - len(failed), failed or "none", model, seconds)
    return {
        "title": state["title"],
        "slides": slides,
        "generation": {
            "version": "image",
            "plannerModel": plan.model,
            "imageModel": model,
            "imagesGenerated": total - len(failed),
            "failedSlides": failed,
            "grounded": plan.grounded,
            "seconds": round(seconds, 1),
            # Slides read back, repainted for their text, and repaints that
            # were no better (the first painting kept).
            "textCheck": text_check,
            "framesCropped": sum(1 for r in results if r.get("unframed")),
            # Painting started before the chapter and fact checks finished;
            # these slides were changed by them and painted again.
            "paintedEarly": early,
            "repaintedAfterCheck": [i + 1 for i in stale],
        },
    }
