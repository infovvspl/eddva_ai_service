"""Design tokens and subject themes.

Two rules this module exists to enforce:

1. **No slide names a colour.** The planner says "accent"; the theme decides
   that Chemistry's accent is orange and Biology's is blue. Changing how a
   subject looks is then one dict, not a sweep through prompt templates.

2. **Text must be readable on what sits behind it.** Every theme is checked
   against WCAG contrast by the test suite, because a good-looking palette that
   puts mid-grey on mid-green is worse than an ugly one: a student at the back
   of the room simply cannot read it.

Sizes are in points, because every renderer we care about (pptxgenjs today,
python-pptx later) speaks points for type. Everything *positional* stays
normalised and lives in :mod:`ai_services.core.ppt_v2.layouts`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ai_services.core.ppt_v2.schema import SlideType, ThemeSpec
from ai_services.core.ppt_v2.textmatch import matches_any

# --- typography -------------------------------------------------------------
# A ~1.25 scale. Projected from the back of a classroom, body copy below about
# 16pt stops being readable, so the scale bottoms out well above the 11pt a
# printed document would use.
TYPE_SCALE: Dict[str, float] = {
    "display": 44.0,   # title-slide headline
    "title": 32.0,     # slide title
    "subtitle": 20.0,
    "heading": 22.0,   # section or column header inside a slide
    "body": 17.0,
    "caption": 13.0,
    "annotation": 12.0,
    "footer": 10.0,
    "stat": 40.0,      # the one big number on a stat block
}

FONTS: Dict[str, str] = {
    # Kept to fonts PowerPoint resolves without embedding; a missing font
    # silently reflows the whole deck on the teacher's machine.
    "heading": "Segoe UI Semibold",
    "body": "Segoe UI",
    "mono": "Consolas",
}

# --- spacing and shape ------------------------------------------------------
# Normalised to the slide, like layout regions.
SPACING: Dict[str, float] = {
    "margin_x": 0.055,
    "margin_y": 0.070,
    "gutter": 0.030,
    "block_gap": 0.022,
    "inset": 0.018,
}

SHAPE: Dict[str, float] = {
    "radius_sm": 0.006,
    "radius_md": 0.012,
    "radius_lg": 0.022,
    "stroke": 0.0015,
    "shadow_blur": 0.010,
}

# --- palettes ---------------------------------------------------------------
# Token meanings, fixed across every theme:
#   primary       dominant subject colour; headings, strong fills
#   primary_soft  tinted background for a card sitting on `surface`
#   accent        the second colour; for contrast, never for body text
#   accent_soft   tinted background for an accent card
#   ink           body text on `surface`
#   ink_muted     secondary text; must still pass contrast on `surface`
#   on_primary    text drawn on top of a `primary` fill
#   surface       slide background
#   surface_alt   a panel distinct from the background
#   border        hairlines, dividers
_BASE: Dict[str, str] = {
    "ink": "#111827",
    "ink_muted": "#4B5563",
    # `heading` is the colour of heading *text*. On most slides that is the
    # subject's primary colour, but on the title slide the primary colour is
    # the background, so the two tokens have to be separable.
    "on_primary": "#FFFFFF",
    "on_accent": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_alt": "#F4F6F8",
    "border": "#D8DEE6",
}

SUBJECT_THEMES: Dict[str, Dict[str, str]] = {
    "biology": {
        "primary": "#1B7A4B", "primary_soft": "#E6F4EC",
        "accent": "#1E5FA8", "accent_soft": "#E6EFF9",
    },
    "physics": {
        "primary": "#1E4FA3", "primary_soft": "#E7EEF9",
        "accent": "#5B3FB5", "accent_soft": "#EDE9F8",
    },
    "chemistry": {
        "primary": "#5B3FB5", "primary_soft": "#EDE9F8",
        "accent": "#B45309", "accent_soft": "#FBEFE0",
    },
    "mathematics": {
        "primary": "#1E4FA3", "primary_soft": "#E7EEF9",
        # Bright yellow cannot carry white text, so the accent is an amber dark
        # enough to, and the yellow itself lives only in `accent_soft`.
        "accent": "#A16207", "accent_soft": "#FEF6DB",
    },
    "history": {
        "primary": "#8A4B2A", "primary_soft": "#F6EDE7",
        "accent": "#B3261E", "accent_soft": "#FAE9E8",
    },
    "geography": {
        "primary": "#136F63", "primary_soft": "#E4F2F0",
        "accent": "#1E5FA8", "accent_soft": "#E6EFF9",
    },
    "default": {
        "primary": "#1F3A5F", "primary_soft": "#EAEEF4",
        "accent": "#0F766E", "accent_soft": "#E3F2F0",
    },
}

# Plain-language names for each palette. Diffusion models cannot read hex, so
# the image prompt gets words; the renderer gets the hex.
PALETTE_WORDS: Dict[str, str] = {
    "biology": "green and blue",
    "physics": "blue and violet",
    "chemistry": "violet and warm orange",
    "mathematics": "blue and amber",
    "history": "warm brown and deep red",
    "geography": "teal and blue",
    "default": "navy and teal",
    # Studio themes (below). Words for the illustration only: a generated
    # picture sits in its own frame, so it never has to match the slide's
    # background, just not clash with it.
    "dark-professional": "coral red and sky blue",
    "ocean-blue": "ocean blue and aqua",
    "warm-sunset": "coral and warm orange",
    "forest-green": "fresh green and golden yellow",
    "royal-purple": "violet and pink",
    "clean-white": "royal blue and teal",
}

# --- studio themes ----------------------------------------------------------
# The themes a teacher can pick in PPT Studio instead of the subject palette.
# The keys are the studio's own `data-theme` values, and the colours follow
# the studio swatches, so a deck looks like the card that was picked: five of
# the six are dark (a deep background, near-white text, a bright accent), as
# they always were for the older generator.
#
# A dark palette inverts the usual roles. `primary` is bright, because it is
# also heading and label text on the dark background; so the text drawn on a
# `primary` fill (`on_primary`) is dark rather than white. Every pair a
# renderer draws is held to WCAG by the test suite (see TEXT_ON_FILL_PAIRS).
STUDIO_PALETTES: Dict[str, Dict[str, str]] = {
    "dark-professional": {
        "surface": "#1A1A2E", "surface_alt": "#16213E", "border": "#34385A",
        "primary": "#F0566F", "primary_soft": "#2A2340",
        "accent": "#5AB4E8", "accent_soft": "#1E2A47",
        "ink": "#F5F7FA", "ink_muted": "#C2C7D6",
        "on_primary": "#14141F", "on_accent": "#0E1626",
    },
    "ocean-blue": {
        "surface": "#0C2D48", "surface_alt": "#123B5C", "border": "#2B5677",
        "primary": "#4FB3E8", "primary_soft": "#123E60",
        "accent": "#7FD6C4", "accent_soft": "#0F3A52",
        "ink": "#F5F9FC", "ink_muted": "#B8D3E4",
        "on_primary": "#062033", "on_accent": "#062033",
    },
    "warm-sunset": {
        "surface": "#2D132C", "surface_alt": "#3E1A3A", "border": "#5E2D52",
        "primary": "#FF6B5E", "primary_soft": "#45183D",
        "accent": "#FFB35C", "accent_soft": "#40203A",
        "ink": "#FFF7F3", "ink_muted": "#E3C2C4",
        "on_primary": "#2A0E1C", "on_accent": "#2A0E1C",
    },
    "forest-green": {
        "surface": "#1B2A1B", "surface_alt": "#243824", "border": "#3C5A3C",
        "primary": "#5CDB95", "primary_soft": "#22402B",
        "accent": "#E8C468", "accent_soft": "#33361F",
        "ink": "#F4FAF4", "ink_muted": "#BCD3BC",
        "on_primary": "#0F1F12", "on_accent": "#1F1A08",
    },
    "royal-purple": {
        "surface": "#1A1035", "surface_alt": "#24184A", "border": "#43357A",
        "primary": "#C27BF7", "primary_soft": "#2C1A55",
        "accent": "#F28FC4", "accent_soft": "#331A4A",
        "ink": "#F7F4FD", "ink_muted": "#C9BDE6",
        "on_primary": "#170B2E", "on_accent": "#2A0B22",
    },
    "clean-white": {
        "surface": "#FFFFFF", "surface_alt": "#F4F6FA", "border": "#DCE3EC",
        # Blue-700 rather than the studio swatch's lighter blue: the title
        # slide's subtitle is drawn on this colour and needs 4.5:1.
        "primary": "#1D4ED8", "primary_soft": "#EAF1FE",
        "accent": "#0E7490", "accent_soft": "#E3F4F7",
        "ink": "#0F172A", "ink_muted": "#475569",
        "on_primary": "#FFFFFF", "on_accent": "#FFFFFF",
    },
}

# The studio's default card. Not a palette: it means "use the subject's".
SUBJECT_THEME_CHOICE = "subject"

# Every (text, fill, minimum) pair the V2 renderers draw, taken from
# pptV2Preview.js and pptV2.js. Badges and the category pill carry small text,
# so they are held to the body threshold; headings, card labels and the bold
# emphasised words are large or bold enough for the large-text one.
TEXT_ON_FILL_PAIRS: Tuple[Tuple[str, str, float], ...] = (
    ("ink", "surface", 4.5), ("ink_muted", "surface", 4.5),
    ("ink", "primary_soft", 4.5), ("ink", "accent_soft", 4.5),
    ("ink", "surface_alt", 4.5), ("ink_muted", "surface_alt", 4.5),
    ("on_primary", "primary", 4.5), ("on_accent", "accent", 4.5),
    ("heading", "surface", 3.0), ("primary", "surface", 3.0),
    ("primary", "primary_soft", 3.0), ("accent", "surface", 3.0),
    ("accent", "accent_soft", 3.0), ("accent", "primary_soft", 3.0),
    ("on_primary", "heading", 3.0),
)


def studio_theme_key(value: Any) -> Optional[str]:
    """The studio palette a request asked for, or None for the subject's.

    Anything that is not a known palette key, including the studio's own
    "subject" choice, means the subject palette: an unknown value must never
    produce an unstyled deck.
    """
    key = str(value or "").strip().lower()
    return key if key in STUDIO_PALETTES else None


def is_dark(colors: Dict[str, str]) -> bool:
    """True when text has to be light: the background itself is dark."""
    return relative_luminance(colors.get("surface", "#FFFFFF")) < 0.18


# Subjects arrive from the syllabus as free text ("Science", "Social Science",
# "Maths"). Map them onto a palette rather than dropping to a grey default,
# because "Science" is by far the most common value we actually see.
_SUBJECT_ALIASES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("biology", "bio", "life science", "botany", "zoology"), "biology"),
    (("physics", "phy"), "physics"),
    (("chemistry", "chem"), "chemistry"),
    (("math", "maths", "mathematics", "algebra", "geometry", "arithmetic",
      "trigonometry", "statistics"), "mathematics"),
    (("history", "civics", "political science", "polity"), "history"),
    (("geography", "geo", "environmental", "evs", "economics"), "geography"),
    (("social science", "social studies", "sst"), "history"),
    (("science", "general science"), "biology"),
)

# When the subject is the catch-all "Science", the chapter is the only signal
# for which branch of it we are actually in.
_CHAPTER_HINTS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("acid", "base", "salt", "metal", "carbon", "reaction", "element",
      "compound", "atom", "molecul", "periodic", "combustion", "coal",
      "petroleum", "coordination"), "chemistry"),
    (("light", "sound", "force", "friction", "motion", "electric", "magnet",
      "energy", "pressure", "gravit", "current", "lens", "reflect", "refract",
      "star", "universe", "solar"), "physics"),
    (("cell", "tissue", "plant", "animal", "photosynthes", "respiration",
      "reproduct", "microorganism", "crop", "nutrition", "heredity",
      "evolution", "ecosystem", "biodiversity"), "biology"),
)


def theme_key_for_subject(subject: str = "", chapter: str = "") -> str:
    """Best palette for a free-text subject, with the chapter as tie-breaker."""
    subject_l = (subject or "").strip().lower()
    chapter_l = (chapter or "").strip().lower()
    if not subject_l and not chapter_l:
        return "default"

    is_generic_science = "science" in subject_l and not any(
        w in subject_l for w in ("social", "political")
    )
    if is_generic_science and chapter_l:
        for words, key in _CHAPTER_HINTS:
            if matches_any(chapter_l, words):
                return key

    for words, key in _SUBJECT_ALIASES:
        if matches_any(subject_l, words):
            return key
    for words, key in _CHAPTER_HINTS:
        if matches_any(chapter_l, words):
            return key
    return "default"


def build_theme(subject: str = "", chapter: str = "",
                slide_type: SlideType = SlideType.CONCEPT,
                palette: Optional[str] = None) -> ThemeSpec:
    """Resolve the token set for one slide.

    ``palette`` is a studio theme key (see STUDIO_PALETTES). When it names a
    known palette, that palette wins over the subject's; otherwise the subject
    decides, exactly as before.
    """
    studio_key = studio_theme_key(palette)
    key = studio_key or theme_key_for_subject(subject, chapter)
    colors = dict(_BASE)
    colors.update(STUDIO_PALETTES[studio_key] if studio_key else SUBJECT_THEMES[key])
    colors["heading"] = colors["primary"]

    background = "surface"
    if slide_type is SlideType.TITLE and is_dark(colors):
        # A dark theme's cover stays dark, as its studio card shows: a
        # gradient between its two background shades, with the light text
        # it already has. Inverting it onto the bright primary would put
        # dark text on coral and look nothing like the theme that was picked.
        background = "gradient:surface->surface_alt"
        colors["heading"] = colors["ink"]
    elif slide_type is SlideType.TITLE:
        # The cover is the one place a full-bleed colour field reads as
        # deliberate rather than as a slide that lost its content.
        background = "gradient:primary->accent"
        colors["ink"] = colors["on_primary"]
        colors["ink_muted"] = "#E9EEF6"
        colors["surface"] = colors["primary"]
        colors["surface_alt"] = colors["accent"]
        colors["heading"] = colors["on_primary"]
    elif slide_type is SlideType.SUMMARY:
        background = "primary_soft"
        colors["surface"] = colors["primary_soft"]
    elif slide_type is SlideType.QUIZ:
        background = "accent_soft"
        colors["surface"] = colors["accent_soft"]

    # What a picture is framed on. Diagrams and textbook figures are drawn
    # for a white page, often with transparent backgrounds and dark labels,
    # which vanish on a dark panel; so a dark theme mounts them on white,
    # like a printed figure. A light theme keeps its own panel colour.
    colors["image_mat"] = "#FFFFFF" if is_dark(colors) else colors["surface_alt"]

    return ThemeSpec(name=key, colors=colors, fonts=dict(FONTS),
                     sizes=dict(TYPE_SCALE), background=background)


# --- contrast ---------------------------------------------------------------

def _srgb_channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    h = (hex_color or "").lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError("not a hex colour: {!r}".format(hex_color))
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb_channel(r) + 0.7152 * _srgb_channel(g) + 0.0722 * _srgb_channel(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.1 contrast ratio: 1.0 (identical) to 21.0 (black on white)."""
    a, b = relative_luminance(fg), relative_luminance(bg)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


# Projection loses contrast to ambient light, so body text is held to the WCAG
# AA threshold for *normal* text even though slide type is large enough that
# the large-text threshold would technically pass.
MIN_BODY_CONTRAST = 4.5
MIN_LARGE_CONTRAST = 3.0
