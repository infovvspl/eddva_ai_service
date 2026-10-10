"""Check painted slides after they are painted: text size against the deck, and garbling.

An image model paints each slide separately, and no prompt makes it keep the
same type sizes from one image to the next: a live 8-slide deck measured card
text from 3.8% to 5.7% of the slide height, and one summary card came back as
gibberish. So each painted slide is read back with OCR and compared with the
deck it belongs to:

  * size  - the height of the lines that match the slide's own card text (and
            title), against the median of the deck;
  * junk  - the share of read words that match nothing the slide was asked to
            say (gibberish, invented cards);
  * coverage - the share of the requested card words that were found at all.

A slide that is off gets one repaint (see image_slides.generate_image_deck);
the better of the two is kept. OCR is easyocr on the CPU, at 1,024 px wide:
about 6s a slide, run as each slide is painted so most of it overlaps the
painting of the others. When OCR is not available the check is skipped and
the deck is returned as painted.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

READ_WIDTH = 1024
_reader: Any = None
_reader_failed = False
_lock = threading.Lock()


def enabled() -> bool:
    return (os.getenv("PPT_IMAGE_SLIDE_CHECK", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def size_tolerance() -> float:
    """How far (as a fraction) card text may sit from the deck's median size.

    OCR heights come in steps of about 8% (one pixel at READ_WIDTH), so 0.15
    allows one step and flags two. At 0.22, replayed on three live decks, it
    let through a two-card slide whose text was visibly bigger than the rest.
    """
    return _float_env("PPT_IMAGE_CHECK_SIZE_TOL", 0.15)


def title_tolerance() -> float:
    return _float_env("PPT_IMAGE_CHECK_TITLE_TOL", 0.18)


def junk_limit() -> float:
    """The share of read words matching nothing asked for that marks a slide garbled."""
    return _float_env("PPT_IMAGE_CHECK_JUNK", 0.30)


def coverage_floor() -> float:
    """The share of requested card words that must be found on the slide."""
    return _float_env("PPT_IMAGE_CHECK_COVERAGE", 0.45)


@dataclass
class Reading:
    """What OCR found on one painted slide. Heights are % of the slide height."""

    title_h: Optional[float] = None
    body_h: Optional[float] = None
    coverage: float = 1.0
    junk: float = 0.0
    tokens: int = 0
    reasons: List[str] = field(default_factory=list)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]{3,}", (text or "").lower())


def _matches(word: str, vocab: Sequence[str]) -> bool:
    if word in vocab:
        return True
    # OCR misreads a letter or two of real text; gibberish is further off. One
    # wrong letter in a short word ("bodv" for "body") is already 0.75.
    cutoff = 0.74 if len(word) <= 5 else 0.8
    return bool(difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff))


def _get_reader():
    global _reader, _reader_failed
    if _reader is not None or _reader_failed:
        return _reader
    try:
        import easyocr  # heavy import (torch); only when a deck is checked
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:  # no OCR here: the deck is returned as painted
        logger.warning("slide check: OCR unavailable (%s); painted slides are not checked", exc)
        _reader_failed = True
    return _reader


def _ocr(path: str) -> Optional[List[tuple]]:
    """(top fraction, height %, text) for every line read, or None without OCR."""
    reader = _get_reader()
    if reader is None:
        return None
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        small = im.resize((READ_WIDTH, max(1, int(READ_WIDTH * im.height / im.width))))
    h = small.height
    with _lock:  # one model, one CPU pool: reads queue rather than contend
        found = reader.readtext(np.array(small))
    lines = []
    for pts, text, conf in found:
        ys = [p[1] for p in pts]
        if conf < 0.2 or not text.strip():
            continue
        lines.append((min(ys) / h, (max(ys) - min(ys)) / h * 100, text))
    return lines


def read_slide(path: str, *, title: str, cards: Iterable[str], other: Iterable[str] = (),
               headings: Iterable[str] = (), ocr=None) -> Optional[Reading]:
    """Read one painted slide against the text it was asked to show.

    ``cards`` are the card headings and lines; ``headings`` the headings alone,
    so card text is measured without them: bold headings are bigger, and on a
    two-card slide they are a larger share of the lines, which hid a live slide
    whose text was a fifth bigger than the deck's. ``other`` is text that may
    also be on the slide (subtitle, header pill, illustration labels, quiz
    letters) and so is not junk. ``ocr`` replaces the OCR call in tests.
    """
    try:
        lines = (ocr or _ocr)(path)
    except Exception:
        logger.exception("slide check: could not read %s", path)
        return None
    if lines is None:
        return None
    title_vocab = sorted(set(_words(title)))
    card_vocab = sorted(set(w for c in cards for w in _words(c)))
    heading_vocab = sorted(set(w for h in headings for w in _words(h)))
    vocab = sorted(set(title_vocab) | set(card_vocab) | set(w for o in other for w in _words(o)))

    title_heights, card_heights, found, total, junk = [], [], set(), 0, 0
    for top, height, text in lines:
        words = _words(text)
        if not words:
            continue
        total += len(words)
        matched = [w for w in words if _matches(w, vocab)]
        junk += sum(1 for w in words if len(w) >= 4 and w not in matched)
        in_title = sum(1 for w in words if _matches(w, title_vocab))
        in_cards = sum(1 for w in words if _matches(w, card_vocab))
        if top < 0.30 and title_vocab and in_title >= max(1, 0.6 * len(words)):
            title_heights.append(height)
        elif card_vocab and in_cards >= max(1, 0.5 * len(words)):
            found.update(w for w in words if _matches(w, card_vocab))
            # A short line made of heading words is a heading, not card text.
            is_heading = (heading_vocab and len(words) <= 6
                          and all(_matches(w, heading_vocab) for w in words))
            if not is_heading:
                card_heights.append(height)

    wanted = [w for w in card_vocab if len(w) >= 4]
    covered = sum(1 for w in wanted if _matches(w, sorted(found))) if found else 0
    return Reading(
        title_h=max(title_heights) if title_heights else None,
        body_h=statistics.median(card_heights) if card_heights else None,
        coverage=(covered / len(wanted)) if wanted else 1.0,
        junk=(junk / total) if total else 0.0,
        tokens=total,
    )


def unframe_enabled() -> bool:
    return (os.getenv("PPT_IMAGE_UNFRAME", "true") or "").strip().lower() in ("1", "true", "yes", "on")


_FRAME_W = 384          # frames are found on a small copy; the crop is made on the full image


def find_frame(pixels) -> Optional[tuple]:
    """(left, top, right, bottom) of a slide painted as a picture of a slide.

    The prompt asks for no page or mock-up frame, and a live cover still came
    back as a white slide on a grey page, shown in the studio with grey bars
    all round. A frame is a band of one colour on every side, enclosing a
    rectangle whose four edges are solid lines of another colour. A slide whose
    content merely sits on a plain background has no such edges, so it is left
    alone. ``pixels`` is an RGB array, height x width x 3.
    """
    import numpy as np

    a = np.asarray(pixels, dtype=np.int16)
    h, w = a.shape[:2]
    if h < 20 or w < 20:
        return None
    ring = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3),
                           a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)])
    edge = np.median(ring, axis=0)
    if (np.abs(ring - edge).max(axis=1) <= 18).mean() < 0.95:
        return None                                   # the border is not one colour
    inside = np.abs(a - edge).max(axis=2) > 30        # pixels that are not the border colour
    rows = np.where(inside.mean(axis=1) > 0.5)[0]
    cols = np.where(inside.mean(axis=0) > 0.5)[0]
    if not len(rows) or not len(cols):
        return None
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if (right - left) * (bottom - top) < 0.5 * w * h:
        return None
    if left < 0.015 * w and w - right < 0.015 * w and top < 0.015 * h and h - bottom < 0.015 * h:
        return None                                   # no band worth removing
    # The enclosed rectangle's edges, a few pixels in (corners may be rounded).
    k = max(2, int(0.01 * min(w, h)))
    span_x, span_y = slice(left + 3 * k, right - 3 * k), slice(top + 3 * k, bottom - 3 * k)
    edges = (inside[top + k, span_x], inside[bottom - 1 - k, span_x],
             inside[span_y, left + k], inside[span_y, right - 1 - k])
    if min(e.mean() for e in edges) < 0.9:
        return None                                   # content on a plain page, not a slide on a page
    # A page is empty: nothing outside the rectangle but its shadow. A dark
    # slide whose title sits above one big pale panel has the same rectangle,
    # and cropping to it would cut the title off.
    pad = max(3, int(0.03 * min(w, h)))
    outside = np.ones_like(inside)
    outside[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad] = False
    if outside.any() and inside[outside].mean() > 0.02:
        return None
    return int(left), int(top), int(right), int(bottom)


def unframe(path: str) -> bool:
    """Crop a painted mock-up frame away, in place, keeping the image size. True if cropped."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
        scale = im.width / _FRAME_W
        small = im.resize((_FRAME_W, max(1, round(im.height / scale))))
        box = find_frame(small)
        if not box:
            return False
        left, top, right, bottom = (round(v * scale) for v in box)
        # Inset past the slide's own edge and shadow, then the largest centred
        # box of the image's proportions, so nothing is stretched.
        inset = round(0.006 * im.width)
        left, top, right, bottom = left + inset, top + inset, right - inset, bottom - inset
        ratio = im.width / im.height
        cw, ch = right - left, bottom - top
        if cw / ch > ratio:
            cw = round(ch * ratio)
        else:
            ch = round(cw / ratio)
        cx, cy = (left + right) // 2, (top + bottom) // 2
        crop = im.crop((cx - cw // 2, cy - ch // 2, cx - cw // 2 + cw, cy - ch // 2 + ch))
        crop.resize(im.size, Image.LANCZOS).save(path)
        logger.info("slide check: cropped a painted page frame from %s", os.path.basename(path))
        return True
    except Exception:
        logger.exception("slide check: could not check %s for a frame", path)
        return False


def _target(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v]
    return statistics.median(vals) if len(vals) >= 2 else None


def judge(readings: Dict[int, Reading]) -> Dict[int, List[str]]:
    """The slides that should be repainted, with why. Sizes are against the deck."""
    body_t = _target(r.body_h for r in readings.values())
    title_t = _target(r.title_h for r in readings.values())
    bad: Dict[int, List[str]] = {}
    for i, r in readings.items():
        why = []
        if r.tokens >= 6 and r.junk > junk_limit():
            why.append("garbled text")
        if r.coverage < coverage_floor():
            why.append("missing text")
        if body_t and r.body_h and abs(r.body_h / body_t - 1) > size_tolerance():
            why.append("card text %s" % ("too big" if r.body_h > body_t else "too small"))
        if title_t and r.title_h and abs(r.title_h / title_t - 1) > title_tolerance():
            why.append("title %s" % ("too big" if r.title_h > title_t else "too small"))
        if why:
            bad[i] = why
    return bad


def score(r: Optional[Reading], body_t: Optional[float], title_t: Optional[float]) -> float:
    """Lower is better: distance from the deck's sizes, plus garbling and missing text."""
    if r is None:
        return 10.0
    s = r.junk * 2 + (1 - r.coverage)
    if body_t and r.body_h:
        s += abs(r.body_h / body_t - 1)
    if title_t and r.title_h:
        s += abs(r.title_h / title_t - 1) * 0.5
    return s


def targets(readings: Dict[int, Reading]):
    return (_target(r.body_h for r in readings.values()),
            _target(r.title_h for r in readings.values()))
