"""Choosing the images and videos shown under an AI Tutor answer.

Search results are picked by their caption text, which is not enough: a caption
can match the topic while the picture is a worksheet screenshot or a title slide.
Images are therefore checked by Gemini vision before they are shown. Videos are
ranked by title relevance, length (short explainers first) and channel.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from ai_services.core.grounding import _terms

logger = logging.getLogger("ai_services.tutor_media")

# Words that say what the student wants, not what it is about. Ignored when
# deciding whether a question, passage, caption or video title is on-topic.
GENERIC_WORDS = frozenset("""
explain explanation simple simply words word tell give show example examples real life
question questions answer answers quiz practice test please help understand meaning mean
define definition describe difference between detail details short long easy easily
notes note topic chapter lesson class grade student students about more some
kya hai hain kaise kyun batao samjhao
""".split())

_MAX_IMAGE_CANDIDATES = 12
_IMAGE_DOWNLOAD_TIMEOUT = 4
_MAX_IMAGE_BYTES = 1_500_000

# Channels that reliably publish school-level explainers; ranked first.
_EDUCATION_CHANNELS = (
    "khan academy", "magnet brains", "ncert", "byju", "vedantu", "physics wallah",
    "doubtnut", "it's aumsum time", "aumsum", "letstute", "toppr", "next generation science",
    "ted-ed", "crashcourse", "extraclass", "manocha academy", "edumantra", "learnohub",
    "sunlike study", "iprep", "infinity learn", "dronstudy",
)
_PREFERRED_MAX_SECONDS = 20 * 60
_HARD_MAX_SECONDS = 60 * 60


def topic_terms(text: str) -> set:
    """Meaningful words of a question/title: no stopwords, no 'explain'-type words."""
    return _terms(text) - GENERIC_WORDS


# ── Images ───────────────────────────────────────────────────────────────────

def _download(url: str) -> "tuple[bytes, str] | None":
    try:
        resp = requests.get(url, timeout=_IMAGE_DOWNLOAD_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not mime.startswith("image/") or mime == "image/svg+xml" or len(resp.content) > _MAX_IMAGE_BYTES:
            return None
        return resp.content, mime
    except Exception:
        return None


def _vision_prompt(count: int, topic: str, class_name: str) -> str:
    return f"""You are choosing pictures to show a {class_name or 'school'} student who is learning about: "{topic}".
Below are {count} numbered images. Approve an image ONLY if it is a clear diagram, labelled figure,
illustration, chart, map or real photo that directly shows or explains "{topic}".

REJECT an image if it is any of these:
- a screenshot or photo of text: worksheets, question papers, notes, book pages, answer keys
- handwritten notes or letters
- a title slide, poster or card that is mostly text
- a person's face or portrait (unless the topic is about that person)
- a meme, advertisement, or anything unrelated to "{topic}"
- anything not suitable for a child

Return JSON only: {{"approved": [list of approved image numbers, best first]}}"""


def verify_images(images: list, topic: str, class_name: str = "", keep: int = 6) -> list:
    """Images Gemini vision approves as clear, relevant and child-safe, best first.

    Returns [] if Gemini is unavailable or fails — showing no picture is better
    than showing an unchecked one to a child.
    """
    candidates = images[:_MAX_IMAGE_CANDIDATES]
    if not candidates or not topic:
        return []
    try:
        from google.genai import types
        from ai_services.core import gemini_client as gc
        if not gc.is_available():
            return []
    except Exception as exc:
        logger.warning("Tutor image check unavailable: %s", exc)
        return []

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        downloads = list(pool.map(lambda img: _download(img.get("thumbnailUrl") or img.get("imageUrl")), candidates))
    usable = [(img, d) for img, d in zip(candidates, downloads) if d]
    if not usable:
        return []

    contents = [_vision_prompt(len(usable), topic, class_name)]
    for n, (_, (data, mime)) in enumerate(usable, 1):
        contents += [f"Image {n}:", types.Part.from_bytes(data=data, mime_type=mime)]
    try:
        result = gc.generate_with_rotation(
            model=gc.DEFAULT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
            what="tutor image check",
        )
        approved = json.loads(getattr(result, "text", "") or "{}").get("approved") or []
    except Exception as exc:
        logger.warning("Tutor image check failed: %s", exc)
        return []

    chosen, seen = [], set()
    for n in approved:
        if isinstance(n, int) and 1 <= n <= len(usable) and n not in seen:
            seen.add(n)
            chosen.append(usable[n - 1][0])
    logger.info("Tutor image check: %d/%d approved for %r", len(chosen), len(usable), topic)
    return chosen[:keep]


# ── Videos ───────────────────────────────────────────────────────────────────

def duration_seconds(text: str) -> "int | None":
    """'3:36' → 216, '1:02:05' → 3725; None if missing or unparseable."""
    parts = (text or "").strip().split(":")
    if not parts or not all(re.fullmatch(r"\d{1,2}", p) for p in parts) or len(parts) > 3:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds


def rank_videos(videos: list, want_terms: set, limit: int = 3) -> list:
    """Videos whose title shares the topic words, short explainers and education channels first."""
    ranked = []
    for v in videos:
        overlap = len(want_terms & topic_terms(v.get("title") or ""))
        if want_terms and overlap == 0:
            continue
        secs = duration_seconds(v.get("duration"))
        if secs is not None and secs > _HARD_MAX_SECONDS:
            continue
        long_video = secs is not None and secs > _PREFERRED_MAX_SECONDS
        channel = (v.get("channel") or "").lower()
        edu = any(c in channel for c in _EDUCATION_CHANNELS)
        ranked.append(((1 if long_video else 0, 0 if edu else 1, -overlap), v))
    ranked.sort(key=lambda t: t[0])  # stable: keeps Google's order within ties
    return [v for _, v in ranked[:limit]]
