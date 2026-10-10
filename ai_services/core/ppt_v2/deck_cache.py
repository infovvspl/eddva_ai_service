"""Finished decks, kept so the same request is answered at once.

Schools on one board teach the same chapters from the same books. Once a deck
for "CBSE Class 10 Science, Life Processes, Respiration, 10 slides, image
slides, Ocean Blue" has been made from a given text, the next teacher asking
for exactly that gets it in a second instead of waiting two minutes for the
same work - and the provider's limits are spent on decks nobody has yet.

A deck is reused only for the same request (see ``match_mode``):

  * the same curriculum names, language and slide style - and, in ``exact``
    mode, the same slide count and theme;
  * the same source text - the key carries a fingerprint of the textbook
    passages, so a school with a different book (or a re-indexed chapter)
    never gets a deck written from someone else's pages;
  * by default across schools (``PPT_DECK_CACHE_SCOPE=global``); set it to
    ``institute`` to share decks only within a school.

Only complete decks are kept: the style that was asked for, every slide
painted, no fallback. "Make a fresh one" (``fresh``) skips the lookup and
replaces what is kept. Decks whose generated images have since been removed
from disk are dropped rather than served with broken pictures.

Redis-backed; without Redis nothing is cached and every deck is generated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# Bump to retire every kept deck after a format change. 2: decks capped at 10
# slides, and image decks always 10 (2026-10-09).
VERSION = 2
_PREFIX = "pptdeck:v%d:" % VERSION
_IMAGE_PATH = re.compile(r"/generated-note-images/([A-Za-z0-9._-]+)")


def enabled() -> bool:
    return (os.getenv("PPT_DECK_CACHE", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def ttl_s() -> int:
    """How long a deck is kept (PPT_DECK_CACHE_TTL_DAYS, default 30 days)."""
    try:
        days = float(os.getenv("PPT_DECK_CACHE_TTL_DAYS", "") or 30)
    except ValueError:
        days = 30
    return max(60, int(days * 86400))


def match_mode() -> str:
    """What makes two requests "the same deck" (PPT_DECK_CACHE_MATCH).

    ``topic`` (default, chosen for the school demos, 2026-10-09): the same
    topic, slide style, language and book - a deck made at 8 slides in Ocean
    Blue answers a request for 10 in Clean White, and "Make a fresh one" makes
    the deck as asked. ``exact``: every setting must match as well.
    """
    mode = (os.getenv("PPT_DECK_CACHE_MATCH", "topic") or "topic").strip().lower()
    return mode if mode in ("topic", "exact") else "topic"


def _shared_across_schools() -> bool:
    return (os.getenv("PPT_DECK_CACHE_SCOPE", "global") or "global").strip().lower() != "institute"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def source_fingerprint(passages: Optional[Iterable[Any]]) -> str:
    """A hash of the text a deck is written from; empty when there is none."""
    items = []
    for p in passages or []:
        if isinstance(p, Mapping):
            items.append([str(p.get("page_no") or ""), " ".join(str(p.get("content") or "").split())])
        else:
            items.append(["", " ".join(str(p).split())])
    if not items:
        return ""
    items.sort()
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


def key_for(data: Mapping[str, Any], *, board: Any = "", vertical: Any = "",
            institute_id: Any = "") -> str:
    """The cache key for a deck request, as the generator would read it."""
    count = _norm(data.get("slideCount") or 5)
    if count != "auto":
        try:
            count = str(int(float(count)))
        except ValueError:
            count = "5"
    loose = match_mode() == "topic"
    parts = {
        "match": match_mode(),
        "topic": _norm(data.get("topic")),
        "className": _norm(data.get("className")),
        "subjectName": _norm(data.get("subjectName")),
        "chapterName": _norm(data.get("chapterName")),
        "topicName": _norm(data.get("topicName")),
        "slideCount": "*" if loose else count,
        "language": _norm(data.get("language") or "English"),
        "style": _norm(data.get("pptVersion")),
        "theme": "*" if loose else _norm(data.get("deckTheme")),
        "board": _norm(board),
        "vertical": _norm(vertical),
        "source": source_fingerprint(data.get("sourcePassages")),
        "school": "" if _shared_across_schools() else _norm(institute_id),
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return _PREFIX + digest


def complete(deck: Any, requested_style: Any = "") -> bool:
    """Worth keeping: the asked-for style, every slide whole, nothing fallen back."""
    if not isinstance(deck, Mapping):
        return False
    slides = deck.get("slides")
    if not isinstance(slides, list) or not slides:
        return False
    gen = deck.get("generation") if isinstance(deck.get("generation"), Mapping) else {}
    if gen.get("fallback") or gen.get("partial") or gen.get("failedSlides") or gen.get("cached"):
        return False
    style = _norm(requested_style)
    if style in ("v2", "image") and _norm(gen.get("version")) != style:
        return False
    if style == "image" and any(not (s.get("slideImage") or {}).get("url") for s in slides
                                if isinstance(s, Mapping)):
        return False
    return True


def _images_dir() -> Path:
    from ai_services.core.image_generation import _generated_images_dir
    return _generated_images_dir()


def _images_present(deck: Mapping[str, Any]) -> bool:
    """Every generated image the deck points at is still on this server's disk."""
    names = set(_IMAGE_PATH.findall(json.dumps(deck, ensure_ascii=False)))
    if not names:
        return True
    folder = _images_dir()
    return all((folder / name).is_file() for name in names)


# get_redis() tries a fresh connection (a 1s timeout) on every call while Redis
# is down; a deck must not pay that twice. Remember the outage for a while.
_RECHECK_S = 30.0
_down_until = 0.0


def _client():
    global _down_until
    if time.monotonic() < _down_until:
        return None
    from ai_services.core.cache import get_redis
    client = get_redis()
    if client is None:
        _down_until = time.monotonic() + _RECHECK_S
    return client


def get(key: str, max_slides: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The kept deck for ``key``, marked as reused; None on a miss.

    A deck longer than ``max_slides`` (kept before the cap was lowered) is
    dropped rather than served.
    """
    if not enabled() or not key:
        return None
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception as exc:
        logger.warning("deck cache: read failed (%s)", exc)
        return None
    if not raw:
        return None
    try:
        entry = json.loads(raw)
        deck = entry["deck"]
    except (ValueError, KeyError, TypeError):
        logger.warning("deck cache: unreadable entry %s; dropped", key[-12:])
        _drop(client, key)
        return None
    if max_slides and len(deck.get("slides") or []) > max_slides:
        logger.info("deck cache: %s has more than %d slides; dropped", key[-12:], max_slides)
        _drop(client, key)
        return None
    if not _images_present(deck):
        logger.info("deck cache: %s points at images no longer on disk; dropped", key[-12:])
        _drop(client, key)
        return None
    deck = copy.deepcopy(deck)
    gen = deck.setdefault("generation", {})
    gen["cached"] = True
    gen["cachedAt"] = entry.get("savedAt")
    return deck


def put(key: str, deck: Any, requested_style: Any = "") -> bool:
    """Keep a finished deck when it is complete. True when kept."""
    if not enabled() or not key or not complete(deck, requested_style):
        return False
    client = _client()
    if client is None:
        return False
    try:
        payload = json.dumps({"deck": deck, "savedAt": time.time()}, ensure_ascii=False)
        client.setex(key, ttl_s(), payload)
        return True
    except Exception as exc:
        logger.warning("deck cache: write failed (%s)", exc)
        return False


def _drop(client, key: str) -> None:
    try:
        client.delete(key)
    except Exception:
        pass
