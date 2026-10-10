"""Which PPT generator a request gets.

V1 is the default and stays the default. V2 is new, it calls an image
generator that costs money, and it has never rendered a slide in front of a
class - so the burden is on V2 to be switched on, never on V1 to be switched
off.

Three levers, checked in order:

  ``PPT_V2_ENABLED``          master switch. Off means V1, always, whatever
                              else is set. This is the kill switch.
  ``PPT_V2_ROLLOUT_PERCENT``  0-100. Deterministic per deck key, so the same
                              topic does not flip between versions on retry
                              and a teacher does not get two different decks
                              for the same chapter.
  ``pptVersion`` on the body  only honoured when ``PPT_V2_ALLOW_OVERRIDE`` is
                              on. It selects a renderer, never a model or a
                              provider, so it carries no cost or routing
                              authority - but it stays gated anyway, because
                              an un-gated client toggle is how an experiment
                              becomes production by accident.
"""
from __future__ import annotations

import hashlib
import os

V1 = "v1"
V2 = "v2"
# Every slide painted whole by an image model (core.ppt_v2.image_slides).
# Never chosen by rollout - only asked for, through the same gated override,
# so the two styles can be compared side by side.
IMAGE = "image"


def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def _rollout_percent() -> int:
    try:
        value = int((os.getenv("PPT_V2_ROLLOUT_PERCENT", "100") or "100").strip())
    except (TypeError, ValueError):
        return 100
    return max(0, min(100, value))


def _bucket(key: str) -> int:
    """0-99, stable for a given key across processes and restarts."""
    digest = hashlib.sha256((key or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def ppt_version(requested: str = "", deck_key: str = "") -> str:
    """Resolve the generator version for one request.

    ``deck_key`` should identify the deck (topic plus chapter is enough), so
    the rollout bucket is stable for a retry of the same deck.
    """
    if not _env_bool("PPT_V2_ENABLED"):
        return V1

    asked = (requested or "").strip().lower()
    if asked in (V1, V2, IMAGE) and _env_bool("PPT_V2_ALLOW_OVERRIDE", "false"):
        return asked

    percent = _rollout_percent()
    if percent >= 100:
        return V2
    if percent <= 0:
        return V1
    return V2 if _bucket(deck_key) < percent else V1


def ai_image_budget(default: int = 3) -> int:
    """How many generated images one deck may pay for."""
    try:
        value = int((os.getenv("PPT_V2_AI_IMAGE_BUDGET", str(default)) or default))
    except (TypeError, ValueError):
        return default
    return max(0, min(value, 20))
