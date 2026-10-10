"""Word-boundary keyword matching.

Plain ``"phy" in subject`` is the obvious way to recognise Physics and it is
wrong: it also matches *geogra-phy*. The same trap catches "atom" inside
*anatomy* and "ion" inside almost everything. Since the keyword lists in this
package are the entire basis for choosing a palette and a visual strategy, a
single false substring match silently mis-renders a whole deck.

Matching is prefix-at-a-word-boundary, so a keyword matches the start of a
word and any suffix of it: "molecul" matches "molecules" and "molecular",
"phy" matches "physics" but not "geography".
"""
from __future__ import annotations

import re
from typing import Dict, Pattern, Sequence

_cache: Dict[tuple, Pattern] = {}


def _pattern(words: Sequence[str]) -> Pattern:
    key = tuple(words)
    cached = _cache.get(key)
    if cached is None:
        # Multi-word keywords ("life science") keep their internal spaces, so
        # whitespace in a keyword is matched flexibly.
        parts = [r"\s+".join(re.escape(p) for p in w.split()) for w in words if w]
        cached = re.compile(r"\b(?:%s)" % "|".join(parts)) if parts else re.compile(r"(?!)")
        _cache[key] = cached
    return cached


def matches_any(text: str, words: Sequence[str]) -> bool:
    """True when any keyword starts a word in ``text``."""
    if not text or not words:
        return False
    return bool(_pattern(tuple(words)).search(re.sub(r"\s+", " ", text.lower())))


def first_match(text: str, words: Sequence[str]) -> str:
    """The matched keyword, for logging why a decision was made."""
    if not text or not words:
        return ""
    found = _pattern(tuple(words)).search(re.sub(r"\s+", " ", text.lower()))
    return found.group(0) if found else ""
