"""The book text an assessment paper is written from.

An assessment used to receive every indexed passage of its chapter (up to the
shared ~30k-token grounding budget) whatever it covered: a topic test on
"Respiration" read the whole "Life Processes" chapter. That is the largest
input of any AI call in the product and it is resent on every regenerate.

This picks a smaller source that still covers what the paper may ask:

* topic test   -> only the topic's own section pages (found from the book's
                  headings by core.chapter_digest), in full;
* chapter test -> the chapter digest (each section's summary and key facts,
                  every fact checked against its page; built once per chapter
                  and cached) plus a small, even share of every section's
                  pages, so short and long answers still read the book's own
                  wording.

Anything unclear keeps today's behaviour: no digest, no headings, a
multi-chapter paper, lecture transcripts mixed in, or the switch off
(ASSESSMENT_SOURCE_OPTIMIZE). It only ever makes the source smaller.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Pages kept alongside the digest for a chapter test, shared evenly across its
# sections (the digest carries the facts; these carry the book's wording for
# short and long answers).
_DEFAULT_CHAPTER_PAGE_TOKENS = 4500
# A topic section this small gets the digest's view of the chapter too, so a
# paper on a one-paragraph section is not starved of material.
_THIN_SECTION_TOKENS = 600

_MULTI_CHAPTER = re.compile(r"^\s*\d+\s+chapters\s*$", re.I)


def enabled() -> bool:
    return (os.getenv("ASSESSMENT_SOURCE_OPTIMIZE", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def chapter_page_tokens() -> int:
    try:
        return max(1000, int(os.getenv("ASSESSMENT_CHAPTER_PAGE_TOKENS", _DEFAULT_CHAPTER_PAGE_TOKENS)))
    except (TypeError, ValueError):
        return _DEFAULT_CHAPTER_PAGE_TOKENS


def passage_tokens(p: Dict[str, Any]) -> int:
    try:
        return int(p.get("tokens") or 0) or len(p.get("content") or "") // 4
    except (TypeError, ValueError):
        return len(p.get("content") or "") // 4


def total_tokens(passages: Optional[Sequence[Any]]) -> int:
    return sum(passage_tokens(p) for p in passages or [] if isinstance(p, dict))


def clean_passage(p: Dict[str, Any]) -> Dict[str, Any]:
    """The passage with faux-bold text collapsed ("55555.....22222" -> "5.2").

    Chapters indexed before core.textbook learned to undo faux-bold still hold
    every heading as repeated glyphs, so no section could be found in them and
    a topic paper read the whole chapter (seen on dev: "Life Processes" came
    out as four page groups). Cleaning here fixes them without re-reading the
    book, and the model reads a fifth of the characters for those words.
    """
    from ai_services.core.textbook import undo_faux_bold
    text = p.get("content") or ""
    cleaned = undo_faux_bold(text)
    if cleaned == text:
        return p
    return {**p, "content": cleaned, "tokens": len(cleaned) // 4}


def _page(p: Dict[str, Any]) -> int:
    try:
        return int(p.get("page_no") or -1)
    except (TypeError, ValueError):
        return -1


def _in_section(passages: Sequence[Dict[str, Any]], section: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = section.get("pages") or []
    if len(pages) != 2:
        return []
    lo, hi = pages
    return [p for p in passages if lo <= _page(p) <= hi]


def even_share(passages: Sequence[Dict[str, Any]], sections: Sequence[Dict[str, Any]],
               budget: int) -> List[Dict[str, Any]]:
    """Up to ``budget`` tokens of pages, an equal share from every section.

    Each section gives its passages in reading order until its share is spent,
    and always at least one, so no section of the chapter goes unread. The
    result is back in reading order.
    """
    with_pages = [s for s in sections if _in_section(passages, s)]
    if not with_pages:
        return []
    share = max(1, budget // len(with_pages))
    chosen: Dict[int, Dict[str, Any]] = {}
    for s in with_pages:
        spent = 0
        for p in sorted(_in_section(passages, s), key=lambda x: (_page(x), x.get("chunk_index") or 0)):
            if spent and spent + passage_tokens(p) > share:
                break
            chosen[id(p)] = p
            spent += passage_tokens(p)
    return [p for p in passages if id(p) in chosen]


def _full(passages, reason: str) -> Dict[str, Any]:
    n = total_tokens(passages)
    return {"passages": list(passages or []), "digest_text": "", "mode": "full", "reason": reason,
            "tokens_before": n, "tokens_after": n, "coverage": None}


def coverage(digest: Dict[str, Any], section: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """What the paper must cover, from the book: a topic section's key terms,
    or every titled section of the chapter. None when the digest has too little.

    This replaces a separate model call that listed a topic's sub-areas from
    general knowledge - one more round trip before the paper, and a list that
    could name things the school's book does not teach.
    """
    if section is not None:
        terms = [t for t in (section.get("key_terms") or []) if isinstance(t, str) and t.strip()]
        return terms[:8] if len(terms) >= 3 else None
    sections = digest.get("sections") or []
    titles = [s["title"].strip().title() for s in sections
              if s.get("number") and (s.get("title") or "").strip()]
    if len(titles) >= 2:
        return titles
    # No numbered headings (a book split into page groups): the groups' key
    # terms, taken in turn so every part of the chapter is represented.
    terms: List[str] = []
    lists = [[t for t in (s.get("key_terms") or []) if isinstance(t, str) and t.strip()] for s in sections]
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x) and x[i].lower() not in {t.lower() for t in terms}:
                terms.append(x[i])
    terms = terms[:8]
    return terms if len(terms) >= 3 else None


def select(passages: Optional[Sequence[Any]], *, topic: str = "", chapter: str = "",
           class_name: str = "", subject: str = "", institute_id: Any = None,
           on_usage: Any = None, digest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The source for one assessment paper.

    Returns ``passages`` (what goes in the SOURCE TEXT), ``digest_text``
    (extra checked facts to put before them, possibly empty), ``mode``
    ("topic_section" | "chapter_digest" | "full"), ``reason`` and the token
    counts before and after. Never raises: any failure is the full source.
    """
    passages = [clean_passage(p) for p in passages or [] if isinstance(p, dict)]
    if not passages:
        return _full(passages, "no passages")
    if not enabled():
        return _full(passages, "switched off")
    if _MULTI_CHAPTER.match(chapter or ""):
        return _full(passages, "multi-chapter paper")
    if any(p.get("source") == "lecture" for p in passages):
        return _full(passages, "lecture transcript in the source")

    try:
        from ai_services.core import chapter_digest
        if digest is None:
            digest = chapter_digest.cached(passages)
            if digest is None:
                # Never make the teacher wait for it: this paper reads the whole
                # chapter, as before, and the digest is ready for the next one.
                started = chapter_digest.build_in_background(
                    passages, chapter=chapter, class_name=class_name, subject=subject,
                    institute_id=institute_id, on_usage=on_usage)
                return _full(passages, "digest building" if started else "no digest")
        if not digest or not digest.get("sections"):
            return _full(passages, "no digest")
        before = total_tokens(passages)

        section = chapter_digest.focus_section(digest, topic) if topic else None
        if section is not None:
            pages = _in_section(passages, section)
            if pages:
                thin = total_tokens(pages) < _THIN_SECTION_TOKENS
                return {
                    "passages": pages,
                    "digest_text": chapter_digest.format_for_prompt(digest, topic) if thin else "",
                    "coverage": coverage(digest, section),
                    "mode": "topic_section",
                    "reason": "section %s %s" % (section.get("number") or "", section.get("title") or ""),
                    "tokens_before": before,
                    "tokens_after": total_tokens(pages),
                }

        pages = even_share(passages, digest["sections"], chapter_page_tokens())
        if not pages:
            return _full(passages, "sections hold no passages")
        digest_text = chapter_digest.format_for_prompt(digest, "")
        after = total_tokens(pages) + len(digest_text) // 4
        if after >= before:
            return _full(passages, "chapter already small")
        return {"passages": pages, "digest_text": digest_text, "mode": "chapter_digest",
                "coverage": coverage(digest, None),
                "reason": "%d sections" % len(digest["sections"]),
                "tokens_before": before, "tokens_after": after}
    except Exception:
        logger.exception("assessment source: selection failed; using every passage")
        return _full(passages, "selection failed")


def digest_block(digest_text: str) -> str:
    """The digest as a labelled part of the SOURCE TEXT."""
    if not digest_text:
        return ""
    return (
        "CHAPTER DIGEST — the chapter's sections with their key facts, each fact checked "
        "against the page shown. These are facts from the book; use them for objective "
        "questions and cite the page given.\n\n" + digest_text + "\n\nBOOK PAGES —\n"
    )
