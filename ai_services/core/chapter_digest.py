"""Chapter digest: a compact, page-cited summary of a chapter, built once.

Every PPT, DPP and assessment used to send the chapter's raw text to the model
on every call. The steps that only PLAN (the deck outline, a paper's
blueprint) do not need the wording, only what the chapter teaches and where:
its sections, their pages, their key facts and terms. That fits in a fraction
of the chapter's 10-15k tokens.

How it is built, and why this way:
  * Sections are found in CODE, from the chapter's own numbered headings
    ("5.3 RESPIRATION"), never by the model. A first version let the model
    choose the sections; on a real NCERT chapter it skipped "5.3 Respiration"
    entirely, which would have hidden the topic from every deck planned from
    the digest. A chapter without numbered headings is split into page groups.
  * One small model call per section (at most SECTION_TOKEN_CAP tokens). A
    whole-chapter call (~11k tokens) cannot fall back to Groq, whose limit is
    8k tokens per minute, and Together was returning 503s on half its calls
    when this was written. Small calls also guarantee every section is read.
  * Every key fact is checked in code: it is kept only when most of its words
    appear on one of its section's pages, on the page that supports it best.
    The digest cannot carry a fact the book does not state.
  * All-or-nothing: if any section comes back unusable the digest is not used,
    because a missing section would be invisible to planning. The caller then
    reads the chapter text, exactly as before.

Steps that WRITE or CHECK wording still read the real pages. Cached by a
fingerprint of the chapter's text (Redis when available, this process
otherwise); the production home is the school database, built when the book
is read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ai_services.chapter_digest")

VERSION = 2
CACHE_TTL_S = 30 * 24 * 3600
# A fact is kept when this share of its content words appears on one page.
FACT_SUPPORT = 0.6
MAX_FACTS_PER_SECTION = 12
MAX_TERMS_PER_SECTION = 10
SECTION_TOKEN_CAP = 4500   # one call per section; fits the Groq fallback's 8k TPM
PAGE_GROUP_TOKENS = 3500   # chapters without numbered headings
MAX_PARALLEL = 3
RETRY_PAUSE_S = 3.0

_STOP = frozenset("""a an the of in on and or to is are was were be been being by for with as at
from that this it its into than then which who whom whose also can may more most such these those
their there they them we our you your he she his her not no do does did has have had will would
shall should could one two three each other all any some only very so if when where while how what
why because about over under between during after before up down out off again further""".split())

# A numbered heading in chapter text: "5.3 RESPIRATION", "5.2.1 Autotrophic
# Nutrition". Figure/activity/table numbers look alike and are excluded by the
# word in front of them. Only top-level numbers (x.y) start a section.
_NOT_AFTER = r"(?<![\w.])(?<!Figure )(?<!Fig\. )(?<!Activity )(?<!Table )(?<!Example )(?<!Exercise )"
# Capitalised titles (NCERT and most Indian textbooks): the title is the run of
# capitalised words, so "5.3 RESPIRATION Pichkari Activity 5.4" and
# "5.4 TRANSPORTATION 5.4.1 ..." both read correctly.
_HEADING_CAPS = re.compile(
    _NOT_AFTER + r"(\d{1,2}\.\d{1,2})(\.\d{1,2})?\s+((?:[A-Z][A-Z'\-]*[A-Z?]?)(?:\s+[A-Z][A-Z'\-]*[A-Z?]?)*)"
    r"(?=\s|$)")
# Mixed-case titles (other publishers): the title runs until a sentence starts.
_HEADING_MIXED = re.compile(
    _NOT_AFTER + r"(\d{1,2}\.\d{1,2})(\.\d{1,2})?\s+([A-Z][A-Za-z?'\- ]{2,70}?)"
    r"(?=\s+[A-Z][a-z]+\s+[a-z]|\s+\d|\s{2,}|\n|$)")
_MAX_HEADING_WORDS = 9

_SECTION_SYSTEM = """You summarise one section of a school textbook chapter for lesson planning.
Use ONLY the text given. Each key fact must be stated in the text and carry the
page number [p.N] it appears on. Do not add facts, numbers, examples or terms
the text does not contain.

Return JSON exactly in this shape:
{"summary": "<1-2 sentences: what this section teaches>",
 "key_facts": [{"fact": "<one short sentence in the book's own terms>", "page": <page>}],
 "key_terms": ["<term>", ...]}

Up to 12 key facts and 10 key terms: the ones a lesson or a test on this section
needs. Keep each fact under 25 words."""

_memory: Dict[str, Dict[str, Any]] = {}
_memory_lock = threading.Lock()


def enabled() -> bool:
    return (os.getenv("CHAPTER_DIGEST", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in _STOP]


def _tokens(p: Dict[str, Any]) -> int:
    return p.get("tokens") or max(1, len(p.get("content", "")) // 4)


def _ordered(passages: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    return sorted((p for p in (passages or []) if isinstance(p, dict) and (p.get("content") or "").strip()),
                  key=lambda p: (p.get("page_no") or 0, p.get("chunk_index") or 0))


def fingerprint(passages: Optional[Sequence[Any]]) -> str:
    """A hash of the chapter text the digest is built from; empty without text."""
    items = sorted([str(p.get("page_no") or ""), " ".join(str(p.get("content") or "").split())]
                   for p in _ordered(passages))
    if not items:
        return ""
    return hashlib.sha256(json.dumps([VERSION, items], ensure_ascii=False).encode("utf-8")).hexdigest()


def _headings(text: str, pattern: "re.Pattern[str]") -> List[Tuple[int, str, str]]:
    """(position, number, title) of each top-level numbered heading in ``text``."""
    out = []
    for m in pattern.finditer(text or ""):
        if m.group(2):                       # x.y.z is a subsection: stays inside its section
            continue
        title = " ".join(m.group(3).split()).rstrip(" .:")
        if len(re.sub(r"[^A-Za-z]", "", title)) >= 3 and len(title.split()) <= _MAX_HEADING_WORDS:
            out.append((m.start(), m.group(1), title))
    return out


def _heading_pattern(ordered: Sequence[Dict[str, Any]]) -> "re.Pattern[str]":
    """Capitalised headings when the chapter has at least two; mixed-case otherwise."""
    numbers = {n for p in ordered for _, n, _ in _headings(p.get("content", ""), _HEADING_CAPS)}
    return _HEADING_CAPS if len(numbers) >= 2 else _HEADING_MIXED


def detect_sections(passages: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """The chapter's top-level sections, found in code.

    Each: {"number", "title", "pages": [first, last], "passages": [...]}. A
    passage holding a heading mid-way is split there, so the text before the
    heading stays with the previous section. With fewer than two distinct
    headings the chapter is split into page groups, so every page is covered.
    """
    ordered = _ordered(passages)
    if not ordered:
        return []
    pattern = _heading_pattern(ordered)
    pieces: List[Tuple[Optional[Tuple[str, str]], Dict[str, Any]]] = []
    for p in ordered:
        text = p.get("content", "")
        last, heading = 0, None
        for pos, number, title in _headings(text, pattern):
            if text[last:pos].strip():
                pieces.append((heading, dict(p, content=text[last:pos])))
            heading, last = (number, title), pos
        pieces.append((heading, dict(p, content=text[last:])))

    if len({h[0] for h, _ in pieces if h}) >= 2:
        sections: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for heading, piece in pieces:
            if heading and (current is None or heading[0] != current["number"]):
                current = {"number": heading[0], "title": heading[1], "passages": []}
                sections.append(current)
            elif current is None:            # text before the first heading
                current = {"number": "", "title": "Introduction", "passages": []}
                sections.append(current)
            current["passages"].append(piece)
    else:
        sections, current, used = [], None, 0
        for _, piece in pieces:
            if current is None or used + _tokens(piece) > PAGE_GROUP_TOKENS:
                current = {"number": "", "title": "", "passages": []}
                sections.append(current)
                used = 0
            current["passages"].append(piece)
            used += _tokens(piece)

    out = []
    for s in sections:
        if not any((x.get("content") or "").strip() for x in s["passages"]):
            continue
        nums = [int(x.get("page_no") or 0) for x in s["passages"]]
        s["pages"] = [min(nums), max(nums)]
        if not s["title"]:
            s["title"] = "Pages %d-%d" % tuple(s["pages"])
        out.append(s)
    return out


def _page_words(passages: Sequence[Dict[str, Any]]) -> Dict[int, set]:
    pages: Dict[int, set] = {}
    for p in passages:
        try:
            n = int(p.get("page_no"))
        except (TypeError, ValueError):
            continue
        pages.setdefault(n, set()).update(_words(p.get("content", "")))
    return pages


def check_facts(facts: Any, pages: Dict[int, set]) -> Tuple[List[Dict[str, Any]], int]:
    """(kept facts, dropped count). Each kept fact sits on the page that best supports it.

    Support = share of the fact's content words found on a page. A correct fact
    cited one page off is kept with its page corrected; a fact no page supports
    is dropped.
    """
    kept, dropped = [], 0
    for f in (facts if isinstance(facts, list) else [])[:MAX_FACTS_PER_SECTION * 2]:
        if not isinstance(f, dict):
            continue
        text = " ".join(str(f.get("fact") or "").split())
        words = _words(text)
        if not text or not words:
            dropped += 1
            continue
        cited = str(f.get("page"))
        best_page, best = None, -1.0
        for n in sorted(pages):
            support = sum(w in pages[n] for w in words) / len(words)
            if support > best or (support == best and str(n) == cited):
                best_page, best = n, support
        if best < FACT_SUPPORT:
            dropped += 1
            continue
        kept.append({"fact": text, "page": best_page})
        if len(kept) >= MAX_FACTS_PER_SECTION:
            break
    return kept, dropped


def _summarise_section(section: Dict[str, Any], scope: str, llm: Any, institute_id: Any,
                       on_usage: Any) -> Optional[Dict[str, Any]]:
    from ai_services.core import grounding

    chosen, used = [], 0
    for p in section["passages"]:
        if chosen and used + _tokens(p) > SECTION_TOKEN_CAP:
            break
        chosen.append(p)
        used += _tokens(p)
    user = "Chapter: %s\nSection: %s\n\nSECTION TEXT:\n\n%s" % (
        scope, section["title"], grounding.format_source_block(chosen))
    content: Any = None
    for attempt in (1, 2):
        try:
            result = llm.complete(
                system_prompt=_SECTION_SYSTEM, user_prompt=user, model=None, temperature=0.1,
                max_tokens=1500, json_mode=True, institute_id=institute_id,
                feature="chapter_digest", capability="content", reasoning_effort="low",
            ) or {}
        except Exception as exc:
            logger.warning("chapter digest: section %r failed (attempt %d): %s",
                           section["title"], attempt, str(exc)[:160])
            if attempt == 1:
                time.sleep(RETRY_PAUSE_S)
            continue
        if on_usage:
            on_usage(result)
        content = result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                content = None
        if isinstance(content, dict):
            break
    if not isinstance(content, dict):
        return None
    facts, dropped = check_facts(content.get("key_facts"), _page_words(section["passages"]))
    terms = [" ".join(str(t).split()) for t in (content.get("key_terms") or []) if str(t).strip()]
    return {"number": section.get("number", ""), "title": section["title"], "pages": section["pages"],
            "summary": " ".join(str(content.get("summary") or "").split()),
            "key_facts": facts, "key_terms": terms[:MAX_TERMS_PER_SECTION], "_dropped": dropped}


def build(passages: Optional[Sequence[Any]], *, chapter: str = "", class_name: str = "",
          subject: str = "", llm: Any = None, institute_id: Any = None,
          on_usage: Any = None) -> Optional[Dict[str, Any]]:
    """Build and check a digest. None when it cannot be made completely."""
    sections = detect_sections(passages)
    if not sections:
        return None
    if llm is None:
        from ai_services.core.llm_client import get_llm
        llm = get_llm()
    scope = " | ".join(x for x in (class_name, subject, chapter) if x) or "this chapter"
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_PARALLEL, len(sections)))) as pool:
        done = list(pool.map(lambda s: _summarise_section(s, scope, llm, institute_id, on_usage), sections))
    missing = [s["title"] for s, d in zip(sections, done) if d is None]
    if missing:
        logger.warning("chapter digest: incomplete (no summary for %s); not used", missing)
        return None
    kept = sum(len(d["key_facts"]) for d in done)
    dropped = sum(d.pop("_dropped") for d in done)
    if not kept:
        return None
    logger.info("chapter digest: %d sections, %d facts kept, %d dropped", len(done), kept, dropped)
    return {"version": VERSION, "sections": done,
            "checks": {"sections": len(done), "facts_kept": kept, "facts_dropped": dropped}}


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        from ai_services.core.ppt_v2.deck_cache import _client
        client = _client()
        if client is not None:
            raw = client.get(key)
            if raw:
                return json.loads(raw)
    except Exception as exc:
        logger.debug("chapter digest: cache read failed (%s)", exc)
    with _memory_lock:
        return _memory.get(key)


def _cache_put(key: str, digest: Dict[str, Any]) -> None:
    with _memory_lock:
        _memory[key] = digest
    try:
        from ai_services.core.ppt_v2.deck_cache import _client
        client = _client()
        if client is not None:
            client.setex(key, CACHE_TTL_S, json.dumps(digest, ensure_ascii=False))
    except Exception as exc:
        logger.debug("chapter digest: cache write failed (%s)", exc)


def cached(passages: Optional[Sequence[Any]]) -> Optional[Dict[str, Any]]:
    """The chapter's digest if it is already built; never builds one."""
    if not enabled():
        return None
    fp = fingerprint(passages)
    return _cache_get("chdigest:v%d:%s" % (VERSION, fp)) if fp else None


_building: set = set()
_building_lock = threading.Lock()


def build_in_background(passages: Optional[Sequence[Any]], **kw) -> bool:
    """Start building the chapter's digest on a background thread; True if started.

    A chapter already being built is not started twice. A paper that finds no
    digest uses the whole chapter instead of waiting for this (measured: a
    first 100-mark paper spent 34 s here before writing a word).
    """
    if not enabled():
        return False
    fp = fingerprint(passages)
    if not fp:
        return False
    with _building_lock:
        if fp in _building:
            return False
        _building.add(fp)

    def work():
        try:
            get_or_build(list(passages or []), **kw)
        except Exception:
            logger.exception("chapter digest: background build failed")
        finally:
            with _building_lock:
                _building.discard(fp)

    threading.Thread(target=work, name="chapter-digest-%s" % fp[:8], daemon=True).start()
    return True


def get_or_build(passages: Optional[Sequence[Any]], **kw) -> Optional[Dict[str, Any]]:
    """The chapter's digest, from the cache or built now and kept. None when off or impossible."""
    if not enabled():
        return None
    fp = fingerprint(passages)
    if not fp:
        return None
    key = "chdigest:v%d:%s" % (VERSION, fp)
    cached = _cache_get(key)
    if cached:
        return cached
    digest = build(passages, **kw)
    if digest:
        _cache_put(key, digest)
    return digest


def focus_section(digest: Dict[str, Any], topic: str) -> Optional[Dict[str, Any]]:
    """The section a topic belongs to; None when unclear.

    The title decides first ("Respiration" -> "5.3 RESPIRATION"); key terms
    only break a tie. A term match alone used to send "Respiration" to the
    introduction, which merely lists respiration among the life processes.
    """
    want = set(_words(topic))
    if not want:
        return None
    best, best_key = None, (0.0, 0.0)
    for s in digest.get("sections") or []:
        title = set(_words(s.get("title", "")))
        terms = {w for t in s.get("key_terms") or [] for w in _words(t)}
        key = (len(want & title) / len(want), len(want & terms) / len(want))
        if key > best_key:
            best, best_key = s, key
    return best if best_key[0] >= 0.5 else None


def focus_pages_enabled() -> bool:
    """A topic's steps read only its section's pages (PPT_FOCUS_SECTION_PAGES)."""
    return (os.getenv("PPT_FOCUS_SECTION_PAGES", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def focus_passages(digest: Optional[Dict[str, Any]], passages: Optional[Sequence[Any]],
                   topic: str) -> Optional[Sequence[Any]]:
    """The passages of the topic's own section; all of them when that is unclear.

    A topic deck used to write each batch of slides from passages ranked across
    the whole chapter (3.5k tokens a batch), which also pulled in other
    sections that merely shared a word. Its section's pages are what the book
    says about the topic. Falls back to every passage when there is no digest,
    no clear section, or the section's pages hold no passage.
    """
    if not (digest and passages and topic and focus_pages_enabled()):
        return passages
    section = focus_section(digest, topic)
    pages = (section or {}).get("pages") or []
    if len(pages) != 2:
        return passages
    lo, hi = pages
    chosen = [p for p in passages if isinstance(p, dict) and lo <= int(p.get("page_no") or -1) <= hi]
    return chosen or passages


def format_for_prompt(digest: Dict[str, Any], topic: str = "") -> str:
    """The digest as planning text: every section briefly, the topic's section in full.

    With no clear topic section, every section is given in full.
    """
    focus = focus_section(digest, topic) if topic else None
    lines = []
    for s in digest.get("sections") or []:
        pages = s.get("pages") or []
        where = (" (pp. %s-%s)" % (pages[0], pages[1]) if len(pages) == 2 and pages[0] != pages[1]
                 else " (p. %s)" % pages[0] if pages else "")
        is_focus = focus is not None and s is focus
        lines.append("%sSECTION %s%s" % ("FOCUS " if is_focus else "",
                                          " ".join(x for x in (s.get("number"), s["title"]) if x), where))
        if s.get("summary"):
            lines.append("  %s" % s["summary"])
        if focus is None or is_focus:
            for f in s.get("key_facts") or []:
                lines.append("  - %s [p.%s]" % (f["fact"], f["page"]))
            if s.get("key_terms"):
                lines.append("  Terms: %s" % ", ".join(s["key_terms"]))
        lines.append("")
    return "\n".join(lines).strip()
