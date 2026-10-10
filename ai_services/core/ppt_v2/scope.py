"""Is the plan inside the chapter? Checked before anything is paid for.

The gate (gate.py) already reports terminology above the class, but only
after the deck is built and its pictures are bought - and then it ships the
deck anyway, so the teacher reads "Krebs cycle, NADH, FADH2" on a Class 10
slide and generates again. Here the same check runs on the PLAN, and a plan
that strays is re-planned once, with the stray terms named, while there is
still time in the request for it.

"Above the class" is decided by the book whenever there is one: a term the
school's own chapter uses is in scope, whatever a global list says - Class 11
biology teaches glycolysis, and its book says so. Only without a book does the
global list apply, and then only up to Class 10, where it was written for.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# The correction rewrites only the slides that strayed (a few seconds for one
# or two slides), but it still has to leave room for pictures inside the
# backend's 240s wait, so it is skipped once the plan itself ran long.
REPLAN_WINDOW_S = 120.0
# Without a book, the global list is trusted only this far up.
_LIST_APPLIES_UP_TO_CLASS = 10


def class_number(class_name: str) -> Optional[int]:
    m = re.search(r"\d{1,2}", class_name or "")
    return int(m.group(0)) if m else None


def plan_text(slides: Sequence[Dict[str, Any]]) -> str:
    """Every word a planned deck will put in front of a student."""
    parts: List[str] = []
    for s in slides or []:
        if not isinstance(s, dict):
            continue
        parts += [str(s.get(k) or "") for k in ("title", "subtitle", "learningObjective")]
        for b in s.get("blocks") or []:
            if isinstance(b, dict):
                parts += [str(b.get("label") or ""), str(b.get("text") or "")]
    return " ".join(p for p in parts if p)


def _book_text(passages: Optional[Sequence[Any]]) -> str:
    out = []
    for p in passages or []:
        text = (p.get("content") or p.get("text")) if isinstance(p, dict) else p
        if text:
            out.append(str(text))
    return " ".join(out)


def off_scope_terms(slides: Sequence[Dict[str, Any]], *,
                    passages: Optional[Sequence[Any]] = None,
                    class_name: str = "") -> List[str]:
    """Terms in the plan that this class's chapter does not teach, as written."""
    from ai_services.core.ppt_v2.relevance import ADVANCED_TERMS

    text = plan_text(slides)
    found: List[str] = []
    for pattern in ADVANCED_TERMS:
        m = re.search(r"\b" + pattern + r"\w*", text, re.I)
        if m:
            found.append((pattern, m.group(0)))
    if not found:
        return []

    book = _book_text(passages)
    if book.strip():
        return [word for pattern, word in found
                if not re.search(r"\b" + pattern, book, re.I)]
    n = class_number(class_name)
    if n is not None and n <= _LIST_APPLIES_UP_TO_CLASS:
        return [word for _pattern, word in found]
    return []


def _slide_terms(slide: Dict[str, Any], terms: Sequence[str]) -> bool:
    text = plan_text([slide])
    return any(re.search(r"\b" + re.escape(t), text, re.I) for t in terms)


def revision_note(terms: Sequence[str], *, class_name: str, grounded: bool) -> str:
    where = ("the SOURCE MATERIAL above does not contain them" if grounded
             else "they are beyond the %s syllabus" % (class_name or "class"))
    return ("REVISION: your previous plan used these terms, but %s: %s. Plan the "
            "deck again without them and without the ideas they stand for. Explain "
            "the topic only at the level and in the words of the %s."
            % (where, ", ".join(terms), "chapter" if grounded else "syllabus"))


def replan_if_off_scope(plan, *, revise: Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
                        passages: Optional[Sequence[Any]], class_name: str,
                        started: float, label: str = "ppt",
                        on_event: Optional[Callable[..., None]] = None) -> Any:
    """Return the plan to build, with any straying slides rewritten once.

    ``revise(note, slides)`` rewrites just the slides that used off-scope
    terms and returns them in the same order. The rewrite replaces them only
    if the deck as a whole ends up closer to the chapter; a failed or worse
    rewrite leaves the plan as it was, so this can never cost a deck.
    """
    grounded = bool(getattr(plan, "grounded", False))

    def say(text, state):
        if on_event:
            try:
                on_event("check", text, state)
            except Exception:
                logger.exception("%s: activity event failed", label)

    say("Checking every slide stays within your textbook" if grounded
         else "Checking every slide stays within the %s syllabus" % (class_name or "class"), "active")
    terms = off_scope_terms(plan.slides, passages=passages, class_name=class_name)
    if not terms:
        say("Every slide stays within your textbook" if grounded
             else "Every slide is within the %s syllabus" % (class_name or "class"), "done")
        return plan
    elapsed = time.perf_counter() - started
    if elapsed >= REPLAN_WINDOW_S:
        logger.info("%s: off-scope terms %s, no time to correct (%.0fs)", label, terms, elapsed)
        return plan
    straying = [i for i, s in enumerate(plan.slides) if _slide_terms(s, terms)]
    if not straying:
        return plan
    logger.info("%s: off-scope terms %s on slide(s) %s; rewriting them once",
                label, terms, [i + 1 for i in straying])
    note = revision_note(terms, class_name=class_name,
                         grounded=bool(getattr(plan, "grounded", False)))
    try:
        rewritten = revise(note, [plan.slides[i] for i in straying])
    except Exception as exc:  # the plan still stands
        logger.warning("%s: rewrite failed (%s); keeping the plan", label, str(exc)[:160])
        return plan
    if len(rewritten) != len(straying):
        return plan
    candidate = list(plan.slides)
    for i, slide in zip(straying, rewritten):
        candidate[i] = slide
    left = off_scope_terms(candidate, passages=passages, class_name=class_name)
    if len(left) >= len(terms):
        logger.info("%s: rewrite no better (%s); keeping the plan", label, left)
        say("Checked the slides stay within your textbook", "done")
        return plan
    logger.info("%s: rewrite kept | off-scope %d -> %d %s", label, len(terms), len(left), left)
    plan.slides = candidate
    say("Rewrote %s to stay within your textbook" % (
        "slide %d" % (straying[0] + 1) if len(straying) == 1
        else "slides " + ", ".join(str(i + 1) for i in straying)), "done")
    return plan
