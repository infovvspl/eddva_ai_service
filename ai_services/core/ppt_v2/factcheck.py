"""Fact check: every statement on the slides, against the school's own book.

The planner is told to use only the book, and the scope check keeps out
terms the book never uses. Neither catches a statement that uses the book's
own words and still gets it wrong - a live Class 10 deck said aerobic
respiration releases "three molecules of carbon dioxide for each molecule of
glucose" where the book says three per PYRUVATE. Here a model reads each
statement beside the book and corrects the ones that contradict it.

It runs on the plan, before layout and pictures, so corrections are in place
before the teacher sees a single slide - nothing on screen changes under
their hands. Only with a book: without one there is nothing to check against.
Any failure leaves the slides exactly as planned; a check must never cost a
deck.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Slides per checking call; the calls run side by side.
BATCH_SLIDES = 5
# Started only while it can finish with room for pictures inside the deck's
# deadline (service.deadline_s, 200s by default).
CHECK_WINDOW_S = 110.0
# Pause before retrying a batch the provider failed.
RETRY_PAUSE_S = 3.0
# A correction may be rewritten, not rewritten into an essay.
MAX_GROWTH = 2.0

_SYSTEM = """You fact-check school slides against the school's own textbook.

For each STATEMENT, compare it with the SOURCE MATERIAL.

A statement is WRONG when it contradicts the source, or states a number,
quantity, cause, sequence, location, or definition differently from the
source. Example: the source says pyruvate gives three molecules of carbon
dioxide; a statement saying glucose gives three molecules is WRONG.

A statement is RIGHT when the source supports it, or it is a faithful
simplification of the source, or it is general framing that asserts no
specific fact (e.g. "Let us look at how cells release energy").

For every WRONG statement, write a corrected sentence: the same length and
style, for the same audience, saying only what the source says.

Return JSON exactly: {"issues": [{"id": "<statement id>",
"problem": "<what is wrong, one short sentence>",
"fix": "<the corrected sentence>"}]}
Return {"issues": []} when every statement is right. Do not list right
statements. Do not invent ids."""


def check_budget() -> int:
    """Book tokens one fact-check batch reads (PPT_FACTCHECK_TOKEN_BUDGET)."""
    try:
        return max(500, int(os.getenv("PPT_FACTCHECK_TOKEN_BUDGET", "") or 4000))
    except ValueError:
        return 4000


def enabled() -> bool:
    return (os.getenv("PPT_FACT_CHECK", "true") or "").strip().lower() in ("1", "true", "yes", "on")


def _statements(slides: Sequence[Dict[str, Any]], start: int, end: int) -> List[Tuple[str, int, int]]:
    """(id, slide index, block index) for every factual sentence in [start, end)."""
    out = []
    for i in range(start, min(end, len(slides))):
        for j, block in enumerate(slides[i].get("blocks") or []):
            if isinstance(block, dict) and (block.get("text") or "").strip():
                out.append(("S%d.%d" % (i + 1, j + 1), i, j))
    return out


def _sane(original: str, fix: str) -> bool:
    fix = (fix or "").strip()
    if not fix or fix == original.strip():
        return False
    return len(fix) <= max(60, MAX_GROWTH * len(original))


def check_facts(slides: List[Dict[str, Any]], *, passages: Optional[Sequence[Any]],
                topic: str = "", chapter: str = "", llm: Any = None,
                institute_id: Any = None, started: Optional[float] = None,
                on_event: Optional[Callable[..., None]] = None,
                on_usage: Optional[Callable[..., None]] = None) -> List[Dict[str, Any]]:
    """Correct, in place, statements that contradict the book. Returns the fixes.

    Each fix is {"slide", "id", "before", "after", "problem"}.
    """
    from ai_services.core.ppt_v2.planner import (backup_capability, call_timeout, max_parallel,
                                                 source_block)

    def say(text: str, state: str) -> None:
        if on_event:
            try:
                on_event("facts", text, state)
            except Exception:
                logger.exception("factcheck: activity event failed")

    if not enabled():
        return []
    book = source_block(passages, topic, chapter)
    if not book:
        return []                    # nothing to check against
    if started is not None and time.perf_counter() - started > CHECK_WINDOW_S:
        logger.info("factcheck: skipped, no time left")
        return []
    if llm is None:
        from ai_services.core.llm_client import get_llm
        llm = get_llm()

    say("Checking every fact against your textbook", "active")
    batches = [_statements(slides, s, s + BATCH_SLIDES)
               for s in range(0, len(slides), BATCH_SLIDES)]
    batches = [b for b in batches if b]

    def check(batch: List[Tuple[str, int, int]]) -> Optional[List[Dict[str, Any]]]:
        """The issues in one batch; None when the batch could not be checked."""
        lines = "\n".join("%s: %s" % (sid, slides[i]["blocks"][j]["text"].strip())
                          for sid, i, j in batch)
        # The pages these statements are about, not the whole chapter again:
        # a statement can only be corrected against the passage that covers
        # it, and leaving unrelated pages out cannot cause a false fix.
        # One query per slide in the batch, so each slide gets its own pages.
        texts: Dict[int, List[str]] = {}
        for _sid, i, j in batch:
            texts.setdefault(i, [str(slides[i].get("title") or "")]).append(slides[i]["blocks"][j]["text"])
        queries = [" ".join(t) for t in texts.values()]
        pages = source_block(passages, topic, chapter, budget=check_budget(), queries=queries) or book
        user = "SOURCE MATERIAL:\n%s\n\nSTATEMENTS:\n%s" % (pages, lines)
        result = None
        for attempt in (1, 2):
            try:
                result = llm.complete(
                    system_prompt=_SYSTEM, user_prompt=user, model=None, temperature=0.0,
                    max_tokens=2500 + 120 * len(batch), json_mode=True,
                    institute_id=institute_id, feature="ppt_v2_factcheck",
                    # The retry goes to the backup model, as the planner's does.
                    capability=(backup_capability() or "content") if attempt == 2 else "content",
                    timeout_s=call_timeout(),
                ) or {}
                break
            except Exception as exc:
                # A provider blip (a live run met three 503s in a row) is worth
                # one more try after a pause - while the deck has time for it.
                late = started is not None and time.perf_counter() - started > CHECK_WINDOW_S
                if attempt == 2 or late:
                    logger.warning("factcheck: batch failed (%s); left as planned", str(exc)[:160])
                    return None
                time.sleep(RETRY_PAUSE_S)
        if on_usage:
            on_usage(result, 1)
        content = result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("factcheck: reply was not JSON; left as planned")
                return None
        if not isinstance(content, dict) or not isinstance(content.get("issues", []), list):
            return None
        return [x for x in content.get("issues") or [] if isinstance(x, dict)]

    with ThreadPoolExecutor(max_workers=max(1, min(len(batches), max_parallel()))) as pool:
        results = list(pool.map(check, batches))
    unchecked = sum(1 for r in results if r is None)
    found = [issue for r in results if r for issue in r]

    where = {sid: (i, j) for batch in batches for sid, i, j in batch}
    fixes = []
    for issue in found:
        sid = str(issue.get("id") or "").strip()
        if sid not in where:
            continue                  # an invented id: nothing to fix
        i, j = where[sid]
        block = slides[i]["blocks"][j]
        before = block.get("text") or ""
        after = re.sub(r"\s+", " ", str(issue.get("fix") or "")).strip()
        if not _sane(before, after):
            continue
        block["text"] = after
        fixes.append({"slide": i + 1, "id": sid, "before": before, "after": after,
                      "problem": str(issue.get("problem") or "")[:200]})
        logger.info("factcheck: %s corrected | %s | %r -> %r",
                    sid, issue.get("problem"), before[:80], after[:80])

    # Never claim a check that did not happen.
    if unchecked == len(batches):
        say("Couldn't complete the fact check right now; slides kept as written", "warn")
    elif fixes:
        on = sorted({f["slide"] for f in fixes})
        say("Corrected %d statement%s on slide%s %s to match your textbook%s" % (
            len(fixes), "" if len(fixes) == 1 else "s", "" if len(on) == 1 else "s",
            ", ".join(str(n) for n in on),
            " (some slides could not be checked)" if unchecked else ""),
            "warn" if unchecked else "done")
    elif unchecked:
        say("Checked most facts against your textbook; some slides could not be checked", "warn")
    else:
        say("Every fact matches your textbook", "done")
    return fixes
