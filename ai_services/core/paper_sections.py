"""Write an assessment paper's sections in parallel, then join them.

A paper used to be one model call writing every question, answer and
explanation in sequence: ~2.7k output tokens and 30-40 s, set by the length of
what is written, not of what is read. The objective sections (MCQ, True/False,
fill in the blanks) and the written ones (short and long answer) do not depend
on each other, so they are written by two calls at once and joined in order:
the wall time becomes the longer of the two instead of their sum.

The backend describes the paper in ``paperPlan`` with the exact instruction
strings it put in ``extraContext``, so each call's instructions are made by
removing the other group's lines - no parsing of free text. Only the first
call writes the paper header and general instructions; the difficulty mix is
split in proportion to each call's questions; textbook figures and drawn
diagrams go to the written call only, so no figure is used twice.

``run`` returns None whenever anything does not line up (no plan, one group,
a string not found, a call failing or coming back without content), and the
caller then makes today's single call. It never makes a paper worse than that.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_ANSWER_KEY = re.compile(r"^\s*#{1,6}\s*\**\s*answer\s*key\b.*$", re.I | re.M)

ONLY_THESE = (
    "Write ONLY the sections listed above, with their exact question counts. "
    "The paper's other sections are written separately and joined to yours, so do "
    "not add them, and do not renumber your sections. Your Answer Key covers only "
    "your sections, under the same section headings."
)
NO_HEADER = (
    "Do NOT write a paper header or General Instructions - start directly with the "
    "first section heading below."
)


def coverage_block(items: List[str]) -> str:
    """The MANDATORY COVERAGE instruction for a whole paper (one call)."""
    return (
        "\nMANDATORY COVERAGE — these sub-areas of this exact topic were identified in advance as core "
        "to a real school textbook treatment of it. Your question set MUST include AT LEAST ONE question "
        "addressing EACH sub-area below, spread across the available sections. Do not skip any of "
        "them, and do not let 2-3 of them dominate while the rest are ignored:\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(items, 1))
        + "\n"
    )


def divide(items: List[str], shares: List[int]) -> List[List[str]]:
    """Each item to exactly one group, in proportion to the groups' question counts.

    Given the same list, both calls covered the same main points and left the
    rest uncovered (measured: the breathing anatomy of "Respiration" in 3 of 3
    parallel papers). Owning a share each, they cover the whole list.
    """
    out: List[List[str]] = [[] for _ in shares]
    for item in items:
        i = min(range(len(shares)), key=lambda g: (len(out[g]) + 1) / max(1, shares[g]))
        out[i].append(item)
    return out


def group_coverage_block(mine: List[str], others: List[str]) -> str:
    """One call's coverage: its own share, and what the rest of the paper covers."""
    if not mine:
        return ""
    text = (
        "\nMANDATORY COVERAGE FOR YOUR SECTIONS — identified in advance from the book. Your "
        "sections MUST include AT LEAST ONE question on EACH of these, spread across your sections:\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(mine, 1))
        + "\n"
    )
    if others:
        text += ("The rest of this paper is written separately and covers: " + "; ".join(others)
                 + ". Do not base your questions on those.\n")
    return text


def divide_pages(pages: List[int], shares: List[int]) -> List[List[int]]:
    """Contiguous runs of pages, sized by the groups' question counts; each group at least one."""
    pages = sorted({int(p) for p in pages if str(p).lstrip("-").isdigit()})
    if len(pages) < len(shares):
        return []
    total = sum(shares) or 1
    out, start = [], 0
    for i, share in enumerate(shares):
        left = len(shares) - i - 1
        n = len(pages) - start - left if left == 0 else max(1, round(len(pages) * share / total))
        n = min(n, len(pages) - start - left)
        out.append(pages[start:start + n])
        start += n
    return out


def _pp(pages: List[int]) -> str:
    return "p.%d" % pages[0] if len(pages) == 1 else "pp.%d-%d" % (pages[0], pages[-1])


def group_pages_block(mine: List[int], others: List[int]) -> str:
    """Where one call's questions come from, so the calls between them use every page."""
    return (
        "\nSOURCE FOCUS FOR YOUR SECTIONS — base your questions on the book's %s. The rest of "
        "this paper is written separately from %s; use those pages only for context, so that "
        "every page of the topic is examined.\n" % (_pp(mine), _pp(others))
    )


def enabled() -> bool:
    return (os.getenv("ASSESSMENT_PARALLEL_SECTIONS", "true") or "").strip().lower() in ("1", "true", "yes", "on")


# Output tokens one question of each section costs, with its answer-key line
# (measured on generated papers: an MCQ with options ~90, a long answer ~450).
_OUTPUT_PER_QUESTION = {"A": 90, "B": 45, "C": 50, "D": 220, "E": 450}
# Above this a paper may be split into more than two calls (max_calls()).
BIG_PAPER_TOKENS = 4000


def max_calls() -> int:
    """Calls a big paper may use. Two by default: measured on a 100-mark paper,
    three calls took 27-48 s and 21k input tokens against 23-26 s and 14k for
    two - the provider slows each call as more run at once from one account."""
    try:
        return max(2, int(os.getenv("ASSESSMENT_PARALLEL_MAX_CALLS", "2")))
    except (TypeError, ValueError):
        return 2


def estimate(section: Dict[str, Any]) -> int:
    return int(section.get("count") or 0) * _OUTPUT_PER_QUESTION.get(str(section.get("letter") or ""), 150)


def _balanced(sections: List[Dict[str, Any]], k: int) -> List[List[Dict[str, Any]]]:
    """``sections`` in order, cut into ``k`` runs whose largest estimated output is smallest."""
    from itertools import combinations
    loads = [estimate(s) for s in sections]
    best, best_max = None, None
    for cuts in combinations(range(1, len(sections)), k - 1):
        bounds = (0,) + cuts + (len(sections),)
        worst = max(sum(loads[a:b]) for a, b in zip(bounds, bounds[1:]))
        if best_max is None or worst < best_max:
            best, best_max = bounds, worst
    return [sections[a:b] for a, b in zip(best, best[1:])]


def groups(plan: Optional[Dict[str, Any]]) -> Optional[List[List[Dict[str, Any]]]]:
    """The calls a paper is written in; None unless it has objective and written sections.

    A normal paper: [objective sections, written sections]. A big one (over
    BIG_PAPER_TOKENS of estimated output): up to max_calls() runs of sections,
    in paper order, balanced by estimated output.
    """
    sections = [s for s in (plan or {}).get("sections") or []
                if isinstance(s, dict) and s.get("text") and int(s.get("count") or 0) > 0]
    objective = [s for s in sections if s.get("kind") == "objective"]
    written = [s for s in sections if s.get("kind") == "written"]
    if not (objective and written):
        return None
    k = min(max_calls(), len(sections))
    if sum(estimate(s) for s in sections) < BIG_PAPER_TOKENS or k <= 2:
        return [objective, written]
    return _balanced(sections, k)


def split_counts(total: Dict[str, int], shares: List[int]) -> List[Dict[str, int]]:
    """Split {easy, medium, hard} across groups; each group's counts add up to its share.

    The levels are laid out evenly interleaved (easy, medium, hard, easy, ...)
    and each group takes a consecutive slice of its own size, so every group
    gets a fair mix and the totals are exact by construction.
    """
    levels = ["easy", "medium", "hard"]
    items = sorted(((k + 0.5) / c, lv) for lv in levels
                   for c in [int(total.get(lv) or 0)] for k in range(c))
    seq = [lv for _, lv in items]
    out, start = [], 0
    for share in shares:
        part = seq[start:start + share]
        start += share
        out.append({lv: part.count(lv) for lv in levels})
    return out


def group_instructions(extra: str, plan: Dict[str, Any], gs: List[List[Dict[str, Any]]],
                       index: int) -> Optional[str]:
    """``extra`` for one call; None when a plan string is not found in it."""
    text = extra
    mine = gs[index]
    for i, g in enumerate(gs):
        if i == index:
            continue
        for s in g:
            if text.count(s["text"]) != 1:
                return None
            text = text.replace(s["text"] + "\n", "", 1) if s["text"] + "\n" in text else text.replace(s["text"], "", 1)

    header = plan.get("header") or ""
    if index > 0 and header:
        if text.count(header) != 1:
            return None
        text = text.replace(header, NO_HEADER, 1)

    diff = plan.get("difficulty") or None
    if diff and diff.get("line"):
        if text.count(diff["line"]) != 1:
            return None
        shares = [sum(int(s.get("count") or 0) for s in g) for g in gs]
        c = split_counts({k: diff.get(k) or 0 for k in ("easy", "medium", "hard")}, shares)[index]
        text = text.replace(diff["line"], (
            "Difficulty mix for your sections: exactly %d easy, %d intermediate and %d hard "
            "questions in total, spread across your sections." % (c["easy"], c["medium"], c["hard"])), 1)

    # Figures and drawn diagrams: the written call only.
    if index != len(gs) - 1:
        for key in ("figures", "plots"):
            rule = plan.get(key) or ""
            if rule:
                if text.count(rule) != 1:
                    return None
                text = text.replace(rule, "", 1)

    last_section = mine[-1]["text"]
    if text.count(last_section) != 1:
        return None
    return text.replace(last_section, last_section + "\n" + ONLY_THESE, 1)


def split_key(content: str):
    """(paper part, answer key body) of one call's output."""
    m = _ANSWER_KEY.search(content or "")
    if not m:
        return (content or "").strip(), ""
    return content[: m.start()].strip(), content[m.end():].strip()


def merge(contents: List[str]) -> str:
    papers, keys = [], []
    for c in contents:
        paper, key = split_key(c)
        papers.append(paper)
        if key:
            keys.append(key)
    out = "\n\n".join(p for p in papers if p)
    if keys:
        out += "\n\n## Answer Key\n\n" + "\n\n".join(keys)
    return out


def run(complete: Callable[[str], Dict[str, Any]], *, user_prompt: str, extra: str,
        plan: Optional[Dict[str, Any]], coverage: Optional[List[str]] = None,
        pages: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Both calls at once and the joined paper, or None to make the single call.

    ``complete(user_prompt)`` makes one model call and returns its result dict
    (``content``, ``usage``, ``model``...). The return value has the joined
    ``content``, the summed ``usage``, the last ``model`` and ``parts``.
    ``coverage`` is the list whose ``coverage_block`` is in the prompt; it is
    divided between the calls. ``pages`` (a topic paper's section pages) are
    divided too: given the same material, both calls examined the same main
    points and left the rest out (measured on "Respiration": the breathing
    anatomy missing from 3 of 3 parallel papers, present in every single-call one).
    """
    if not (enabled() and plan and extra):
        return None
    gs = groups(plan)
    if not gs or user_prompt.count(extra) != 1:
        return None
    whole_cov = coverage_block(coverage) if coverage else ""
    if whole_cov and user_prompt.count(whole_cov) != 1:
        whole_cov = ""
    shares = [sum(int(s.get("count") or 0) for s in g) for g in gs]
    owned = divide(list(coverage or []), shares) if whole_cov else [[] for _ in gs]
    page_parts = divide_pages(list(pages or []), shares) if pages else []
    prompts = []
    for i in range(len(gs)):
        g = group_instructions(extra, plan, gs, i)
        if g is None:
            logger.warning("paper sections: plan does not match the instructions; single call")
            return None
        prompt = user_prompt.replace(extra, g, 1)
        if whole_cov:
            others = [x for j, part in enumerate(owned) if j != i for x in part]
            prompt = prompt.replace(whole_cov, group_coverage_block(owned[i], others), 1)
        if page_parts:
            rest = sorted(p for j, part in enumerate(page_parts) if j != i for p in part)
            prompt += group_pages_block(page_parts[i], rest)
        prompts.append(prompt)
    try:
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            results = list(pool.map(complete, prompts))
    except Exception as exc:
        logger.warning("paper sections: a parallel call failed (%s); single call", exc)
        return None
    contents = [str((r or {}).get("content") or "") for r in results]
    if not all(c.strip() for c in contents):
        logger.warning("paper sections: a parallel call returned nothing; single call")
        return None
    usage: Dict[str, int] = {}
    for r in results:
        for k, v in ((r or {}).get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + int(v)
    return {"content": merge(contents), "usage": usage, "model": (results[-1] or {}).get("model"),
            "latency_ms": max(int((r or {}).get("latency_ms") or 0) for r in results), "parts": len(results)}
