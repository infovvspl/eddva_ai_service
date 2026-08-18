"""
Source-grounded generation — write slides from the school's own textbook.

The ungrounded generator writes from the model's own knowledge, which is why a
teacher could not vouch for it: correct-sounding content may not be what their
book actually teaches. Grounded generation inverts that. The chapter's own
passages are supplied as the only permitted source, every bullet must be
traceable to a page, and anything the book does not cover is simply left out.

Retrieval is deliberately simple. The teacher has already chosen class, subject,
chapter and topic, so the candidate set is one chapter — the part of RAG that is
normally hard is already solved by the curriculum tree. Passages are ranked by
term overlap with the topic, which is enough at this scale; embeddings would add
infrastructure without changing which nine passages of a nine-page chapter win.
"""
import logging
import os
import re

logger = logging.getLogger("ai_services.grounding")

_STOPWORDS = frozenset("""
a an the and or of in on for to from with without into by is are was were be been
its it this that these those as at how what why when which who whom whose
introduction chapter topic class subject exercise example
""".split())

# How much of the chapter may go into one prompt.
#
# This was 12,000, inherited from Groq's per-request ceiling — but grounded
# generation runs on Gemini, whose context window is three orders of magnitude
# larger. The old figure was cutting real chapters: measured against the indexed
# set, "Metals and Non-metals" lost 7 of its 22 passages on the content path and
# 2 on the slide path, silently, every time a teacher generated.
#
# 30,000 clears every chapter currently indexed with room to spare. It costs
# nothing for a chapter that already fitted — only what is actually sent is
# billed — so the extra budget is paid for exactly by the chapters that were
# being truncated, which is where the money should go.
#
# Kept well below the model's real limit so instructions and the answer still
# fit comfortably, and env-overridable so a deployment can tune it without a
# release.
_DEFAULT_SOURCE_TOKEN_BUDGET = int(os.getenv("GROUNDING_TOKEN_BUDGET", "30000"))


def _terms(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def rank_passages(passages: list, topic: str, chapter: str = "") -> list:
    """Order passages by relevance to the topic, keeping the book's own order as
    the tie-breaker so a selection still reads in teaching sequence."""
    want = _terms(topic) | _terms(chapter)
    if not want:
        return list(passages)

    scored = []
    for p in passages:
        have = _terms(p.get("content", ""))
        if not have:
            continue
        overlap = len(want & have)
        # Normalise so a long passage does not win on length alone.
        score = overlap / (len(want) ** 0.5 or 1)
        scored.append((score, p.get("page_no", 0), p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored]


def select_source(
    passages: list,
    topic: str,
    chapter: str = "",
    token_budget: int = _DEFAULT_SOURCE_TOKEN_BUDGET,
) -> dict:
    """Pick the passages that fit the budget, then restore reading order.

    Whole chapters usually fit, in which case nothing is dropped and the model
    sees the book exactly as written. Ranking only matters for long chapters.
    """
    if not passages:
        return {"passages": [], "tokens": 0, "truncated": False, "pages": []}

    ranked = rank_passages(passages, topic, chapter)
    chosen, used = [], 0
    for p in ranked:
        t = p.get("tokens") or max(1, len(p.get("content", "")) // 4)
        if used + t > token_budget:
            continue
        chosen.append(p)
        used += t

    chosen.sort(key=lambda p: (p.get("page_no", 0), p.get("chunk_index", 0)))
    return {
        "passages": chosen,
        "tokens": used,
        "truncated": len(chosen) < len(passages),
        "pages": sorted({p.get("page_no") for p in chosen if p.get("page_no")}),
    }


def format_source_block(passages: list) -> str:
    """Render passages with page markers the model is told to cite."""
    return "\n\n".join(
        f"[p.{p.get('page_no', '?')}] {p.get('content', '').strip()}"
        for p in passages
    )


GROUNDED_SYSTEM_PROMPT = """\
You are preparing classroom slides STRICTLY from the textbook extract supplied below.

═══ THE SOURCE IS THE ONLY PERMITTED AUTHORITY ═══
1. Every fact, definition, formula, number, name and example must come from the
   SOURCE TEXT. If it is not in the source, it does not go on a slide.
2. Do NOT add material from your own knowledge, even when you are certain it is
   correct and relevant. A true statement that is not in this book is still wrong
   here, because the teacher must be able to point to it in their copy.
3. Do NOT rephrase a definition into something more general or more advanced.
   Keep the book's own terminology and notation.
4. If the source does not contain enough material for the requested number of
   slides, produce FEWER slides. Never pad.
5. Every bullet must carry the page it came from in "pages" (e.g. [3]). A bullet
   you cannot cite is a bullet you must delete.
6. Worked examples, exercises and numbers must be reproduced faithfully — do not
   invent alternative numbers or "similar" examples.

═══ WHAT GOOD LOOKS LIKE ═══
  ✗ WRONG (correct in general, absent from the source):
      "Euclid's division algorithm is a special case of the division algorithm
       used in modern computer science."
  ✓ RIGHT (present in the source, cited):
      "Euclid's division algorithm is based on Euclid's division lemma, which
       states that for given positive integers a and b there exist unique
       integers q and r satisfying a = bq + r." [pages: 2]

═══ SLIDE STRUCTURE ═══
Slide 1 is type "title": a short title and a one-sentence subtitle; bullets [].
Middle slides are type "content": a 3-6 word title and 3-5 bullets, each one
complete sentence drawn from the source.
The final slide is type "summary": key takeaways, each still cited.

═══ OUTPUT ═══
Return ONLY valid JSON, no markdown fence:
{
  "title": "Presentation title taken from the chapter",
  "slides": [
    {"slideNumber": 1, "type": "title", "title": "...", "subtitle": "...",
     "bullets": [], "pages": [], "speakerNotes": "...", "imageSearchTerm": "..."},
    {"slideNumber": 2, "type": "content", "title": "...", "subtitle": "",
     "bullets": ["..."], "pages": [2], "speakerNotes": "...", "imageSearchTerm": "..."}
  ]
}"""


def build_grounded_user_prompt(
    *,
    slide_count: int,
    language: str,
    topic: str,
    ctx: dict,
    source_block: str,
) -> str:
    scope = " | ".join(
        v for v in (
            ctx.get("className"), ctx.get("subjectName"),
            ctx.get("chapterName"), ctx.get("topicName"),
        ) if v
    )
    return (
        f"Curriculum scope: {scope or topic}\n"
        f"Requested: up to {slide_count} slides, in {language}.\n"
        f"Focus: \"{topic}\".\n\n"
        "Use fewer slides if the source does not support that many.\n\n"
        "═══ SOURCE TEXT (the textbook extract — your only permitted facts) ═══\n"
        f"{source_block}\n"
        "═══ END OF SOURCE TEXT ═══"
    )
