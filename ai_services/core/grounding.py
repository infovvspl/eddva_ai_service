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


def citation_label(p: dict) -> str:
    """The inline marker a passage is cited by, e.g. "p.12" or "Lecture: Ch 4 intro".

    Passages default to "ebook" when untagged, so callers that never learned
    about lecture grounding (ppt.py, older cached requests) keep citing pages
    exactly as before.
    """
    if p.get("source") == "lecture":
        title = (p.get("source_title") or "the lecture").strip()
        return f"Lecture: {title}"
    page = p.get("page_no")
    return f"p.{page}" if page else "p.?"


def rank_passages(passages: list, topic: str, chapter: str = "") -> list:
    """Order passages by relevance to the topic, keeping the source's own order
    as the tie-breaker so a selection still reads in teaching sequence."""
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
        scored.append((score, p.get("page_no", 0) or 0, p.get("chunk_index", 0) or 0, p))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [p for _, _, _, p in scored]


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
        return {"passages": [], "tokens": 0, "truncated": False, "pages": [], "citations": []}

    ranked = rank_passages(passages, topic, chapter)
    chosen, used = [], 0
    for p in ranked:
        t = p.get("tokens") or max(1, len(p.get("content", "")) // 4)
        if used + t > token_budget:
            continue
        chosen.append(p)
        used += t

    # Group by source so a mixed selection still reads as "the book, then the
    # lecture" rather than interleaved out of teaching order; page/chunk order
    # is preserved within each group.
    chosen.sort(key=lambda p: (
        p.get("source", "ebook"), p.get("page_no", 0) or 0, p.get("chunk_index", 0) or 0,
    ))
    return {
        "passages": chosen,
        "tokens": used,
        "truncated": len(chosen) < len(passages),
        "pages": sorted({p.get("page_no") for p in chosen if p.get("page_no")}),
        "citations": sorted({citation_label(p) for p in chosen}),
    }


def format_source_block(passages: list) -> str:
    """Render passages with the citation marker the model is told to cite."""
    return "\n\n".join(
        f"[{citation_label(p)}] {p.get('content', '').strip()}"
        for p in passages
    )


def build_grounded_system_prompt(has_ebook: bool = True, has_lecture: bool = False) -> str:
    """The grounded-deck system prompt, worded for whichever source(s) are supplied.

    Callers that never pass has_lecture (the default) get exactly the original
    textbook-only wording, so this is a drop-in replacement for the old
    GROUNDED_SYSTEM_PROMPT constant.
    """
    if has_ebook and has_lecture:
        source_desc = "the textbook extract AND the lecture transcript excerpts"
        citation_note = (
            "5. Every bullet drawn from the textbook must carry the page it came from in \"pages\"\n"
            "   (e.g. [3]). A bullet drawn from the lecture transcript instead can leave \"pages\": []\n"
            "   — the transcript has no page numbers — but every fact must still be traceable to one\n"
            "   source or the other.\n"
            "6. Prefer the textbook's own wording for definitions and terminology; treat the lecture\n"
            "   transcript as the teacher's own spoken explanations and examples — useful context, but\n"
            "   a rough transcript, so write it up cleanly without changing what was actually said.\n"
        )
    elif has_lecture:
        source_desc = "the lecture transcript excerpts"
        citation_note = (
            "5. The transcript has no page numbers, so leave \"pages\": [] on every slide — the rule\n"
            "   is that every fact must come from the transcript below, not that it be individually\n"
            "   cited.\n"
            "6. The transcript is raw speech-to-text: expect filler words, run-on sentences and\n"
            "   occasional recognition errors. Write clean, well-formed slide content from it without\n"
            "   changing what the teacher actually said or inventing detail it does not support.\n"
        )
    else:
        source_desc = "the textbook extract"
        citation_note = (
            "5. Every bullet must carry the page it came from in \"pages\" (e.g. [3]). A bullet you\n"
            "   cannot cite is a bullet you must delete.\n"
            "6. Do NOT rephrase a definition into something more general or more advanced. Keep the\n"
            "   book's own terminology and notation.\n"
        )

    return f"""\
You are preparing classroom slides STRICTLY from {source_desc} supplied below.

═══ THE SOURCE IS THE ONLY PERMITTED AUTHORITY ═══
1. Every fact, definition, formula, number, name and example must come from the
   SOURCE TEXT. If it is not in the source, it does not go on a slide.
2. Do NOT add material from your own knowledge, even when you are certain it is
   correct and relevant. A true statement that is not in the source is still wrong
   here, because the teacher must be able to point to it.
3. If the source does not contain enough material for the requested number of
   slides, produce FEWER slides. Never pad.
4. Worked examples, exercises and numbers must be reproduced faithfully — do not
   invent alternative numbers or "similar" examples.
{citation_note}
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
The final slide is type "summary": key takeaways, each still cited where possible.

═══ OUTPUT ═══
Return ONLY valid JSON, no markdown fence:
{{
  "title": "Presentation title taken from the source",
  "slides": [
    {{"slideNumber": 1, "type": "title", "title": "...", "subtitle": "...",
     "bullets": [], "pages": [], "speakerNotes": "...", "imageSearchTerm": "..."}},
    {{"slideNumber": 2, "type": "content", "title": "...", "subtitle": "",
     "bullets": ["..."], "pages": [2], "speakerNotes": "...", "imageSearchTerm": "..."}}
  ]
}}"""


# Kept for any caller that still imports the constant directly — identical to
# build_grounded_system_prompt() with its defaults (ebook-only).
GROUNDED_SYSTEM_PROMPT = build_grounded_system_prompt()


def build_grounded_user_prompt(
    *,
    slide_count: int,
    language: str,
    topic: str,
    ctx: dict,
    source_block: str,
    has_ebook: bool = True,
    has_lecture: bool = False,
) -> str:
    scope = " | ".join(
        v for v in (
            ctx.get("className"), ctx.get("subjectName"),
            ctx.get("chapterName"), ctx.get("topicName"),
        ) if v
    )
    source_label = (
        "the textbook extract AND the lecture transcript excerpts — your only permitted facts"
        if has_ebook and has_lecture
        else "the lecture transcript excerpts — your only permitted facts" if has_lecture
        else "the textbook extract — your only permitted facts"
    )
    return (
        f"Curriculum scope: {scope or topic}\n"
        f"Requested: up to {slide_count} slides, in {language}.\n"
        f"Focus: \"{topic}\".\n\n"
        "Use fewer slides if the source does not support that many.\n\n"
        f"═══ SOURCE TEXT ({source_label}) ═══\n"
        f"{source_block}\n"
        "═══ END OF SOURCE TEXT ═══"
    )
