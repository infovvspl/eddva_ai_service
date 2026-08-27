"""
PPT Generation — NestJS ai-bridge endpoints.

POST /ppt/generate         → Full presentation: Groq slide content + Serper images
POST /ppt/regenerate-slide → Single slide regeneration with fresh image
POST /ppt/search-image     → Serper image search only

The actual .pptx file is assembled client-side via PptxGenJS — this service
only returns structured JSON.  The image-proxy endpoint stays in NestJS because
browsers cannot send Authorization headers on bare <img src> requests.
"""
import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import requests as _requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ai_services.core.boards import board_instruction
from ai_services.core.serpapi_images import search_google_images
from ai_services.core.usage_logger import log_usage
from .base import get_llm

logger = logging.getLogger("ai_services.ppt")

_VISUAL_HINTS = frozenset({
    "map", "diagram", "photograph", "photo", "chart",
    "illustration", "artifact", "image", "picture",
    "excavation", "figure", "drawing",
})

# Stock libraries rank highly for "educational diagram" while showing generic
# classroom scenes. Demoted rather than dropped, so a slide still gets a picture
# when nothing better can be downloaded.
_STOCK_SOURCES = (
    "shutterstock", "alamy", "dreamstime", "istockphoto", "gettyimages",
    "123rf", "depositphotos", "bigstock", "magnific", "vecteezy",
    "freepik", "abposters", "stockphoto", "adobestock",
)

# Social and video results match the topic by their caption, not their picture.
# Facebook and Instagram answer image requests from lookaside.* with the poster's
# PROFILE PHOTO, and a YouTube thumbnail is usually the presenter's face — which
# is how a slide about metals ended up showing a portrait of a man in a suit.
_SOCIAL_SOURCES = (
    "instagram", "facebook", "fbsbx", "lookaside", "twitter", "x.com",
    "pinterest", "linkedin", "tiktok", "reddit", "threads",
    "ytimg", "youtube", "dailymotion",
)

# Sites that publish the kind of labelled figure a teacher actually wants.
# Promoted ahead of everything else when they appear in the results.
_EDUCATIONAL_SOURCES = (
    "byjus", "teachoo", "vedantu", "toppr", "ncert", "learncbse", "cbse",
    "geeksforgeeks", "khanacademy", "wikipedia", "wikimedia", "britannica",
    "sciencenotes", "chemistrylearner", "bbc.co.uk", "rsc.org", "olympiad",
    "shaalaa", "meritnation", "embibe", "physicswallah", "doubtnut",
    "tutorialspoint", "sciencedirect", "nasa.gov", "libretexts",
)

_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/",
    "Accept": "image/webp,image/apng,image/jpeg,image/png,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────────────────────────────────────────────────────────────
#  System prompts — ported verbatim from school-ppt.service.ts
#  Placeholders use {{TAG}} syntax so raw JSON braces in examples are safe.
# ─────────────────────────────────────────────────────────────────────────────
_GENERATE_PROMPT_TEMPLATE = """\
You are an expert educational presentation designer creating classroom PPT slides.

═══ CURRICULUM SCOPE (read this before anything else) ═══
{{CURRICULUM_CONTEXT}}

Every slide must teach content that genuinely belongs to the scope stated above, pitched at
the stated class level. Before writing each slide, check: "does this slide's content belong
to the STATED chapter/topic, or does it actually belong to a different topic — even one from
a different chapter or subject that is loosely thematically related?" If the latter, cut it
and write a slide that is in scope instead.
  ✗ If the topic is Nutrition (within a Life Processes chapter), do NOT add slides about food
    chains, producers/consumers, or ecosystems — those belong to a separate Ecology chapter.
  ✗ Do NOT introduce more advanced theory a student at the stated class level has not been
    taught yet, even when it is technically correct.

Two independent checks must BOTH pass for every slide:
  1. IN SCOPE — the content belongs to the stated chapter/topic.
  2. AT LEVEL — the content is taught at the stated class, in that class's own terms.
A slide that passes only one of these is a failed slide. Rewrite it.
  ✗ Class 10 "Corrosion": pitting corrosion, crevice corrosion, stress corrosion cracking,
    corrosion of polymers/ceramics, corrosion standards, materials-selection theory.
    These are in scope but far ABOVE Class 10 — they are wrong here.
  ✓ Class 10 "Corrosion": rusting of iron and its equation, conditions required for rusting,
    tarnishing of silver, green coating on copper, painting/oiling/greasing, galvanisation,
    anodising, alloying.
Use the terminology the stated board's textbook uses for this class — not a synonym from a
higher class or a different board.
═══ NO FILLER SLIDES (the single most common failure mode to avoid) ═══
Every content slide must teach actual subject matter a student is examined on. Do NOT produce
scaffold slides that could belong to any topic in any subject. These titles are forbidden:
  "Introduction", "Key Concepts Overview", "Important Definitions List", "Historical Background",
  "Theoretical Framework", "Practical Applications", "Case Studies Analysis", "Real World Examples",
  "Step by Step Procedures", "Common Misconceptions", "Frequently Asked Questions",
  "Glossary of Terms", "Chapter Summary Review", "Assessment and Evaluation", "Future Scope",
  "Conclusion".
If you are about to write one of those, you have run out of plan — go deeper into the real
material instead (a specific reaction, a named process, a worked numerical, a labelled structure).
  ✗ WRONG: a slide titled "Key Concepts Overview" on a Corrosion topic.
  ✓ RIGHT: "Rusting of Iron: Chemical Equation", "Conditions Required for Rusting",
    "Galvanisation as Rust Prevention".
{{COVERAGE_BLOCK}}
═══ JSON FORMAT RULES ═══
• All strings: DOUBLE QUOTES only — never single quotes.
• "bullets": proper JSON array — "bullets": ["...", "..."]
• Never write bullets=[...] — always use a colon.
• No trailing commas before } or ].

═══ BULLET POINT STANDARD (most important rule) ═══
Each bullet must be ONE complete sentence of 15–30 words. Under 15 words is REJECTED.
Each bullet must carry at least one CONCRETE SPECIFIC — a named substance or person, a number,
a date, a balanced equation, or a formula. A bullet that only restates the slide title in
general words is worthless; delete it and write a substantive one.
  ✗ VAGUE (rejected — no specific, too short):
      "The reactivity series is an important tool in understanding metal chemistry."
      "Metals can react with acids to produce hydrogen gas and a salt."
  ✓ SUBSTANTIVE (15–30 words, carries a specific):
      "Zinc displaces hydrogen from dilute hydrochloric acid to form zinc chloride and hydrogen gas: $$Zn + 2HCl \\rightarrow ZnCl_2 + H_2$$."
      "In the reactivity series potassium, sodium and calcium sit above hydrogen, so all three displace hydrogen from cold water."

═══ FACTUAL ACCURACY (check every bullet before you finalise it) ═══
Ask of each bullet: "is this the correct term, and would a subject teacher mark this right?"
Never reach for a related-sounding term that means something different.
  ✗ "The reaction of metals with acids is known as neutralization." — WRONG. Metal + acid is a
    DISPLACEMENT (redox) reaction. Neutralisation is acid + base → salt + water.
State relationships precisely: do not call something proportional, equal, or a type of something
else unless it genuinely is.

═══ ACADEMIC RIGOUR & COURSE CONTENT ═══
• Write the real course content for the stated class, subject, and chapter — not a generic overview.
• Any slide about a chemical reaction MUST include the balanced chemical equation in LaTeX.
• Any slide about a numerical, physics, chemistry or maths concept MUST include the relevant formula, a derivation step, or a worked numerical example with real numbers.
• Formulas must use standard LaTeX wrapped in double dollar signs: $$formula$$ (e.g. $$E = mc^2$$ or $$\\vec{F} = m\\vec{a}$$). Ensure double backslashes are used for LaTeX commands (e.g. \\\\frac for fraction) so they survive JSON parsing.

  ✗ TOO SHORT (fragment — rejected): "Located in Pakistan" / "Formula is F=ma"
  ✓ CORRECT LENGTH & FORMULA (15–30 words):
      "The Harappan civilisation (3300–1300 BCE) covered 1.25 million sq km across modern Pakistan and India."
      "Newton's Second Law states that force is directly proportional to acceleration, expressed as the formula $$\\vec{F} = m\\vec{a}$$."

═══ SLIDE STRUCTURE ═══
Generate EXACTLY {{SLIDE_COUNT}} slides about "{{TOPIC}}" in {{LANGUAGE}}.

Slide 1  → type "title"
  • title: short compelling title (5–8 words)
  • subtitle: one sentence (10–18 words) that previews what students will learn
  • bullets: []

Slides 2 – {{LAST_CONTENT}}  → type "content"
  • title: clear 3–6 word heading for this specific sub-topic
  • bullets: EXACTLY 5 bullet points — each a complete sentence of 15–30 words containing core definitions, mathematical derivations, formula steps, or numerical practice problems.
  • Each slide must cover a DIFFERENT sub-topic. Never repeat a sub-topic across two slides.
  • Work through the sub-topics in a sensible teaching order, spending proportionally more
    slides on the parts of the scope a textbook treats in most depth.

Slide {{SLIDE_COUNT}}  → type "summary"
  • title: "Key Takeaways" or similar
  • bullets: 5 bullet points, each a complete sentence of 15–30 words summarising one key fact or fundamental formula.

═══ IMAGE SEARCH TERM RULES ═══
imageSearchTerm MUST name a REAL, depictable subject that visually shows THAT slide's
specific content — built from the concrete nouns and examples in THAT slide's bullets,
NOT the abstract slide title and NOT the chapter name.
  1. Use the concrete examples the bullets mention. If a slide's bullets say metals occur
     natively as gold, silver and platinum → "native gold nugget specimen", NOT "metals in nature".
  2. Prefer a real object, mineral specimen, labelled science diagram, or clear photograph —
     something an image search returns a clean, on-topic picture of.
  3. Avoid generic industrial or process words (mining, factory, extraction, plant, machinery)
     UNLESS that slide is actually about that process.
  4. 4–7 words, ending with one visual type word: photograph, diagram, specimen, map, chart, illustration.

ALL slides:
  • speakerNotes: 2–3 sentences — a teaching tip or discussion question for the teacher

═══ OUTPUT ═══
Return ONLY valid JSON — no markdown fences, no commentary:
{
  "title": "Presentation Title",
  "slides": [
    { "slideNumber": 1, "type": "title", "title": "Engaging Main Title", "subtitle": "One sentence previewing what students will learn.", "bullets": [], "speakerNotes": "Ask students what they already know.", "imageSearchTerm": "topic overview educational photograph" },
    { "slideNumber": 2, "type": "content", "title": "Sub-Topic Heading", "subtitle": "", "bullets": ["Complete sentence of 15–30 words with a specific fact.", "Step-by-step formula derivation: $$y = mx + c$$.", "A sentence explaining a cause, effect, or significance.", "Worked numerical example: find $$x$$ given $$y=5$$.", "An interesting detail that deepens understanding."], "speakerNotes": "Ask students: which fact surprised you most?", "imageSearchTerm": "specific 4-7 word term with visual hint" }
  ]
}"""


def _build_curriculum_context(ctx: dict) -> str:
    """Render the class/subject/chapter/topic block that scopes the whole deck.

    Falls back to a bare topic line when the caller sent no curriculum context,
    so free-text decks (studio opened outside Topic Management) still work.
    """
    lines = []
    if ctx.get("className"):
        cls = ctx["className"]
        lines.append(f"• Class / grade level: {cls}  ← PITCH EVERY SLIDE AT THIS LEVEL")
        lines.append(
            f"• Write for a {cls} student sitting this board's exam. Use only the vocabulary, "
            f"depth and worked examples that {cls}'s own textbook uses. If a fact is true but "
            f"is taught in a higher class, LEAVE IT OUT — a technically correct slide above "
            f"{cls} level is a wrong slide here."
        )
    if ctx.get("subjectName"):
        lines.append(f"• Subject: {ctx['subjectName']}")
    if ctx.get("chapterName"):
        lines.append(f"• Chapter: {ctx['chapterName']}")
    if ctx.get("topicName"):
        lines.append(f"• Topic: {ctx['topicName']}")
        lines.append(
            "• SCOPE: this deck covers ONLY the topic named above — not the rest of the chapter."
        )
    elif ctx.get("chapterName"):
        lines.append(
            "• SCOPE: this deck covers the WHOLE chapter named above, distributed across its "
            "topics — not a single topic, and not material from any other chapter."
        )
    if lines and not ctx.get("className"):
        # No grade level resolved. Without an anchor the model drifts upward into
        # college-level treatment, so pin it to school level explicitly — and put
        # it first, where the known-class line would have been.
        lines.insert(
            0,
            "• Class / grade level: NOT SPECIFIED — infer the school grade this topic is "
            "normally taught in and stay at that level throughout. This is a SCHOOL classroom "
            "deck: never write at undergraduate or professional level.",
        )

    if not lines:
        return (
            f"• Topic: {ctx.get('topic') or 'as stated below'}\n"
            "• No class/subject/chapter was supplied — infer the school grade this topic is "
            "normally taught in and stay at that level. Never write above school level."
        )
    return "\n".join(lines)


def _build_coverage_block(sub_areas: "list[str] | None", slide_count: int) -> str:
    """Turn the pre-computed coverage plan into a fixed 1:1 slide plan.

    One sub-area per content slide, stated as an exact numbered mapping. An
    earlier version listed the sub-areas and let the model allocate slides across
    them; when there were more content slides than sub-areas it padded the deck
    with generic filler ("Key Concepts Overview", "Glossary of Terms",
    "Assessment and Evaluation") instead of going deeper on the real material.
    Removing the allocation decision removes that failure.
    """
    if not sub_areas:
        return ""
    listed = "\n".join(f"  Slide {i + 2}: {area}" for i, area in enumerate(sub_areas))
    return (
        "\n═══ FIXED SLIDE PLAN (this is the deck — follow it exactly) ═══\n"
        "Each content slide below is already assigned its sub-topic. Write that slide on that "
        "sub-topic. Do not reorder, do not merge, do not substitute, do not skip, and do not "
        "invent extra slides beyond this plan.\n"
        f"{listed}\n"
        "The slide title you write should name the actual concept, not the plan text verbatim.\n"
    )


def _build_generate_prompt(
    slide_count: int,
    language: str,
    topic: str,
    ctx: dict,
    sub_areas: "list[str] | None" = None,
) -> str:
    return (
        _GENERATE_PROMPT_TEMPLATE
        .replace("{{CURRICULUM_CONTEXT}}", _build_curriculum_context(ctx))
        .replace("{{COVERAGE_BLOCK}}", _build_coverage_block(sub_areas, slide_count))
        .replace("{{SLIDE_COUNT}}", str(slide_count))
        .replace("{{TOPIC}}", topic)
        .replace("{{LANGUAGE}}", language)
        .replace("{{LAST_CONTENT}}", str(slide_count - 1))
    )


def _decompose_ppt_coverage(ctx: dict, slide_count: int, institute_id) -> "list[str] | None":
    """Enumerate the distinct sub-areas the deck must cover, as a separate short call.

    Deciding coverage BEFORE writing any slides — then feeding it back as an explicit
    structural requirement — is what actually gets followed. Asking the main generation
    call to distribute content across a chapter on its own was already shown to fail in
    this codebase: it narrows to one sub-area and silently skips the rest. See
    _decompose_topic_coverage in bridge.py for the original finding.

    Returns None on any failure so generation falls back to the in-line instruction.
    """
    scope_name = ctx.get("topicName") or ctx.get("chapterName") or ctx.get("topic")
    if not scope_name:
        return None
    # A chapter-wide deck legitimately spans more ground than a single-topic deck.
    is_chapter_scope = not ctx.get("topicName") and bool(ctx.get("chapterName"))
    # Slide 1 is the title slide and the last is the summary — the rest are content.
    needed = max(1, slide_count - 2)
    try:
        result = get_llm().complete(
            # Deliberately never mentions a target/ceiling slide count. Naming a number
            # here anchors the model: asked for "at most 20" on a narrow Class 10 topic
            # it padded to 20 by escalating into university-level material (pitting
            # corrosion, stress corrosion cracking, corrosion of polymers). Asked
            # open-endedly it returns the honest ~11 class-appropriate sub-areas. The
            # slide budget is applied by the caller instead.
            system_prompt=(
                "You are a school curriculum expert. Given a class/subject/chapter/topic, list the "
                "distinct sub-areas a real school textbook covers under that exact scope, in proper "
                "teaching order.\n"
                "Rules:\n"
                "1. Everything you list must be material actually taught at the STATED CLASS LEVEL, "
                "from that class's own textbook treatment. This is the most important rule.\n"
                "   ✗ For Class 10 'Corrosion', do NOT list university materials-engineering topics "
                "like pitting corrosion, crevice corrosion, stress corrosion cracking, corrosion of "
                "polymers/ceramics, or corrosion standards and regulations.\n"
                "   ✓ For Class 10 'Corrosion', list what the school textbook covers: rusting of iron "
                "and its equation, conditions required for rusting, tarnishing of silver, green "
                "coating on copper, prevention by painting/oiling/greasing, galvanisation, anodising, "
                "alloying.\n"
                "2. List every genuine sub-area and then STOP. Do not extend the list with more "
                "advanced material, adjacent topics, or commentary once the class-level content is "
                "covered. A narrow topic may honestly have only 8-12 sub-areas; a whole chapter has "
                "more. Both are correct answers.\n"
                "3. Stay strictly inside the stated scope. If a topic is named, cover only that topic "
                "and not the rest of its chapter.\n"
                "4. Do not omit a major branch of the scope (e.g. do not cover only the human/animal "
                "side of something that also covers plants).\n"
                "5. NEVER include generic non-subject entries. These are forbidden: 'Introduction', "
                "'Key Concepts Overview', 'Important Definitions', 'Historical Background', "
                "'Theoretical Framework', 'Case Studies', 'Real World Examples', 'Glossary of Terms', "
                "'Frequently Asked Questions', 'Chapter Summary', 'Assessment and Evaluation', "
                "'Education and Awareness', 'Future Scope', 'Future Directions', 'Conclusion'. Every "
                "item must name actual subject matter a student is examined on."
            ),
            user_prompt=(
                f"Class: {ctx.get('className') or 'not specified'}\n"
                f"Subject: {ctx.get('subjectName') or 'not specified'}\n"
                f"Chapter: {ctx.get('chapterName') or 'not specified'}\n"
                f"Topic: {ctx.get('topicName') or '(whole chapter — no single topic selected)'}\n\n"
                f"List the distinct sub-areas of THIS "
                f"{'chapter' if is_chapter_scope else 'topic'} exactly as the "
                f"{ctx.get('className') or 'stated class'} textbook treats it. Each item: a short "
                "phrase (3-9 words) naming one specific concept.\n"
                'Respond as JSON: {"sub_areas": ["...", "...", ...]}'
            ),
            model=_MODEL,
            temperature=0.3,
            max_tokens=1200,
            json_mode=True,
            institute_id=institute_id,
        )
        sub_areas = result["content"].get("sub_areas")
        if not isinstance(sub_areas, list):
            return None
        cleaned = [str(s).strip() for s in sub_areas if str(s).strip()]
        if not cleaned:
            return None
        # The requested slide count is a ceiling applied here, not a target given to
        # the model. A scope with less material yields a shorter, honest deck.
        return cleaned[:needed]
    except Exception as exc:
        logger.warning("PPT coverage decomposition failed (%s) — falling back to in-line instruction", exc)
        return None


def _build_regenerate_prompt(
    slide_index: int, total_slides: int, topic: str, slide_type: str, ctx: "dict | None" = None,
) -> str:
    rules = {
        "title": (
            "Title slide: engaging title (5–8 words) + subtitle (10–18 words previewing "
            "what students will learn). bullets must be []."
        ),
        "summary": (
            "Summary slide: 5 bullet points, each a complete sentence of 15–30 words "
            "summarising one key fact or fundamental formula from the presentation."
        ),
        "content": (
            "Content slide: EXACTLY 5 bullet points, each a complete sentence of 15–30 words "
            "containing core definitions, mathematical derivations, formula steps, or numerical practice problems."
        ),
    }
    slide_rule = rules.get(slide_type, rules["content"])
    scope_block = (
        f"\n═══ CURRICULUM SCOPE ═══\n{_build_curriculum_context(ctx)}\n"
        "The replacement slide must stay strictly inside this scope — do not drift into a "
        "different topic or chapter, and do not exceed the stated class level.\n"
        if ctx else ""
    )
    return f"""\
You are a senior curriculum writer. Regenerate slide {slide_index + 1} of {total_slides} \
for a presentation about "{topic}". Type: "{slide_type}".
{scope_block}
═══ ACADEMIC RIGOUR & COURSE CONTENT ═══
• Write the real course content for the stated class, subject, and chapter — not a generic overview.
• For numerical, physics, chemistry, or math chapters, you MUST include clear step-by-step mathematical derivations, relevant formulas, and worked numerical examples.
• Formulas must use standard LaTeX wrapped in double dollar signs: $$formula$$ (e.g. $$E = mc^2$$ or $$\\vec{{F}} = m\\vec{{a}}$$). Ensure double backslashes are used for LaTeX commands (e.g. \\\\frac for fraction) so they survive JSON parsing.

BULLET RULE — each bullet must be ONE complete sentence of 15–30 words with a specific fact, formula, derivation step, or worked numerical example. \
Not a fragment. Not a long paragraph.
  ✗ TOO SHORT: "Located in Pakistan" / "Formula is F=ma"
  ✓ CORRECT: "Newton's Second Law states that force is directly proportional to acceleration, expressed as the formula $$\\vec{{F}} = m\\vec{{a}}$$."

{slide_rule}

IMAGE RULE: imageSearchTerm must name a REAL, depictable subject that visually shows THIS slide’s \
specific content — built from the concrete nouns and examples in the bullets above, not the abstract \
slide title and not the chapter name. Use the bullets’ real examples (e.g. metals occurring natively \
as gold/silver/platinum → "native gold nugget specimen", NOT "metals in nature"). Avoid generic \
industrial/process words (mining, extraction, factory, machinery) unless this slide is about that \
process. 4–7 words ending with a visual type hint (photograph, diagram, specimen, map, chart).

Return ONLY valid JSON:
{{
  "slideNumber": {slide_index + 1},
  "type": "{slide_type}",
  "title": "Slide Title",
  "subtitle": "",
  "bullets": ["Complete sentence with specific facts.", "Another complete sentence..."],
  "speakerNotes": "3-4 sentences of teaching tips and discussion questions for the teacher",
  "imageSearchTerm": "specific 4-7 word term matching THIS slide’s exact sub-topic with visual type hint"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dedupe_words(text: str) -> str:
    """Drop repeated words, keeping first occurrence and original order."""
    seen, out = set(), []
    for word in text.split():
        key = word.lower()
        if key not in seen:
            seen.add(key)
            out.append(word)
    return " ".join(out)


def _image_context(ctx: "dict | None") -> str:
    """Curriculum words to disambiguate an image search: "chapter subject class".

    Without these a slide titled "Development" returns child-development stock
    art and "Solutions" returns problem-solving clip art, because the word alone
    carries none of the subject it was written for.
    """
    if not ctx:
        return ""
    parts = [ctx.get("chapterName"), ctx.get("subjectName"), ctx.get("className")]
    return " ".join(str(p).strip() for p in parts if p and str(p).strip())


def _search_queries(term: str, slide_title: str, ctx: "dict | None" = None) -> "list[str]":
    """Queries to try for one slide, most specific first.

    A single query cannot serve both cases. "Bacteria: Basic Facts" needs the
    slide's own words — adding "Microorganisms Science Class 8" pulls back
    chapter-overview collages of algae and protozoa instead of bacteria. But a
    slide titled "Development" needs the chapter, or it returns child-development
    charts from an unrelated subject.

    So they are tried in order: the slide's own term first, the chapter-scoped
    version if that finds nothing usable, and the chapter itself as a last
    resort. The last one is deliberately broad — a slide with a roughly relevant
    picture beats a slide with an empty frame.
    """
    context = _image_context(ctx)
    term = (term or "").strip()
    title = (slide_title or "").strip()

    def visual(q: str) -> str:
        return q if any(h in q.lower() for h in _VISUAL_HINTS) else f"{q} diagram"

    queries: "list[str]" = []

    # 1. The slide's own term, when the model gave something usable.
    if len(term.split()) >= 3:
        queries.append(visual(term))

    # 2. Scoped to the chapter — disambiguates common words like "Development"
    #    or "Solutions" that mean something else outside this subject.
    if context:
        base = term if len(term.split()) >= 3 else title
        if base:
            queries.append(visual(_dedupe_words(f"{base} {context}")))
        queries.append(visual(_dedupe_words(context)))
    elif title:
        queries.append(visual(f"{title} educational"))

    # 3. Nothing else to go on.
    if not queries and title:
        queries.append(visual(f"{title} educational"))

    # Preserve order, drop repeats.
    seen, out = set(), []
    for q in queries:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _enrich_search_term(term: str, slide_title: str, ctx: "dict | None" = None) -> str:
    """The single best query for a slide — the first of _search_queries.

    Kept because /ppt/search-image exposes a one-shot search to the editor.
    """
    qs = _search_queries(term, slide_title, ctx)
    return qs[0] if qs else (slide_title or term or "").strip()


def _download_image_as_base64(url: str) -> "str | None":
    """Fetch an external image URL and return it as a base64 data URI."""
    try:
        r = _requests.get(url, headers=_IMAGE_HEADERS, timeout=8, allow_redirects=True)
        if not r.ok:
            return None
        ct = r.headers.get("content-type", "")
        if not ct.startswith("image/"):
            return None
        if len(r.content) < 1024:
            return None
        mime = ct.split(";")[0].strip()
        return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"
    except Exception:
        return None


def _source_rank(item: dict) -> int:
    """Order a result by how likely its picture matches its caption.

    0 — a teaching site: the image is the figure the caption describes.
    1 — anything else.
    2 — stock library: on-topic words, generic classroom photo.
    3 — social or video: the caption is on-topic but the image is a profile
        photo or a presenter's face, so these go last.
    """
    haystack = f"{item.get('source', '')} {item.get('imageUrl', '')}".lower()
    if any(s in haystack for s in _SOCIAL_SOURCES):
        return 3
    if any(s in haystack for s in _STOCK_SOURCES):
        return 2
    if any(s in haystack for s in _EDUCATIONAL_SOURCES):
        return 0
    return 1


def _is_stock_photo(item: dict) -> bool:
    """Kept for callers that only need the stock/not-stock distinction."""
    haystack = f"{item.get('source', '')} {item.get('imageUrl', '')}".lower()
    return any(s in haystack for s in _STOCK_SOURCES)


# Words that carry no topic meaning, so they must not count as a caption match.
_RELEVANCE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "with", "from",
    "by", "at", "is", "are", "this", "that", "these", "those", "its", "their",
    "between", "within", "into", "over", "educational", "slide", "topic",
    "overview", "content", "class", "chapter", "science", "study", "concept",
})


def _keywords(text: str) -> "set[str]":
    """Meaningful lowercase words from a phrase — drops stopwords, visual-type
    hints (diagram/photograph/…) and very short tokens."""
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) >= 3 and w not in _RELEVANCE_STOPWORDS and w not in _VISUAL_HINTS
    }


def _caption_is_relevant(caption: str, strong: "set[str]", broad: "set[str]") -> bool:
    """A returned image is on-topic if its caption shares the slide's specific
    subject words. `strong` = words from the slide's own search term (the exact
    sub-topic); `broad` = strong + slide-title words. One strong-word match, or
    two broad-word matches, is enough — captions are short."""
    cap = _keywords(caption)
    if not cap:
        return False
    if strong & cap:
        return True
    return len(broad & cap) >= 2


def _fetch_image_for_slide(search_term: str, slide_title: str, ctx: "dict | None" = None) -> dict:
    """Search Serper and return the first downloadable, ON-TOPIC image as base64.

    Results are ranked by source before anything is downloaded, so a teaching
    figure is taken ahead of a stock photo. We then only accept a result whose
    caption actually shares the slide's subject words (`_caption_is_relevant`),
    so a slightly-off search can't put an unrelated picture on the slide. If no
    query yields a relevant image, the best downloadable one is used as a
    fallback rather than leaving the slide blank.
    """
    queries = _search_queries(search_term, slide_title, ctx)
    strong = _keywords(search_term)
    broad = strong | _keywords(slide_title)
    first_url = None
    fallback: "dict | None" = None  # best downloadable image even if off-caption

    for attempt, query in enumerate(queries, 1):
        try:
            results = search_google_images(query, limit=10)
            if not results:
                continue
            # sorted() is stable, so Serper's own ranking survives within a tier.
            ordered = sorted(results, key=_source_rank)
            if first_url is None:
                first_url = ordered[0].get("imageUrl")

            for item in ordered:
                url = item.get("imageUrl")
                if not url:
                    continue
                relevant = _caption_is_relevant(item.get("title") or "", strong, broad)
                # Skip off-topic results once we already hold a fallback — no need
                # to download more of them.
                if not relevant and fallback is not None:
                    continue
                b64 = _download_image_as_base64(url)
                if not b64:
                    continue
                if relevant:
                    if attempt > 1:
                        logger.info("Slide image found on attempt %d (query %r)", attempt, query)
                    return {"imageUrl": url, "imageBase64": b64}
                if fallback is None:
                    fallback = {"imageUrl": url, "imageBase64": b64}
        except Exception as exc:
            logger.warning("Image search failed for %r: %s", query, exc)

    # No caption-relevant image anywhere — use the best downloadable one so the
    # slide isn't blank, then the raw best URL as a last resort (client proxies it).
    if fallback is not None:
        logger.info("No caption-relevant image for %r; using best available", search_term)
        return fallback
    if first_url:
        logger.info("No downloadable image for %r; returning URL only", search_term)
        return {"imageUrl": first_url, "imageBase64": None}

    logger.warning("No image at all for slide %r (tried %d queries)", slide_title, len(queries))
    return {"imageUrl": None, "imageBase64": None}


# A single-backslash LaTeX command whose first letter is also a JSON string escape
# gets eaten by the JSON parser: "\rightarrow" arrives as CR + "ightarrow",
# "\frac" as FF + "rac", "\times" as TAB + "imes", "\beta" as BS + "eta".
# The prompt asks for double backslashes, but models comply inconsistently, so
# repair it here instead of shipping mangled formulas to the slide.
_JSON_ESCAPE_TO_LETTER = {
    "\r": "r",   # rightarrow, rightleftharpoons, rho, right
    "\n": "n",   # neq, nu, nabla
    "\t": "t",   # times, theta, to, text, tau
    "\f": "f",   # frac, forall
    "\b": "b",   # beta, bar, binom
    "\v": "v",   # vec
}
_MATH_SPAN_RE = re.compile(r"\$\$[\s\S]*?\$\$|\$[^$\n]*?\$")


def _repair_latex_escapes(text: str) -> str:
    """Restore backslashes the JSON parser consumed inside $...$ math spans.

    Scoped to math spans on purpose: a control character next to a letter is
    meaningless there, whereas elsewhere (e.g. speaker notes) a real newline is
    legitimate and must be left alone.
    """
    if not isinstance(text, str) or "$" not in text:
        return text

    def fix_span(match: "re.Match[str]") -> str:
        span = match.group(0)
        for ch, letter in _JSON_ESCAPE_TO_LETTER.items():
            if ch in span:
                # Only when followed by letters — that is the mangled-macro shape.
                span = re.sub(re.escape(ch) + r"(?=[a-zA-Z])", "\\\\" + letter, span)
                # Any leftover bare control character is still not slide content.
                span = span.replace(ch, " ")
        return span

    return _MATH_SPAN_RE.sub(fix_span, text)


def _repair_slide_latex(slide: dict) -> dict:
    """Apply _repair_latex_escapes to every rendered text field of a slide."""
    if not isinstance(slide, dict):
        return slide
    out = dict(slide)
    for key in ("title", "subtitle"):
        if isinstance(out.get(key), str):
            out[key] = _repair_latex_escapes(out[key])
    if isinstance(out.get("bullets"), list):
        out["bullets"] = [
            _repair_latex_escapes(b) if isinstance(b, str) else b for b in out["bullets"]
        ]
    return out


def _parse_llm_json(content: "dict | str", endpoint: str) -> "tuple[dict | None, str | None]":
    """Return (parsed_dict, error_message).  Handles both pre-parsed and raw-string content."""
    if isinstance(content, dict):
        return content, None
    if isinstance(content, str):
        try:
            return json.loads(content), None
        except json.JSONDecodeError as exc:
            logger.error("%s: LLM returned invalid JSON — %s", endpoint, exc)
            return None, "AI returned malformed JSON. Please try again."
    return None, f"Unexpected LLM content type: {type(content).__name__}"


def _log(institute_id, vertical, model, result=None, success=True, error=None):
    """Fire-and-forget usage log — never blocks the response."""
    log_usage(
        institute_id=institute_id,
        institute_type=vertical or "school",
        feature_id="ppt_generate",
        feature_category="content_generation",
        model_used=result.get("model", model) if result else model,
        tokens_input=result.get("tokens_input", 0) if result else 0,
        tokens_output=result.get("tokens_output", 0) if result else 0,
        latency_ms=int(result.get("latency_ms", 0)) if result else 0,
        success=success,
        error_message=str(error) if error else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Views
# ─────────────────────────────────────────────────────────────────────────────
_MODEL = "openai/gpt-oss-120b"

_MAX_SLIDES = 25

# Output budget, constrained by Groq's per-key tokens-per-minute ceiling.
#
# Groq rejects a request outright (413) when prompt + max_tokens exceeds the
# key's TPM limit — regardless of how much quota is actually free. The on-demand
# keys in .env are 8000 TPM (an earlier comment here assumed 12000, which is why
# max_tokens=8500 produced "Request too large: Limit 8000, Requested 11113" and
# 500'd every ungrounded deck whenever the Gemini grounded path fell back). The
# real ceiling is enforced dynamically at the call site (see _groq_max_tokens):
# prompt + max_tokens must stay under GROQ_TPM_CEILING. This is only the ceiling
# on the OUTPUT half; a measured 16-slide deck emits ~3700 completion tokens.
_MAX_TOKENS = 6000

# The per-request TPM cap on the on-demand Groq keys. prompt + max_tokens must be
# below this or Groq 413s before generating a single token. Env-overridable so a
# Dev-tier upgrade (higher TPM) can lift it without a release.
_GROQ_TPM_CEILING = int(os.getenv("GROQ_TPM_CEILING", "8000"))


def _groq_max_tokens(*prompts: str, want: int = _MAX_TOKENS) -> int:
    """Largest output budget that still fits under the Groq per-request TPM cap.

    Groq counts prompt + max_tokens against the ceiling, so a fixed max_tokens
    413s as soon as the prompt grows (the coverage-planned PPT prompt is large).
    Estimate the prompt at ~4 chars/token, leave headroom, and never ask for more
    than fits — capped so a slide deck cannot be starved below a usable length.
    """
    # Estimate at ~3.5 chars/token (conservative — dense JSON/LaTeX prompts run
    # fewer chars per token than prose, so a 4-char estimate can undercount and
    # tip back over the ceiling). Better to shave a little output than to 413.
    est_prompt = int(sum(len(p) for p in prompts) / 3.5)
    room = _GROQ_TPM_CEILING - est_prompt - 500  # 500-token safety margin
    return max(1200, min(want, room))

# Serper allows ~1 req/s sustained; a small pool keeps 25 slides well under the
# NestJS 240s timeout without bursting hard enough to get rate-limited.
_IMAGE_WORKERS = 5

# Wall-clock budget for the optional coverage-planning call.
#
# LLMClient.complete() already retries 3 rounds across all ~20 keys before it
# raises, which takes roughly two minutes when Groq is unreachable. The planning
# step is best-effort — generation falls back to the in-line instruction without
# it — so it must not spend that budget: the caller (NestJS) gives up at 240s,
# and anything the planner burns is taken from the actual slide generation.
_COVERAGE_DEADLINE_S = 25


def _classify_gemini_failure(exc: Exception) -> str:
    """Turn a grounded-call exception into a precise, teacher-legible reason.

    All three collapse to "General knowledge" on the badge, but the reason is the
    difference between "top up the Gemini quota" and "the keys are wrong" — so it
    must survive to the log and the response instead of being flattened away.
    """
    from ai_services.core import gemini_client as _gc

    if _gc._looks_rate_limited(exc):
        return "gemini_exhausted"      # 429 / quota — the common one, and fixable
    if _gc._looks_overloaded(exc):
        return "gemini_overloaded"     # 503 high demand — transient, just retry
    if _gc._looks_key_rejected(exc):
        return "gemini_key_rejected"   # bad/disabled key
    if _gc._looks_model_unavailable(exc):
        return "gemini_model_unavailable"
    return "gemini_error"              # malformed JSON, empty response, transport


def _generate_grounded(
    *, passages, ctx, topic, slide_count, language, institute_id, vertical,
):
    """Write the deck from the school's own chapter.

    Returns (Response, None) on success, or (None, reason) to let the caller fall
    back to general-knowledge generation. The reason is threaded through to the
    teacher's badge so an indexed chapter that lands on "General knowledge" is
    self-explaining (Gemini quota vs bad key vs no usable passages) rather than a
    silent mystery.
    """
    from ai_services.core import gemini_client as _gc
    from ai_services.core import grounding as _gr

    if not _gc.is_available():
        # No key present or the SDK is not installed on this host. This is the
        # blanket failure that turns EVERY grounded deck into general knowledge.
        logger.warning(
            "Grounded PPT skipped: Gemini unavailable (no key or google-genai "
            "missing) — deck for %r will be general knowledge despite %d passages",
            ctx.get("chapterName") or topic, len(passages),
        )
        return None, "gemini_unavailable"

    # Slides need far fewer passages than detailed notes, and the grounded Gemini
    # call scales with prompt size — so cap the source context for PPT well below
    # the notes default (30k). This keeps the deck grounded in the book while
    # cutting the wait substantially. Env-tunable.
    _ppt_budget = int(os.getenv("PPT_GROUNDING_TOKEN_BUDGET", "12000"))
    selection = _gr.select_source(
        passages, ctx.get("topicName") or topic, ctx.get("chapterName") or "",
        token_budget=_ppt_budget,
    )
    if not selection["passages"]:
        logger.warning(
            "Grounded PPT skipped: %d passages supplied for %r but none had usable "
            "text after ranking — check the chapter's OCR quality",
            len(passages), ctx.get("chapterName") or topic,
        )
        return None, "no_relevant_passages"

    # Output budget scales with the deck: ~320 tokens/slide is ample for slide
    # bullets, and a smaller ceiling means Gemini finishes sooner. Capped at 8000.
    _out_tokens = max(2500, min(8000, slide_count * 320))
    try:
        result = _gc.complete_json(
            system_prompt=_gr.GROUNDED_SYSTEM_PROMPT,
            user_prompt=_gr.build_grounded_user_prompt(
                slide_count=slide_count, language=language, topic=topic,
                ctx=ctx, source_block=_gr.format_source_block(selection["passages"]),
            ),
            max_output_tokens=_out_tokens,
        )
    except Exception as exc:
        reason = _classify_gemini_failure(exc)
        logger.warning(
            "Grounded PPT fell back (%s) for %r: %s",
            reason, ctx.get("chapterName") or topic, exc,
        )
        _log(institute_id, vertical, "gemini", success=False, error=exc)
        return None, reason

    _log(institute_id, vertical, result.get("model", "gemini"), result=result)

    data = result["content"]
    slides = data.get("slides") or []
    if not slides:
        logger.warning("Grounded PPT fell back: Gemini returned no slides for %r",
                       ctx.get("chapterName") or topic)
        return None, "gemini_error"

    with ThreadPoolExecutor(max_workers=_IMAGE_WORKERS) as pool:
        images = list(pool.map(
            lambda s: _fetch_image_for_slide(
                s.get("imageSearchTerm", ""), s.get("title", ""), ctx,
            ),
            slides,
        ))
    data["slides"] = [{**s, **img} for s, img in zip(slides, images)]

    # The caller shows this to the teacher, so a grounded deck is never mistaken
    # for one written from general knowledge.
    data["source"] = {
        "grounded": True,
        "passagesUsed": len(selection["passages"]),
        "passagesAvailable": len(passages),
        "pages": selection["pages"],
        "truncated": selection["truncated"],
    }
    return Response({"success": True, "data": data}), None


@api_view(["POST"])
def generate_presentation(request):
    """
    POST /ppt/generate

    Body: { topic, slideCount?, language?,
            className?, subjectName?, chapterName?, topicName? }
    Returns: { success: true, data: { title, slides: [ SlideObject ] } }

    SlideObject includes imageUrl + imageBase64 (Serper images, fetched concurrently).
    PPT assembly (.pptx) is done client-side via PptxGenJS — only JSON is returned here.
    """
    topic = (request.data.get("topic") or "").strip()
    if not topic:
        return Response({"error": "topic is required"}, status=status.HTTP_400_BAD_REQUEST)

    slide_count = max(3, min(_MAX_SLIDES, int(request.data.get("slideCount") or 5)))
    language = (request.data.get("language") or "English").strip()
    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", None) or request.headers.get("X-Vertical") or request.headers.get("x-vertical") or "school"
    board = getattr(request, "board", None) or request.headers.get("X-Board") or request.headers.get("x-board") or None

    ctx = {
        "topic": topic,
        "className": (request.data.get("className") or "").strip(),
        "subjectName": (request.data.get("subjectName") or "").strip(),
        "chapterName": (request.data.get("chapterName") or "").strip(),
        "topicName": (request.data.get("topicName") or "").strip(),
    }

    # ── Grounded path ────────────────────────────────────────────────────────
    # When the caller supplies passages from the school's own chapter, the deck is
    # written from those alone. This runs on Gemini rather than Groq: a chapter is
    # several thousand tokens and Groq rejects any request whose prompt plus
    # max_tokens exceeds the key's 12k TPM ceiling.
    passages = request.data.get("sourcePassages") or []
    # When passages were supplied but grounding fell back, this holds the precise
    # reason so the teacher's badge can explain an indexed chapter that still shows
    # "General knowledge" (Gemini quota vs bad key vs unusable OCR) instead of a
    # generic, unactionable message.
    grounded_fallback_reason = None
    if passages:
        grounded, grounded_fallback_reason = _generate_grounded(
            passages=passages, ctx=ctx, topic=topic, slide_count=slide_count,
            language=language, institute_id=institute_id, vertical=vertical,
        )
        if grounded is not None:
            return grounded
        logger.warning(
            "Grounded generation unavailable (%s) — falling back to general knowledge",
            grounded_fallback_reason,
        )

    # Decide coverage before writing any slides — see _decompose_ppt_coverage.
    # Run it under a wall-clock deadline: if the LLM is unhealthy this step would
    # otherwise consume the whole request budget retrying, and it is optional.
    sub_areas = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            sub_areas = pool.submit(
                _decompose_ppt_coverage, ctx, slide_count, institute_id
            ).result(timeout=_COVERAGE_DEADLINE_S)
    except FuturesTimeout:
        logger.warning(
            "PPT coverage planning exceeded %ss — generating without a slide plan",
            _COVERAGE_DEADLINE_S,
        )
    except Exception as exc:  # never let the optional pre-step break generation
        logger.warning("PPT coverage planning failed (%s) — generating without a slide plan", exc)

    if sub_areas:
        # Keep the deck exactly as long as the plan (+ title + summary). A scope with
        # genuinely less material yields a shorter, honest deck rather than one padded
        # out to the requested length with filler slides.
        slide_count = len(sub_areas) + 2

    system_prompt = board_instruction(board) + _build_generate_prompt(
        slide_count, language, topic, ctx, sub_areas
    )
    scope_line = " | ".join(
        v for v in (ctx["className"], ctx["subjectName"], ctx["chapterName"], ctx["topicName"]) if v
    )
    user_prompt = (
        f'Write a {slide_count}-slide educational presentation about: "{topic}". '
        + (f"Curriculum scope: {scope_line}. " if scope_line else "")
        + f"Language: {language}. "
        "Each bullet must be one complete sentence of 15–30 words with a specific fact. "
        "Not fragments. Not essays."
    )

    # Single attempt on purpose. LLMClient.complete() already rotates through all
    # ~20 keys three times (~2 min) before raising, so an outer retry loop only
    # multiplied that: three attempts took ~5.5 min against a caller that times
    # out at 240s, turning a fast failure into a long hang with a generic error.
    # The loop also could not help with malformed JSON — that is handled below,
    # outside the call — so its temperature fallback never actually applied.
    llm = get_llm()
    # Fit the output budget under the Groq per-request TPM ceiling. Without this,
    # a large coverage-planned prompt plus a fixed max_tokens exceeds the 8000 cap
    # and Groq 413s before producing anything — which is exactly how an ungrounded
    # fallback (Gemini overloaded) turned into a 500 for the teacher.
    _gen_max_tokens = _groq_max_tokens(system_prompt, user_prompt)
    if _gen_max_tokens < _MAX_TOKENS:
        logger.info(
            "PPT ungrounded: capping max_tokens to %d (prompt ~%d tok) to fit Groq TPM %d",
            _gen_max_tokens, (len(system_prompt) + len(user_prompt)) // 4, _GROQ_TPM_CEILING,
        )
    try:
        llm_result = llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=_MODEL,
            temperature=0.6,
            max_tokens=_gen_max_tokens,
            json_mode=True,
            institute_id=institute_id,
        )
    except Exception as exc:
        logger.warning("PPT generate failed: %s", exc)
        _log(institute_id, vertical, _MODEL, success=False, error=exc)
        return Response(
            {"error": "Failed to generate slide content. Please try again."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    _log(institute_id, vertical, _MODEL, result=llm_result)

    data, err = _parse_llm_json(llm_result["content"], "ppt/generate")
    if err:
        return Response({"error": err}, status=status.HTTP_502_BAD_GATEWAY)

    slides = [_repair_slide_latex(s) for s in (data.get("slides") or [])]
    # Fetched concurrently: sequentially this cost 1s of sleep plus a download per
    # slide, which at 25 slides ran up against the caller's request timeout.
    with ThreadPoolExecutor(max_workers=_IMAGE_WORKERS) as pool:
        images = list(pool.map(
            lambda s: _fetch_image_for_slide(
                s.get("imageSearchTerm", ""), s.get("title", ""), ctx,
            ),
            slides,
        ))

    data["slides"] = [{**slide, **image} for slide, image in zip(slides, images)]
    # State the source explicitly on the general-knowledge path too. When passages
    # were supplied but grounding fell back, carry the precise reason; when none
    # were supplied the chapter simply is not indexed. Either way the badge stops
    # guessing. (If the caller sent no passages, NestJS labels it 'not_indexed'.)
    if passages:
        data["source"] = {
            "grounded": False,
            "reason": grounded_fallback_reason or "unavailable",
            "passagesAvailable": len(passages),
        }
    return Response({"success": True, "data": data})


@api_view(["POST"])
def regenerate_slide(request):
    """
    POST /ppt/regenerate-slide

    Body: { slideIndex, topic, currentSlide, totalSlides }
    Returns: { success: true, data: SlideObject }
    """
    slide_index = request.data.get("slideIndex")
    topic = request.data.get("topic")
    if topic is None or slide_index is None:
        return Response(
            {"error": "slideIndex and topic are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    slide_index = int(slide_index)
    total_slides = int(request.data.get("totalSlides") or 5)
    current_slide = request.data.get("currentSlide") or {}
    slide_type = (
        "title" if slide_index == 0
        else "summary" if slide_index == total_slides - 1
        else "content"
    )

    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", None) or request.headers.get("X-Vertical") or request.headers.get("x-vertical") or "school"
    board = getattr(request, "board", None) or request.headers.get("X-Board") or request.headers.get("x-board") or None

    ctx = {
        "topic": topic,
        "className": (request.data.get("className") or "").strip(),
        "subjectName": (request.data.get("subjectName") or "").strip(),
        "chapterName": (request.data.get("chapterName") or "").strip(),
        "topicName": (request.data.get("topicName") or "").strip(),
    }
    has_ctx = any(ctx[k] for k in ("className", "subjectName", "chapterName", "topicName"))

    llm = get_llm()
    try:
        llm_result = llm.complete(
            system_prompt=board_instruction(board) + _build_regenerate_prompt(
                slide_index, total_slides, topic, slide_type, ctx if has_ctx else None
            ),
            user_prompt=(
                f'Regenerate slide {slide_index + 1} about "{topic}". '
                f"Current slide data: {json.dumps(current_slide)}"
            ),
            model=_MODEL,
            temperature=0.8,
            max_tokens=2048,
            json_mode=True,
            institute_id=institute_id,
        )
    except Exception as exc:
        logger.error("PPT regenerate-slide LLM error: %s", exc)
        _log(institute_id, vertical, _MODEL, success=False, error=exc)
        return Response(
            {"error": "Failed to regenerate slide. Please try again."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    _log(institute_id, vertical, _MODEL, result=llm_result)

    new_slide, err = _parse_llm_json(llm_result["content"], "ppt/regenerate-slide")
    if err:
        return Response({"error": err}, status=status.HTTP_502_BAD_GATEWAY)

    new_slide = _repair_slide_latex(new_slide)
    image = _fetch_image_for_slide(
        new_slide.get("imageSearchTerm", ""),
        new_slide.get("title", ""),
        ctx,
    )
    new_slide.update(image)
    return Response({"success": True, "data": new_slide})


@api_view(["POST"])
def search_image(request):
    """
    POST /ppt/search-image

    Body: { searchTerm }
    Returns: { success: true, imageUrl, imageBase64 }
    """
    search_term = (request.data.get("searchTerm") or "").strip()
    if not search_term:
        return Response(
            {"error": "searchTerm is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    image = _fetch_image_for_slide(search_term, search_term)
    return Response({"success": True, **image})
