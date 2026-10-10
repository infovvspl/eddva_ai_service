"""The V2 planner: a storyboard, not a deck.

The planner answers one question per slide - *what should this slide teach,
and what should the learner see?* It does not choose a layout, a colour, an
image source or a coordinate. Those belong to the composer and the decision
engine, and keeping them out of the prompt is what makes the rest of the
pipeline testable without an LLM.

It deliberately reuses the existing content generation rather than replacing
it: the same ``LLMClient``, the same router capability, the same grounding
passages. What is new is the *shape* of what is asked for. V1 asks for a title
and some bullets and one search term. V2 asks for a teaching objective, typed
content blocks, the terms worth emphasising, and a description of what the
learner should see - which is the information the visual pipeline needs and
V1 never had.

Structured output: this client exposes ``json_mode`` but no provider-level
JSON schema parameter, so "structured" here means schema-in-the-prompt plus
hard validation on the way back, which is the convention the rest of the
service already uses. Malformed output is a controlled error, never a crash,
and never a half-built deck.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Slide types the planner may choose from. Deliberately a subset of SlideType:
# the planner should not be reaching for DATA or ANATOMY unless the content
# really is that, and a shorter list measurably improves compliance.
PLANNER_SLIDE_TYPES = (
    "title", "concept", "process", "comparison", "cause_effect",
    "anatomy", "timeline", "formula", "example", "summary", "quiz",
)

# The finer-grained shape of what a slide teaches. Declared by the planner,
# then VALIDATED and if necessary corrected by core.ppt_v2.content - the
# planner is the starting point, never the last word.
PLANNER_CONTENT_TYPES = (
    "process", "cycle", "anatomy_structure", "geometric_diagram", "comparison",
    "classification", "cause_effect", "timeline", "geographic_map", "graph_data",
    "formula_equation", "worked_example", "flowchart_algorithm",
    "system_architecture", "experiment_procedure", "case_study",
    "definition_terminology", "concept_explanation", "summary", "revision_quiz",
    "title",
)

# Advisory only. The decision engine owns the final call; see
# ``visuals.preferred_strategy``. The planner knows the pedagogy, the engine
# knows what the deployment can actually produce.
PLANNER_VISUAL_KINDS = ("photo", "illustration", "diagram", "chart", "none")

_SCHEMA = """{
  "title": "<deck title, <= 70 characters>",
  "slides": [
    {
      "slideType": "one of: %(types)s",
      "contentType": "one of: %(content_types)s",
      "title": "<slide title, <= 60 characters>",
      "emphasisText": "<the 1-3 words inside title that carry the idea, or \\"\\">",
      "subtitle": "<one short clarifying line, or \\"\\">",
      "learningObjective": "<what a student can do after this slide, one sentence>",
      "blocks": [
        {
          "kind": "bullet | step | definition | stat | equation | callout | compare_left | compare_right | question | answer",
          "label": "<2-4 word handle, e.g. \\"Sunlight\\" or \\"1. Absorb\\">",
          "text": "<one complete sentence, 12-28 words, one specific fact>",
          "icon": "<single plain noun for an icon, e.g. \\"sun\\", \\"water\\", \\"leaf\\">"
        }
      ],
      "highlightedTerms": ["<term a student must remember>"],
      "visual": {
        "kind": "one of: %(kinds)s",
        "description": "<what the picture shows, as objects and their arrangement>",
        "objects": ["<concrete thing visible in THE SAME single picture, at most 4>"],
        "mustShow": ["<detail the picture is useless without>"],
        "relationships": ["<what the picture must SHOW happening, e.g. \"an incident ray strikes the mirror\">"],
        "avoid": ["<concept that must NOT appear in the picture, or omit>"],
        "annotations": [
          {"text": "<label pointing at part of the picture, <= 6 words>",
           "anchor": [0.5, 0.5]}
        ],
        "altText": "<one sentence describing the picture>"
      },
      "speakerNotes": "<what the teacher says or asks, 1-2 sentences>"
    }
  ]
}""" % {"types": " | ".join(PLANNER_SLIDE_TYPES),
        "content_types": " | ".join(PLANNER_CONTENT_TYPES),
        "kinds": " | ".join(PLANNER_VISUAL_KINDS)}

_RULES = """RULES

1. Produce EXACTLY %(count)d slide objects. Not one more, not one fewer.
2. Every slide needs a learningObjective. If you cannot name what the student
   learns, the slide should not exist.
3. 2 to 4 blocks per slide. A slide with six bullets is a handout, not a slide.
4. Each block "text" is ONE complete sentence with ONE specific fact. No
   fragments, no essays, no restating the title. Each block "label" NAMES
   its content ("Glycolysis", "Lactic acid") - never "Step 1" or "Point 2";
   numbering is drawn for you.
5. "slideType" must describe the SHAPE of the content:
   - process      an ordered sequence of stages
   - comparison   exactly two things set against each other
   - cause_effect one thing leading to another
   - anatomy      a structure whose named parts matter
   - formula      an equation and what its terms mean
   - concept      an idea and its supporting facts (use when unsure)
   Choosing a type you cannot fill is worse than choosing "concept".
6. EVERY slide except "summary" and "quiz" MUST fill "visual.description"
   and "visual.objects". A slide with an empty visual block gets no picture
   at all, which is not what you meant when you left it blank.
7. "visual.objects" lists things that appear together in ONE picture, at
   most four. It is not a list of examples. "block on a rough floor, block on
   a smooth floor, wheel on a road, boat in water" describes four different
   pictures, and asking for all of them at once means no real figure can
   match and a good one gets rejected. Pick the single scene the slide needs.
8. "visual.description" says WHAT IS IN THE PICTURE - objects and how they are
   arranged. It is not a teaching sentence.
   GOOD: "a broad green leaf in sunlight, with water rising through the stem
          and gas moving at the leaf surface"
   BAD:  "an image explaining the importance of photosynthesis"
9. "visual.annotations" are labels that will be drawn as real text pointing at
   the picture. Anchors are [x, y] from 0 to 1 inside the picture. At most 6.
   Only add them when the picture has parts worth naming.
10. "visual.relationships" is what the picture must SHOW HAPPENING, not just
   contain. This is the difference between a picture that is on topic and one
   that teaches the slide: "a mirror" is on topic, "an incident ray strikes
   the mirror and a reflected ray leaves it" is the lesson. One to three
   short phrases. Omit for a slide whose picture is a single object.
   Each one must be VISIBLE IN A PICTURE. "sunlight falls on the leaf" is
   visible; "each organelle labelled with its function" is a requirement
   about the lesson, not something a drawing can show, and asking for it
   makes good diagrams look wrong.
11. "visual.avoid" names concepts that would make the picture WRONG for this
   slide, not merely absent from it. Use it when the topic has a well-known
   counterpart that search results tend to bundle in. An introductory
   photosynthesis slide should avoid "respiration"; a mitosis slide should
   avoid "meiosis". Leave it out when nothing competes, and never list
   something the slide is actually comparing.
12. "visual.kind" is a hint, not a decision:
   - photo        a real camera could take this (a volcano, apparatus, a crop)
   - illustration a drawn figure is clearer (a process, something microscopic)
   - diagram      a labelled structure
   - chart        quantitative - a graph, axes, proportions
   - none         the slide teaches better with no picture at all
13. STAY INSIDE THE SYLLABUS FOR THE STATED CLASS. This is the rule most often
   broken and it makes a deck useless. Asked for Class 8 Photosynthesis, a
   deck that reaches for thylakoid membranes, the Calvin cycle, ATP/NADPH or
   C6H12O6 is wrong - not too advanced, WRONG, because the student has not
   met those terms and the exam will not ask for them. Explain the same idea
   with the vocabulary that class actually has. If you are unsure whether a
   term is in the syllabus, do not use it.
14. Use ONLY the supplied source material when source material is given below.
    Do not add facts from general knowledge to a grounded deck.
15. "contentType" is the SHAPE of what the student must understand, and it
    decides how the slide is drawn. Decide it BEFORE writing the blocks:
    - process                an ordered sequence of stages
    - cycle                  stages that come back to the start
    - anatomy_structure      one structure whose named parts matter
    - geometric_diagram      shapes, angles, rays or lengths that must be exact
    - comparison             two things set against each other
    - classification         groups, and what belongs in each group
    - cause_effect           one thing leading to another
    - timeline               events in date order (needs real dates)
    - geographic_map         where things are, on a map
    - graph_data             quantities on axes (needs the numbers or the trend)
    - formula_equation       an equation and what each symbol means
    - worked_example         a problem solved step by step (needs the working)
    - flowchart_algorithm    steps with decisions and branches
    - system_architecture    named components and how they connect
    - experiment_procedure   apparatus, method, observation, conclusion
    - case_study             a real situation that illustrates the idea
    - definition_terminology terms, what they mean, and an example of each
    - concept_explanation    an idea and its supporting facts (use when unsure)
    - summary, revision_quiz, title   as named
    The content must be able to carry the type: a timeline with no dates or a
    worked example with no working is replaced automatically, so choose
    honestly. Still fill "slideType" from its own list.

Return ONLY the JSON object. No commentary, no markdown fences."""


class PlannerError(Exception):
    """A planner failure the caller can act on.

    ``reason`` is a short stable code (``llm_error``, ``malformed_json``,
    ``invalid_schema``, ``slide_count``) so it can go into a log line and a
    fallback decision without string matching on a message.
    """

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class PlannerResult:
    title: str
    slides: List[Dict[str, Any]]
    attempts: int = 1
    grounded: bool = False
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


def build_planner_prompts(topic: str, slide_count: int, language: str = "English",
                          class_name: str = "", subject: str = "", chapter: str = "",
                          board: str = "", objectives: Optional[Sequence[str]] = None,
                          passages: Optional[Sequence[str]] = None,
                          board_rules: str = "",
                          part: Optional[Dict[str, Any]] = None,
                          focus_topic: str = "",
                          source_queries: Optional[Sequence[str]] = None) -> Tuple[str, str]:
    """(system_prompt, user_prompt) for one planning call.

    With ``part`` - {"outline": [...], "start": i, "deck_size": n} - the call
    writes only some slides of an already-outlined deck (see plan_deck's
    parallel path); everything else about the request is identical.
    """
    scope = " | ".join(x for x in (class_name, subject, chapter, board) if x)

    system = "\n\n".join(filter(None, [
        board_rules.strip(),
        "You plan educational slide decks. You decide what each slide TEACHES "
        "and what the learner should SEE. You do not choose layouts, colours, "
        "fonts or positions - another system does that.",
        "Return JSON matching this shape exactly:",
        _SCHEMA,
        _RULES % {"count": slide_count},
    ]))

    if part:
        outline = part["outline"]
        start = part["start"]
        mine = outline[start:start + slide_count]
        parts = [
            'You are writing slides %d-%d of a %d-slide lesson on: "%s". The whole '
            "deck is outlined below so your slides fit with the others; write ONLY "
            "the slides listed under YOUR SLIDES, in that order, keeping each one's "
            "title, slideType and focus." % (start + 1, start + slide_count,
                                               part["deck_size"], topic),
            "DECK OUTLINE:\n" + "\n".join(
                "%d. [%s] %s - %s" % (i + 1, o.get("slideType") or "concept",
                                      o.get("title") or "", o.get("focus") or "")
                for i, o in enumerate(outline)),
            "YOUR SLIDES:\n" + "\n".join(
                "%d. [%s] %s - %s" % (start + i + 1, o.get("slideType") or "concept",
                                      o.get("title") or "", o.get("focus") or "")
                for i, o in enumerate(mine)),
        ]
    else:
        parts = ['Plan a %d-slide lesson on: "%s".' % (slide_count, topic)]
    if class_name:
        # Repeated at the top of the user prompt because a scope line buried
        # among four other fields does not change what the model writes: the
        # first real run of this planner produced Calvin-cycle content for a
        # Class 8 request.
        parts.append(
            "AUDIENCE: %s students%s. Every sentence, term and example must be "
            "inside the %s syllabus. Do not use vocabulary from a higher class."
            % (class_name, " following the %s syllabus" % board if board else "",
               class_name))
    if scope:
        parts.append("Curriculum scope: %s." % scope)
    if focus_topic:
        # A topic deck is given its whole chapter as the source (the book is
        # indexed by chapter), and was told to follow the book's order - so a
        # deck asked for one topic could drift into the chapter's others.
        parts.append(
            'FOCUS: this deck covers ONLY the topic "%s", one part of the chapter '
            '"%s". Every slide must teach something about "%s". The source material '
            "below is the whole chapter: use only what it says about this topic, and "
            "do not make slides about the chapter's other topics."
            % (focus_topic, chapter or "this chapter", focus_topic))
    parts.append("Language: %s." % language)
    if objectives:
        parts.append("Learning objectives to cover: %s."
                     % "; ".join(o for o in objectives if o))

    if part:
        # A batch reads the pages its own slides teach.
        body = source_block(passages, topic, chapter, budget=slice_budget(),
                            queries=slide_queries(mine, focus_topic or topic))
    elif source_queries:
        body = source_block(passages, topic, chapter, budget=slice_budget(), queries=source_queries)
    elif focus_topic:
        body = source_block(passages, topic, chapter, budget=topic_budget(), query=focus_topic)
    else:
        body = source_block(passages, topic, chapter)
    if body:
        parts.append(
            "\nSOURCE MATERIAL (from the school's own chapter). Use ONLY these "
            "facts; do not supplement them from general knowledge. Follow the "
            "book: its definitions, its terms, its examples, its numbers and its "
            "order. Do not introduce a process, term, number or example the book "
            "does not contain - if the book stops at a level of detail, so does "
            "the deck. Where the book is silent on something, leave it out.\n\n%s"
            % body)

    parts.append("\nReturn exactly %d slides." % slide_count)
    return system, "\n".join(parts)


# The book, as the planner reads it.
#
# Passages arrive from the backend as {"page_no", "chunk_index", "content",
# "tokens", "source"}. This module used to read only "text", so every passage
# of every real chapter was silently skipped: decks logged as grounded were
# written from general knowledge, under a "use ONLY these facts" heading
# followed by nothing. The selection is now the one V1's grounded path uses
# (core.grounding): ranked by relevance to the topic when the chapter is too
# long, then put back in reading order, each passage tagged with its page.
_SOURCE_TOKEN_BUDGET = 10000
# A topic deck needs that topic's pages, not its whole chapter.
_TOPIC_SOURCE_TOKEN_BUDGET = 4500
# One batch of slides (or of fact checks) needs the pages its own slides teach.
# Sending the whole chapter with every batch was ~48k input tokens for one
# 7-slide deck in a minute - enough to be rate-limited and fall back to V1.
_SLICE_TOKEN_BUDGET = 3500


def _env_budget(name: str, default: int) -> int:
    try:
        return max(500, int(os.getenv(name, "") or default))
    except ValueError:
        return default


def topic_budget() -> int:
    return _env_budget("PPT_TOPIC_TOKEN_BUDGET", _TOPIC_SOURCE_TOKEN_BUDGET)


def slice_budget() -> int:
    return _env_budget("PPT_SLICE_TOKEN_BUDGET", _SLICE_TOKEN_BUDGET)


def max_parallel() -> int:
    """How many LLM calls one deck may have in flight at once."""
    try:
        return max(1, int(os.getenv("PPT_PLANNER_MAX_PARALLEL", "") or 3))
    except ValueError:
        return 3


def call_timeout() -> float:
    """Seconds one outline/batch/fact-check call may take (PPT_PLANNER_CALL_TIMEOUT_S).

    Healthy calls take 15-40s. A live deck lost its V2 build to one request
    that hung, unanswered, until the 120s capability timeout; cut at this cap
    instead, the call is retried while the deck still has time.
    """
    try:
        return max(20.0, float(os.getenv("PPT_PLANNER_CALL_TIMEOUT_S", "") or 60))
    except ValueError:
        return 60.0


def backup_capability() -> str:
    """The routing capability a second attempt uses (PPT_PLANNER_BACKUP_CAPABILITY).

    "grounded" routes to GLM-5.3-Flash on the same account. While the primary
    planner model was stalling requests for a minute or more, GLM answered in
    about 9s and wrote a complete, valid V2 plan in 17s - so a call that
    stalled on the primary is retried there. Empty keeps the retry on the
    primary.
    """
    return (os.getenv("PPT_PLANNER_BACKUP_CAPABILITY", "grounded") or "").strip()


def batch_retry_until() -> float:
    """How far into a deck a failed batch may still retry (PPT_PLANNER_RETRY_UNTIL_S)."""
    try:
        return max(30.0, float(os.getenv("PPT_PLANNER_RETRY_UNTIL_S", "") or 150))
    except ValueError:
        return 150.0


def slides_query(slides: Sequence[Any]) -> str:
    """What some planned (or outlined) slides are about, for picking their pages."""
    parts: List[str] = []
    for s in slides or []:
        if not isinstance(s, dict):
            continue
        parts += [str(s.get(k) or "") for k in ("title", "focus", "subtitle", "emphasisText")]
        for b in s.get("blocks") or []:
            if isinstance(b, dict):
                parts += [str(b.get("label") or ""), str(b.get("text") or "")]
    return " ".join(p for p in parts if p)
# A hard ceiling on the source text whatever the token estimate says.
_MAX_SOURCE_CHARS = 40000


def _passage_items(passages: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in passages or []:
        if isinstance(p, dict):
            text = str(p.get("content") or p.get("text") or "").strip()
            if text:
                items.append(dict(p, content=text))
        else:
            text = str(p or "").strip()
            if text:
                items.append({"content": text})
    return items


def source_block(passages: Optional[Sequence[Any]], topic: str = "",
                 chapter: str = "", *, query: str = "",
                 queries: Optional[Sequence[str]] = None,
                 budget: Optional[int] = None) -> str:
    """The book text one call is given, or "" when there is none.

    By default the chapter, ranked by topic and chapter, up to the full budget
    (PPT_GROUNDING_TOKEN_BUDGET). With ``query`` the passages are ranked by it
    alone - a topic, or what a batch of slides is about - and ``budget`` caps
    how much is sent, so a call carries the pages it needs rather than the
    whole chapter every time.

    With ``queries`` (one per slide) every slide gets a fair share: each
    slide's best passage, then each slide's next best, and so on until the
    budget is spent. Ranked as one query, a batch about mirrors and lenses
    filled its whole budget with lens pages and left the mirror slide with
    nothing to be checked against.
    """
    items = _passage_items(passages)
    if not items:
        return ""
    from ai_services.core import grounding

    if budget is None:
        budget = _env_budget("PPT_GROUNDING_TOKEN_BUDGET", _SOURCE_TOKEN_BUDGET)
    wanted = [q for q in (queries or []) if q and q.strip()]
    if wanted:
        chosen = _fair_share(items, wanted, budget)
    elif query:
        chosen = _fair_share(items, [query], budget)
    else:
        chosen = grounding.select_source(items, topic, chapter, token_budget=budget)["passages"]
    # One passage longer than the whole budget is still the book: take its start.
    chosen = chosen or items[:1]
    if any(p.get("page_no") or p.get("source") == "lecture" for p in chosen):
        body = grounding.format_source_block(chosen)
    else:
        body = "\n\n---\n\n".join(p["content"] for p in chosen)
    return body[:_MAX_SOURCE_CHARS]


# --- validation -------------------------------------------------------------

def _rank_in_chapter(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Passages by relevance to ``query``, weighting each word by how rare it is here.

    In a chapter called "Light", "light" is on every page; matching it says
    nothing, while "reflection" picks out the right pages. Plain word overlap
    gave "Reflection of Light" the refraction pages ("speed of light") as
    readily as the reflection ones. Book order breaks ties.
    """
    import math

    from ai_services.core import grounding

    want = grounding._terms(query)
    terms = [grounding._terms(p.get("content", "")) for p in items]
    n = len(items)
    df = {w: sum(1 for t in terms if w in t) for w in want}
    weight = {w: math.log((n + 1) / (df[w] + 1)) for w in want}
    scored = [(sum(weight[w] for w in want & t), p.get("page_no", 0) or 0,
               p.get("chunk_index", 0) or 0, k, p)
              for k, (p, t) in enumerate(zip(items, terms)) if t]
    scored.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return [x[-1] for x in scored]


def _fair_share(items: List[Dict[str, Any]], queries: Sequence[str], budget: int) -> List[Dict[str, Any]]:
    """Round-robin over each query's ranking until the budget is spent; book order."""
    rankings = [_rank_in_chapter(items, q) for q in queries]
    chosen, taken, used = [], set(), 0
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth >= len(ranking) or id(ranking[depth]) in taken:
                continue
            passage = ranking[depth]
            cost = passage.get("tokens") or max(1, len(passage.get("content", "")) // 4)
            if used + cost > budget:
                continue
            taken.add(id(passage))
            chosen.append(passage)
            used += cost
    chosen.sort(key=lambda p: (p.get("source", "ebook"), p.get("page_no", 0) or 0,
                               p.get("chunk_index", 0) or 0))
    return chosen


def slide_queries(slides: Sequence[Any], fallback: str = "") -> List[str]:
    """One page-picking query per slide (see source_block's ``queries``)."""
    out = [slides_query([s]) for s in slides or []]
    return [q or fallback for q in out if q or fallback]


def validate_payload(data: Any, slide_count: int) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """(title, slides, problems).

    Structural only. Anything a tolerant composer can repair - a missing
    subtitle, an odd slide type, an out-of-range anchor - is left to
    ``slide_from_plan``; this catches what makes a deck unusable.
    """
    problems: List[str] = []
    if not isinstance(data, dict):
        return None, [], ["planner returned %s, not an object" % type(data).__name__]

    raw_slides = data.get("slides")
    if not isinstance(raw_slides, list):
        return None, [], ["planner returned no 'slides' array"]

    slides: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_slides):
        if not isinstance(item, dict):
            problems.append("slide %d is %s, not an object" % (i + 1, type(item).__name__))
            continue
        if not str(item.get("title") or "").strip():
            problems.append("slide %d has no title" % (i + 1))
            continue
        slides.append(item)

    if not slides:
        problems.append("planner produced no usable slides")

    title = str(data.get("title") or "").strip() or None
    return title, slides, problems


def enforce_slide_count(slides: List[Dict[str, Any]], wanted: int) -> Tuple[List[Dict[str, Any]], str]:
    """(slides, action). Never silently violates the requested count.

    Trimming is safe - the planner front-loads the important slides, and a
    caller who asked for 2 must get 2. Padding is not safe: inventing a slide
    the planner did not write means inventing content, so a short deck is
    reported rather than filled.
    """
    if len(slides) == wanted:
        return slides, "exact"
    if len(slides) > wanted:
        logger.info("ppt_v2: planner returned %d slides for a %d-slide request; trimming",
                    len(slides), wanted)
        return slides[:wanted], "trimmed"
    return slides, "short"


# --- the call ---------------------------------------------------------------

# Output budget for one planner call. A fixed 6000 tokens fitted a 4-slide
# deck and cut a 10-slide one off mid-JSON: the planner model reasons before
# it answers, and the reasoning is billed against the same budget (a 10-slide
# replay stopped at finish_reason=length with 2394 of 6000 tokens spent
# thinking). So the budget grows with the deck. It is a ceiling, not a target -
# the model stops when the JSON is complete - and the cap keeps a long deck
# inside the content route's 120s timeout at the observed ~150 tokens/s.
PLANNER_BASE_TOKENS = 3000      # reasoning plus the deck envelope
PLANNER_TOKENS_PER_SLIDE = 800  # one slide's blocks, visual and notes
PLANNER_MIN_TOKENS = 6000
PLANNER_MAX_TOKENS = 16000


def planner_output_budget(slide_count: int) -> int:
    """max_tokens for a deck of ``slide_count`` slides."""
    wanted = PLANNER_BASE_TOKENS + PLANNER_TOKENS_PER_SLIDE * max(1, int(slide_count or 1))
    return max(PLANNER_MIN_TOKENS, min(PLANNER_MAX_TOKENS, wanted))


def revise_slides(*, slides: Sequence[Dict[str, Any]], note: str, topic: str,
                  language: str = "English", class_name: str = "", subject: str = "",
                  chapter: str = "", board: str = "", board_rules: str = "",
                  passages: Optional[Sequence[Any]] = None,
                  institute_id: Any = None, llm: Any = None, model: str = "",
                  capability: str = "content",
                  on_usage: Optional[Any] = None,
                  focus_topic: str = "") -> List[Dict[str, Any]]:
    """Rewrite a few planned slides to a correction note. Same count, same order.

    Used to fix the slides that strayed from the chapter without paying for a
    whole new plan: a live 8-slide deck spent 68s re-planning everything to
    remove one word from one slide. Raises PlannerError when the rewrite is
    unusable; the caller then keeps what it had.
    """
    if llm is None:
        from ai_services.core.llm_client import get_llm
        llm = get_llm()
    count = len(slides)
    system, user = build_planner_prompts(
        topic=topic, slide_count=count, language=language, class_name=class_name,
        subject=subject, chapter=chapter, board=board, passages=passages,
        board_rules=board_rules, focus_topic=focus_topic,
        source_queries=slide_queries(slides, focus_topic or topic))
    user += ("\n\nThese %d slides were already planned for this deck:\n%s\n\n%s\n"
             "Rewrite ONLY these slides, in the same order, keeping each one's "
             "slideType and purpose. Return {\"title\": \"\", \"slides\": [...]} "
             "with exactly %d slide objects."
             % (count, json.dumps(list(slides), ensure_ascii=False), note, count))
    try:
        result = llm.complete(
            system_prompt=system, user_prompt=user, model=model or None,
            temperature=0.3, max_tokens=planner_output_budget(count), json_mode=True,
            institute_id=institute_id, feature="ppt_v2_plan", capability=capability,
        ) or {}
    except Exception as exc:
        raise PlannerError("llm_error", str(exc)[:300])
    if on_usage:
        on_usage(result, 1)
    content = result.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlannerError("malformed_json", str(exc)[:200])
    _title, revised, _problems = validate_payload(content, count)
    if len(revised) != count:
        raise PlannerError("slide_count", "rewrite returned %d of %d slides"
                           % (len(revised), count))
    return revised


def focus_topic_for(ctx: Dict[str, Any]) -> str:
    """The topic a deck must stay inside, or "" for a whole-chapter deck.

    A deck opened from a topic carries both names; one opened from a chapter
    (or with the topic named the same as its chapter) covers the chapter.
    """
    topic = str((ctx or {}).get("topicName") or "").strip()
    chapter = str((ctx or {}).get("chapterName") or "").strip()
    return topic if topic and topic.lower() != chapter.lower() else ""


def revise_args(plan_args: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of plan_deck's arguments revise_slides takes."""
    keep = ("topic", "language", "class_name", "subject", "chapter", "board",
            "board_rules", "passages", "institute_id", "llm", "on_usage", "focus_topic")
    return {k: plan_args[k] for k in keep if k in plan_args}


# --- parallel planning -------------------------------------------------------
#
# One call writing a whole 10-slide deck took 60-140s live - most of a
# teacher's wait. Instead: a short outline call (titles, shapes, what each
# slide covers, in the book's order), then the slides written in parallel
# batches, each seeing the whole outline so the deck still reads as one.
# Any failure falls back to the single call, so this can only be faster.
PARALLEL_MIN_SLIDES = 5
PARALLEL_BATCH_SIZE = 3

_OUTLINE_SCHEMA = """{
  "title": "<deck title, <= 70 characters>",
  "slides": [
    {"slideType": "one of: %(types)s",
     "title": "<slide title, <= 60 characters>",
     "focus": "<what this slide teaches, one sentence>"}
  ]
}""" % {"types": " | ".join(PLANNER_SLIDE_TYPES)}

_OUTLINE_RULES = """RULES

1. Produce EXACTLY %(count)d outline entries.
2. The first slide is the "title" slide. The last is a "summary" (a "quiz"
   may come just before it).
3. Follow the order of the SOURCE MATERIAL when there is one. No two slides
   teach the same thing.
4. "slideType" describes the SHAPE of the content: process (ordered stages),
   comparison (exactly two things), cause_effect, anatomy (named parts),
   formula (an equation and its terms), concept (when unsure).
5. Every slide must be inside the audience's syllabus and the source."""


# --- automatic slide count -------------------------------------------------------
#
# A teacher asking for "the deck for this chapter" should not have to guess how
# many slides it takes. The outline reads the chapter and gives one slide to
# each distinct idea the book teaches; the range only keeps that sane. A topic
# is narrower than its chapter (whose whole text arrives either way), so it
# gets a narrower range; with no book, the range is the usual lesson size.
AUTO_MIN_SLIDES = 5
AUTO_MAX_SLIDES = 10     # never above views/ppt._MAX_SLIDES
AUTO_TOPIC_RANGE = (5, 10)
AUTO_NO_BOOK_RANGE = (6, 10)


def auto_slide_range(passages: Optional[Sequence[Any]], topic: str = "", chapter: str = "",
                     topic_only: bool = False) -> Tuple[int, int]:
    """(fewest, most) slides the content can sensibly take."""
    body = source_block(passages, topic, chapter)
    if not body:
        return AUTO_NO_BOOK_RANGE
    if topic_only:
        return AUTO_TOPIC_RANGE
    words = len(body.split())
    lo = max(AUTO_MIN_SLIDES, min(AUTO_MAX_SLIDES - 2, round(words / 300) + 3))
    hi = max(lo + 2, min(AUTO_MAX_SLIDES, round(words / 150) + 4))
    return lo, hi


def parallel_enabled() -> bool:
    return (os.getenv("PPT_PLANNER_PARALLEL", "true") or "").strip().lower() \
        in ("1", "true", "yes", "on")


def build_outline_prompts(*, topic: str, slide_count: int, language: str = "English",
                          class_name: str = "", subject: str = "", chapter: str = "",
                          board: str = "", objectives: Optional[Sequence[str]] = None,
                          passages: Optional[Sequence[Any]] = None,
                          board_rules: str = "",
                          slide_range: Optional[Tuple[int, int]] = None,
                          focus_topic: str = "") -> Tuple[str, str]:
    """(system, user) for the outline call. Same audience, scope and source.

    With ``slide_range`` the outline decides the count itself, within it.
    """
    rules = _OUTLINE_RULES % {"count": slide_count}
    if slide_range:
        lo, hi = slide_range
        rules = rules.replace(
            "1. Produce EXACTLY %d outline entries." % slide_count,
            "1. Decide how many slides this content needs, between %d and %d. Give "
            "one slide to each distinct idea the SOURCE MATERIAL teaches about the "
            "topic: do not pad thin material to fill slides, and do not cram two "
            "ideas onto one slide. Cover what the source teaches about the topic, "
            "and nothing it does not." % (lo, hi))
    system = "\n\n".join(filter(None, [
        board_rules.strip(),
        "You outline educational slide decks: which slides, in what order, what "
        "each one teaches and what shape its content has. Another step writes the "
        "slides themselves.",
        "Return JSON matching this shape exactly:",
        _OUTLINE_SCHEMA,
        rules,
    ]))
    _system, user = build_planner_prompts(
        topic=topic, slide_count=slide_count, language=language, class_name=class_name,
        subject=subject, chapter=chapter, board=board, objectives=objectives,
        passages=passages, board_rules=board_rules, focus_topic=focus_topic)
    if slide_range:
        lo, hi = slide_range
        user = user.replace('Plan a %d-slide lesson on:' % slide_count,
                            'Plan a lesson (%d to %d slides, as the content needs) on:' % (lo, hi))
        user = user.replace("Return exactly %d slides." % slide_count,
                            "Return between %d and %d outline entries." % (lo, hi))
    else:
        user = user.replace("Return exactly %d slides." % slide_count,
                            "Return exactly %d outline entries." % slide_count)
    return system, user


def _json_content(result: Dict[str, Any]) -> Any:
    content = result.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlannerError("malformed_json", str(exc)[:200])
    return content


def _plan_in_parallel(*, topic: str, slide_count: int, language: str, class_name: str,
                      subject: str, chapter: str, board: str, board_rules: str,
                      objectives, passages, institute_id, llm, model: str,
                      capability: str, temperature: float, on_usage,
                      started: float, note=None,
                      auto_range: Optional[Tuple[int, int]] = None,
                      on_batch: Optional[Any] = None,
                      focus_topic: str = "") -> PlannerResult:
    note = note or _event_sink(None)
    common = dict(topic=topic, language=language, class_name=class_name,
                  subject=subject, chapter=chapter, board=board,
                  objectives=objectives, passages=passages, board_rules=board_rules,
                  focus_topic=focus_topic)

    def ask(system: str, user: str, max_tokens: int, backup: bool = False) -> Dict[str, Any]:
        # A second attempt goes to the backup model when one is configured.
        use = (backup_capability() or capability) if backup else capability
        try:
            result = llm.complete(
                system_prompt=system, user_prompt=user, model=model or None,
                temperature=temperature, max_tokens=max_tokens, json_mode=True,
                institute_id=institute_id, feature="ppt_v2_plan",
                capability=use, timeout_s=call_timeout(),
            ) or {}
        except Exception as exc:
            raise PlannerError("llm_error", str(exc)[:300])
        if on_usage:
            on_usage(result, 1)
        return result

    # 1. outline
    note("outline", "Planning the lesson outline")
    system, user = build_outline_prompts(slide_count=slide_count, slide_range=auto_range,
                                         **common)
    outline_tokens = 3000 + 120 * (auto_range[1] if auto_range else slide_count)
    try:
        first = ask(system, user, outline_tokens)
    except PlannerError as exc:
        # A stalled or rate-limited outline: once more, on the backup model.
        if not (backup_capability() and _batch_retry_ok(str(exc), started)):
            raise
        logger.warning("ppt_v2: outline failed (%s); retrying on %s", str(exc)[:120], backup_capability())
        time.sleep(RETRY_PAUSE_S)
        first = ask(system, user, outline_tokens, backup=True)
    content = _json_content(first)
    if not isinstance(content, dict) or not isinstance(content.get("slides"), list):
        raise PlannerError("invalid_schema", "outline has no slides array")
    outline = [o for o in content["slides"] if isinstance(o, dict) and o.get("title")]
    if auto_range:
        lo, hi = auto_range
        if len(outline) < lo:
            raise PlannerError("slide_count", "outline chose %d slides, fewer than %d"
                               % (len(outline), lo))
        outline = outline[:hi]
        slide_count = len(outline)
    elif len(outline) < slide_count:
        raise PlannerError("slide_count", "outline has %d of %d slides"
                           % (len(outline), slide_count))
    outline = outline[:slide_count]
    logger.info("ppt_v2: outline done | %d slides %.1fs",
                len(outline), time.perf_counter() - started)
    note("outline", "%s %d slides: %s" % (
        "Chose" if auto_range else "Outlined", len(outline),
        ", ".join((o.get("title") or "")[:28] for o in outline[:3])
        + ("…" if len(outline) > 3 else "")), "done")

    # 2. the slides, in parallel batches
    starts = list(range(0, slide_count, PARALLEL_BATCH_SIZE))

    def fill(start: int) -> List[Dict[str, Any]]:
        count = min(PARALLEL_BATCH_SIZE, slide_count - start)
        span = ("slide %d" % (start + 1) if count == 1
                else "slides %d–%d" % (start + 1, start + count))
        key = "write-%d" % start
        note(key, "Writing %s" % span)
        system, user = build_planner_prompts(
            slide_count=count, part={"outline": outline, "start": start,
                                     "deck_size": slide_count}, **common)
        last: Optional[PlannerError] = None
        for attempt in (1, 2):
            retry = "" if attempt == 1 else "\n\n" + _BROKEN_REPLY_NOTE
            if attempt == 2:
                time.sleep(RETRY_PAUSE_S)     # a rate-limited provider needs a moment
            try:
                result = ask(system, user + retry, planner_output_budget(count), backup=attempt == 2)
                _title, slides, problems = validate_payload(_json_content(result), count)
                if problems and not slides:
                    raise PlannerError("invalid_schema", "; ".join(problems[:3]))
                slides, action = enforce_slide_count(slides, count)
                if action == "short":
                    raise PlannerError("slide_count", "batch at %d returned %d of %d"
                                       % (start + 1, len(slides), count))
                note(key, "Wrote %s" % span, "done")
                if on_batch:
                    # Image decks start painting these now, not when every
                    # batch is in: a live 15-slide deck waited 49s for the
                    # last two batches before painting anything.
                    try:
                        on_batch(start, slides, slide_count)
                    except Exception:
                        logger.exception("ppt_v2: on_batch failed for slides from %d", start + 1)
                return slides
            except PlannerError as exc:
                last = exc
                if not _batch_retry_ok(str(exc), started):
                    break
        raise last or PlannerError("llm_error", "batch failed")

    # A 16-slide chapter used to fire six batches at once.
    with ThreadPoolExecutor(max_workers=max(1, min(len(starts), max_parallel()))) as pool:
        batches = list(pool.map(fill, starts))
    slides = [slide for batch in batches for slide in batch]
    return PlannerResult(
        title=str(content.get("title") or topic), slides=slides, attempts=1,
        grounded=bool(source_block(passages, topic, chapter)),
        model=str(first.get("model") or model), usage=first,
        latency_s=time.perf_counter() - started)


def _event_sink(on_event):
    """note(key, text, state="active") - a no-op when nobody is listening."""
    def note(key, text, state="active"):
        if on_event:
            try:
                on_event(key, text, state)
            except Exception:
                logger.exception("ppt_v2: activity event failed")
    return note


RETRY_PAUSE_S = 2.0
_BROKEN_REPLY_NOTE = ("Your previous reply was cut off or was not valid JSON. "
                      "Return ONLY the complete JSON object, nothing before or after it.")
# A retry is only worth making while there is room for it inside the request:
# the backend waits 240s for the whole deck, and the planner is the first stage.
_RETRY_WINDOW_S = 90.0
_NOT_RETRYABLE = ("402", "401", "403", "credit", "not configured", "api key",
                  "unauthor", "forbidden", "payment")


def _batch_retry_ok(error_text: str, started: float) -> bool:
    """A batch retry: one call of at most call_timeout(), so it may run later."""
    text = (error_text or "").lower()
    if any(marker in text for marker in _NOT_RETRYABLE):
        return False
    return (time.perf_counter() - started) < batch_retry_until()


def _worth_retrying(error_text: str, started: float) -> bool:
    text = (error_text or "").lower()
    if any(marker in text for marker in _NOT_RETRYABLE):
        return False
    return (time.perf_counter() - started) < _RETRY_WINDOW_S


def plan_deck(*, topic: str, slide_count: int, language: str = "English",
              class_name: str = "", subject: str = "", chapter: str = "",
              board: str = "", board_rules: str = "",
              objectives: Optional[Sequence[str]] = None,
              passages: Optional[Sequence[Any]] = None,
              institute_id: Any = None, llm: Any = None,
              model: str = "", capability: str = "content",
              max_tokens: Optional[int] = None, temperature: float = 0.5,
              on_usage: Optional[Any] = None,
              revision_note: str = "",
              on_event: Optional[Any] = None,
              auto_range: Optional[Tuple[int, int]] = None,
              focus_topic: str = "",
              on_batch: Optional[Any] = None) -> PlannerResult:
    """Produce a validated storyboard, or raise :class:`PlannerError`.

    At most two LLM calls. The second is made when the first produced a deck
    of the wrong length, or a reply that broke (cut off, not JSON, a provider
    hiccup) - provided there is time left for it. A broken first reply used to
    end the V2 deck on the spot, and the teacher got the old-style fallback
    and generated again; one retry is cheaper than that second generation.
    Errors that will repeat - no credit, a rejected key - are never retried.

    ``revision_note`` is appended to the request when the caller is asking for
    a corrected plan (see scope.revision_note).

    ``on_batch(start, slides, deck_size)`` is called as each batch of a
    parallel plan is written (slides ``start``.. of a ``deck_size``-slide
    deck), before the others finish. Not called on the one-call path.
    """
    if llm is None:
        from ai_services.core.llm_client import get_llm
        llm = get_llm()

    if max_tokens is None:
        max_tokens = planner_output_budget(slide_count)
    attempts = 0
    last_problem = ""
    retry_note = ""
    deadline_start = time.perf_counter()
    grounded = bool(source_block(passages, topic, chapter))

    note = _event_sink(on_event)
    # With auto_range the outline chooses the count; slide_count is then only
    # the target the one-call fallback uses if the outline cannot be had.
    if ((slide_count >= PARALLEL_MIN_SLIDES or auto_range)
            and parallel_enabled() and not revision_note):
        try:
            return _plan_in_parallel(
                topic=topic, slide_count=slide_count, language=language,
                class_name=class_name, subject=subject, chapter=chapter, board=board,
                board_rules=board_rules, objectives=objectives, passages=passages,
                institute_id=institute_id, llm=llm, model=model, capability=capability,
                temperature=temperature, on_usage=on_usage, started=deadline_start,
                note=note, auto_range=auto_range, focus_topic=focus_topic,
                on_batch=on_batch)
        except PlannerError as exc:
            # A repeatable error (no credit, a bad key) will fail the single
            # call too; say so now rather than wait for it.
            if not _worth_retrying(str(exc), deadline_start):
                raise
            logger.warning("ppt_v2: parallel planning failed (%s: %s); planning in one call",
                           exc.reason, str(exc)[:160])
    note("write", "Writing all %d slides" % slide_count)

    for attempt in range(1, 3):
        attempts = attempt
        system, user = build_planner_prompts(
            topic=topic, slide_count=slide_count, language=language,
            class_name=class_name, subject=subject, chapter=chapter, board=board,
            objectives=objectives, passages=passages, board_rules=board_rules,
            focus_topic=focus_topic)
        if revision_note:
            user += "\n\n" + revision_note
        if retry_note:
            user += "\n\n" + retry_note

        try:
            result = llm.complete(
                system_prompt=system, user_prompt=user, model=model or None,
                temperature=temperature, max_tokens=max_tokens, json_mode=True,
                institute_id=institute_id, feature="ppt_v2_plan",
                capability=capability,
            ) or {}
        except Exception as exc:
            logger.warning("ppt_v2: planner call failed (attempt %d): %s", attempt, exc)
            if attempt == 1 and _worth_retrying(str(exc), deadline_start):
                retry_note = _BROKEN_REPLY_NOTE
                continue
            raise PlannerError("llm_error", str(exc)[:300])

        if on_usage:
            on_usage(result, attempt)

        content = result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as exc:
                logger.warning("ppt_v2: planner returned invalid JSON (attempt %d): %s",
                               attempt, exc)
                if attempt == 1 and _worth_retrying("", deadline_start):
                    retry_note = _BROKEN_REPLY_NOTE
                    continue
                raise PlannerError("malformed_json", str(exc)[:200])

        title, slides, problems = validate_payload(content, slide_count)
        if problems and not slides:
            logger.warning("ppt_v2: planner output unusable: %s", "; ".join(problems[:3]))
            raise PlannerError("invalid_schema", "; ".join(problems[:3]))
        if problems:
            logger.info("ppt_v2: planner output had %d repairable problems", len(problems))

        slides, action = enforce_slide_count(slides, slide_count)
        if action != "short":
            note("write", "Wrote %d slides" % len(slides), "done")
            return PlannerResult(
                title=title or topic, slides=slides, attempts=attempts,
                grounded=grounded,
                model=str(result.get("model") or model),
                usage=result,
                latency_s=time.perf_counter() - deadline_start,
            )

        last_problem = ("planner returned %d slides, %d were requested"
                        % (len(slides), slide_count))
        logger.info("ppt_v2: %s (attempt %d)", last_problem, attempt)
        retry_note = ("Your previous attempt returned the wrong number of "
                      "slides. Return EXACTLY %d slide objects this time." % slide_count)

    # Two attempts, still the wrong length. Returning a 3-slide deck to a
    # caller who asked for 2 is a silent contract violation, so this is an
    # error the caller handles rather than a deck we quietly reshape.
    raise PlannerError("slide_count", last_problem)
