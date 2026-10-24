"""
School learning path: AI roadmaps for what a student is weak in.

- chapter_roadmap: for a weak chapter, the topics a student should learn in
  teaching order, each with its subtopics, and which of the school's
  recorded topics each roadmap topic covers (so marks recorded against the
  school's topics still count).
- topic_roadmap: for a weak topic in a chapter that is otherwise fine, just
  that topic's subtopics.
- topic_video: the one best YouTube video for a topic, for the student's
  class and board (the learning path's videos step).
- topic_scope: whether a question asked in a topic's chat is about that
  topic (the chat only answers questions on its topic).

The NestJS learning path stores each roadmap once and shows it as the
student's mind map; the subtopics are also the concepts quizzes are tagged
with.
"""

import logging
import re

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ..core.tutor_media import (
    _EDUCATION_CHANNELS, _HARD_MAX_SECONDS, _PREFERRED_MAX_SECONDS, duration_seconds, topic_terms,
)
from ..core.web_search import search_videos
from .base import _log_usage_to_db, get_llm, metered
from .test import _clean_concepts, _clean_text

logger = logging.getLogger("ai_services.llm")

_MIN_TOPICS = 3
_MAX_TOPICS = 12
_MAX_SUBTOPICS = 8
_MAX_EXISTING = 40
_MAX_PASSAGE_CHARS = 6000
_GENERIC = {"introduction", "summary", "exercises", "exercise", "conclusion", "overview", "revision"}

_SYSTEM_PROMPT = (
    "You are an expert school curriculum designer for Indian boards (CBSE/ICSE/State). "
    "You turn a textbook chapter into an accurate learning roadmap: the topics a student learns, "
    "in teaching order, each broken into specific subtopics. Respond with ONLY a JSON object."
)


def _clean_name(value) -> str:
    return _clean_text(str(value or "")).replace('"', "").replace("'", "")[:120].strip()


def _key(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def build_roadmap_prompt(subject, chapter, class_name, board, existing, passages) -> str:
    listed = "\n".join(f"  - {t}" for t in existing) if existing else "  (none recorded)"
    grounding = (
        f"\nTextbook extract for this chapter (follow its headings, order and scope):\n<<<\n{passages}\n>>>\n"
        if passages else ""
    )
    return (
        f"Subject: {subject or 'not given'}. Chapter: {chapter}. Class: {class_name or 'not given'}. "
        f"Board: {board or 'CBSE'}.\n"
        f"Topic names the school has recorded for this chapter (may be incomplete, misspelt or too broad):\n{listed}\n"
        f"{grounding}\n"
        "Write the learning roadmap for this chapter:\n"
        f"  • {_MIN_TOPICS} to {_MAX_TOPICS} topics in the order they are taught, each a clear, correctly spelt name.\n"
        f"  • Each topic has 3 to {_MAX_SUBTOPICS} subtopics, 2-6 words each, specific and testable "
        "(e.g. 'Discriminant formula'), never 'Introduction', 'Summary' or 'Exercises'. No subtopic repeats.\n"
        "  • Only what this class and board actually cover in this chapter.\n"
        "  • 'covers': the recorded school topic names (copied exactly from the list above) that this roadmap topic "
        "includes. Every recorded name must appear in exactly one topic's 'covers' unless it is not part of this chapter.\n"
        'Return JSON: {"topics": [{"topic": "...", "subtopics": ["..."], "covers": ["..."]}]}'
    )


def parse_roadmap(content, existing: list) -> list:
    """Validated roadmap topics; 'covers' keeps only real recorded names, each used once."""
    existing_by_key = {_key(t): t for t in existing}
    used_covers, seen_topics, seen_subs = set(), set(), set()
    items = content.get("topics") if isinstance(content, dict) else content
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("topic"))
        if not name or _key(name) in seen_topics or _key(name) in _GENERIC:
            continue
        subs = []
        for s in _clean_concepts(item.get("subtopics")):
            k = _key(s)
            if k and k not in _GENERIC and k not in seen_subs:
                seen_subs.add(k)
                subs.append(s)
        covers = []
        for c in item.get("covers") or []:
            original = existing_by_key.get(_key(c))
            if original and original not in used_covers:
                used_covers.add(original)
                covers.append(original)
        seen_topics.add(_key(name))
        out.append({"topic": name, "subtopics": subs[:_MAX_SUBTOPICS], "covers": covers})
        if len(out) >= _MAX_TOPICS:
            break
    return out


def build_topic_prompt(subject, chapter, topic, class_name, board, passages) -> str:
    grounding = (
        f"\nTextbook extract for this chapter (follow its headings and scope):\n<<<\n{passages}\n>>>\n"
        if passages else ""
    )
    return (
        f"Subject: {subject or 'not given'}. Chapter: {chapter or 'not given'}. Class: {class_name or 'not given'}. "
        f"Board: {board or 'CBSE'}.\n"
        f"Topic as recorded by the school (may be misspelt or in capitals): {topic}\n{grounding}\n"
        "Write the learning roadmap for THIS TOPIC ONLY:\n"
        "  • 'topic': the topic's correct, clearly written name.\n"
        f"  • 'subtopics': 3 to {_MAX_SUBTOPICS} subtopics in the order they are taught, 2-6 words each, specific and "
        "testable, never 'Introduction', 'Summary' or 'Exercises'. Only what this class and board cover.\n"
        'Return JSON: {"topic": "...", "subtopics": ["..."]}'
    )


def parse_topic_roadmap(content) -> dict:
    item = content if isinstance(content, dict) else {}
    name = _clean_name(item.get("topic"))
    subs, seen = [], set()
    for s in _clean_concepts(item.get("subtopics")):
        k = _key(s)
        if k and k not in _GENERIC and k not in seen:
            seen.add(k)
            subs.append(s)
    return {"topic": name, "subtopics": subs[:_MAX_SUBTOPICS]}


@api_view(["POST"])
@metered("learning_topic_roadmap")
def topic_roadmap(request):
    """The subtopic roadmap of one weak topic (its chapter is not weak, so no chapter roadmap is made)."""
    data = request.data
    topic = _clean_name(data.get("topic"))
    if not topic:
        return Response({"error": "Missing topic"}, status=400)
    passages = data.get("passages") or []
    passage_text = ""
    if isinstance(passages, list):
        passage_text = "\n\n".join(_clean_text(str(p))[:1500] for p in passages if p)[:_MAX_PASSAGE_CHARS]

    user_prompt = build_topic_prompt(
        _clean_name(data.get("subject")), _clean_name(data.get("chapter")), topic,
        _clean_name(data.get("class_name")), _clean_name(data.get("board")), passage_text,
    )
    institute_id = getattr(request, "institute_id", "default")
    try:
        result = get_llm().complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=1200,
            json_mode=True,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("topic_roadmap failed: %s", e)
        return Response({"error": str(e)}, status=502)

    try:
        _log_usage_to_db(getattr(request, "institute", None), institute_id, "learning_topic_roadmap", result,
                         cache_hit=False, vertical=getattr(request, "vertical", "school"))
    except Exception as log_e:  # usage logging must never fail the request
        logger.error("Failed to log usage for topic_roadmap: %s", log_e)

    roadmap = parse_topic_roadmap(result.get("content"))
    if len(roadmap["subtopics"]) < 2:
        return Response({"error": "The roadmap came back incomplete"}, status=502)
    roadmap["topic"] = roadmap["topic"] or topic
    return Response({**roadmap, "_meta": {"model": result.get("model"), "grounded": bool(passage_text)}})


@api_view(["POST"])
@metered("learning_chapter_roadmap")
def chapter_roadmap(request):
    data = request.data
    chapter = _clean_name(data.get("chapter"))
    if not chapter:
        return Response({"error": "Missing chapter"}, status=400)
    existing, seen = [], set()
    for t in data.get("existing_topics") or []:
        name = _clean_name(t)
        if name and _key(name) not in seen:
            seen.add(_key(name))
            existing.append(name)
    existing = existing[:_MAX_EXISTING]

    passages = data.get("passages") or []
    passage_text = ""
    if isinstance(passages, list):
        passage_text = "\n\n".join(_clean_text(str(p))[:1500] for p in passages if p)[:_MAX_PASSAGE_CHARS]

    user_prompt = build_roadmap_prompt(
        _clean_name(data.get("subject")), chapter, _clean_name(data.get("class_name")),
        _clean_name(data.get("board")), existing, passage_text,
    )
    institute_id = getattr(request, "institute_id", "default")
    try:
        result = get_llm().complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=4000,
            json_mode=True,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("chapter_roadmap failed: %s", e)
        return Response({"error": str(e)}, status=502)

    try:
        _log_usage_to_db(getattr(request, "institute", None), institute_id, "learning_chapter_roadmap", result,
                         cache_hit=False, vertical=getattr(request, "vertical", "school"))
    except Exception as log_e:  # usage logging must never fail the request
        logger.error("Failed to log usage for chapter_roadmap: %s", log_e)

    topics = parse_roadmap(result.get("content"), existing)
    if len(topics) < _MIN_TOPICS:
        logger.warning("chapter_roadmap: only %d usable topics for %s", len(topics), chapter)
        return Response({"error": "The roadmap came back incomplete"}, status=502)
    return Response({"topics": topics, "_meta": {"model": result.get("model"), "grounded": bool(passage_text)}})


# ── Topic video ──────────────────────────────────────────────────────────────

# Words that show a video follows a board's syllabus (CBSE follows NCERT books).
_BOARD_WORDS = {
    "cbse": ("cbse", "ncert"),
    "icse": ("icse", "isc", "cisce", "selina"),
}
_VIDEO_CANDIDATES = 10


def _board_words(board: str) -> tuple:
    b = (board or "").strip().lower()
    if not b:
        return ()
    for key, words in _BOARD_WORDS.items():
        if key in b:
            return words
    return (b,)  # a state board: its own name


def _class_number(class_name: str) -> "str | None":
    m = re.search(r"\d{1,2}", class_name or "")
    return m.group(0) if m else None


def best_video(videos: list, topic: str, class_name: str = "", board: str = "") -> "dict | None":
    """
    The one video that best teaches `topic` to this class and board:
      1. covers the topic (share of the topic's words in the title),
      2. follows the board (CBSE/NCERT, ICSE, the state board),
      3. is for the class ("class 10", "10th"),
      4. comes from a known education channel,
      5. is a short explainer (under 20 min); over an hour is never picked;
    ties keep the search engine's order.
    """
    want = topic_terms(topic)
    board_words = _board_words(board)
    n = _class_number(class_name)
    class_re = re.compile(rf"\b(?:class|grade|std|standard)\s*{n}\b|\b{n}\s*(?:th|st|nd|rd)\b", re.I) if n else None
    ranked = []
    for i, v in enumerate(videos or []):
        title = v.get("title") or ""
        text = f"{title} {v.get('channel') or ''}".lower()
        overlap = len(want & topic_terms(title))
        if want and overlap == 0:
            continue
        secs = duration_seconds(v.get("duration"))
        if secs is not None and secs > _HARD_MAX_SECONDS:
            continue
        coverage = round(overlap / len(want), 1) if want else 0
        key = (
            -coverage,
            0 if board_words and any(w in text for w in board_words) else 1,
            0 if class_re and class_re.search(text) else 1,
            0 if any(c in text for c in _EDUCATION_CHANNELS) else 1,
            1 if secs is not None and secs > _PREFERRED_MAX_SECONDS else 0,
            i,
        )
        ranked.append((key, v))
    ranked.sort(key=lambda t: t[0])
    return ranked[0][1] if ranked else None


@api_view(["POST"])
@metered("learning_topic_video")
def topic_video(request):
    """
    POST /learning/topic-video
    Body: { topic, chapter?, subject?, class_name?, board? }
    Returns: { video: {videoId, title, url, thumbnailUrl, channel, duration} | null }
    """
    data = request.data
    topic = _clean_name(data.get("topic"))
    if not topic:
        return Response({"error": "Missing topic"}, status=400)
    chapter = _clean_name(data.get("chapter"))
    subject = _clean_name(data.get("subject"))
    class_name = _clean_name(data.get("class_name"))
    board = _clean_name(data.get("board"))
    n = _class_number(class_name)

    parts = [topic, chapter if chapter and _key(chapter) != _key(topic) else "", subject, f"class {n}" if n else "", board, "explained"]
    found = search_videos(" ".join(p for p in parts if p), _VIDEO_CANDIDATES)
    if not found and board:  # a rarely-tagged board can leave nothing; the topic alone still finds good videos
        found = search_videos(" ".join(p for p in [topic, subject, f"class {n}" if n else "", "explained"] if p), _VIDEO_CANDIDATES)
    video = best_video(found, topic, class_name, board)
    return Response({"video": video, "_meta": {"candidates": len(found)}})


# ── Topic scope ──────────────────────────────────────────────────────────────

_SCOPE_SYSTEM_PROMPT = (
    "You decide whether a school student's message belongs in a study chat about one topic. "
    "Respond with ONLY a JSON object."
)
_MAX_SCOPE_HISTORY = 4


def build_scope_prompt(subject: str, chapter: str, topic: str, subtopics: list, history: list, message: str) -> str:
    lines = [
        f"Subject: {subject or 'unknown'}",
        f"Chapter: {chapter or 'unknown'}",
        f"Topic: {topic}",
    ]
    if subtopics:
        lines.append("Subtopics: " + "; ".join(subtopics))
    if history:
        lines.append("Recent messages (oldest first):")
        lines += [f"- {h['role']}: {h['content']}" for h in history]
    lines += [
        f"New message from the student: {message}",
        "",
        "The message is ON topic when it is about this topic or this chapter, including:",
        "- the basic ideas or earlier concepts a student needs to understand the topic;",
        "- worked examples, applications, diagrams, formulas or definitions from it;",
        "- a follow-up to the recent messages (\"why?\", \"explain again\", \"give an example\");",
        "- asking to be tested or quizzed on the topic or chapter;",
        "- the student's notes, quiz, assessment, scores or study plan for it;",
        "- a greeting or thanks.",
        "It is OFF topic when it is about anything else: another chapter or subject, general knowledge,",
        "news, sport, entertainment, personal advice, or other homework.",
        'Return {"on_topic": true} or {"on_topic": false}.',
    ]
    return "\n".join(lines)


def _clean_history(raw) -> list:
    out = []
    for h in (raw if isinstance(raw, list) else [])[-_MAX_SCOPE_HISTORY:]:
        if not isinstance(h, dict):
            continue
        role = "student" if h.get("role") == "student" else "tutor"
        content = _clean_text(str(h.get("content") or ""))[:400]
        if content:
            out.append({"role": role, "content": content})
    return out


def parse_scope(content) -> "bool | None":
    """True / False from the model's JSON, or None when it gave no clear answer."""
    value = content.get("on_topic") if isinstance(content, dict) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


@api_view(["POST"])
@metered("learning_topic_scope")
def topic_scope(request):
    """
    POST /learning/topic-scope
    Body: { topic, chapter?, subject?, subtopics?: [str], history?: [{role, content}], message }
    Returns: { on_topic: bool }   (on_topic is true when the model gives no clear answer)
    """
    data = request.data
    topic = _clean_name(data.get("topic"))
    message = _clean_text(str(data.get("message") or ""))[:1000]
    if not topic or not message:
        return Response({"error": "Missing topic or message"}, status=400)
    subtopics = _clean_concepts(data.get("subtopics"))[:_MAX_SUBTOPICS]
    user_prompt = build_scope_prompt(
        _clean_name(data.get("subject")), _clean_name(data.get("chapter")), topic, subtopics,
        _clean_history(data.get("history")), message,
    )
    institute_id = getattr(request, "institute_id", "default")
    try:
        result = get_llm().complete(
            system_prompt=_SCOPE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model="openai/gpt-oss-120b",
            temperature=0.0,
            max_tokens=300,
            json_mode=True,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("topic_scope failed: %s", e)
        return Response({"error": str(e)}, status=502)

    try:
        _log_usage_to_db(getattr(request, "institute", None), institute_id, "learning_topic_scope", result,
                         cache_hit=False, vertical=getattr(request, "vertical", "school"))
    except Exception as log_e:  # usage logging must never fail the request
        logger.error("Failed to log usage for topic_scope: %s", log_e)

    on_topic = parse_scope(result.get("content"))
    # No clear answer: let the question through rather than refuse a real one.
    return Response({"on_topic": True if on_topic is None else on_topic})
