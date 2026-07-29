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
import time

import requests as _requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ai_services.core.serpapi_images import search_google_images
from ai_services.core.usage_logger import log_usage
from .base import get_llm

logger = logging.getLogger("ai_services.ppt")

_VISUAL_HINTS = frozenset({
    "map", "diagram", "photograph", "photo", "chart",
    "illustration", "artifact", "image", "picture",
    "excavation", "figure", "drawing",
})

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

═══ JSON FORMAT RULES ═══
• All strings: DOUBLE QUOTES only — never single quotes.
• "bullets": proper JSON array — "bullets": ["...", "..."]
• Never write bullets=[...] — always use a colon.
• No trailing commas before } or ].

═══ BULLET POINT STANDARD (most important rule) ═══
Each bullet point must be ONE complete, informative sentence of 12–30 words.
It must state a clear fact, include a specific detail, and be understandable on its own.

═══ ACADEMIC RIGOUR & COURSE CONTENT ═══
• Generates high-quality actual course content for the specific class, subject, and chapter.
• For numerical, physics, chemistry, or math chapters, you MUST include clear step-by-step mathematical derivations, relevant formulas, and worked numerical examples.
• Formulas must use standard LaTeX wrapped in double dollar signs: $$formula$$ (e.g. $$E = mc^2$$ or $$\\vec{F} = m\\vec{a}$$). Ensure double backslashes are used for LaTeX commands (e.g. \\\\frac for fraction) so they survive JSON parsing.

  ✗ TOO SHORT (fragment — rejected): "Located in Pakistan" / "Formula is F=ma"
  ✓ CORRECT LENGTH & FORMULA (12–30 words):
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
  • bullets: EXACTLY 5 bullet points — each a complete sentence of 12–30 words containing core definitions, mathematical derivations, formula steps, or numerical practice problems.
  • Each slide must cover a DIFFERENT sub-topic.

Slide {{SLIDE_COUNT}}  → type "summary"
  • title: "Key Takeaways" or similar
  • bullets: 5 bullet points, each a complete sentence of 12–30 words summarising one key fact or fundamental formula.

═══ IMAGE SEARCH TERM RULES ═══
imageSearchTerm MUST match the exact sub-topic of THAT slide — never the general topic.
  1. Term must name the concept specific to THAT slide.
  2. 4–7 words long.
  3. Include a visual type word: map, diagram, photograph, excavation, artifact, chart, illustration.

ALL slides:
  • speakerNotes: 2–3 sentences — a teaching tip or discussion question for the teacher

═══ OUTPUT ═══
Return ONLY valid JSON — no markdown fences, no commentary:
{
  "title": "Presentation Title",
  "slides": [
    { "slideNumber": 1, "type": "title", "title": "Engaging Main Title", "subtitle": "One sentence previewing what students will learn.", "bullets": [], "speakerNotes": "Ask students what they already know.", "imageSearchTerm": "topic overview educational photograph" },
    { "slideNumber": 2, "type": "content", "title": "Sub-Topic Heading", "subtitle": "", "bullets": ["Complete sentence of 12–30 words with a specific fact.", "Step-by-step formula derivation: $$y = mx + c$$.", "A sentence explaining a cause, effect, or significance.", "Worked numerical example: find $$x$$ given $$y=5$$.", "An interesting detail that deepens understanding."], "speakerNotes": "Ask students: which fact surprised you most?", "imageSearchTerm": "specific 4-7 word term with visual hint" }
  ]
}"""


def _build_generate_prompt(slide_count: int, language: str, topic: str) -> str:
    return (
        _GENERATE_PROMPT_TEMPLATE
        .replace("{{SLIDE_COUNT}}", str(slide_count))
        .replace("{{TOPIC}}", topic)
        .replace("{{LANGUAGE}}", language)
        .replace("{{LAST_CONTENT}}", str(slide_count - 1))
    )


def _build_regenerate_prompt(slide_index: int, total_slides: int, topic: str, slide_type: str) -> str:
    rules = {
        "title": (
            "Title slide: engaging title (5–8 words) + subtitle (10–18 words previewing "
            "what students will learn). bullets must be []."
        ),
        "summary": (
            "Summary slide: 5 bullet points, each a complete sentence of 12–30 words "
            "summarising one key fact or fundamental formula from the presentation."
        ),
        "content": (
            "Content slide: EXACTLY 5 bullet points, each a complete sentence of 12–30 words "
            "containing core definitions, mathematical derivations, formula steps, or numerical practice problems."
        ),
    }
    slide_rule = rules.get(slide_type, rules["content"])
    return f"""\
You are a senior curriculum writer. Regenerate slide {slide_index + 1} of {total_slides} \
for a presentation about "{topic}". Type: "{slide_type}".

═══ ACADEMIC RIGOUR & COURSE CONTENT ═══
• Generates high-quality actual course content for the specific class, subject, and chapter.
• For numerical, physics, chemistry, or math chapters, you MUST include clear step-by-step mathematical derivations, relevant formulas, and worked numerical examples.
• Formulas must use standard LaTeX wrapped in double dollar signs: $$formula$$ (e.g. $$E = mc^2$$ or $$\\vec{F} = m\\vec{a}$$). Ensure double backslashes are used for LaTeX commands (e.g. \\\\frac for fraction) so they survive JSON parsing.

BULLET RULE — each bullet must be ONE complete sentence of 12–30 words with a specific fact, formula, derivation step, or worked numerical example. \
Not a fragment. Not a long paragraph.
  ✗ TOO SHORT: "Located in Pakistan" / "Formula is F=ma"
  ✓ CORRECT: "Newton's Second Law states that force is directly proportional to acceleration, expressed as the formula $$\\vec{F} = m\\vec{a}$$."

{slide_rule}

IMAGE RULE: imageSearchTerm must describe the exact visual content of THIS slide’s sub-topic — \
not the general topic. Include a visual type hint (map, diagram, photograph, artifact, chart).

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
def _enrich_search_term(term: str, slide_title: str) -> str:
    """Ensure the Serper query is specific and includes a visual type hint."""
    if not term or len(term.strip().split()) < 3:
        base = (slide_title or term or "").strip()
        return f"{base} educational diagram photograph"
    if not any(h in term.lower() for h in _VISUAL_HINTS):
        return f"{term} photograph"
    return term


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


def _fetch_image_for_slide(search_term: str, slide_title: str) -> dict:
    """Search Serper and return the first downloadable image as a base64 data URI."""
    try:
        enriched = _enrich_search_term(search_term, slide_title)
        results = search_google_images(enriched, limit=5)
        for item in results:
            url = item.get("imageUrl")
            if not url:
                continue
            b64 = _download_image_as_base64(url)
            if b64:
                return {"imageUrl": url, "imageBase64": b64}
        # Return the URL even if we couldn't download it — client can still try
        if results:
            return {"imageUrl": results[0].get("imageUrl"), "imageBase64": None}
    except Exception as exc:
        logger.warning("Image fetch error for %r: %s", search_term, exc)
    return {"imageUrl": None, "imageBase64": None}


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
_MODEL = "llama-3.3-70b-versatile"


@api_view(["POST"])
def generate_presentation(request):
    """
    POST /ppt/generate

    Body: { topic, slideCount?, language? }
    Returns: { success: true, data: { title, slides: [ SlideObject ] } }

    SlideObject includes imageUrl + imageBase64 (Serper fetch, 1s delay between slides).
    PPT assembly (.pptx) is done client-side via PptxGenJS — only JSON is returned here.
    """
    topic = (request.data.get("topic") or "").strip()
    if not topic:
        return Response({"error": "topic is required"}, status=status.HTTP_400_BAD_REQUEST)

    slide_count = max(3, min(15, int(request.data.get("slideCount") or 5)))
    language = (request.data.get("language") or "English").strip()
    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", "school")

    system_prompt = _build_generate_prompt(slide_count, language, topic)
    user_prompt = (
        f'Write a {slide_count}-slide educational presentation about: "{topic}". '
        f"Language: {language}. "
        "Each bullet must be one complete sentence of 12–20 words with a specific fact. "
        "Not fragments. Not essays."
    )

    llm = get_llm()
    llm_result = None
    for attempt in range(1, 4):
        try:
            llm_result = llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=_MODEL,
                temperature=0.7 if attempt == 1 else 0.4,
                max_tokens=8000,
                json_mode=True,
                institute_id=institute_id,
            )
            break
        except Exception as exc:
            logger.warning("PPT generate attempt %d/%d failed: %s", attempt, 3, exc)
            if attempt == 3:
                _log(institute_id, vertical, _MODEL, success=False, error=exc)
                return Response(
                    {"error": "Failed to generate slide content. Please try again."},
                    status=status.HTTP_502_BAD_GATEWAY,
                )
            time.sleep(1)

    _log(institute_id, vertical, _MODEL, result=llm_result)

    data, err = _parse_llm_json(llm_result["content"], "ppt/generate")
    if err:
        return Response({"error": err}, status=status.HTTP_502_BAD_GATEWAY)

    slides = data.get("slides") or []
    slides_with_images = []
    for i, slide in enumerate(slides):
        if i > 0:
            time.sleep(1)  # Serper rate limit — 1 req/s
        image = _fetch_image_for_slide(
            slide.get("imageSearchTerm", ""),
            slide.get("title", ""),
        )
        slides_with_images.append({**slide, **image})

    data["slides"] = slides_with_images
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
    vertical = getattr(request, "vertical", "school")

    llm = get_llm()
    try:
        llm_result = llm.complete(
            system_prompt=_build_regenerate_prompt(slide_index, total_slides, topic, slide_type),
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

    image = _fetch_image_for_slide(
        new_slide.get("imageSearchTerm", ""),
        new_slide.get("title", ""),
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
