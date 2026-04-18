import logging
import re
import threading

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.batch_processor import BatchProcessor
from .base import ai_call

# ── Text cleaning ─────────────────────────────────────────────────────────────

# PDF encoding artifact patterns (from finetuning data)
_PDF_ARTIFACTS = re.compile(
    r'\(cid:\d+\)'          # (cid:90), (cid:32) etc.
    r'|\\x[0-9a-fA-F]{2}'  # \x90 hex escapes
    r'|\ufffd'              # replacement character
    r'|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'  # control characters
)

# Detects options leaked into question text: " A. ", " B) ", " C- "
_EMBEDDED_OPTIONS = re.compile(r'\s+[A-D][.):\-]\s+', re.IGNORECASE)


def _clean_text(text: str) -> str:
    """Remove PDF encoding artifacts and normalize whitespace."""
    if not text:
        return ""
    text = _PDF_ARTIFACTS.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _question_has_embedded_options(question: str) -> bool:
    """Return True if the question text contains options leaked into it."""
    return bool(_EMBEDDED_OPTIONS.search(question))

logger = logging.getLogger("ai_services.llm")

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@api_view(["POST"])
def generate_practice_test(request):
    """
    Generate MCQ practice questions for a topic.

    Request body:
        topic          (str, required)  — topic name e.g. "Newton's Laws of Motion"
        num_questions  (int, default 5) — how many questions (capped at 10)
        difficulty     (str, default "medium") — easy | medium | hard

    Response:
        {
          "topic": "...",
          "difficulty": "...",
          "questions": [
            {
              "id": 1,
              "question": "...",
              "options": ["opt A", "opt B", "opt C", "opt D"],
              "answer": "A",          ← always exactly one of A/B/C/D
              "explanation": "..."
            }
          ]
        }
    """
    data = request.data
    topic = (data.get("topic") or "").strip()
    if not topic:
        return Response({"error": "Missing topic"}, status=400)

    try:
        num_questions = max(1, min(int(data.get("num_questions", 5)), 10))
    except (TypeError, ValueError):
        num_questions = 5

    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"

    template = get_template("test_generate")
    user_prompt = template.user_template.format(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
    )

    logger.info(
        "generate_practice_test | topic=%s | n=%d | difficulty=%s",
        topic, num_questions, difficulty,
    )

    # ai_call uses JSON mode — edvaqwen returns structured JSON
    response = ai_call(
        request,
        feature="test_generate",
        user_prompt=user_prompt,
        temperature=0.3,      # low temp = factual, consistent
        skip_cache=False,
        max_tokens=2048,
    )

    # Validate response has the expected questions array
    if hasattr(response, "data") and isinstance(response.data, dict):
        questions = response.data.get("questions")
        if not isinstance(questions, list) or len(questions) == 0:
            logger.error(
                "generate_practice_test: LLM returned no questions | topic=%s | raw=%s",
                topic, str(response.data)[:300],
            )
            return Response(
                {
                    "error": "ai_no_questions",
                    "detail": "The AI did not return any questions. Please try again.",
                    "topic": topic,
                },
                status=502,
            )

        # Validate and sanitize each question
        sanitized = []
        for q in questions:
            if not isinstance(q, dict):
                continue

            ans = (q.get("answer") or "").strip().upper()
            # Accept "A", "B", "C", "D" — also handle "A." or "A)"
            if ans and ans[0] in "ABCD":
                ans = ans[0]
            opts = q.get("options") or []
            question_text = _clean_text(q.get("question") or "")

            # Skip if answer not valid
            if ans not in ("A", "B", "C", "D"):
                logger.warning(
                    "generate_practice_test: skipping — invalid answer=%s id=%s",
                    q.get("answer"), q.get("id"),
                )
                continue

            # Skip if options count is wrong
            if len(opts) != 4:
                logger.warning(
                    "generate_practice_test: skipping — wrong option count=%d id=%s",
                    len(opts), q.get("id"),
                )
                continue

            # Skip if question contains embedded option text (PDF recall artifact)
            if _question_has_embedded_options(question_text):
                logger.warning(
                    "generate_practice_test: skipping — options leaked into question: %s",
                    question_text[:80],
                )
                continue

            # Skip empty question
            if len(question_text) < 10:
                logger.warning("generate_practice_test: skipping — question too short id=%s", q.get("id"))
                continue

            clean_opts = [_clean_text(str(o)) for o in opts]

            # Skip if any option is empty after cleaning
            if any(len(o) < 2 for o in clean_opts):
                logger.warning("generate_practice_test: skipping — empty option id=%s", q.get("id"))
                continue

            sanitized.append({
                "id": q.get("id"),
                "question": question_text,
                "options": clean_opts,
                "answer": ans,
                "explanation": _clean_text(q.get("explanation") or ""),
            })

        if not sanitized:
            logger.error(
                "generate_practice_test: all questions failed validation | topic=%s", topic
            )
            return Response(
                {
                    "error": "ai_validation_failed",
                    "detail": "All generated questions failed validation. Please try again.",
                    "topic": topic,
                },
                status=502,
            )

        response.data["questions"] = sanitized
        response.data["topic"] = topic
        response.data["difficulty"] = difficulty
        logger.info(
            "generate_practice_test: OK | topic=%s | returned=%d/%d questions",
            topic, len(sanitized), num_questions,
        )

    return response


@api_view(["POST"])
def batch_generate_tests(request):
    """Batch generate tests for multiple topics."""
    institute = getattr(request, "institute", None)
    if institute and not institute.is_feature_enabled("batch"):
        return Response(
            {"error": "Batch processing is not enabled for your plan", "upgrade": "Contact support"},
            status=403,
        )

    data = request.data
    topics = data.get("topics", [])
    if not topics:
        return Response({"error": "Missing topics list"}, status=400)

    try:
        num_questions = max(1, min(int(data.get("num_questions", 5)), 10))
    except (TypeError, ValueError):
        num_questions = 5

    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("test_generate")

    prompts = [
        template.user_template.format(
            topic=t,
            num_questions=num_questions,
            difficulty=difficulty,
        )
        for t in topics
    ]

    processor = BatchProcessor()
    job = processor.create_job("test_generate", institute_id, prompts)
    threading.Thread(target=processor.run, args=(job,), daemon=True).start()

    return Response({
        "job_id": job.job_id,
        "status": "queued",
        "total_items": len(topics),
        "institute": institute_id,
        "message": "Poll /test/batch/status/?job_id=<id> for progress",
    }, status=202)


@api_view(["GET"])
def batch_status(request):
    """Check status of a batch test generation job."""
    job_id = request.query_params.get("job_id")
    if not job_id:
        return Response({"error": "Missing job_id"}, status=400)

    job = BatchProcessor.get_job(job_id)
    if not job:
        return Response({"error": "Job not found"}, status=404)

    institute_id = getattr(request, "institute_id", "default")
    if job.institute_id != institute_id:
        return Response({"error": "Job not found"}, status=404)

    result = job.progress
    if job.status.value in ("completed", "partial"):
        result["results"] = [
            {"topic_index": i, "result": item.result, "error": item.error}
            for i, item in enumerate(job.items)
        ]

    return Response(result)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "ai_test",
        "model": get_model_for_task("test_generate"),
    })