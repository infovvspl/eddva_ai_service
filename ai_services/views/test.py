import logging
import re
import threading

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.batch_processor import BatchProcessor
from .base import ai_call, get_llm

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


def parse_mcq_text(raw_text, topic, difficulty):
    questions = []
    # Split on blank line before each new "Question:" block
    blocks = re.split(r'\n\n(?=Question\s*\d*\s*:)', raw_text.strip())
    for i, block in enumerate(blocks):
        if not block.strip():
            continue
        try:
            q_match = re.search(r'Question\s*\d*\s*:\s*(.+?)(?=\nA\))', block, re.DOTALL)
            a_match = re.search(r'A\)\s*(.+?)(?=\nB\))', block, re.DOTALL)
            b_match = re.search(r'B\)\s*(.+?)(?=\nC\))', block, re.DOTALL)
            c_match = re.search(r'C\)\s*(.+?)(?=\nD\))', block, re.DOTALL)
            d_match = re.search(r'D\)\s*(.+?)(?=\nCorrect Answer:)', block, re.DOTALL)
            ans_match = re.search(r'Correct Answer:\s*([ABCD])', block)
            exp_match = re.search(r'Explanation:\s*(.+?)$', block, re.DOTALL)

            if q_match and ans_match:
                answer_letter = ans_match.group(1).strip()
                options = [
                    a_match.group(1).strip() if a_match else '',
                    b_match.group(1).strip() if b_match else '',
                    c_match.group(1).strip() if c_match else '',
                    d_match.group(1).strip() if d_match else '',
                ]
                questions.append({
                    'id': i + 1,
                    'question': q_match.group(1).strip(),
                    'options': options,
                    'answer': answer_letter,
                    'explanation': exp_match.group(1).strip() if exp_match else '',
                })
        except Exception:
            continue

    return {
        'topic': topic,
        'difficulty': difficulty,
        'questions': questions,
    }


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
    user_prompt = (
        f"Generate {num_questions} MCQ questions on {topic} for JEE/NEET.\n"
        f"Difficulty: {difficulty}.\n\n"
        "For each question write exactly:\n"
        "Question: [question text]\n"
        "A) [option 1]\n"
        "B) [option 2]\n"
        "C) [option 3]\n"
        "D) [option 4]\n"
        "Correct Answer: [A/B/C/D]\n"
        "Explanation: [one sentence]\n\n"
        f"Write all {num_questions} questions now."
    )

    institute_id = getattr(request, "institute_id", "default")

    logger.info(
        "generate_practice_test | topic=%s | n=%d | difficulty=%s",
        topic, num_questions, difficulty,
    )

    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model="groq",
            temperature=0.3,
            max_tokens=2048,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)

    parsed = parse_mcq_text(result["content"], topic, difficulty)
    parsed["_meta"] = {
        "source": "llm",
        "model": result["model"],
        "latency_ms": round(result["latency_ms"]),
        "institute": institute_id,
    }

    logger.info(
        "generate_practice_test: OK | topic=%s | returned=%d questions",
        topic, len(parsed["questions"]),
    )
    return JsonResponse(parsed)


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