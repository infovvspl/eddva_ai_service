import logging
import re
import threading
import json

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.batch_processor import BatchProcessor
from .base import ai_call, get_llm

# ── Text cleaning ─────────────────────────────────────────────────────────────

# PDF encoding artifact patterns
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


def _dedupe_mcq_key(question: str) -> str:
    if not question:
        return ""
    t = re.sub(r"\s+", " ", question).strip().lower()
    t = re.sub(r"[^\w\s\u0900-\u0fff]", "", t, flags=re.UNICODE)
    return t[:300]


def parse_ai_result(result, topic, difficulty, qtype):
    """
    Parse the LLM result. Handles both raw JSON (from json_mode=True)
    and potential listwrappers.
    """
    questions = []
    content = result.get("content", {})
    
    # Content might be a list or a dict with "questions" key
    raw_qs = []
    if isinstance(content, list):
        raw_qs = content
    elif isinstance(content, dict):
        raw_qs = content.get("questions") or content.get("results") or content.get("data") or content.get("list") or []
        if not raw_qs and ("question" in content or "questionText" in content): # Single question case
            raw_qs = [content]
    
    if not isinstance(raw_qs, list):
        raw_qs = []

    seen_keys = set()
    for i, q in enumerate(raw_qs):
        try:
            qtext = _clean_text(q.get("question") or q.get("questionText") or q.get("text") or "")
            if not qtext:
                continue
            
            dk = _dedupe_mcq_key(qtext)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)

            parsed_q = {
                "id": len(questions) + 1,
                "question": qtext,
                "explanation": _clean_text(q.get("explanation") or q.get("solution") or q.get("solutionText") or ""),
            }

            # Handle Model Answer (for descriptive)
            model_answer = q.get("answer") or q.get("correctAnswer") or q.get("modelAnswer") or ""
            if isinstance(model_answer, dict):
                model_answer = "\n".join([f"{str(v)}" for v in model_answer.values()])
            elif isinstance(model_answer, list):
                model_answer = "\n".join([str(v) for v in model_answer])
            
            if qtype in ("mcq", "mcq_single", "mcq_multi", "assertion_reason", "statement"):
                options = q.get("options") or []
                if isinstance(options, list) and len(options) >= 2:
                    # Case: list of strings
                    if all(isinstance(o, (str, int, float)) for o in options):
                        parsed_q["options"] = [str(o) for o in options][:4]
                    # Case: list of objects {label, text}
                    else:
                        norm_opts = []
                        for opt in options:
                            o_text = _clean_text(str(opt.get("text") or opt.get("content") or opt.get("value") or opt))
                            norm_opts.append(o_text)
                        parsed_q["options"] = norm_opts[:4]
                
                # Answer for MCQ
                ans = str(q.get("answer") or q.get("correctOption") or q.get("correct_option") or "").strip().upper()
                parsed_q["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
                
                # Multi-correct
                if qtype == "mcq_multi":
                    m_ans = q.get("correctOptions") or q.get("answers") or []
                    if isinstance(m_ans, list):
                        parsed_q["correctOptions"] = [str(a).strip().upper() for a in m_ans if str(a).strip().upper() in ("A", "B", "C", "D")]
                    elif isinstance(m_ans, str):
                        parsed_q["correctOptions"] = [a.strip().upper() for a in re.split(r'[,; ]+', m_ans) if a.strip().upper() in ("A", "B", "C", "D")]

            elif qtype == "integer":
                # Integer answer
                val = str(q.get("answer") or q.get("integerAnswer") or q.get("value") or "0").strip()
                # Extract digits
                val = "".join(re.findall(r'[-\d]+', val))
                parsed_q["integerAnswer"] = val if val else "0"
            
            elif qtype in ("descriptive", "short_answer", "long_answer", "match", "diagram"):
                # Use answer field as solution text
                parsed_q["answer"] = _clean_text(str(model_answer))
                # For bridge compatibility, also set it as solutionText if present
                parsed_q["solutionText"] = parsed_q["answer"]

            if "_meta" in q:
                parsed_q["_meta"] = q["_meta"]

            questions.append(parsed_q)
        except Exception as e:
            logger.warning("Failed to parse individual question: %s", e)
            continue

    return {"questions": questions}


@api_view(["POST"])
def generate_practice_test(request):
    """
    Generate a practice test (MCQ, Integer, or MSQ) based on a topic and difficulty.
    Uses JSON mode for reliability.
    """
    data = request.data
    topic = data.get("topic")
    if not topic:
        return JsonResponse({"error": "Missing topic"}, status=400)

    try:
        num_questions = max(1, min(int(data.get("num_questions", 2)), 10))
    except (TypeError, ValueError):
        num_questions = 2

    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"

    exam_target = (data.get("exam_target") or "").strip().lower()

    qtype = (data.get("type") or data.get("question_type") or "mcq").strip().lower()
    
    # Map high-level types to prompt instructions
    if qtype == "integer":
        type_instr = (
            "Each question is an Integer type (answer is a single number, e.g. 42 or -5). "
            "In your JSON, provide the numerical answer in the 'answer' field."
        )
    elif qtype == "mcq_multi":
        type_instr = (
            "Each question is a Multiple Correct MCQ (one or more options can be correct). "
            "Provide exactly 4 options (A, B, C, D). All 4 options MUST be completely distinct and different from each other. "
            "In your JSON, provide 'options' (list of 4 strings) and 'correctOptions' (list of correct labels, e.g. ['A', 'C'])."
        )
    elif qtype in ("descriptive", "short_answer", "long_answer"):
        type_instr = (
            "Each question is a Descriptive / Structured Answer type (no A/B/C/D options). "
            "Provide a detailed model answer in the 'answer' field as a single string. "
            "Break down the model answer into clear steps or bullet points using newlines."
        )
    elif qtype in ("match", "match_the_following"):
        type_instr = (
            "Each question is a Match the Following type. "
            "Provide two columns (Column I and Column II) with 4 items each. "
            "Render the columns as part of the 'question' text. "
            "Provide the correct mapping (e.g. A-1, B-4, C-2, D-3) in the 'answer' field."
        )
    elif qtype == "diagram":
        type_instr = (
            "Each question is Diagram-based. "
            "Describe the diagram using text/characters (ASCII art) or a clear labeled description in the 'question' text. "
            "Ask the student to identify parts or explain a process shown. "
            "Provide a detailed model answer in the 'answer' field."
        )
    else:
        # Default MCQ
        type_instr = (
            "Each question is a Single Correct MCQ. Give exactly 4 options. All 4 options MUST be completely distinct and different from each other. "
            "In your JSON, provide 'options' (list of 4 strings) and 'answer' (the correct option label A/B/C/D)."
        )

    heuristic_block = ""
    if exam_target:
        exam_key = "jee advanced" if "advance" in exam_target else "jee main" if "main" in exam_target else "neet" if "neet" in exam_target else ""
        if exam_key:
            if exam_key == "jee advanced":
                formula = "D = 0.25C + 0.20S + 0.20M + 0.15T + 0.20X"
                exam_rules = "Multi-step, multi-concept problems."
            elif exam_key == "jee main":
                formula = "D = 0.22C + 0.20S + 0.18M + 0.20T + 0.20X"
                exam_rules = "Balanced MCQ + numerical."
            else:
                formula = "D = 0.30C + 0.10S + 0.05M + 0.15T + 0.40X"
                exam_rules = "Prefer statement-based or assertion-reason, conceptual clarity, high trickiness."

            heuristic_block = f"""
----------------------------------------
DIFFICULTY MODEL
Variables:
C = Concept level (1-10) [1=Direct formula, 5=Concept application, 10=Multi-concept deep understanding]
S = Steps required (1-10) [1=1-2 steps, 5=3-5 steps, 10=6+ steps]
M = Calculation level (1-10) [1=Basic math, 5=Algebra/moderate, 10=Heavy calculus/vector]
T = Time required (1-10) [1=<1 min, 5=1-2 min, 10=3+ min]
X = Trickiness (1-10) [1=Direct, 10=Confusing exceptions/statements]

Target Exam: {exam_key.upper()}
Difficulty Formula: {formula}
Exam Rules: {exam_rules}
Subject Rules:
- Physics/Math: prioritize multi-step reasoning and calculation
- Physical Chemistry: calculation-heavy
- Organic Chemistry: reaction mechanisms and multi-step conversions
- Inorganic Chemistry: NCERT facts, exceptions, memory traps
- Biology (NEET): statement-based, conceptual clarity, high trickiness

Targets:
- Easy -> Target Final D <= 3
- Medium -> Target 4 <= Final D <= 7
- Hard -> Target Final D >= 8

STRICT RULES (VERY IMPORTANT):
- Use the difficulty model internally.
- DO NOT display C, S, M, T, X, or D in the question.
- DO NOT show any calculations of difficulty.
- Ensure the question matches the target difficulty strictly ({difficulty}).
- Include a '_meta' object with C, S, M, T, X, D inside each question JSON to report your scores.
----------------------------------------
"""

    diff_context = (
        "highly competitive JEE Advanced / NEET level question with deep conceptual clarity"
        if difficulty == "hard"
        else "competitive JEE Mains / NEET level question"
        if difficulty == "medium"
        else "foundation level concept testing question"
    )

    user_prompt = (
        f"Generate {num_questions} distinct {qtype} questions on the topic: {topic}.\n"
        f"Difficulty: {difficulty}. This should be a {diff_context}.\n"
        f"{type_instr}\n\n"
        f"{heuristic_block}\n"
        "Return the result as a JSON object with a 'questions' array. "
        "Each question object MUST have: 'question', 'answer', and 'explanation'. "
        "Include an 'options' field only for MCQs."
    )

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("test_generate")

    logger.info(
        "generate_practice_test | topic=%s | type=%s | n=%d",
        topic[:50], qtype, num_questions
    )

    try:
        # Switch to 'quiz' model (8b-instant)
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model="quiz",
            temperature=0.4,
            max_tokens=4000,
            json_mode=True,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("LLM complete failed: %s", e)
        return JsonResponse({"error": str(e)}, status=502)

    parsed = parse_ai_result(result, topic, difficulty, qtype)
    parsed["_meta"] = {
        "source": "llm",
        "model": result["model"],
        "latency_ms": round(result["latency_ms"]),
        "institute": institute_id,
        "type": qtype,
    }

    logger.info(
        "generate_practice_test: OK | type=%s | returned=%d questions",
        qtype, len(parsed["questions"]),
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
        f"Generate {num_questions} MCQ questions on {t} (Difficulty: {difficulty}). Return JSON."
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
    """Retrieve the status of a batch processing job."""
    job_id = request.query_params.get("job_id")
    if not job_id:
        return Response({"error": "Missing job_id"}, status=400)

    processor = BatchProcessor()
    job = processor.get_job(job_id)
    if not job:
        return Response({"error": "Job not found"}, status=404)

    return Response({
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "results": job.results if job.status == "completed" else None,
        "error": job.errors if job.status == "failed" else None
    })


@api_view(["GET"])
def health(request):
    """Health check for the test generation service."""
    return Response({"status": "healthy", "service": "test_generate"})