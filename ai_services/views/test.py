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

_SUBJECT_RULES_TEST = {
    "physics": (
        "PHYSICS: Focus on numerical problems requiring formula application, unit analysis, "
        "and multi-step calculation. Cover mechanics, thermodynamics, electrostatics, optics, "
        "and modern physics as relevant to the topic. All answers must include SI units."
    ),
    "chemistry": (
        "CHEMISTRY: Include stoichiometric calculations, molar mass problems, reaction mechanisms, "
        "and NCERT-level factual/conceptual questions. Cover physical, organic, and inorganic "
        "chemistry as appropriate to the topic. Ensure numerical answers show balanced equations."
    ),
    "mathematics": (
        "MATHEMATICS: Focus on application problems requiring step-wise derivation or proof. "
        "State the relevant theorem or formula clearly in the question. Cover calculus, algebra, "
        "coordinate geometry, and trigonometry as appropriate to the topic."
    ),
    "maths": (
        "MATHEMATICS: Focus on application problems requiring step-wise derivation or proof. "
        "State the relevant theorem or formula clearly in the question. Cover calculus, algebra, "
        "coordinate geometry, and trigonometry as appropriate to the topic."
    ),
    "biology": (
        "BIOLOGY: Prefer assertion-reason, statement-based, and process-identification questions. "
        "Cover cell biology, genetics, ecology, human physiology, and plant biology as relevant "
        "to the topic. Use NCERT terminology and diagrams where applicable."
    ),
}


def _dedupe_mcq_key(question: str) -> str:
    if not question:
        return ""
    t = re.sub(r"\s+", " ", question).strip().lower()
    t = re.sub(r"[^\w\s\u0900-\u0fff]", "", t, flags=re.UNICODE)
    return t[:300]


def parse_ai_result(result, topic, difficulty, qtype, style=""):
    """
    Parse the LLM result. Handles both raw JSON (from json_mode=True)
    and potential list-wrappers.
    `style` is used to determine the effective parsing branch when the style
    overrides what the JSON structure looks like (e.g. assertion_reason → MCQ parse).
    """
    # Determine the effective parse type from style when it overrides the base type
    _MCQ_STYLES = {"assertion_reason", "statement"}
    _DESC_STYLES = {"match", "diagram", "case_study", "short_answer", "detailed_answer"}
    if style in _MCQ_STYLES:
        parse_as = "mcq_single"
    elif style in _DESC_STYLES:
        parse_as = "descriptive"
    else:
        parse_as = qtype

    questions = []
    content = result.get("content", {})

    raw_qs = []
    if isinstance(content, list):
        raw_qs = content
    elif isinstance(content, dict):
        raw_qs = content.get("questions") or content.get("results") or content.get("data") or content.get("list") or []
        if not raw_qs and ("question" in content or "questionText" in content):
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

            model_answer = q.get("answer") or q.get("correctAnswer") or q.get("modelAnswer") or ""
            if isinstance(model_answer, dict):
                model_answer = "\n".join([str(v) for v in model_answer.values()])
            elif isinstance(model_answer, list):
                model_answer = "\n".join([str(v) for v in model_answer])

            if parse_as in ("mcq", "mcq_single", "mcq_multi"):
                options = q.get("options") or []
                if isinstance(options, list) and len(options) >= 2:
                    if all(isinstance(o, (str, int, float)) for o in options):
                        parsed_q["options"] = [str(o) for o in options][:4]
                    else:
                        parsed_q["options"] = [
                            _clean_text(str(opt.get("text") or opt.get("content") or opt.get("value") or opt))
                            for opt in options
                        ][:4]

                ans = str(q.get("answer") or q.get("correctOption") or q.get("correct_option") or "").strip().upper()
                parsed_q["answer"] = ans if ans in ("A", "B", "C", "D") else "A"

                if parse_as == "mcq_multi" or qtype == "mcq_multi":
                    m_ans = q.get("correctOptions") or q.get("answers") or []
                    if isinstance(m_ans, list):
                        parsed_q["correctOptions"] = [str(a).strip().upper() for a in m_ans if str(a).strip().upper() in ("A", "B", "C", "D")]
                    elif isinstance(m_ans, str):
                        parsed_q["correctOptions"] = [a.strip().upper() for a in re.split(r'[,; ]+', m_ans) if a.strip().upper() in ("A", "B", "C", "D")]

            elif parse_as == "integer":
                val = str(q.get("answer") or q.get("integerAnswer") or q.get("value") or "0").strip()
                val = "".join(re.findall(r'[-\d]+', val))
                parsed_q["integerAnswer"] = val if val else "0"

            else:
                # descriptive / match / diagram / case_study / short_answer / detailed_answer
                parsed_q["answer"] = _clean_text(str(model_answer))
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

    notes = data.get("notes")
    notes_context = ""
    if notes:
        if isinstance(notes, list):
            notes_context = "\n".join([f"- {n}" for n in notes if n])
        elif isinstance(notes, str):
            notes_context = notes.strip()
        
        if notes_context:
            notes_context = f"\n\nADDITIONAL CONTEXT (from lecture notes):\n{notes_context}\n"

    try:
        num_questions = max(1, min(int(data.get("num_questions", 2)), 10))
    except (TypeError, ValueError):
        num_questions = 2

    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"

    subject = (data.get("subject") or "").strip()
    chapter = (data.get("chapter") or "").strip()
    exam_target = (data.get("exam_target") or "").strip().lower()

    qtype = (data.get("type") or data.get("question_type") or "mcq").strip().lower()
    style = (data.get("style") or "").strip().lower()

    # ── Style-specific prompt templates (style takes priority over bare qtype) ─
    if style == "assertion_reason":
        type_instr = (
            "ASSERTION-REASON PATTERN — follow this format exactly for every question:\n"
            "  Question text must be:\n"
            "    \"Assertion (A): [a clear factual/conceptual statement about the topic]\n"
            "     Reason (R): [an explanation, cause, or mechanism related to the assertion]\"\n\n"
            "  The 4 options MUST always be exactly these standard assertion-reason choices:\n"
            "    A: Both Assertion (A) and Reason (R) are true, and R is the correct explanation of A\n"
            "    B: Both Assertion (A) and Reason (R) are true, but R is NOT the correct explanation of A\n"
            "    C: Assertion (A) is true, but Reason (R) is false\n"
            "    D: Assertion (A) is false, but Reason (R) may be true or false\n\n"
            "  In your JSON: 'options' must be the list of these 4 strings exactly, "
            "'answer' is the correct label (A/B/C/D), "
            "'explanation' justifies which assertion/reason relationship holds and why."
        )
    elif style == "statement":
        type_instr = (
            "STATEMENT-BASED PATTERN — follow this format exactly for every question:\n"
            "  Question text must present 2–3 numbered statements about the topic, with mixed correctness.\n"
            "  Format:\n"
            "    \"Consider the following statements:\n"
            "     Statement 1: [...]\n"
            "     Statement 2: [...]\n"
            "     Statement 3: [...]\n"
            "     Which of the above statements is/are correct?\"\n\n"
            "  Options must cover combinations of which statements are true, e.g.:\n"
            "    A: 1 only   B: 1 and 2 only   C: 2 and 3 only   D: All three\n"
            "  (adjust options to match actual correctness of the statements you wrote)\n\n"
            "  In your JSON: 'options' (4 strings), 'answer' (correct label A/B/C/D), "
            "'explanation' clarifying which statements are true/false and why."
        )
    elif style == "match":
        type_instr = (
            "MATCH THE FOLLOWING PATTERN — follow this format exactly for every question:\n"
            "  The question text MUST contain two labeled columns:\n"
            "    Column I  — four items labeled 1, 2, 3, 4 (concepts/terms/processes)\n"
            "    Column II — four items labeled A, B, C, D (definitions/examples/descriptions)\n"
            "  Each Column I item maps to exactly one Column II item.\n\n"
            "  In your JSON:\n"
            "    'question': the full question text including both columns\n"
            "    'answer': the correct mapping string, e.g. '1-C, 2-A, 3-D, 4-B'\n"
            "    'explanation': one sentence per pair explaining each mapping\n"
            "  Do NOT include A/B/C/D MCQ option lists."
        )
    elif style == "diagram":
        type_instr = (
            "DIAGRAM-BASED PATTERN — follow this format exactly for every question:\n"
            "  The question MUST include a labeled text/ASCII diagram or clearly described labeled structure.\n"
            "  Subject-specific guidance:\n"
            "    Biology — labeled cell/organelle diagram, organ system cross-section, or process flowchart\n"
            "    Chemistry — labelled apparatus, molecular bonding structure, or reaction setup\n"
            "    Physics — labelled circuit diagram, ray diagram, or force/vector diagram\n"
            "  The question must ask the student to:\n"
            "    (a) identify labeled parts, OR (b) explain the process shown, OR (c) predict an outcome\n\n"
            "  In your JSON:\n"
            "    'question': includes the ASCII/text diagram with clear labels (A, B, C, D or i, ii, iii)\n"
            "    'answer': detailed model answer referencing diagram labels explicitly\n"
            "    'explanation': additional examiner insight\n"
            "  Do NOT include A/B/C/D MCQ option lists."
        )
    elif style == "case_study":
        type_instr = (
            "CASE STUDY / PASSAGE-BASED PATTERN — follow this format exactly for every question:\n"
            "  Begin with a 60–80 word scenario, experiment result, or data extract about the topic.\n"
            "  Follow immediately with a 3-part structured question:\n"
            "    (a) [1 mark] Identify the principle / phenomenon / law involved\n"
            "    (b) [2 marks] Explain the mechanism / show the calculation / derive the result\n"
            "    (c) [1 mark] Predict what changes if [a specific variable] is modified\n\n"
            "  In your JSON:\n"
            "    'question': passage text + all three sub-questions (a), (b), (c)\n"
            "    'answer': answers labeled (a), (b), (c) with complete content\n"
            "    'explanation': key insight connecting the case to the core topic concept\n"
            "  Do NOT include A/B/C/D MCQ option lists."
        )
    elif style == "short_answer":
        type_instr = (
            "SHORT ANSWER (2-mark pattern) — follow this format exactly for every question:\n"
            "  Question style: concise single-sentence stem, e.g.:\n"
            "    'Define X.'  /  'State Y with an example.'  /  'Differentiate between A and B.'\n"
            "  Answer must have exactly 2 clearly numbered points:\n"
            "    Point 1 (1 mark): definition / statement / first distinction\n"
            "    Point 2 (1 mark): example / elaboration / second distinction\n\n"
            "  In your JSON:\n"
            "    'question': concise 1-sentence question\n"
            "    'answer': exactly 2 numbered points, each 1–2 sentences\n"
            "    'explanation': brief note on why this answer is correct\n"
            "  Do NOT include A/B/C/D MCQ option lists."
        )
    elif style == "detailed_answer":
        type_instr = (
            "DETAILED / LONG ANSWER (5-mark pattern) — follow this format exactly for every question:\n"
            "  Question style: multi-part or broad stem demanding extended structured response, e.g.:\n"
            "    'Explain in detail...'  /  'Derive the expression for...'  /  'Describe with examples and diagram...'\n"
            "  Answer MUST have 4–5 clearly numbered points:\n"
            "    Point 1 (1m): Introduction / definition\n"
            "    Point 2 (1m): Core mechanism / principle / derivation step 1\n"
            "    Point 3 (1m): Continuation / derivation step 2 / application\n"
            "    Point 4 (1m): Example / diagram description / conclusion\n"
            "    Point 5 (1m, if applicable): Special case / exception / real-world relevance\n\n"
            "  In your JSON:\n"
            "    'question': detailed question stem\n"
            "    'answer': 4–5 numbered points with sub-explanations\n"
            "    'explanation': examiner tip or common mistake to avoid\n"
            "  Do NOT include A/B/C/D MCQ option lists."
        )
    # ── Base type instructions (used when no style override) ─────────────────
    elif qtype == "integer":
        type_instr = (
            "INTEGER TYPE — each question has a single numerical answer (integer, e.g. 42 or -5).\n"
            "No options. Show the working/formula in the question if needed.\n"
            "In your JSON: 'answer' field must be the integer value as a string."
        )
    elif qtype == "mcq_multi":
        type_instr = (
            "MULTIPLE CORRECT MCQ — one or more options can be correct.\n"
            "Provide exactly 4 options (A, B, C, D). All 4 must be completely distinct.\n"
            "At least 2 options must be correct.\n"
            "In your JSON: 'options' (list of 4 strings), "
            "'correctOptions' (list of correct labels, e.g. ['A', 'C'])."
        )
    elif qtype in ("descriptive", "short_answer", "long_answer"):
        type_instr = (
            "DESCRIPTIVE / STRUCTURED ANSWER — no A/B/C/D options.\n"
            "Provide a detailed model answer in the 'answer' field.\n"
            "Break it into clear numbered steps or bullet points using newlines."
        )
    elif qtype in ("match", "match_the_following"):
        type_instr = (
            "MATCH THE FOLLOWING — two columns (Column I items 1–4, Column II items A–D).\n"
            "Render both columns inside the 'question' text.\n"
            "'answer' must give the mapping as '1-B, 2-D, 3-A, 4-C'."
        )
    elif qtype == "diagram":
        type_instr = (
            "DIAGRAM-BASED — include a labeled ASCII/text diagram in the question stem.\n"
            "Ask student to identify parts or explain the process.\n"
            "'answer' must reference diagram labels explicitly."
        )
    else:
        type_instr = (
            "SINGLE CORRECT MCQ — exactly 4 options. All 4 MUST be completely distinct.\n"
            "In your JSON: 'options' (list of 4 strings), 'answer' (correct label A/B/C/D)."
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

    # Build curriculum breadcrumb for the prompt
    curriculum_parts = []
    if subject:
        curriculum_parts.append(f"Subject: {subject}")
    if chapter:
        curriculum_parts.append(f"Chapter: {chapter}")
    curriculum_parts.append(f"Topic: {topic}")
    curriculum_context = " → ".join(curriculum_parts)

    scope_constraint = ""
    if subject or chapter:
        scope_constraint = (
            f"IMPORTANT: All questions MUST be strictly within the scope of {curriculum_context}. "
            f"Do NOT generate questions from unrelated topics or subjects.\n"
        )

    subject_rule = _SUBJECT_RULES_TEST.get(subject.lower(), "") if subject else ""

    user_prompt = (
        f"Generate {num_questions} distinct {qtype} questions.\n"
        f"Curriculum scope: {curriculum_context}\n"
        f"{scope_constraint}"
        f"Difficulty: {difficulty}. This should be a {diff_context}.\n"
        f"{type_instr}\n\n"
        f"{subject_rule}\n\n"
        f"{heuristic_block}\n"
        f"{notes_context}\n"
        "Return the result as a JSON object with a 'questions' array. "
        "Each question object MUST have: 'question', 'answer', and 'explanation'. "
        "Include an 'options' field only for MCQs."
    )

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("test_generate")

    logger.info(
        "generate_practice_test | subject=%s | chapter=%s | topic=%s | type=%s | style=%s | n=%d",
        subject or "—", chapter or "—", topic[:50], qtype, style or "—", num_questions
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

    parsed = parse_ai_result(result, topic, difficulty, qtype, style)
    parsed["_meta"] = {
        "source": "llm",
        "model": result["model"],
        "latency_ms": round(result["latency_ms"]),
        "institute": institute_id,
        "type": qtype,
        "style": style or None,
    }

    logger.info(
        "generate_practice_test: OK | type=%s | style=%s | returned=%d questions",
        qtype, style or "—", len(parsed["questions"]),
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