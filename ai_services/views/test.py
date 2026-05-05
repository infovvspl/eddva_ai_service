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
    # chapters = list of exact DB chapter names for subject tests (overrides AI's own chapter selection)
    chapters_raw = data.get("chapters") or []
    db_chapters = [str(c).strip() for c in chapters_raw if str(c).strip()] if isinstance(chapters_raw, list) else []
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

    # ── Normalise exam_target so bare "jee", "jee_mains", "both", etc. all map correctly ──
    # This is the value that comes from student.examTarget (could be enum value like "jee" or "jee_mains")
    et_raw = (exam_target or "").lower().replace("_", " ").strip()
    if et_raw == "both":
        et_raw = "jee main"  # Treat "both JEE+NEET" as JEE Main by default (more general)
    elif et_raw == "jee":
        et_raw = "jee main"  # Bare "jee" defaults to JEE Main, NOT school/CBSE
    exam_target = et_raw  # Use the normalized value downstream

    heuristic_block = ""
    if exam_target:
        exam_key = "jee advanced" if "advance" in exam_target else "jee main" if ("main" in exam_target or exam_target == "jee") else "neet" if "neet" in exam_target else ""
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

    # ── Exam-aware difficulty description (matches the heuristic block above) ──
    if exam_target:
        et = exam_target.lower()
        if "advance" in et:  # JEE Advanced
            diff_context = (
                "IIT JEE Advanced level — multi-concept multi-step problem requiring deep reasoning, "
                "non-trivial mathematics, and synthesis of 2+ concepts. Absolutely NOT school/NCERT-direct."
                if difficulty == "hard"
                else "JEE Advanced level — competitive multi-step problem with at least one non-obvious insight"
                if difficulty == "medium"
                else "JEE Advanced foundation — single-concept but still competitive (above NCERT direct)"
            )
        elif "main" in et:  # JEE Main
            diff_context = (
                "JEE Main hard tier — calculation-heavy 4-6 step problem, NCERT-extension level, "
                "matches actual JEE Main paper difficulty (NOT school/board level)"
                if difficulty == "hard"
                else "JEE Main level — competitive single/two-concept MCQ, balanced numerical + conceptual"
                if difficulty == "medium"
                else "JEE Main entry tier — competitive MCQ on direct concept, slightly above NCERT"
            )
        elif "neet" in et:
            diff_context = (
                "NEET hard tier — statement-based or assertion-reason, multi-fact integration, "
                "tricky exceptions and confusing distractors (NOT school/CBSE-direct)"
                if difficulty == "hard"
                else "NEET level — NCERT-deep conceptual MCQ with at least one common-misconception trap"
                if difficulty == "medium"
                else "NEET foundation — direct NCERT-fact MCQ with mild trickiness"
            )
        else:  # CBSE / school
            diff_context = (
                "CBSE board hard tier — HOTS (Higher Order Thinking Skill) question with application twist"
                if difficulty == "hard"
                else "CBSE board standard — direct concept application, board-paper style"
                if difficulty == "medium"
                else "CBSE foundation — definition or simple recall question"
            )
    else:
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

    # ── STRICT SCOPE LOCK — top-of-prompt enforcement ──────────────────────────
    # Without this, the model freely cross-pollinates from other chapters/subjects
    # (e.g. "Liquid Solution" chapter → Organic Chemistry questions).
    scope_constraint = ""
    if subject or chapter or topic:
        scope_lock_lines = ["🔒 STRICT SCOPE LOCK — READ CAREFULLY 🔒"]
        if subject:
            scope_lock_lines.append(f"  • Subject: {subject}  (FORBIDDEN to use any other subject)")
        if chapter:
            scope_lock_lines.append(f"  • Chapter: {chapter}  (FORBIDDEN to use any other chapter)")
        if topic and len(topic) < 80:  # Only add structured topic if it's a clean name, not a prose paragraph
            scope_lock_lines.append(f"  • Topic: {topic}")
        scope_lock_lines.append("")
        scope_lock_lines.append(
            "  Every question MUST be specifically about the chapter content above. "
            "Before finalizing each question, ask yourself: 'Could this question appear in a textbook chapter "
            f"titled \"{chapter or topic or subject}\"?' If the answer is NO, DELETE the question and write a new one that fits."
        )
        scope_lock_lines.append(
            "  CROSS-SUBJECT / CROSS-CHAPTER CONTAMINATION IS NOT ALLOWED. "
            "If you are inclined to write about Organic Chemistry but the chapter is Inorganic, STOP and pivot."
        )

        # ── Subject-specific anti-contamination hints (covers full JEE/NEET/CBSE syllabus) ──
        # Each branch matches keywords against the chapter name and emits a tight scope hint
        # telling the model exactly which sub-domain is allowed and which are FORBIDDEN.
        sub_lower = (subject or "").lower()
        chap_lower = (chapter or "").lower()
        topic_lower = (topic or "").lower()
        cmb = f"{chap_lower} {topic_lower}"  # combined chapter+topic for keyword matching

        def has(*kws):
            return any(k in cmb for k in kws)

        # ═══ CHEMISTRY ═══════════════════════════════════════════════════════
        if "chem" in sub_lower:
            # ── Physical Chemistry ──
            if has("solution", "colligative", "raoult", "henry", "molarity", "molality", "normality",
                   "concentration", "vapour pressure", "vapor pressure", "osmotic"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Solutions chapter. Allowed: Raoult's law, Henry's law, ideal/non-ideal solutions, colligative properties, depression of freezing point, elevation of boiling point, osmotic pressure, van't Hoff factor. FORBIDDEN: organic reactions, inorganic compounds, atomic structure.")
            elif has("solid state", "crystal", "lattice", "unit cell", "packing", "bcc", "fcc", "hcp"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Solid State chapter. Allowed: crystal lattice, unit cells (BCC/FCC/HCP), packing efficiency, density of unit cell, defects, Schottky/Frenkel defects, semiconductors. FORBIDDEN: organic, inorganic, solutions.")
            elif has("thermodynamic", "enthalpy", "entropy", "gibbs", "spontaneity", "calorimet", "hess"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Thermodynamics chapter. Allowed: First/Second/Third laws, enthalpy, entropy, Gibbs free energy, spontaneity, Hess's law, bond enthalpies, calorimetry. FORBIDDEN: organic synthesis, inorganic facts, equilibrium.")
            elif has("equilibrium", "le chatelier", "kp", "kc", "ka", "kb", "kw", "buffer", "ph ", "acid base", "ionic equilibrium", "solubility product", "ksp"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Equilibrium chapter (chemical/ionic). Allowed: Kp, Kc, Le Chatelier's principle, Ka, Kb, Kw, pH calculations, buffers, Ksp, common ion effect. FORBIDDEN: organic, inorganic, electrochemistry.")
            elif has("electrochem", "galvanic", "electrolyt", "nernst", "emf", "battery", "fuel cell", "corrosion", "kohlrausch", "conductivity"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Electrochemistry chapter. Allowed: galvanic/electrolytic cells, EMF, Nernst equation, Kohlrausch law, conductivity, batteries, fuel cells, corrosion, electrolysis. FORBIDDEN: organic, inorganic compound facts, kinetics.")
            elif has("kinetic", "rate", "order of reaction", "molecularity", "arrhenius", "half life", "activation energy", "pseudo first"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Chemical Kinetics chapter. Allowed: rate law, order vs molecularity, Arrhenius equation, activation energy, half-life, pseudo-first-order, integrated rate equations. FORBIDDEN: equilibrium constants, thermodynamics, organic mechanisms.")
            elif has("surface chem", "adsorption", "colloid", "emulsion", "catalyst", "tyndall", "brownian", "micelle", "freundlich"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Surface Chemistry chapter. Allowed: adsorption, Freundlich isotherm, catalysis, colloids, emulsions, micelles, Tyndall effect. FORBIDDEN: organic synthesis, inorganic facts.")
            elif has("atomic structure", "bohr", "quantum", "schrodinger", "orbital", "azimuthal", "magnetic quantum", "de broglie", "heisenberg"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Atomic Structure chapter. Allowed: Bohr model, quantum numbers, orbitals, electronic configuration, de Broglie, Heisenberg uncertainty, Schrödinger. FORBIDDEN: chemical bonding, periodic trends, organic.")
            elif has("mole concept", "stoichiometr", "equivalent weight", "empirical", "molecular formula", "limiting reagent", "percentage compos", "some basic"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Mole Concept / Stoichiometry chapter. Allowed: mole calculations, empirical/molecular formula, limiting reagent, percentage composition, equivalent weight, gravimetric analysis. FORBIDDEN: organic, inorganic, advanced topics.")
            elif has("gaseous", "states of matter", "ideal gas", "real gas", "van der waals", "compressibility", "kinetic theory of gas"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — States of Matter / Gases. Allowed: ideal gas equation, gas laws (Boyle/Charles/Avogadro), real gases, van der Waals equation, compressibility factor, kinetic theory of gases. FORBIDDEN: solid state, solutions, organic.")
            elif has("redox", "oxidation number", "balancing redox"):
                scope_lock_lines.append("  • PHYSICAL CHEMISTRY — Redox Reactions chapter. Allowed: oxidation states, balancing redox equations (ion-electron / oxidation number method), redox titrations. FORBIDDEN: electrochemistry, organic, inorganic facts.")

            # ── Inorganic Chemistry ──
            elif has("periodic", "periodicity", "ionization energy", "electron affinity", "electronegativity"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Periodic Table / Periodicity. Allowed: trends in atomic radius, ionization energy, electron affinity, electronegativity, anomalies. FORBIDDEN: organic, physical chemistry calculations, individual block reactions.")
            elif has("chemical bonding", "molecular structure", "vsepr", "hybridi", "bond order", "molecular orbital", "valence bond"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Chemical Bonding chapter. Allowed: VSEPR theory, hybridization, MO theory, bond order, valence bond, polarity, dipole moment, H-bonding. FORBIDDEN: organic reactions, periodic trends, physical chemistry.")
            elif has("hydrogen", "hydride", "heavy water", "hydrogen peroxide"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Hydrogen chapter. Allowed: position of H in periodic table, isotopes, hydrides (ionic/covalent/interstitial), H₂O, heavy water, H₂O₂, hard/soft water. FORBIDDEN: other elements, organic, physical chemistry.")
            elif has("s-block", "s block", "alkali metal", "alkaline earth", "group 1", "group 2", "sodium", "potassium", "lithium", "magnesium", "calcium"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — s-Block Elements chapter. Allowed: Group 1 & 2 metals, their oxides/hydroxides/carbonates/halides, anomalous behaviour of Li/Be, biological importance, plaster of Paris, NaOH, NaCl, Na₂CO₃. FORBIDDEN: p-block, d-block, organic.")
            elif has("p-block", "p block", "group 13", "group 14", "group 15", "group 16", "group 17", "group 18", "boron", "carbon family", "nitrogen family", "oxygen family", "halogen", "noble gas"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — p-Block Elements chapter. Allowed: Groups 13-18, anomalous behaviour, allotropes, oxides/oxyacids, interhalogens, noble gas compounds. FORBIDDEN: s-block, d-block, organic.")
            elif has("d-block", "d block", "transition", "f-block", "f block", "lanthanoid", "actinoid", "lanthanide", "actinide"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — d-Block / f-Block Elements chapter. Allowed: transition metal trends, variable oxidation states, magnetic/colour properties, alloys, KMnO₄, K₂Cr₂O₇, lanthanoid/actinoid contraction. FORBIDDEN: s/p-block, coordination chemistry, organic.")
            elif has("coordination", "complex compound", "ligand", "werner", "cfse", "crystal field", "isomerism in coord"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Coordination Compounds chapter. Allowed: Werner's theory, ligands, IUPAC nomenclature of complexes, isomerism, VBT/CFT, CFSE, magnetic moment, applications. FORBIDDEN: simple inorganic salts, organic, physical chemistry.")
            elif has("metallurg", "extraction", "ore", "mineral", "concentration of ore", "froth flotation", "smelting", "calcination", "roasting", "isolation of element"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Metallurgy / Isolation of Elements chapter. Allowed: ore concentration (gravity/froth flotation/magnetic), calcination, roasting, smelting, electrolytic refining, Ellingham diagram, extraction of Fe/Cu/Al/Zn. FORBIDDEN: organic, individual element families.")
            elif has("salt analysis", "qualitative analysis", "anion", "cation", "group reagent"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Qualitative Salt Analysis. Allowed: cation groups (I-VI), anion tests, group reagents, confirmatory tests, dry/wet tests. FORBIDDEN: organic, physical chemistry.")
            elif has("environmental chem", "pollution", "smog", "ozone depletion", "acid rain", "greenhouse"):
                scope_lock_lines.append("  • INORGANIC CHEMISTRY — Environmental Chemistry chapter. Allowed: air/water/soil pollution, smog, acid rain, ozone depletion, greenhouse effect, BOD/COD, green chemistry. FORBIDDEN: organic synthesis, physical chemistry calculations.")

            # ── Organic Chemistry ──
            elif has("goc", "general organic", "hyperconjugation", "inductive", "mesomeric", "resonance effect", "electromeric", "carbocation", "carbanion", "free radical stability"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — GOC (General Organic Chemistry) chapter. Allowed: inductive/mesomeric/electromeric/hyperconjugation effects, stability of carbocations/carbanions/free radicals, reaction intermediates, types of reactions (addition/substitution/elimination), bond cleavage. FORBIDDEN: physical, inorganic, specific functional groups in detail.")
            elif has("iupac", "nomenclature", "isomer"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — IUPAC Nomenclature / Isomerism chapter. Allowed: IUPAC naming rules, structural/stereoisomerism, geometric, optical isomerism, R/S, E/Z, chirality, conformers. FORBIDDEN: reactions, physical, inorganic.")
            elif has("hydrocarbon", "alkane", "alkene", "alkyne", "aromatic", "benzene", "wurtz", "kolbe"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Hydrocarbons chapter. Allowed: alkanes/alkenes/alkynes/aromatic compounds, preparation methods (Wurtz, Kolbe, Sabatier), addition reactions, Markovnikov/anti-Markovnikov, Friedel-Crafts. FORBIDDEN: physical, inorganic, oxygen-containing functional groups in detail.")
            elif has("haloalkane", "haloarene", "alkyl halide", "aryl halide", "sn1", "sn2", "e1", "e2"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Haloalkanes & Haloarenes chapter. Allowed: SN1/SN2/E1/E2 mechanisms, Walden inversion, polyhalogen compounds (chloroform/CCl₄/freons), Grignard reagents prep, comparison alkyl vs aryl halides. FORBIDDEN: alcohols/ethers in detail, physical, inorganic.")
            elif has("alcohol", "phenol", "ether", "williamson"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Alcohols, Phenols & Ethers chapter. Allowed: prep (from haloalkanes, hydration of alkenes), reactions (esterification, dehydration, oxidation), Williamson ether synthesis, Kolbe's, Reimer-Tiemann, distinction tests (Lucas, iodoform). FORBIDDEN: aldehydes/ketones in detail, physical, inorganic.")
            elif has("aldehyde", "ketone", "carboxylic acid", "ester", "acid derivative", "acid halide", "acid anhydride", "amide", "cannizzaro", "aldol", "perkin", "claisen"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Aldehydes, Ketones & Carboxylic Acids chapter. Allowed: prep (Rosenmund, Stephen, Etard), nucleophilic addition reactions, aldol/Cannizzaro/Perkin/Claisen, Tollens/Fehling/iodoform tests, esterification, decarboxylation. FORBIDDEN: amines, alcohols in detail, physical, inorganic.")
            elif has("amine", "diazonium", "aniline", "hofmann", "gabriel", "carbylamine", "azo coupling"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Amines chapter. Allowed: 1°/2°/3° amines, prep (Hofmann/Gabriel/reduction of nitriles or nitro), basicity comparison, carbylamine test, diazonium salts, Sandmeyer/Gattermann, azo coupling. FORBIDDEN: aldehydes/ketones, alcohols, inorganic.")
            elif has("biomolecule", "carbohydrate", "protein", "amino acid", "lipid", "vitamin", "nucleic", "dna", "rna", "enzyme", "monosaccharide", "polysaccharide", "peptide"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Biomolecules chapter. Allowed: carbohydrates (mono/di/poly-saccharides), proteins (α-amino acids, peptide bond, structure), enzymes, vitamins (water/fat-soluble), nucleic acids (DNA/RNA structure), hormones. FORBIDDEN: small organic molecules, physical, inorganic.")
            elif has("polymer", "polymerization", "monomer", "thermoplast", "thermoset", "elastomer", "natural polymer", "synthetic polymer", "addition polymer", "condensation polymer", "rubber", "nylon", "teflon", "bakelite"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Polymers chapter. Allowed: classification (natural/synthetic, addition/condensation), polymerization mechanisms, common polymers (polythene, PVC, Teflon, Bakelite, nylon, terylene), biodegradable polymers. FORBIDDEN: small organic molecules, physical, inorganic.")
            elif has("everyday life", "drug", "antibiotic", "analgesic", "antiseptic", "antacid", "soap", "detergent", "food preservative"):
                scope_lock_lines.append("  • ORGANIC CHEMISTRY — Chemistry in Everyday Life chapter. Allowed: drugs (analgesics, antipyretics, antibiotics, antiseptics, antacids, antihistamines), soaps/detergents (saponification), food preservatives, artificial sweeteners. FORBIDDEN: pure organic synthesis, physical, inorganic.")

        # ═══ PHYSICS ═════════════════════════════════════════════════════════
        elif "physics" in sub_lower:
            if has("unit", "measurement", "dimensional", "significant figure"):
                scope_lock_lines.append("  • PHYSICS — Units & Measurements chapter. Allowed: SI units, dimensional analysis, significant figures, errors in measurement, accuracy vs precision. FORBIDDEN: any other physics topic.")
            elif has("kinematic", "motion in straight", "motion in plane", "projectile", "relative velocity", "uniformly accelerated"):
                scope_lock_lines.append("  • PHYSICS — Kinematics chapter (1D/2D motion). Allowed: equations of motion, projectile motion, relative velocity, uniform circular motion, vectors of position/velocity/acceleration. FORBIDDEN: forces (Newton's laws), energy, rotation.")
            elif has("newton", "law of motion", "friction", "pseudo force", "free body", "tension", "normal force"):
                scope_lock_lines.append("  • PHYSICS — Laws of Motion chapter. Allowed: Newton's three laws, free-body diagrams, friction (static/kinetic/rolling), pseudo forces, banking of roads, tension in strings, pulley problems. FORBIDDEN: kinematics in isolation, energy, rotation, gravitation.")
            elif has("work", "energy", "power", "kinetic energy", "potential energy", "conservation of energy", "collision", "spring potential"):
                scope_lock_lines.append("  • PHYSICS — Work, Energy & Power chapter. Allowed: work done by constant/variable force, KE, PE (gravitational, spring), conservation of energy, elastic/inelastic collisions, power. FORBIDDEN: forces alone, rotation in detail, gravitation.")
            elif has("system of particle", "centre of mass", "rotation", "rotational motion", "torque", "angular momentum", "moment of inertia", "rolling motion"):
                scope_lock_lines.append("  • PHYSICS — Rotational Motion chapter. Allowed: centre of mass, torque, angular momentum, moment of inertia, parallel/perpendicular axis theorems, rolling without slipping, rotational kinematics/dynamics. FORBIDDEN: linear motion alone, gravitation, oscillations.")
            elif has("gravitation", "gravity", "kepler", "satellite", "escape velocity", "orbital velocity", "geostationary"):
                scope_lock_lines.append("  • PHYSICS — Gravitation chapter. Allowed: Newton's law of gravitation, gravitational PE, Kepler's laws, satellite motion, escape/orbital velocity, geostationary orbit, weightlessness. FORBIDDEN: rotational motion, mechanics-only problems.")
            elif has("elasticity", "stress", "strain", "young modulus", "bulk modulus", "shear modulus"):
                scope_lock_lines.append("  • PHYSICS — Mechanical Properties of Solids (Elasticity) chapter. Allowed: stress-strain curve, Young's/Bulk/Shear moduli, Hooke's law, elastic energy. FORBIDDEN: fluids, thermal, mechanics-only.")
            elif has("fluid", "pressure", "bernoulli", "viscosity", "stokes", "surface tension", "capillarity", "reynolds", "torricelli", "pascal"):
                scope_lock_lines.append("  • PHYSICS — Mechanical Properties of Fluids chapter. Allowed: hydrostatic pressure, Pascal's law, Archimedes principle, Bernoulli's equation, viscosity, Stokes' law, surface tension, capillarity. FORBIDDEN: solids, thermal, mechanics.")
            elif has("thermal", "temperature", "heat transfer", "calorimetry", "specific heat", "latent heat", "stefan", "wien", "newton's law of cooling", "expansion"):
                scope_lock_lines.append("  • PHYSICS — Thermal Properties of Matter chapter. Allowed: thermal expansion, calorimetry, specific/latent heat, conduction/convection/radiation, Stefan-Boltzmann, Wien's law, Newton's law of cooling. FORBIDDEN: thermodynamics laws, kinetic theory in detail.")
            elif has("thermodynamic", "first law", "second law", "carnot", "isothermal", "adiabatic", "isobaric", "isochoric", "entropy"):
                scope_lock_lines.append("  • PHYSICS — Thermodynamics chapter. Allowed: zeroth/first/second laws, thermodynamic processes (isothermal/adiabatic/isobaric/isochoric), Carnot engine, refrigerators, entropy, efficiency. FORBIDDEN: kinetic theory in detail, thermal expansion alone.")
            elif has("kinetic theory", "rms speed", "mean free path", "degree of freedom", "maxwell distribution"):
                scope_lock_lines.append("  • PHYSICS — Kinetic Theory of Gases chapter. Allowed: pressure of gas, RMS/mean/most probable speeds, mean free path, degrees of freedom, equipartition of energy, specific heats. FORBIDDEN: thermodynamics laws, thermal in isolation.")
            elif has("oscillat", "shm", "simple harmonic", "pendulum", "spring oscill", "damped", "forced oscill", "resonance"):
                scope_lock_lines.append("  • PHYSICS — Oscillations chapter. Allowed: SHM, displacement/velocity/acceleration in SHM, energy in SHM, simple/compound pendulum, spring-mass system, damped/forced oscillations, resonance. FORBIDDEN: waves on string, sound, mechanics.")
            elif has("wave motion", "wave on string", "transverse wave", "longitudinal wave", "sound", "doppler", "stationary wave", "beats", "organ pipe", "superposition"):
                scope_lock_lines.append("  • PHYSICS — Waves chapter. Allowed: wave equation, transverse/longitudinal waves, waves on string, sound waves, superposition, beats, stationary waves, organ pipes, Doppler effect. FORBIDDEN: oscillations alone, light/optics.")
            elif has("electrostatic", "electric field", "electric potential", "gauss", "capacitor", "dielectric", "coulomb"):
                scope_lock_lines.append("  • PHYSICS — Electrostatics chapter. Allowed: Coulomb's law, electric field/potential, Gauss's law, conductors/insulators, dipole, capacitors (parallel plate, series/parallel), dielectrics. FORBIDDEN: current electricity, magnetism, EM induction.")
            elif has("current electric", "resistance", "kirchhoff", "ohm", "wheatstone", "potentiometer", "galvanometer", "ammeter", "voltmeter", "drift velocity", "emf"):
                scope_lock_lines.append("  • PHYSICS — Current Electricity chapter. Allowed: Ohm's law, resistance/resistivity, Kirchhoff's laws, Wheatstone bridge, potentiometer, galvanometer/ammeter/voltmeter, EMF vs terminal voltage, internal resistance. FORBIDDEN: electrostatics in isolation, magnetism, EM induction.")
            elif has("magnetic effect", "biot-savart", "ampere", "lorentz force", "moving charge", "cyclotron", "solenoid", "torque on current loop", "magnetic dipole"):
                scope_lock_lines.append("  • PHYSICS — Moving Charges & Magnetism chapter. Allowed: Biot-Savart, Ampere's circuital law, Lorentz force, force on current-carrying conductor, cyclotron, torque on current loop, moving coil galvanometer. FORBIDDEN: magnetism of materials in isolation, EM induction, AC.")
            elif has("magnetism and matter", "magnetic property", "diamagnet", "paramagnet", "ferromagnet", "hysteresis", "earth magnetism"):
                scope_lock_lines.append("  • PHYSICS — Magnetism & Matter chapter. Allowed: bar magnet as magnetic dipole, Earth's magnetism, magnetic properties of materials (dia/para/ferro), hysteresis. FORBIDDEN: moving charges, EM induction, AC.")
            elif has("electromagnetic induction", "faraday", "lenz", "self induction", "mutual induction", "eddy current", "induced emf"):
                scope_lock_lines.append("  • PHYSICS — Electromagnetic Induction chapter. Allowed: Faraday's law, Lenz's law, motional EMF, self/mutual inductance, eddy currents, energy stored in inductor. FORBIDDEN: electrostatics, AC circuits in isolation.")
            elif has("alternating current", "ac circuit", "rms value", "lcr", "impedance", "reactance", "transformer", "resonance in lcr", "wattless"):
                scope_lock_lines.append("  • PHYSICS — Alternating Current chapter. Allowed: RMS/peak values of AC, AC through R/L/C/LCR, impedance/reactance, resonance in LCR, power in AC, transformers, wattless current. FORBIDDEN: DC current, EM induction in isolation.")
            elif has("electromagnetic wave", "em wave", "displacement current", "maxwell equation", "spectrum"):
                scope_lock_lines.append("  • PHYSICS — Electromagnetic Waves chapter. Allowed: displacement current, Maxwell's equations qualitatively, EM wave properties, EM spectrum (radio/microwave/IR/visible/UV/X-ray/gamma) and uses. FORBIDDEN: AC circuits, optics in detail.")
            elif has("ray optic", "reflection", "refraction", "lens", "mirror", "prism", "total internal reflection", "optical instrument", "human eye", "telescope", "microscope"):
                scope_lock_lines.append("  • PHYSICS — Ray Optics chapter. Allowed: reflection/refraction, mirror/lens formulae, prism (deviation, dispersion), TIR, optical instruments (eye, microscope, telescope). FORBIDDEN: wave optics, EM waves in isolation.")
            elif has("wave optic", "interference", "diffraction", "polarisation", "polarization", "young double slit", "huygens", "single slit"):
                scope_lock_lines.append("  • PHYSICS — Wave Optics chapter. Allowed: Huygens' principle, interference (YDSE), diffraction (single slit), polarisation, Brewster's law, resolving power. FORBIDDEN: ray optics, EM waves in isolation.")
            elif has("dual nature", "photoelectric", "matter wave", "de broglie", "davisson germer"):
                scope_lock_lines.append("  • PHYSICS — Dual Nature of Matter & Radiation chapter. Allowed: photoelectric effect, Einstein's equation, work function, threshold frequency/wavelength, de Broglie wavelength, Davisson-Germer experiment. FORBIDDEN: atoms/nuclei, semiconductors.")
            elif has("atom", "bohr model", "spectral line", "hydrogen spectrum", "rutherford"):
                scope_lock_lines.append("  • PHYSICS — Atoms chapter. Allowed: Rutherford's α-scattering, Bohr's model of H atom, spectral series (Lyman/Balmer/Paschen/Brackett/Pfund), energy levels, ionization energy. FORBIDDEN: nuclei, photoelectric effect in detail, quantum.")
            elif has("nuclei", "nuclear", "binding energy", "mass defect", "radioactive", "alpha decay", "beta decay", "gamma decay", "fission", "fusion", "half life"):
                scope_lock_lines.append("  • PHYSICS — Nuclei chapter. Allowed: nuclear binding energy, mass defect, radioactive decay (α/β/γ), half-life, decay constant, nuclear fission/fusion. FORBIDDEN: atoms (Bohr), photoelectric, semiconductors.")
            elif has("semiconductor", "p-n junction", "diode", "transistor", "logic gate", "rectifier", "zener", "transistor amplifier", "led"):
                scope_lock_lines.append("  • PHYSICS — Semiconductor Electronics chapter. Allowed: intrinsic/extrinsic semiconductors, n-type/p-type, p-n junction diode, half/full-wave rectifier, Zener diode as voltage regulator, transistor (CB/CE/CC), logic gates. FORBIDDEN: nuclei, atoms, classical physics.")
            elif has("communication", "modulation", "amplitude modulation", "frequency modulation", "transmission", "antenna"):
                scope_lock_lines.append("  • PHYSICS — Communication Systems chapter. Allowed: elements of comm system, bandwidth, propagation modes, AM/FM modulation/demodulation, sky/space wave propagation. FORBIDDEN: semiconductors, EM waves in isolation.")

        # ═══ MATHEMATICS ═════════════════════════════════════════════════════
        elif "math" in sub_lower:
            if has("set", "relation", "function", "domain", "range", "one-one", "onto", "inverse function", "binary operation"):
                scope_lock_lines.append("  • MATHEMATICS — Sets, Relations & Functions chapter. Allowed: types of sets, set operations, Venn diagrams, types of relations (reflexive/symmetric/transitive/equivalence), types of functions (one-one/onto/bijective), inverse functions, composition. FORBIDDEN: trigonometry, calculus, algebra in isolation.")
            elif has("trigonometric function", "trigonometry", "trigo", "sin", "cos", "tan", "trigonometric ratio", "trigonometric identit", "trigonometric equation", "general solution", "height and distance"):
                scope_lock_lines.append("  • MATHEMATICS — Trigonometric Functions chapter. Allowed: trigonometric ratios, identities, sum/difference formulae, double/half angle, trigonometric equations, general solutions, heights & distances. FORBIDDEN: inverse trigonometry, calculus, algebra.")
            elif has("inverse trigonometric", "inverse trigo", "principal value", "sin inverse", "cos inverse", "tan inverse"):
                scope_lock_lines.append("  • MATHEMATICS — Inverse Trigonometric Functions chapter. Allowed: definitions, principal value branches, properties, identities involving sin⁻¹/cos⁻¹/tan⁻¹, simplification. FORBIDDEN: ordinary trigonometry, calculus.")
            elif has("complex number", "iota", "argand", "modulus", "argument", "polar form", "de moivre", "cube root of unity"):
                scope_lock_lines.append("  • MATHEMATICS — Complex Numbers chapter. Allowed: i (iota), Argand plane, modulus, argument, polar form, conjugate, De Moivre's theorem, cube roots of unity, equations involving complex numbers. FORBIDDEN: real algebra, geometry, calculus.")
            elif has("quadratic", "discriminant", "roots of quadratic", "nature of roots", "vieta", "sum of roots", "product of roots"):
                scope_lock_lines.append("  • MATHEMATICS — Quadratic Equations chapter. Allowed: discriminant, nature of roots, sum/product of roots (Vieta), forming quadratic from roots, range of quadratic expression. FORBIDDEN: complex numbers in isolation, sequences, calculus.")
            elif has("sequence", "series", "ap", "gp", "hp", "arithmetic progression", "geometric progression", "harmonic progression", "arithmetic mean", "geometric mean"):
                scope_lock_lines.append("  • MATHEMATICS — Sequences & Series chapter. Allowed: AP/GP/HP definitions, nth term, sum to n terms, AM/GM/HM inequalities, special series (Σn, Σn², Σn³). FORBIDDEN: complex, calculus, algebra.")
            elif has("permutation", "combination", "factorial", "ncr", "npr", "circular permutation"):
                scope_lock_lines.append("  • MATHEMATICS — Permutations & Combinations chapter. Allowed: fundamental principle of counting, factorial, nPr, nCr, circular permutations, distributions, combinations with restrictions. FORBIDDEN: probability in isolation, binomial theorem.")
            elif has("binomial theorem", "binomial expansion", "general term", "middle term", "binomial coefficient"):
                scope_lock_lines.append("  • MATHEMATICS — Binomial Theorem chapter. Allowed: binomial expansion for positive integral index, general/middle term, binomial coefficients, properties, applications. FORBIDDEN: P&C in isolation, calculus.")
            elif has("matrix", "matrices", "matrix multiplication", "transpose"):
                scope_lock_lines.append("  • MATHEMATICS — Matrices chapter. Allowed: types of matrices, addition/multiplication, transpose, symmetric/skew-symmetric, elementary operations, inverse using row operations. FORBIDDEN: determinants in isolation, system of equations using Cramer.")
            elif has("determinant", "cofactor", "adjoint", "cramer", "minor"):
                scope_lock_lines.append("  • MATHEMATICS — Determinants chapter. Allowed: properties of determinants, minors and cofactors, adjoint and inverse of matrix, Cramer's rule, system of linear equations consistency. FORBIDDEN: matrix operations in isolation.")
            elif has("limit", "continuity", "differentiabil", "left hand limit", "right hand limit", "l hopital", "lhopital"):
                scope_lock_lines.append("  • MATHEMATICS — Limits, Continuity & Differentiability chapter. Allowed: limit definitions, standard limits, L'Hôpital, continuity at a point/interval, differentiability, types of discontinuities. FORBIDDEN: differentiation rules in isolation, integration.")
            elif has("differentiation", "derivative", "chain rule", "implicit", "logarithmic differentiation", "parametric"):
                scope_lock_lines.append("  • MATHEMATICS — Differentiation chapter. Allowed: derivatives of standard functions, chain/product/quotient rule, implicit/parametric/logarithmic differentiation, higher order derivatives. FORBIDDEN: limits in isolation, integration, application of derivatives.")
            elif has("application of derivative", "rate of change", "tangent", "normal", "monotonic", "increasing", "decreasing", "maxima", "minima", "rolle", "mean value theorem", "approximation"):
                scope_lock_lines.append("  • MATHEMATICS — Application of Derivatives chapter. Allowed: rate of change, tangent/normal, monotonicity, maxima/minima (1st/2nd derivative test), Rolle's & Mean Value theorem, approximations. FORBIDDEN: pure differentiation, integration.")
            elif has("indefinite integral", "integration by part", "integration by substitut", "partial fraction", "integral formula"):
                scope_lock_lines.append("  • MATHEMATICS — Indefinite Integrals chapter. Allowed: integration as anti-derivative, methods (substitution, by parts, partial fractions), standard integrals, special integrals. FORBIDDEN: definite integrals in isolation, application of integrals.")
            elif has("definite integral", "limit of sum", "fundamental theorem of calculus", "property of definite integ"):
                scope_lock_lines.append("  • MATHEMATICS — Definite Integrals chapter. Allowed: definition as limit of a sum, fundamental theorem, properties of definite integrals, evaluation. FORBIDDEN: indefinite integrals in isolation, area under curves.")
            elif has("area under curve", "area between curve", "application of integral"):
                scope_lock_lines.append("  • MATHEMATICS — Application of Integrals chapter. Allowed: area under simple curves, area bounded by two curves. FORBIDDEN: pure integration, differential equations.")
            elif has("differential equation", "order of differential", "degree of differential", "homogeneous differential", "linear differential", "variable separable"):
                scope_lock_lines.append("  • MATHEMATICS — Differential Equations chapter. Allowed: order/degree, formation, solving (variable separable, homogeneous, linear DE with integrating factor). FORBIDDEN: integration in isolation, application of derivatives.")
            elif has("straight line", "slope of line", "equation of line", "distance between line", "angle between line", "perpendicular distance"):
                scope_lock_lines.append("  • MATHEMATICS — Straight Lines chapter. Allowed: various forms of line equation, slope, angle between lines, perpendicular distance, distance between parallel lines, family of lines. FORBIDDEN: circles, conic sections, 3D geometry.")
            elif has("circle", "tangent to circle", "chord of circle", "system of circle", "common chord"):
                scope_lock_lines.append("  • MATHEMATICS — Circles chapter. Allowed: equation of circle (general/standard), tangent and normal, chord of contact, system of circles, family of circles. FORBIDDEN: straight lines, parabola/ellipse/hyperbola.")
            elif has("conic section", "parabola", "ellipse", "hyperbola", "directrix", "eccentricity", "asymptote of hyperbola"):
                scope_lock_lines.append("  • MATHEMATICS — Conic Sections chapter. Allowed: parabola/ellipse/hyperbola standard forms, focus, directrix, eccentricity, latus rectum, tangent and normal. FORBIDDEN: straight lines, circles, 3D geometry.")
            elif has("vector", "scalar product", "dot product", "cross product", "vector triple", "scalar triple"):
                scope_lock_lines.append("  • MATHEMATICS — Vector Algebra chapter. Allowed: vector addition, dot/cross product, scalar/vector triple product, projection, equation of line and plane in vector form. FORBIDDEN: 3D coordinate geometry in isolation, calculus.")
            elif has("3d geometry", "three-dimensional", "direction cosine", "direction ratio", "equation of plane", "skew lines"):
                scope_lock_lines.append("  • MATHEMATICS — 3D Geometry chapter. Allowed: direction cosines/ratios, equation of line/plane in space, distance between two lines, angle between line & plane, distance from a point to plane. FORBIDDEN: vector algebra in isolation, conic sections.")
            elif has("probability", "bayes", "conditional probability", "binomial distribution", "random variable", "expected value", "independent event"):
                scope_lock_lines.append("  • MATHEMATICS — Probability chapter. Allowed: events, conditional probability, multiplication theorem, total probability, Bayes' theorem, random variables, binomial distribution. FORBIDDEN: statistics in isolation, P&C in detail.")
            elif has("statistic", "mean deviation", "standard deviation", "variance", "median", "mode"):
                scope_lock_lines.append("  • MATHEMATICS — Statistics chapter. Allowed: measures of dispersion (mean deviation, variance, standard deviation), analysis of frequency distribution. FORBIDDEN: probability in isolation, P&C.")
            elif has("linear programming", "lpp", "feasible region", "objective function", "constraint"):
                scope_lock_lines.append("  • MATHEMATICS — Linear Programming chapter. Allowed: LPP definition, mathematical formulation, graphical method (feasible region, objective function, optimisation). FORBIDDEN: pure algebra, calculus.")
            elif has("mathematical reasoning", "statement", "tautology", "contradiction", "negation", "implication"):
                scope_lock_lines.append("  • MATHEMATICS — Mathematical Reasoning chapter. Allowed: statements, logical connectives, negation, implications, tautology, contradiction, validity of arguments. FORBIDDEN: any other math topic.")

        # ═══ BIOLOGY ═════════════════════════════════════════════════════════
        elif "bio" in sub_lower or "neet" in (exam_target or "").lower():
            if has("living world", "diversity in living", "taxonomy", "binomial nomenclature", "kingdom"):
                scope_lock_lines.append("  • BIOLOGY — Diversity of Living Organisms chapter. Allowed: taxonomy, binomial nomenclature, taxonomic categories/hierarchy, five-kingdom classification (Monera/Protista/Fungi/Plantae/Animalia), viruses/viroids/lichens. FORBIDDEN: cell biology, physiology, genetics.")
            elif has("plant kingdom", "cryptogam", "phanerogam", "thallophyta", "bryophyta", "pteridophyta", "gymnosperm", "angiosperm", "algae"):
                scope_lock_lines.append("  • BIOLOGY — Plant Kingdom chapter. Allowed: classification of plants (algae/bryophytes/pteridophytes/gymnosperms/angiosperms), life cycles, alternation of generations. FORBIDDEN: animal kingdom, plant physiology in detail, anatomy.")
            elif has("animal kingdom", "porifera", "coelenterate", "platyhelminthes", "annelida", "arthropoda", "mollusca", "echinodermata", "chordata", "phylum"):
                scope_lock_lines.append("  • BIOLOGY — Animal Kingdom chapter. Allowed: classification of animals (Porifera through Chordata), characteristic features, examples of each phylum. FORBIDDEN: plant kingdom, physiology, structural organization.")
            elif has("morphology of flowering", "root", "stem", "leaf", "inflorescence", "flower", "fruit", "seed", "modification of root", "modification of stem"):
                scope_lock_lines.append("  • BIOLOGY — Morphology of Flowering Plants chapter. Allowed: root systems, stem (modifications), leaf (types, venation, phyllotaxy), inflorescence, flower (parts, aestivation, floral formula), fruit, seed types. FORBIDDEN: anatomy (internal), physiology, animal anatomy.")
            elif has("anatomy of flowering", "tissue system", "epidermis of plant", "vascular bundle", "secondary growth"):
                scope_lock_lines.append("  • BIOLOGY — Anatomy of Flowering Plants chapter. Allowed: plant tissues (meristematic/permanent), tissue systems (epidermal/ground/vascular), anatomy of dicot/monocot root/stem/leaf, secondary growth. FORBIDDEN: morphology, animal histology, physiology.")
            elif has("structural organisation in animal", "animal tissue", "epithelial tissue", "connective tissue", "muscular tissue", "nervous tissue", "frog anatomy", "cockroach anatomy"):
                scope_lock_lines.append("  • BIOLOGY — Structural Organisation in Animals chapter. Allowed: animal tissues (epithelial/connective/muscular/nervous), morphology and anatomy of frog/earthworm/cockroach. FORBIDDEN: animal kingdom classification, human physiology, plant tissues.")
            elif has("cell unit of life", "cell theory", "prokaryotic cell", "eukaryotic cell", "cell organelle", "nucleus", "ribosome", "mitochondria", "chloroplast", "endoplasmic", "golgi", "lysosome", "cytoskeleton", "centrosome"):
                scope_lock_lines.append("  • BIOLOGY — Cell: The Unit of Life chapter. Allowed: cell theory, prokaryotic vs eukaryotic, cell organelles (nucleus/ER/Golgi/ribosomes/mitochondria/chloroplasts/lysosomes/peroxisomes), cytoskeleton, cell envelope. FORBIDDEN: biomolecules, cell cycle, genetics.")
            elif has("biomolecule", "amino acid", "protein", "carbohydrate", "lipid", "nucleic acid in cell", "enzyme", "metabolism overview"):
                scope_lock_lines.append("  • BIOLOGY — Biomolecules chapter. Allowed: biomolecules in living systems (amino acids, proteins, carbohydrates, lipids, nucleic acids), enzymes (cofactors, classification, kinetics), metabolism overview. FORBIDDEN: cell organelles, genetics, physiology.")
            elif has("cell cycle", "cell division", "mitosis", "meiosis", "interphase", "prophase", "metaphase", "anaphase", "telophase"):
                scope_lock_lines.append("  • BIOLOGY — Cell Cycle and Cell Division chapter. Allowed: phases of cell cycle (G1/S/G2/M), mitosis stages, meiosis stages, significance of mitosis vs meiosis. FORBIDDEN: genetics, molecular biology, physiology.")
            elif has("transport in plant", "diffusion", "osmosis", "active transport in plant", "ascent of sap", "transpiration", "stomata"):
                scope_lock_lines.append("  • BIOLOGY — Transport in Plants chapter. Allowed: diffusion/facilitated/active transport, osmosis, water potential, plasmolysis, ascent of sap, transpiration, stomatal regulation, phloem transport. FORBIDDEN: mineral nutrition, photosynthesis, animal physiology.")
            elif has("mineral nutrition", "essential mineral", "macronutrient", "micronutrient", "deficiency symptom", "nitrogen cycle in plant", "biological nitrogen fixation"):
                scope_lock_lines.append("  • BIOLOGY — Mineral Nutrition chapter. Allowed: essential nutrients (macro/micro), criteria of essentiality, deficiency symptoms, nitrogen cycle, biological N₂ fixation, mechanism of absorption. FORBIDDEN: photosynthesis, plant respiration, animal nutrition.")
            elif has("photosynthesis", "calvin cycle", "light reaction", "dark reaction", "c3 plant", "c4 plant", "kranz anatomy", "photorespirat", "chlorophyll"):
                scope_lock_lines.append("  • BIOLOGY — Photosynthesis chapter. Allowed: site of photosynthesis, pigments, light reactions (PSI/PSII), Calvin cycle (C3), Hatch-Slack pathway (C4), photorespiration, factors affecting photosynthesis. FORBIDDEN: respiration, plant transport, animal physiology.")
            elif has("respiration in plant", "glycolysis", "krebs", "tca", "electron transport chain", "oxidative phosphorylation", "fermentation", "respiratory quotient"):
                scope_lock_lines.append("  • BIOLOGY — Respiration in Plants chapter. Allowed: glycolysis, fermentation, Krebs cycle, ETC, oxidative phosphorylation, respiratory quotient, amphibolic pathway. FORBIDDEN: photosynthesis, breathing in animals, plant transport.")
            elif has("plant growth", "plant development", "phytohormone", "auxin", "gibberellin", "cytokinin", "ethylene", "abscisic", "photoperiod", "vernal"):
                scope_lock_lines.append("  • BIOLOGY — Plant Growth & Development chapter. Allowed: growth phases, plant hormones (auxin/gibberellin/cytokinin/ethylene/ABA), photoperiodism, vernalization, seed dormancy. FORBIDDEN: photosynthesis, mineral nutrition, animal hormones.")
            elif has("digest", "alimentary canal", "absorption of food", "liver function", "pancreas function", "digestive enzyme"):
                scope_lock_lines.append("  • BIOLOGY — Digestion & Absorption chapter. Allowed: human digestive system, alimentary canal, digestive glands, digestion of carbohydrates/proteins/fats, absorption mechanisms, disorders. FORBIDDEN: respiration, circulation, other systems.")
            elif has("breathing", "respiration in human", "respiratory system", "lung", "alveoli", "haldane", "bohr effect", "transport of o2", "transport of co2"):
                scope_lock_lines.append("  • BIOLOGY — Breathing & Exchange of Gases chapter. Allowed: respiratory organs, mechanism of breathing, gas exchange (alveoli), O₂ and CO₂ transport, Bohr effect, regulation of respiration, disorders. FORBIDDEN: digestion, circulation, other systems.")
            elif has("body fluid", "circulation", "blood", "heart", "cardiac cycle", "ecg", "blood vessel", "lymph", "blood group"):
                scope_lock_lines.append("  • BIOLOGY — Body Fluids & Circulation chapter. Allowed: blood composition, blood groups (ABO/Rh), coagulation, lymph, human circulatory system, cardiac cycle, ECG, double circulation, disorders. FORBIDDEN: respiration, excretion, other systems.")
            elif has("excret", "kidney", "nephron", "urine formation", "osmoregulation", "dialysis", "renal"):
                scope_lock_lines.append("  • BIOLOGY — Excretory Products & Their Elimination chapter. Allowed: modes of excretion, human excretory system, nephron structure, urine formation, regulation of kidney function, dialysis, kidney disorders. FORBIDDEN: digestion, circulation, other systems.")
            elif has("locomotion", "movement", "muscle contraction", "skeletal system", "joint type", "sliding filament"):
                scope_lock_lines.append("  • BIOLOGY — Locomotion & Movement chapter. Allowed: types of movement, muscle types, sliding filament theory of muscle contraction, skeletal system, joints, disorders (arthritis/osteoporosis). FORBIDDEN: nervous control, other systems.")
            elif has("neural control", "nervous system", "neuron", "synapse", "reflex arc", "brain", "spinal cord", "cranial nerve", "central nervous"):
                scope_lock_lines.append("  • BIOLOGY — Neural Control & Coordination chapter. Allowed: neuron structure, generation of nerve impulse, transmission across synapse, CNS (brain, spinal cord), PNS, reflex action, sensory reception. FORBIDDEN: chemical coordination, other systems.")
            elif has("chemical coordination", "endocrine", "hormone in human", "pituitary", "thyroid", "parathyroid", "pancreas hormone", "adrenal", "gonad", "hypothalamus"):
                scope_lock_lines.append("  • BIOLOGY — Chemical Coordination & Integration chapter. Allowed: endocrine glands (pituitary/thyroid/parathyroid/pancreas/adrenal/gonads), hormones, mechanism of hormone action, hormonal disorders. FORBIDDEN: neural control, plant hormones.")
            elif has("reproduction in organism", "asexual reproduction", "sexual reproduction overview", "vegetative propagation"):
                scope_lock_lines.append("  • BIOLOGY — Reproduction in Organisms chapter. Allowed: modes of reproduction (asexual, sexual), pre-fertilization/fertilization/post-fertilization events overview, life span. FORBIDDEN: specific human or plant reproduction in detail.")
            elif has("sexual reproduction in flowering plant", "microsporogenesis", "megasporogenesis", "double fertilization", "endosperm", "polyembryony", "apomixis"):
                scope_lock_lines.append("  • BIOLOGY — Sexual Reproduction in Flowering Plants chapter. Allowed: pre-fertilization (microsporogenesis/megasporogenesis), pollination types/agents, double fertilization, endosperm/embryo development, apomixis, polyembryony. FORBIDDEN: human reproduction, animal reproduction.")
            elif has("human reproduction", "spermatogenesis", "oogenesis", "menstrual cycle", "fertilization in human", "implantation", "pregnancy", "parturition", "lactation"):
                scope_lock_lines.append("  • BIOLOGY — Human Reproduction chapter. Allowed: male/female reproductive systems, gametogenesis, menstrual cycle, fertilization, pregnancy, parturition, lactation. FORBIDDEN: plant reproduction, reproductive health.")
            elif has("reproductive health", "contracept", "amniocentesis", "infertility", "std", "sexually transmitted"):
                scope_lock_lines.append("  • BIOLOGY — Reproductive Health chapter. Allowed: reproductive health, population stabilization, contraception, MTP, STDs, infertility, ART (IVF/ZIFT/GIFT). FORBIDDEN: pure reproduction biology, genetics.")
            elif has("inheritance", "principles of inheritance", "mendel", "law of segregation", "independent assortment", "linkage", "crossing over", "sex determination", "pedigree", "monohybrid", "dihybrid"):
                scope_lock_lines.append("  • BIOLOGY — Principles of Inheritance & Variation chapter. Allowed: Mendel's laws, monohybrid/dihybrid cross, incomplete dominance, codominance, polygenic inheritance, linkage, sex determination, pedigree analysis, mutations, genetic disorders. FORBIDDEN: molecular biology, evolution.")
            elif has("molecular basis of inheritance", "dna structure", "dna replication", "transcription", "translation", "genetic code", "gene regulation", "operon", "human genome"):
                scope_lock_lines.append("  • BIOLOGY — Molecular Basis of Inheritance chapter. Allowed: DNA as genetic material, structure, replication, transcription, genetic code, translation, regulation of gene expression (lac operon), Human Genome Project, DNA fingerprinting. FORBIDDEN: classical genetics, evolution.")
            elif has("evolution", "darwin", "natural selection", "lamarck", "hardy weinberg", "speciation", "adaptive radiation", "fossil"):
                scope_lock_lines.append("  • BIOLOGY — Evolution chapter. Allowed: theories of evolution (Darwin, Lamarck), evidences, Hardy-Weinberg principle, natural selection, adaptive radiation, human evolution, origin of life. FORBIDDEN: genetics, ecology.")
            elif has("health and disease", "infectious disease", "vaccination", "immunity", "innate immunity", "acquired immunity", "antigen", "antibody", "aids", "cancer"):
                scope_lock_lines.append("  • BIOLOGY — Human Health & Disease chapter. Allowed: pathogens & diseases (malaria, typhoid, pneumonia, etc.), immunity (innate/acquired), vaccination, AIDS, cancer, drug abuse. FORBIDDEN: microbes in welfare, biotechnology.")
            elif has("food production", "improvement in food", "plant breeding", "tissue culture", "single cell protein", "animal husbandry"):
                scope_lock_lines.append("  • BIOLOGY — Strategies for Enhancement in Food Production chapter. Allowed: animal husbandry, plant breeding for disease resistance, single cell protein, tissue culture. FORBIDDEN: biotechnology applications, microbes.")
            elif has("microbes in human welfare", "industrial product", "household product", "biofertilizer", "biocontrol", "sewage treatment"):
                scope_lock_lines.append("  • BIOLOGY — Microbes in Human Welfare chapter. Allowed: microbes in household products, industrial products, sewage treatment, biogas production, biocontrol agents, biofertilizers. FORBIDDEN: human diseases, biotechnology.")
            elif has("biotechnology principle", "recombinant dna", "restriction enzyme", "cloning vector", "pcr", "transgenic"):
                scope_lock_lines.append("  • BIOLOGY — Biotechnology: Principles & Processes chapter. Allowed: principles of biotech, tools (restriction enzymes, vectors, host organism), processes (rDNA technology, PCR, gel electrophoresis, transformation). FORBIDDEN: biotechnology applications, molecular biology.")
            elif has("biotechnology and its application", "transgenic plant", "transgenic animal", "gene therapy", "molecular diagnos", "biopirac"):
                scope_lock_lines.append("  • BIOLOGY — Biotechnology & Its Applications chapter. Allowed: applications in agriculture (Bt crops, RNAi), medicine (insulin, gene therapy, molecular diagnosis), transgenic animals, biosafety, biopiracy, GEAC. FORBIDDEN: biotech principles in isolation.")
            elif has("organism and population", "habitat and niche", "population attribute", "population growth", "population interaction", "abiotic factor"):
                scope_lock_lines.append("  • BIOLOGY — Organisms & Populations chapter. Allowed: organism and environment, major abiotic factors, population attributes (density/natality/mortality), population growth (exponential/logistic), interactions (mutualism/competition/predation/parasitism). FORBIDDEN: ecosystem, biodiversity.")
            elif has("ecosystem", "energy flow", "trophic level", "food chain", "food web", "ecological pyramid", "biogeochemical cycle", "primary productivity", "decomposer"):
                scope_lock_lines.append("  • BIOLOGY — Ecosystem chapter. Allowed: ecosystem structure & function, productivity, decomposition, energy flow, ecological pyramids, nutrient cycles (carbon/phosphorus). FORBIDDEN: organisms & populations, biodiversity.")
            elif has("biodiversity", "biodiversity hot spot", "extinct", "endemic", "in situ", "ex situ", "biosphere reserve", "national park"):
                scope_lock_lines.append("  • BIOLOGY — Biodiversity & Conservation chapter. Allowed: biodiversity (genetic/species/ecosystem), patterns, loss of biodiversity, conservation strategies (in-situ, ex-situ), hot spots, endangered species, IUCN red list. FORBIDDEN: ecosystem, environmental issues.")
            elif has("environmental issue", "air pollution", "water pollution", "noise pollution", "soil pollution", "global warming", "ozone depletion", "deforestation"):
                scope_lock_lines.append("  • BIOLOGY — Environmental Issues chapter. Allowed: air/water/soil/noise pollution, global warming, ozone depletion, deforestation, e-waste, case studies (Chipko, Bhopal). FORBIDDEN: biodiversity, ecosystem in detail.")

        scope_lock_lines.append("")
        chapter_label = (chapter or topic or subject).strip()[:60]
        scope_lock_lines.append(
            f"  • MANDATORY: each question's JSON MUST include a field 'scope_check' = '{chapter_label}' "
            "(exact chapter name). The post-processor will REJECT any question whose scope_check is missing or doesn't match."
        )
        scope_lock_lines.append(
            f"  • Before generating each question, ask yourself one final time: 'Is this question DIRECTLY from chapter \"{chapter_label}\"?' "
            "If you cannot answer YES with absolute confidence, REWRITE it on a topic that IS in this chapter."
        )
        scope_constraint = "\n".join(scope_lock_lines) + "\n\n"

    subject_rule = _SUBJECT_RULES_TEST.get(subject.lower(), "") if subject else ""

    exam_banner = ""
    if exam_target:
        et = exam_target.lower()
        if "advance" in et:
            exam_banner = (
                "🎯 TARGET EXAM: **JEE ADVANCED** (IIT entrance — toughest engineering exam in India).\n"
                "  HARD TIER means: the question requires combining 2+ concepts AND 4+ reasoning/calculation steps,\n"
                "  has a non-obvious trap or twist, and a top-1% student should need 4+ minutes to solve.\n"
                "  Examples of acceptable HARD JEE Advanced questions:\n"
                "    • A 4-step kinematics-energy-rotation combined problem with a non-trivial constraint\n"
                "    • A reaction mechanism asking the major product after 3 sequential transformations\n"
                "    • A definite integral that requires substitution + partial fractions + reduction formula\n"
                "  REJECT and rewrite if your draft question:\n"
                "    ✗ Could appear in a Class 9/10 NCERT exercise\n"
                "    ✗ Asks for a single-formula plug-in (e.g. 'apply v=u+at')\n"
                "    ✗ Tests only definition recall (e.g. 'What is colligative property?')\n"
                "    ✗ Has a single-step calculation\n\n"
            )
        elif "main" in et:
            exam_banner = (
                "🎯 TARGET EXAM: **JEE MAIN** (NTA, undergraduate engineering entrance).\n"
                "  HARD TIER means: 3-4 reasoning/calculation steps, requires applying a concept to a non-standard situation,\n"
                "  and an above-average JEE-prep student should need 2-3 minutes to solve. NOT board/CBSE/school level.\n"
                "  Examples of acceptable HARD JEE Main questions:\n"
                "    • A 3-step physics calculation requiring identification + formula application + algebra\n"
                "    • A chemistry question requiring two-step product prediction with stereochemistry hint\n"
                "    • A maths problem combining 2 standard techniques (e.g. integration by substitution + parts)\n"
                "  REJECT and rewrite if your draft question:\n"
                "    ✗ Is a direct NCERT line/formula recall\n"
                "    ✗ Has only 1 step or 1 concept\n"
                "    ✗ Could be solved in under 30 seconds by a Class 11 student\n"
                "    ✗ Asks 'state the law' or 'define X' (board-style)\n\n"
            )
        elif "neet" in et:
            exam_banner = (
                "🎯 TARGET EXAM: **NEET-UG** (NTA medical entrance).\n"
                "  HARD TIER means: NCERT-deep statement-based or assertion-reason testing edge cases, exceptions,\n"
                "  or composite facts spanning multiple paragraphs of NCERT. NOT board-style recall.\n"
                "  Examples of acceptable HARD NEET questions:\n"
                "    • Statement-based: 4 statements where 2 are subtly wrong NCERT facts\n"
                "    • Assertion-reason where R is correct but does not actually explain A\n"
                "    • A multi-fact integration (e.g. one organism, multiple kingdoms-level traits)\n"
                "  REJECT and rewrite if your draft question:\n"
                "    ✗ Asks 'define X' or 'what is Y' (board-style)\n"
                "    ✗ Has all options obviously different (no NCERT-trap)\n"
                "    ✗ Could be solved by reading any one NCERT line\n\n"
            )

    # ── VARIETY RULE — prevents the model from clustering on the easiest sub-topic ──
    # When chapter is given: rotate across sub-topics within that chapter.
    # When ONLY subject is given (subject test): rotate across MULTIPLE chapters of the subject.
    variety_rule = ""
    if num_questions >= 3:
        is_subject_only = bool(subject) and not (chapter and chapter.strip())
        if is_subject_only:
            if db_chapters:
                # Constrain to ONLY the chapters that exist in the institution's DB for this course
                max_per_chap = max(1, -(-num_questions // max(len(db_chapters), 1)))  # ceiling division
                chapter_list_str = ", ".join(f'"{c}"' for c in db_chapters)
                variety_rule = (
                    "🎨 VARIETY REQUIREMENT — STRICT CHAPTER CONSTRAINT (CRITICAL):\n"
                    f"  This course has ONLY these {len(db_chapters)} chapter(s) for {subject}: {chapter_list_str}.\n"
                    f"  YOU MUST ONLY generate questions from THESE chapters. DO NOT use any other chapter.\n"
                    f"  • Distribute {num_questions} questions evenly — max {max_per_chap} per chapter.\n"
                    "  • In each question's JSON, 'chapter' = exactly one of the chapter names listed above (copy it exactly).\n"
                    "  • 'subtopic' = a 2-4 word sub-topic label within that chapter.\n"
                    f"  • FORBIDDEN: any chapter NOT in this list: {chapter_list_str}.\n\n"
                )
            else:
                variety_rule = (
                    "🎨 VARIETY REQUIREMENT — SUBJECT-WIDE TEST (CRITICAL):\n"
                    f"  This is a {subject} subject test — questions MUST span MULTIPLE chapters of {subject}, NOT cluster on one.\n"
                    f"  • Distribute the {num_questions} questions across at least min({num_questions}, 8) different chapters.\n"
                    "  • DO NOT generate more than 2-3 questions from any single chapter.\n"
                    "  • Mix Physical / Inorganic / Organic for Chemistry; Mechanics / E&M / Optics / Modern for Physics; etc.\n"
                    "  • In each question's JSON, include a field 'chapter' = the chapter name this question is from "
                    "(e.g. 'Solutions', 'Coordination Compounds', 'Aldehydes and Ketones'), and 'subtopic' = a 2-4 word label.\n"
                    "  • Before finalizing, count: if you have 3+ questions from the same chapter, replace one with a different chapter.\n\n"
                )
        else:
            variety_rule = (
                "🎨 VARIETY REQUIREMENT (CRITICAL):\n"
                f"  You MUST generate {num_questions} questions on DIFFERENT sub-topics within the chapter.\n"
                "  • Maximum 2 questions per sub-topic across the whole batch (cluster-of-3 is BANNED).\n"
                "  • Rotate across ALL the major sub-topics listed in the SCOPE LOCK above; do NOT pick the easiest one and stay there.\n"
                "  • If the chapter has, say, 6 named sub-topics, your questions must cover at least min(num_questions, 5) distinct ones.\n"
                "  • Vary the question STYLE too: mix numerical / conceptual / application / trap-style.\n"
                "  • Vary the OPENING phrasing — do not start every question with 'Assertion (A): The...' or 'What is the...'.\n"
                "  • Before finalizing the JSON, scan your N questions: if any two test the same fact/formula/sub-topic, "
                "delete one and write a new one on a different sub-topic.\n"
                "  • In each question's JSON, include a field 'subtopic' = a 2-4 word label of the specific sub-topic "
                "(e.g. 'Raoult's law', 'osmotic pressure', 'depression of FP', 'azeotrope', 'molarity calculation').\n\n"
            )

    formatting_rule = (
        "FORMATTING RULES — STRICT (READ CAREFULLY):\n"
        "  • DO NOT use LaTeX delimiters anywhere — no $...$, no $$...$$, no \\( \\), no \\[ \\].\n"
        "  • DO NOT use LaTeX commands like \\text{}, \\frac{}{}, \\sqrt{}, \\;, \\,, \\quad.\n"
        "  • Write all math in plain text using Unicode symbols:\n"
        "      multiplication = ×    division = ÷    plus/minus = ±    degrees = °\n"
        "      square root = √(x)    fraction = (a/b)    powers = x²,x³,xⁿ    subscripts = H₂O, CO₂\n"
        "      Greek letters as Unicode: π, θ, α, β, λ, μ, Ω, Δ\n"
        "      arrows: → ← ⇌ (chemistry equilibrium)    relations: ≤ ≥ ≠ ≈\n"
        "  • Examples of CORRECT formatting:\n"
        "      'What is the volume of 0.1 mol of an ideal gas at STP?'  ✓\n"
        "      '2H₂ + O₂ → 2H₂O'  ✓\n"
        "      'v = u + at where a = 9.8 m/s²'  ✓\n"
        "  • Examples of WRONG formatting (DO NOT DO THIS):\n"
        "      'What is the volume of $0.1 \\text{mol}$ of...'  ✗  (uses $ and \\text)\n"
        "      '$v = u + at$'  ✗  (uses $)\n"
        "      'H_2O' or 'H^2'  ✗  (use H₂O / H²)\n"
        "  • Units always go in plain text after the number with one space: '22.4 L', '9.8 m/s²', '273 K'.\n\n"
    )

    user_prompt = (
        f"{scope_constraint}"             # SCOPE LOCK comes FIRST so it dominates the model's attention
        f"{exam_banner}"
        f"{variety_rule}"                 # VARIETY enforcement to prevent sub-topic clustering
        f"{formatting_rule}"
        f"Generate {num_questions} distinct {qtype} questions.\n"
        f"Curriculum scope: {curriculum_context}\n"
        f"Difficulty: {difficulty}. This should be a {diff_context}.\n"
        f"{type_instr}\n\n"
        f"{subject_rule}\n\n"
        f"{heuristic_block}\n"
        f"{notes_context}\n"
        "Return the result as a JSON object with a 'questions' array. "
        "Each question object MUST have: 'question', 'answer', 'explanation', "
        f"'scope_check' (echo back '{(chapter or topic or subject or '').strip()[:60]}' to confirm scope), "
        "'subtopic' (2-4 word label of the specific sub-topic this question tests), "
        + (
            f"AND 'chapter' = '{chapter}' (fixed — every question in this batch is from this chapter). "
            if chapter else
            f"AND 'chapter' (the SPECIFIC chapter title this question is from — e.g. 'Units and Measurement', "
            f"'Thermal Physics', 'Electrochemistry' — must be a real chapter from {subject or 'the subject'} syllabus, "
            f"NOT the subject name '{subject or ''}' itself, and NOT a vague label like 'General'). "
        )
        + "Include an 'options' field only for MCQs."
    )

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("test_generate")

    logger.info(
        "generate_practice_test | exam=%s | diff=%s | subject=%s | chapter=%s | topic=%s | type=%s | style=%s | n=%d",
        exam_target or "—", difficulty, subject or "—", chapter or "—", topic[:50], qtype, style or "—", num_questions
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