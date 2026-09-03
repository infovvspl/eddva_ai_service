import json
import logging
import time

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.pii import restore_name, to_prompt_name
from ai_services.core.prompt_templates import get_template
from ai_services.core.usage_logger import log_usage
from .base import ai_call, ai_call_text, get_llm

logger = logging.getLogger("ai_services.career")


def interpret_holland_code(code: str) -> str:
    descriptions = {
        'R': 'Realistic — practical, hands-on, technical',
        'I': 'Investigative — analytical, curious, research-oriented',
        'A': 'Artistic — creative, expressive, imaginative',
        'S': 'Social — helpful, caring, people-oriented',
        'E': 'Enterprising — leadership, ambitious, persuasive',
        'C': 'Conventional — organised, detail-oriented, systematic',
    }
    result = []
    for letter in str(code or '').upper():
        if letter in descriptions:
            result.append(descriptions[letter])
    return ' | '.join(result)


_TOP_CAREER_REQUIRED_FIELDS = (
    'title', 'reasoning', 'exams', 'topColleges', 'salaryRange',
    'educationPath', 'keySkills', 'jobRoles',
)


def _validate_career_report(report: dict) -> list:
    """
    Checks a parsed career_guidance response for the failure modes the prompt
    explicitly asks the model to avoid but that json.loads() can't catch on its
    own: missing catalog fields (these get cached into school_career_paths
    verbatim — an empty one here recreates the blank-detail-page bug), fit
    scores that aren't actually differentiated, and boilerplate-length text.
    Returns a list of human-readable problems; empty means the report passed.
    """
    problems = []
    top_careers = report.get('topCareers')
    if not isinstance(top_careers, list) or len(top_careers) != 3:
        problems.append('topCareers must be a list of exactly 3 careers')
        top_careers = top_careers if isinstance(top_careers, list) else []

    for i, career in enumerate(top_careers):
        if not isinstance(career, dict):
            problems.append(f'topCareers[{i}] is not an object')
            continue
        for field in _TOP_CAREER_REQUIRED_FIELDS:
            if career.get(field) in (None, '', []):
                problems.append(f'topCareers[{i}].{field} is empty')
        if len(str(career.get('reasoning') or '')) < 30:
            problems.append(f'topCareers[{i}].reasoning is too short to be specific to this student')

    fit_scores = [c.get('fitScore') for c in top_careers if isinstance(c, dict)]
    if fit_scores and all(isinstance(f, (int, float)) and f >= 90 for f in fit_scores):
        problems.append('all fitScores are 90+ — not realistic/differentiated')

    if len(str(report.get('overallAnalysis') or '')) < 40:
        problems.append('overallAnalysis is too short/generic')
    if len(str(report.get('encouragement') or '')) < 20:
        problems.append('encouragement is too short/generic')
    if not report.get('immediateActions'):
        problems.append('immediateActions is empty')

    return problems


def _parse_career_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    clean = str(raw).strip()
    if clean.startswith('```'):
        clean = clean.split('```')[1]
        if clean.lower().startswith('json'):
            clean = clean[4:]
    # Be lenient: extract the outermost JSON object if extra prose surrounds it.
    start, end = clean.find('{'), clean.rfind('}')
    if start != -1 and end > start:
        clean = clean[start:end + 1]
    return json.loads(clean.strip())


@api_view(["POST"])
def career_guidance(request):
    """AI #17 — School Career Guidance. Auth handled by global X-API-Key middleware."""
    _start_time = time.time()
    data = request.data or {}
    grade = data.get('grade', 10)
    board = data.get('board', 'CBSE')
    # The real name never reaches the provider. The prompt gets a sentinel and
    # the name is put back into the report before it is returned — the analysis
    # is driven by marks, attendance and interest scores, none of which need to
    # know who the student is. See core/pii.py.
    student_name = data.get('studentName', 'Student')
    prompt_name = to_prompt_name(student_name)
    subject_marks = data.get('subjectMarks', []) or []
    strong_subjects = data.get('strongSubjects', []) or []
    weak_subjects = data.get('weakSubjects', []) or []
    quiz_test_summary = data.get('quizTestSummary', 'No data available')
    attendance = data.get('attendancePercentage', 0)
    homework_rate = data.get('homeworkRate', 0)
    holland_code = data.get('hollandCode', 'IS')
    holland_scores = data.get('hollandScores', {}) or {}
    top_career_matches = data.get('topCareerMatches', []) or []
    institute_id = data.get('instituteId', '') or getattr(request, 'institute_id', '')
    # Falling back to studentName here wrote the student's name into the usage
    # log and forwarded it to the billing dashboard, which is the same exposure
    # this endpoint otherwise avoids. An absent id stays absent.
    user_id_cg = data.get('userId') or data.get('user_id') or ''

    marks_text = '\n'.join([
        f"  {s.get('subject')}: {s.get('percentage')}% ({s.get('grade')})"
        for s in subject_marks
    ]) or 'No marks data available'
    strong_text = ', '.join(strong_subjects) or 'None identified yet'
    weak_text = ', '.join(weak_subjects) or 'None identified'
    careers_text = '\n'.join([
        f"  {i + 1}. {c.get('title')} (fit score: {c.get('fitScore')})"
        for i, c in enumerate(top_career_matches)
    ]) or '  No pre-calculated matches'

    model_id = get_model_for_task('career_guidance')
    template = get_template('career_guidance')
    user_prompt = template.user_template.format(
        student_name=prompt_name,
        grade=grade,
        board=board,
        academic_year='2025-26',
        subject_marks=marks_text,
        strong_subjects=strong_text,
        weak_subjects=weak_text,
        quiz_test_summary=quiz_test_summary,
        attendance_percentage=attendance,
        homework_rate=homework_rate,
        holland_code=holland_code,
        holland_scores=json.dumps(holland_scores),
        holland_interpretation=interpret_holland_code(holland_code),
        top_career_matches=careers_text,
    )

    tokens_input_total = 0
    tokens_output_total = 0
    result = None
    report = None
    problems = []
    last_error = None

    # Up to 2 attempts. A malformed or incomplete first response (missing
    # catalog fields, all fitScores maxed out, boilerplate text) gets retried
    # once with the specific problems appended, so the model can fix them
    # in-request rather than the report either reaching the student half-empty
    # or a custom career getting permanently cached with blank detail fields
    # (see saveAiCareers). Cheaper and faster than failing the job outright and
    # waiting for Bull's job-level retry to redo the whole call from scratch.
    for attempt in range(2):
        prompt = user_prompt
        if problems:
            prompt += (
                "\n\nYour previous response had these problems — fix ALL of them:\n"
                + "\n".join(f"- {p}" for p in problems)
            )
        try:
            result = get_llm().complete(
                system_prompt=template.system,
                user_prompt=prompt,
                model=model_id,
                temperature=0.4,
                # 3 careers now each carry exams/colleges/salary/educationPath/
                # skills/roles/prosCons too (so a cached custom career has full
                # detail, not an empty stub) — 2048 was cutting the JSON off.
                max_tokens=3072,
                json_mode=False,
                institute_id=institute_id,
            )
        except Exception as exc:
            last_error = exc
            logger.error("career_guidance LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue

        tokens_input_total += result.get('tokens_input', 0) or 0
        tokens_output_total += result.get('tokens_output', 0) or 0

        try:
            report = _parse_career_json(result.get('content', '{}'))
        except Exception as exc:
            last_error = exc
            logger.error("career_guidance JSON parse failed (attempt %d): %s", attempt + 1, exc)
            report = None
            problems = ['response was not valid JSON']
            continue

        problems = _validate_career_report(report)
        if not problems:
            break
        logger.warning("career_guidance report incomplete (attempt %d): %s", attempt + 1, problems)

    if report is None:
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='career_guidance_report',
                feature_category='student',
                model_used=model_id,
                tokens_input=tokens_input_total,
                tokens_output=tokens_output_total,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(last_error)[:500] if last_error else 'unknown',
                user_id=user_id_cg,
            )
        except Exception:
            pass
        return Response({'error': str(last_error) if last_error else 'Failed to generate report'}, status=502)

    if problems:
        # Both attempts still had gaps — return it anyway (better than failing
        # a job that's already spent two LLM calls) but keep the signal visible
        # so a recurring pattern here is easy to spot in the logs.
        logger.warning("career_guidance returning report with unresolved issues after retry: %s", problems)

    # Put the student's name back wherever the model personalised the report.
    # No-op when the caller sent no real name.
    report = restore_name(report, student_name)

    try:
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='career_guidance_report',
            feature_category='student',
            model_used=result.get('model', model_id) if result else model_id,
            tokens_input=tokens_input_total,
            tokens_output=tokens_output_total,
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_cg,
        )
    except Exception:
        pass

    return Response({'report': report, 'latency_ms': int((time.time() - _start_time) * 1000)})


@api_view(["POST"])
def generate_career_plan(request):
    data = request.data
    goal = data.get("goal")
    if not goal:
        return Response({"error": "Missing goal"}, status=400)

    template = get_template("career_roadmap")
    user_prompt = template.user_template.format(
        goal=goal,
        interests=", ".join(data.get("interests", [])),
        current_skills=", ".join(data.get("current_skills", [])),
        timeline_months=data.get("timeline_months", 12),
    )

    return ai_call_text(request, "career_roadmap", user_prompt,
                        wrap_fn=lambda t: {"career_path": t, "roadmap": t, "skills": [], "timeline": []})


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "career",
        "model": get_model_for_task("career_guidance"),
    })
