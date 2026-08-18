import json
import logging
import time

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
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


@api_view(["POST"])
def career_guidance(request):
    """AI #17 — School Career Guidance. Auth handled by global X-API-Key middleware."""
    _start_time = time.time()
    data = request.data or {}
    grade = data.get('grade', 10)
    board = data.get('board', 'CBSE')
    student_name = data.get('studentName', 'Student')
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
    user_id_cg = data.get('userId') or data.get('user_id') or data.get('studentName') or ''

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

    template = get_template('career_guidance')
    user_prompt = template.user_template.format(
        student_name=student_name,
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

    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model='openai/gpt-oss-120b',
            temperature=0.4,
            max_tokens=2048,
            json_mode=False,
            institute_id=institute_id,
        )
    except Exception as exc:
        logger.error("career_guidance LLM failed: %s", exc)
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='career_guidance_report',
                feature_category='student',
                model_used='openai/gpt-oss-120b',
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(exc)[:500],
                user_id=user_id_cg,
            )
        except Exception:
            pass
        return Response({'error': str(exc)}, status=502)

    raw = result.get('content', '{}')
    try:
        if isinstance(raw, dict):
            report = raw
        else:
            clean = str(raw).strip()
            if clean.startswith('```'):
                clean = clean.split('```')[1]
                if clean.lower().startswith('json'):
                    clean = clean[4:]
            # Be lenient: extract the outermost JSON object if extra prose surrounds it.
            start, end = clean.find('{'), clean.rfind('}')
            if start != -1 and end > start:
                clean = clean[start:end + 1]
            report = json.loads(clean.strip())
    except Exception as exc:
        logger.error("career_guidance JSON parse failed: %s", exc)
        return Response({'error': 'Failed to parse AI response', 'raw': str(raw)[:500]}, status=500)

    try:
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='career_guidance_report',
            feature_category='student',
            model_used=result.get('model', 'openai/gpt-oss-120b'),
            tokens_input=result.get('tokens_input', 0),
            tokens_output=result.get('tokens_output', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_cg,
        )
    except Exception:
        pass

    return Response({'report': report, 'latency_ms': result.get('latency_ms', 0)})


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
        "service": "career_roadmap",
        "model": get_model_for_task("career_roadmap"),
    })
