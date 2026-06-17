import time

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.usage_logger import log_usage
from .base import ai_call, ai_call_text


@api_view(["POST"])
def generate_study_plan(request):
    _start_time = time.time()
    data = request.data
    student_id = data.get("student_id")
    if not student_id:
        return Response({"error": "Missing student_id"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    user_id_sp = data.get('userId') or data.get('user_id') or student_id or ''

    template = get_template("study_plan")
    user_prompt = template.user_template.format(
        student_id=student_id,
        learning_style=data.get("learning_style", "visual"),
        subjects_to_focus=", ".join(data.get("subjects_to_focus", [])),
        available_hours_per_day=data.get("available_hours_per_day", 2),
    )

    response = ai_call_text(request, "study_plan", user_prompt,
                        wrap_fn=lambda t: {"daily_schedule": t, "weekly_plan": [], "recommendations": []})
    try:
        _sp_success = getattr(response, 'status_code', 200) < 400
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='personalised_study_plan',
            feature_category='student',
            model_used=get_model_for_task('study_plan'),
            tokens_input=int(len(user_prompt) / 4),
            tokens_output=0,
            latency_ms=int((time.time() - _start_time) * 1000),
            success=_sp_success,
            user_id=user_id_sp,
        )
    except Exception:
        pass
    return response


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "personalization",
        "model": get_model_for_task("study_plan"),
    })
