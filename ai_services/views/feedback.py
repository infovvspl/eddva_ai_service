import time

from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.usage_logger import log_usage
from .base import ai_call, ai_call_text


def root(request):
    return JsonResponse({
        "message": "AI Study API v2.0 — Production Multi-Tenant",
        "services": [
            "/feedback", "/notes", "/content", "/test",
            "/career", "/performance", "/personalization", "/cheating",
        ],
        "auth": "Include X-API-Key header with your institute API key",
    })



@api_view(["POST"])
def analyze_feedback(request):
    _start_time = time.time()
    data = request.data
    subject = data.get("subject")
    student_answer = data.get("student_answer")
    marking_scheme = data.get("marking_scheme")
    institute_id = getattr(request, "institute_id", "default")
    user_id_fb = data.get('userId') or data.get('user_id') or data.get('studentId') or ''

    if not all([subject, student_answer, marking_scheme]):
        return Response({"error": "Missing required fields: subject, student_answer, marking_scheme"}, status=400)

    template = get_template("feedback_analyze")
    user_prompt = template.user_template.format(
        subject=subject,
        marking_scheme=marking_scheme,
        student_answer=student_answer,
        extra_context=data.get("extra_context", "N/A"),
    )

    response = ai_call_text(request, "feedback_analyze", user_prompt,
                        wrap_fn=lambda t: {"score": 0, "feedback": t, "suggestions": []})
    try:
        _fb_success = getattr(response, 'status_code', 200) < 400
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='retranscribe_regenerate',
            feature_category='teacher',
            model_used=get_model_for_task('feedback_analyze'),
            tokens_input=int(len(user_prompt) / 4),
            tokens_output=0,
            latency_ms=int((time.time() - _start_time) * 1000),
            success=_fb_success,
            user_id=user_id_fb,
        )
    except Exception:
        pass
    return response


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "feedback",
        "model": get_model_for_task("feedback_analyze"),
    })
