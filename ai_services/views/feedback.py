from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call


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
    data = request.data
    subject = data.get("subject")
    student_answer = data.get("student_answer")
    marking_scheme = data.get("marking_scheme")

    if not all([subject, student_answer, marking_scheme]):
        return Response({"error": "Missing required fields: subject, student_answer, marking_scheme"}, status=400)

    template = get_template("feedback_analyze")
    user_prompt = template.user_template.format(
        subject=subject,
        marking_scheme=marking_scheme,
        student_answer=student_answer,
        extra_context=data.get("extra_context", "N/A"),
    )

    return ai_call(request, feature="feedback_analyze", user_prompt=user_prompt)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "feedback",
        "model": get_model_for_task("feedback_analyze"),
    })
