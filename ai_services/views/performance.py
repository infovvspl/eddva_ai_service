import json
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call


@api_view(["POST"])
def analyze_performance(request):
    data = request.data
    student_id = data.get("student_id")
    if not student_id:
        return Response({"error": "Missing student_id"}, status=400)

    subjects_data = data.get("subjects_data", [])

    template = get_template("performance_analyze")
    user_prompt = template.user_template.format(
        student_id=student_id,
        subjects_data=json.dumps(subjects_data),
    )

    return ai_call(request, feature="performance_analyze", user_prompt=user_prompt)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "performance_analysis",
        "model": get_model_for_task("performance_analyze"),
    })
