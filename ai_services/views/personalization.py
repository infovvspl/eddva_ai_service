from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call


@api_view(["POST"])
def generate_study_plan(request):
    data = request.data
    student_id = data.get("student_id")
    if not student_id:
        return Response({"error": "Missing student_id"}, status=400)

    template = get_template("study_plan")
    user_prompt = template.user_template.format(
        student_id=student_id,
        learning_style=data.get("learning_style", "visual"),
        subjects_to_focus=", ".join(data.get("subjects_to_focus", [])),
        available_hours_per_day=data.get("available_hours_per_day", 2),
    )

    return ai_call(request, feature="study_plan", user_prompt=user_prompt)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "personalization",
        "model": get_model_for_task("study_plan"),
    })
