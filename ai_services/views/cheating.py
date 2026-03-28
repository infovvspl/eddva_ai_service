import json
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call


@api_view(["POST"])
def analyze_exam_logs(request):
    logs = request.data
    if not logs:
        return Response({"error": "Missing logs"}, status=400)

    template = get_template("cheating_analyze")
    user_prompt = template.user_template.format(logs_json=json.dumps(logs))

    return ai_call(
        request,
        feature="cheating_analyze",
        user_prompt=user_prompt,
        skip_cache=True,  # cheating detection must always be real-time
    )


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "cheating_detection",
        "model": get_model_for_task("cheating_analyze"),
    })
