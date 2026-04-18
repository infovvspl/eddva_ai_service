from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call, ai_call_text


@api_view(["GET"])
def suggest_resources(request):
    topic = request.query_params.get("topic")
    if not topic:
        return Response({"error": "Missing topic"}, status=400)

    template = get_template("content_suggest")
    user_prompt = template.user_template.format(topic=topic)

    return ai_call_text(request, "content_suggest", user_prompt,
                        wrap_fn=lambda t: {"resources": [{"title": "AI Recommendation", "content": t}]})


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "ai_content",
        "model": get_model_for_task("content_suggest"),
    })
