from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call, ai_call_text


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
