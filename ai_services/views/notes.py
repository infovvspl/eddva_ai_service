import os
import glob as globmod
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from .base import ai_call

NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "video_notes")


@api_view(["POST"])
def upload_and_generate_notes(request):
    file = request.FILES.get("file")
    student_id = request.data.get("student_id", "student_001")
    topic = request.data.get("topic", "Untitled")

    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    # In production: transcribe with Whisper, then pass transcript to LLM
    transcript = f"[Transcript of uploaded file: {file.name} on topic: {topic}]"

    template = get_template("notes_generate")
    user_prompt = template.user_template.format(transcript=transcript)

    return ai_call(request, feature="notes_generate", user_prompt=user_prompt)


@api_view(["GET"])
def list_saved_notes(request):
    student_id = request.query_params.get("student_id")
    if not student_id:
        return Response({"error": "Missing student_id"}, status=400)

    pattern = os.path.join(NOTES_DIR, f"{student_id}_*.json")
    files = globmod.glob(pattern)
    return Response({
        "student_id": student_id,
        "notes": [os.path.basename(f) for f in sorted(files, reverse=True)],
    })


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "ai_notes",
        "model": get_model_for_task("notes_generate"),
    })
