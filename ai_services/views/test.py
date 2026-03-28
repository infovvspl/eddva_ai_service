import threading
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.batch_processor import BatchProcessor
from .base import ai_call


@api_view(["POST"])
def generate_practice_test(request):
    """Generate a practice test. For bulk generation, use /test/batch/."""
    data = request.data
    topic = data.get("topic")
    if not topic:
        return Response({"error": "Missing topic"}, status=400)

    num_questions = data.get("num_questions", 5)
    difficulty = data.get("difficulty", "medium")
    question_types = data.get("question_types", "mcq, true_false, short_answer")

    template = get_template("test_generate")
    user_prompt = template.user_template.format(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        question_types=question_types,
    )

    return ai_call(request, feature="test_generate", user_prompt=user_prompt)


@api_view(["POST"])
def batch_generate_tests(request):
    """
    Batch generate multiple tests — 50% cheaper via batch pricing.
    Requires 'batch' feature to be enabled for the tenant.
    """
    # Check batch feature access
    institute = getattr(request, "institute", None)
    if institute and not institute.is_feature_enabled("batch"):
        return Response(
            {"error": "Batch processing is not enabled for your plan", "upgrade": "Contact support"},
            status=403,
        )

    data = request.data
    topics = data.get("topics", [])
    if not topics:
        return Response({"error": "Missing topics list"}, status=400)

    num_questions = data.get("num_questions", 5)
    difficulty = data.get("difficulty", "medium")
    question_types = data.get("question_types", "mcq, true_false, short_answer")
    institute_id = getattr(request, "institute_id", "default")

    template = get_template("test_generate")
    prompts = [
        template.user_template.format(
            topic=t,
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=question_types,
        )
        for t in topics
    ]

    processor = BatchProcessor()
    job = processor.create_job("test_generate", institute_id, prompts)
    threading.Thread(target=processor.run, args=(job,), daemon=True).start()

    return Response({
        "job_id": job.job_id,
        "status": "queued",
        "total_items": len(topics),
        "institute": institute_id,
        "message": "Poll /test/batch/status/?job_id=<id> for progress",
    }, status=202)


@api_view(["GET"])
def batch_status(request):
    """Check status of a batch test generation job."""
    job_id = request.query_params.get("job_id")
    if not job_id:
        return Response({"error": "Missing job_id"}, status=400)

    job = BatchProcessor.get_job(job_id)
    if not job:
        return Response({"error": "Job not found"}, status=404)

    # Tenant isolation: only allow access to own jobs
    institute_id = getattr(request, "institute_id", "default")
    if job.institute_id != institute_id:
        return Response({"error": "Job not found"}, status=404)

    result = job.progress
    if job.status.value in ("completed", "partial"):
        result["results"] = [
            {"topic_index": i, "result": item.result, "error": item.error}
            for i, item in enumerate(job.items)
        ]

    return Response(result)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "ai_test",
        "model": get_model_for_task("test_generate"),
    })
