"""
Batch question evaluator — QA pass over AI-generated questions.

Checks a batch of generated questions for factual correctness, coherence, and
grade-appropriate difficulty. Registry-backed (vertical-aware prompt + model),
tenant-feature-gated, and usage-logged like the rest of the bridge.
"""

import json
import logging
import time

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.prompt_templates import get_template
from ai_services.core.model_tier import get_model_for_task
from ai_services.core.usage_logger import log_usage
from .base import get_llm, _log_usage_to_db

logger = logging.getLogger("ai_services.evaluate")

FEATURE = "evaluate_batch"


@api_view(["POST"])
def evaluate_batch(request):
    """
    POST body: {"questions": [{"id", "question", "options", "answer", "explanation", "_meta": {"difficulty"}}, ...]}
    Returns:   {"evaluations": [{"id", "is_accurate", "score", "feedback", "corrections"}, ...]}
    """
    _start_time = time.time()
    institute = getattr(request, "institute", None)
    institute_id = getattr(request, "institute_id", "default")
    vertical = getattr(request, "vertical", "coaching")
    profile = getattr(request, "profile", None)
    user_id_ev = (request.data or {}).get('userId') or (request.data or {}).get('user_id') or ''

    # Feature gating: vertical-level then tenant-level
    if profile is not None and not profile.allows_feature(FEATURE):
        return Response({"error": f"Feature '{FEATURE}' is not available for the {vertical} vertical"}, status=403)
    if institute is not None and not institute.is_feature_enabled(FEATURE):
        return Response({"error": f"Feature '{FEATURE}' is not enabled for your plan", "plan": institute.plan}, status=403)

    questions = (request.data or {}).get("questions", [])
    if not questions:
        return Response({"error": "No questions provided"}, status=400)

    # Keep only the fields the evaluator needs (trim noisy/huge payloads).
    simplified = []
    for q in questions:
        meta = q.get("_meta")
        simplified.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "options": q.get("options", []),
            "answer": q.get("answer", "") or q.get("integerAnswer", "") or q.get("solutionText", ""),
            "explanation": q.get("explanation", ""),
            "assigned_difficulty": meta.get("difficulty", "unknown") if isinstance(meta, dict) else "unknown",
        })

    template = get_template(FEATURE, vertical)
    model = get_model_for_task(FEATURE, vertical)
    user_prompt = template.user_template.format(
        questions_json=json.dumps(simplified, indent=2, ensure_ascii=False)
    )

    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=model,
            temperature=0.2,  # low temp for strict, deterministic evaluation
            json_mode=True,
            institute_id=institute_id,
        )
    except Exception as e:
        logger.error("Evaluation failed (institute=%s, vertical=%s): %s", institute_id, vertical, e)
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='in_video_quiz_generator',
                feature_category='teacher',
                model_used=model,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(e)[:500],
                user_id=user_id_ev,
            )
        except Exception:
            pass
        return Response({"error": str(e)}, status=502)

    content = result.get("content", {})
    evaluations = content.get("evaluations", []) if isinstance(content, dict) else []
    if not evaluations and isinstance(content, list):  # model returned a bare list
        evaluations = content

    _log_usage_to_db(institute, institute_id, FEATURE, result, cache_hit=False, vertical=vertical)

    try:
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='in_video_quiz_generator',
            feature_category='teacher',
            model_used=result.get('model', model),
            tokens_input=result.get('tokens_input', 0),
            tokens_output=result.get('tokens_output', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_ev,
        )
    except Exception:
        pass

    return Response({
        "evaluations": evaluations,
        "_meta": {
            "source": "llm",
            "model": result.get("model", model),
            "latency_ms": round(result.get("latency_ms", 0)),
            "institute": institute_id,
            "vertical": vertical,
            "count": len(evaluations),
        },
    })


@api_view(["GET"])
def health(request):
    return Response({"status": "ok", "feature": FEATURE})
