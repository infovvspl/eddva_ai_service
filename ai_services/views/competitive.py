"""
Competitive Exam Prep — question extraction endpoint for NestJS ai-bridge.

POST /competitive/extract-questions → read a PYQ / question-bank PDF (and
optionally a separate answer-key PDF) and return structured, discrete
questions for the global competitive_questions bank.

Stateless by design, same as textbook.py: this service extracts, the NestJS
backend persists (always as unverified, pending the Super Admin review
queue — see modules/school/competitive/competitive-master.service.ts).
"""
import logging

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ai_services.core.competitive_extraction import (
    extract_answer_key,
    extract_questions,
    merge_answer_key,
)
from ai_services.core.textbook import get_ingest_progress
from ai_services.core.usage_logger import log_usage

logger = logging.getLogger("ai_services.competitive_extraction")


@api_view(["POST"])
def extract_competitive_questions(request):
    """
    POST /competitive/extract-questions

    Body: { fileUrl, answerKeyFileUrl?, progressKey? }
    Returns: { success, data: { questions[], pages, totalFound, quality, truncated } }
    """
    file_url = (request.data.get("fileUrl") or "").strip()
    if not file_url:
        return Response({"error": "fileUrl is required"}, status=status.HTTP_400_BAD_REQUEST)

    answer_key_url = (request.data.get("answerKeyFileUrl") or "").strip() or None
    progress_key = (request.data.get("progressKey") or "").strip() or None

    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", None) or "school"

    try:
        result = extract_questions(file_url, progress_key=progress_key)

        if answer_key_url:
            try:
                key_result = extract_answer_key(answer_key_url)
                result["questions"] = merge_answer_key(result["questions"], key_result["answers"])
            except Exception as exc:
                # The paper itself extracted fine; a bad/unreachable key file
                # should not fail the whole run, just leave answers unfilled.
                logger.warning("Answer-key extraction failed for %s: %s", answer_key_url[:120], exc)

    except ValueError as exc:
        logger.warning("Competitive extraction rejected for %s: %s", file_url[:120], exc)
        log_usage(
            institute_id=institute_id, institute_type=vertical,
            feature_id="competitive_question_extraction", feature_category="content_ingestion",
            model_used="gemini", success=False, error_message=str(exc),
        )
        return Response({"error": str(exc)}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    except Exception as exc:
        logger.error("Competitive extraction failed for %s: %s", file_url[:120], exc)
        log_usage(
            institute_id=institute_id, institute_type=vertical,
            feature_id="competitive_question_extraction", feature_category="content_ingestion",
            model_used="gemini", success=False, error_message=str(exc),
        )
        return Response(
            {"error": "Could not extract questions from this PDF. Check the file is reachable and not password protected."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    log_usage(
        institute_id=institute_id, institute_type=vertical,
        feature_id="competitive_question_extraction", feature_category="content_ingestion",
        model_used="gemini", success=True,
    )

    if result.get("quality") == "low_yield":
        logger.warning("Competitive extraction found no usable questions in %s", file_url[:120])

    return Response({
        "success": True,
        "data": {
            "questions": result["questions"],
            "pages": result["pages"],
            "totalFound": result["total_found"],
            "quality": result["quality"],
            "truncated": result["truncated"],
        },
    })


@api_view(["GET"])
def extract_progress(request):
    """
    GET /competitive/extract-progress?key=<progressKey>

    Live page-level progress for one in-flight extract_competitive_questions
    call — same mechanism textbook ingestion uses (textbook.get_ingest_progress
    reads the same Redis key namespace this module's extract_questions()
    writes to via set_ingest_progress). Returns {} once nothing has been
    published (not started, already finished, or the key expired) — never an
    error, since the backend's polling treats this as best-effort.
    """
    key = (request.query_params.get("key") or "").strip()
    progress = get_ingest_progress(key) if key else None
    return Response({"success": True, "data": progress or {}})
