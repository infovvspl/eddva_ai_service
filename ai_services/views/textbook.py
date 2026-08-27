"""
Textbook ingestion endpoint — NestJS ai-bridge.

POST /textbook/ingest  → read a chapter PDF and return page-tagged passages

Stateless by design: this service extracts, the NestJS backend persists. The
school database is owned by the backend, and keeping it that way avoids a second
writer against tenant data.
"""
import logging

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ai_services.core.textbook import ingest_pdf
from ai_services.core.usage_logger import log_usage

logger = logging.getLogger("ai_services.textbook")


@api_view(["POST"])
def ingest_textbook(request):
    """
    POST /textbook/ingest

    Body: { fileUrl, allowOcr? }
    Returns: { success, data: { pages, chunks[], quality, method, ... } }

    A scanned chapter (no text layer) is transcribed automatically unless
    allowOcr is false — that path costs an LLM call, so callers doing a bulk
    dry-run can turn it off and just see which files would need it.
    """
    file_url = (request.data.get("fileUrl") or "").strip()
    if not file_url:
        return Response({"error": "fileUrl is required"}, status=status.HTTP_400_BAD_REQUEST)

    allow_ocr = request.data.get("allowOcr")
    allow_ocr = True if allow_ocr is None else bool(allow_ocr)

    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", None) or "school"

    try:
        result = ingest_pdf(file_url, allow_ocr=allow_ocr)
    except ValueError as exc:
        # Size ceiling (or an obviously malformed download) — report specifically
        # so the teacher is told to split/replace the file, not "try again".
        logger.warning("Textbook ingest rejected for %s: %s", file_url[:120], exc)
        log_usage(
            institute_id=institute_id, institute_type=vertical,
            feature_id="textbook_ingest", feature_category="content_ingestion",
            model_used="pdfplumber", success=False, error_message=str(exc),
        )
        return Response({"error": str(exc)}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    except Exception as exc:
        logger.error("Textbook ingest failed for %s: %s", file_url[:120], exc)
        log_usage(
            institute_id=institute_id, institute_type=vertical,
            feature_id="textbook_ingest", feature_category="content_ingestion",
            model_used="pdfplumber", success=False, error_message=str(exc),
        )
        return Response(
            {"error": "Could not read this PDF. Check the file is reachable and not password protected."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    log_usage(
        institute_id=institute_id, institute_type=vertical,
        feature_id="textbook_ingest", feature_category="content_ingestion",
        model_used=result.get("method", "pdfplumber"), success=True,
    )

    if result.get("needs_ocr"):
        # Reached only when OCR was disabled or failed — the caller must be able
        # to distinguish "book has no usable text" from "ingest succeeded, empty".
        logger.warning("Textbook produced no usable text (%s)", file_url[:120])

    return Response({"success": True, "data": result})
