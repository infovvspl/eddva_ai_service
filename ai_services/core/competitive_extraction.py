"""
Competitive-exam question extraction — turn a PYQ paper or question-bank PDF
into structured, discrete questions for the Competitive Exam Prep vertical's
global bank.

This deliberately does NOT reuse the regex/pdfplumber approach the earlier
`extract1.py` prototype used. Spot-checking real files from the source
folders showed why: most official NEET papers and older JEE Advanced papers
are scanned images with no text layer at all, and even "clean" digitally
typeset JEE Main papers lay out their four options as a 2x2 grid that plain
linear text extraction flattens out of order. A vision model reading the
actual page — the same way `ai_services.core.textbook` already reads a
scanned textbook chapter — sees the real layout and the real pixels, so it
sidesteps both failure modes in one pass instead of needing a text-layer
branch plus a separate OCR branch.

Like `textbook.py`, this module is stateless: it extracts and returns
structured questions. The NestJS backend persists them (as unverified rows,
pending human review in the Super Admin verify queue).
"""
import json
import logging
import os

from ai_services.core.textbook import (
    TruncatedTranscript,
    fetch_pdf,
    set_ingest_progress,
    split_pdf_batches,
    _pdf_page_count,
)

logger = logging.getLogger("ai_services.competitive_extraction")

# Smaller than textbook.py's 20 pages/batch: a page of densely-packed MCQs
# with options/answers/explanations produces a much more verbose JSON array
# per page than a plain prose transcript does, so the same output-token
# ceiling is reached sooner.
_EXTRACT_PAGES_PER_BATCH = int(os.getenv("COMPETITIVE_EXTRACT_PAGES_PER_BATCH", "10"))
_EXTRACT_MAX_OUTPUT_TOKENS = 30000

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_QUESTION_TYPES = {"mcq_single", "mcq_multiple", "integer", "subjective"}

_EXTRACT_PROMPT = (
    "You are extracting exam questions from a page of an Indian competitive-exam "
    "(JEE / NEET) question paper or practice-question document, for a verified "
    "question bank. A human will review every question you extract before it is "
    "ever shown to a student, so accuracy matters more than completeness — it is "
    "fine to extract fewer questions than to invent or merge content.\n\n"
    "For EACH distinct question on these pages, extract:\n"
    "- question_number: the printed question number (integer), or null if unnumbered\n"
    "- question_text: the full question exactly as printed, preserving equations/"
    "notation as plain text\n"
    "- options: an object keyed \"A\",\"B\",\"C\",\"D\" (map printed (1)(2)(3)(4) to "
    "A/B/C/D in that order). Omit entirely for a non-MCQ (e.g. integer-answer) "
    "question.\n"
    "- correct_answer: the option letter A-D if an answer key or marked answer is "
    "visible on these pages, else null — do NOT guess the answer yourself\n"
    "- explanation: the worked solution if printed on these pages, else null\n"
    "- subject: Physics/Chemistry/Maths/Biology/Botany/Zoology if a section header "
    "indicates it, else null\n"
    "- chapter: the specific chapter/topic name if a heading or label on the page "
    "names it, else null\n"
    "- difficulty: your best-effort \"easy\"|\"medium\"|\"hard\" estimate, else null\n"
    "- question_type: \"mcq_single\"|\"mcq_multiple\"|\"integer\"|\"subjective\"\n"
    "- has_image: true if the question references or requires a diagram/graph/figure\n\n"
    "Skip page headers, instructions, running footers, and page numbers — those are "
    "not questions. Do not invent anything not printed on the page.\n\n"
    'Return JSON only: {"questions": [...]}'
)


def _extract_batch(data: bytes, first_page_no: int) -> "list[dict]":
    """Vision-extract structured questions from one page-batch's PDF bytes."""
    from ai_services.core import gemini_client as _gc
    from google.genai import types

    result = _gc.generate_with_rotation(
        model=_gc.DEFAULT_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type="application/pdf"),
            _EXTRACT_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=_EXTRACT_MAX_OUTPUT_TOKENS,
        ),
        what=f"competitive question extraction, pages {first_page_no}+",
    )

    candidates = getattr(result, "candidates", None) or []
    finish = str(getattr(candidates[0], "finish_reason", "") if candidates else "")
    if "MAX_TOKENS" in finish.upper():
        raise TruncatedTranscript(
            f"extraction hit the {_EXTRACT_MAX_OUTPUT_TOKENS}-token output limit "
            f"at page {first_page_no}"
        )

    raw = (getattr(result, "text", "") or "").strip() or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if raw.count("{") > raw.count("}"):
            raise TruncatedTranscript(
                f"extraction ended mid-object at page {first_page_no}"
            ) from exc
        raise RuntimeError(f"extraction response was not valid JSON: {exc}") from exc

    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    cleaned = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        text = str(q.get("question_text") or "").strip()
        if len(text) < 10:
            continue
        options = q.get("options") if isinstance(q.get("options"), dict) else {}
        options = {k: str(v).strip() for k, v in options.items() if k in ("A", "B", "C", "D") and str(v).strip()}

        difficulty = q.get("difficulty")
        difficulty = difficulty if difficulty in _VALID_DIFFICULTIES else None

        q_type = q.get("question_type")
        q_type = q_type if q_type in _VALID_QUESTION_TYPES else ("mcq_single" if options else "subjective")

        cleaned.append({
            "question_number": q.get("question_number") if isinstance(q.get("question_number"), int) else None,
            "question_text": text,
            "options": options,
            "correct_answer": q.get("correct_answer") if q.get("correct_answer") in ("A", "B", "C", "D") else None,
            "explanation": (str(q.get("explanation")).strip() or None) if q.get("explanation") else None,
            "subject": (str(q.get("subject")).strip() or None) if q.get("subject") else None,
            "chapter": (str(q.get("chapter")).strip() or None) if q.get("chapter") else None,
            "difficulty": difficulty,
            "question_type": q_type,
            "has_image": bool(q.get("has_image")),
            "source_page": first_page_no,
        })
    return cleaned


def extract_questions(source: "str | bytes", progress_key: "str | None" = None) -> dict:
    """Extract structured questions from a PYQ / question-bank PDF.

    Batches the PDF the same way `textbook.ocr_pdf_pages` does, so a large
    multi-hundred-page compilation costs more requests rather than silently
    losing its tail. Every batch is a direct vision read of the page images
    via Gemini's native PDF support — there is no text-layer branch, since
    a vision read handles both a scanned paper and a digitally typeset one
    the same way, which is the whole point (see module docstring).

    Returns `{questions: [...], pages, total_found, quality, truncated}` —
    `quality` is "ok" when at least one question was found, "low_yield" when
    the document had pages but nothing extractable (e.g. a cover/instruction
    page, or a page the model genuinely found nothing on), never "unreadable"
    the way textbook ingestion can be, since a vision read doesn't depend on
    a text layer existing at all.
    """
    data = fetch_pdf(source) if isinstance(source, str) else source

    batches = split_pdf_batches(data, _EXTRACT_PAGES_PER_BATCH)
    total_pages = _pdf_page_count(data) or (len(batches) * _EXTRACT_PAGES_PER_BATCH)
    set_ingest_progress(progress_key, 0, total_pages, "extract")

    questions: "list[dict]" = []
    pages_done = 0
    truncated = False
    for first_page_no, chunk in batches:
        try:
            batch_questions = _extract_batch(chunk, first_page_no)
            questions.extend(batch_questions)
        except TruncatedTranscript as exc:
            # One batch running out of budget doesn't invalidate the rest —
            # keep what was already extracted and flag the run as partial.
            truncated = True
            logger.warning("Competitive extraction truncated: %s", exc)
        pages_done = min(pages_done + _EXTRACT_PAGES_PER_BATCH, total_pages)
        set_ingest_progress(progress_key, pages_done, total_pages, "extract")
        logger.info(
            "Extracted %d questions from pages %d-%d",
            len(questions), first_page_no, first_page_no + _EXTRACT_PAGES_PER_BATCH - 1,
        )

    quality = "ok" if questions else "low_yield"
    return {
        "questions": questions,
        "pages": total_pages,
        "total_found": len(questions),
        "quality": quality,
        "truncated": truncated,
    }


def extract_answer_key(source: "str | bytes") -> dict:
    """Extract a {question_number: {correct_answer, explanation}} map from a
    separate answer-key/solutions PDF — used when a paper's key lives in its
    own file (common for NEET papers and institute grand tests) rather than
    inline with each question.
    """
    data = fetch_pdf(source) if isinstance(source, str) else source
    batches = split_pdf_batches(data, _EXTRACT_PAGES_PER_BATCH)

    answers: "dict[int, dict]" = {}
    for first_page_no, chunk in batches:
        try:
            batch_questions = _extract_batch(chunk, first_page_no)
        except TruncatedTranscript:
            continue
        for q in batch_questions:
            num = q.get("question_number")
            if num is None:
                continue
            if q.get("correct_answer") or q.get("explanation"):
                answers[num] = {
                    "correct_answer": q.get("correct_answer"),
                    "explanation": q.get("explanation"),
                }
    return {"answers": answers}


def merge_answer_key(questions: "list[dict]", answer_key: "dict[int, dict]") -> "list[dict]":
    """Fill in correct_answer/explanation from a separately-extracted key,
    matched by question_number. Never overwrites an answer already found
    inline on the question's own page — the key only fills gaps.
    """
    for q in questions:
        num = q.get("question_number")
        if num is None or num not in answer_key:
            continue
        key_entry = answer_key[num]
        if not q.get("correct_answer") and key_entry.get("correct_answer"):
            q["correct_answer"] = key_entry["correct_answer"]
        if not q.get("explanation") and key_entry.get("explanation"):
            q["explanation"] = key_entry["explanation"]
    return questions
