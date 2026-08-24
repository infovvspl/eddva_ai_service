"""
Textbook ingestion — turn a chapter PDF into page-tagged passages.

The school's own book is the authoritative source for generated content, so
everything downstream (slides, notes, questions) is written from these passages
rather than from what the model happens to know. Passages keep their page number
so generated material can cite the book, which is what makes it checkable by a
teacher.

This module is deliberately stateless: it extracts and returns passages. The
school database is owned by the NestJS backend, which persists what it gets back
— mirroring how the rest of this service is called.
"""
import io
import logging
import os
import re

import requests as _requests

logger = logging.getLogger("ai_services.textbook")

# Roughly 4 characters per token for English prose. Passages are sized so a
# handful fit in a prompt alongside the instructions without crowding them out.
_CHARS_PER_TOKEN = 4
_TARGET_CHARS = 2400          # ≈600 tokens
_MAX_CHARS = 3600             # hard ceiling before a forced split
_MIN_CHARS = 120              # below this a "page" is a header or scan artefact

# Chapter PDFs are usually small, but whole-book or high-resolution scanned
# uploads can be large. The file is buffered in memory up to this size, so raise
# it in step with the AI service's available RAM. Env-overridable.
_MAX_PDF_MB = int(os.getenv("TEXTBOOK_MAX_PDF_MB", "250"))
_MAX_PDF_BYTES = _MAX_PDF_MB * 1024 * 1024

# Pages per vision request. Measured on real indexed chapters, transcription runs
# 570-700 output tokens per page, so a single request against a 30k output ceiling
# runs out at roughly 45 pages. Batching keeps page count from being a ceiling at
# all: a long scan costs more requests rather than silently losing its tail.
_OCR_PAGES_PER_BATCH = 20
_OCR_MAX_OUTPUT_TOKENS = 30000


class TruncatedTranscript(RuntimeError):
    """The model hit its output ceiling mid-transcript.

    Distinct from an unreadable scan: the page images were fine and the text is
    partial, so telling the user to re-scan would send them after the wrong
    problem. Splitting the PDF is the fix.
    """


def approx_tokens(text: str) -> int:
    return max(1, len(text or "") // _CHARS_PER_TOKEN)


def clean_page_text(text: str) -> str:
    """Normalise extracted page text without altering wording.

    Only layout noise is removed — hyphenation introduced by line wrapping,
    repeated whitespace, and stray form feeds. Nothing is paraphrased, because
    the whole point of grounding is that the words stay the book's own.
    """
    if not text:
        return ""
    t = text.replace("\x0c", "\n")
    # Join words split across a line break: "algo-\nrithm" -> "algorithm".
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Collapse single newlines inside a paragraph into spaces, keep blank lines.
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fetch_pdf(url: str) -> bytes:
    """Download a PDF, refusing anything over the size ceiling.

    The ceiling is enforced while streaming rather than after the fact: reading
    the whole body first and then measuring it means an oversized file has
    already been held in memory, which is what the limit exists to prevent.
    """
    with _requests.get(url, timeout=60, stream=True) as resp:
        resp.raise_for_status()

        # Trust a declared length only to fail fast; it is absent or wrong often
        # enough that the running count below is the real guard.
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > _MAX_PDF_BYTES:
            raise ValueError(
                f"PDF is {int(declared) // 1024 // 1024}MB; limit is "
                f"{_MAX_PDF_BYTES // 1024 // 1024}MB"
            )

        buf = io.BytesIO()
        size = 0
        for block in resp.iter_content(chunk_size=256 * 1024):
            size += len(block)
            if size > _MAX_PDF_BYTES:
                raise ValueError(
                    f"PDF exceeds the {_MAX_PDF_BYTES // 1024 // 1024}MB limit"
                )
            buf.write(block)
        return buf.getvalue()


def extract_pdf_pages(source: "str | bytes") -> "list[dict]":
    """Return [{page_no, text, chars}] for a PDF given as a URL or raw bytes."""
    data = fetch_pdf(source) if isinstance(source, str) else source

    # Imported here, not at module scope: this module is reachable from urls.py,
    # so a top-level import makes the whole URL conf — and therefore the entire
    # service — fail wherever pdfplumber is absent, such as the CI image. It is
    # only needed to read a PDF, which is exactly this function.
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required to read a chapter PDF but is not installed"
        ) from exc

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:          # one bad page must not lose the book
                logger.warning("Page %d extraction failed: %s", i, exc)
                raw = ""
            text = clean_page_text(raw)
            pages.append({"page_no": i, "text": text, "chars": len(text)})
    return pages


def _split_long(text: str, page_no: int) -> "list[dict]":
    """Break an over-long page on paragraph, then sentence, boundaries."""
    out, buf = [], ""
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 1 <= _TARGET_CHARS:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(para) <= _MAX_CHARS:
            buf = para
            continue
        # A single paragraph longer than the ceiling — split on sentence ends.
        sentence = ""
        for piece in re.split(r"(?<=[.!?])\s+", para):
            if len(sentence) + len(piece) + 1 > _TARGET_CHARS and sentence:
                out.append(sentence.strip())
                sentence = piece
            else:
                sentence = f"{sentence} {piece}".strip()
        if sentence:
            buf = sentence
    if buf:
        out.append(buf)
    return [{"page_no": page_no, "content": c} for c in out if len(c) >= _MIN_CHARS]


def chunk_pages(pages: "list[dict]") -> "list[dict]":
    """Group pages into passages of ~600 tokens, never spanning a page boundary.

    Passages stay within one page so every one carries an unambiguous citation.
    A page shorter than the target is emitted whole rather than merged with its
    neighbour, because a merged passage could only cite a range.
    """
    chunks = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if len(text) < _MIN_CHARS:
            continue
        if len(text) <= _MAX_CHARS:
            chunks.append({"page_no": page["page_no"], "content": text})
        else:
            chunks.extend(_split_long(text, page["page_no"]))

    for idx, c in enumerate(chunks):
        c["chunk_index"] = idx
        c["tokens"] = approx_tokens(c["content"])
    return chunks


def split_pdf_batches(data: bytes, per_batch: int) -> "list[tuple[int, bytes]]":
    """Split a PDF into (first_page_no, pdf_bytes) slices for transcription.

    Falls back to a single slice when pypdf is unavailable, so a minimal install
    still transcribes — it just regains the old length ceiling, which the caller
    now detects and reports instead of silently truncating.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        logger.warning("pypdf unavailable; transcribing the PDF in a single pass")
        return [(1, data)]

    reader = PdfReader(io.BytesIO(data))
    total = len(reader.pages)
    if total <= per_batch:
        return [(1, data)]

    batches = []
    for start in range(0, total, per_batch):
        writer = PdfWriter()
        for page in reader.pages[start:start + per_batch]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        batches.append((start + 1, buf.getvalue()))
    return batches


def _ocr_batch(data: bytes, first_page_no: int) -> "list[dict]":
    """Transcribe one slice, returning pages numbered from first_page_no."""
    import json as _json

    from ai_services.core import gemini_client as _gc
    from google.genai import types

    result = _gc.generate_with_rotation(
        model=_gc.DEFAULT_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type="application/pdf"),
            "Transcribe this textbook chapter to plain text, page by page. "
            "Preserve wording, headings, numbered sections, equations and examples "
            "exactly as printed. Do not summarise, reorder or add anything. "
            'Return JSON: {"pages":[{"page_no":1,"text":"..."}]}',
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=_OCR_MAX_OUTPUT_TOKENS,
        ),
        what=f"OCR pages {first_page_no}+",
    )

    # Checked before parsing: a response cut off at the ceiling is invalid JSON,
    # and reporting that as malformed output would blame the wrong thing.
    candidates = getattr(result, "candidates", None) or []
    finish = str(getattr(candidates[0], "finish_reason", "") if candidates else "")
    if "MAX_TOKENS" in finish.upper():
        raise TruncatedTranscript(
            f"transcript hit the {_OCR_MAX_OUTPUT_TOKENS}-token output limit "
            f"at page {first_page_no}"
        )

    raw = (getattr(result, "text", "") or "").strip() or "{}"
    try:
        payload = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        # Unbalanced JSON is the other signature of a cut-off response, for the
        # cases where finish_reason does not say so.
        if raw.count("{") > raw.count("}"):
            raise TruncatedTranscript(
                f"transcript ended mid-object at page {first_page_no}"
            ) from exc
        raise RuntimeError(f"vision transcript was not valid JSON: {exc}") from exc

    pages = []
    for offset, p in enumerate(payload.get("pages", [])):
        text = clean_page_text(p.get("text") or "")
        pages.append({
            "page_no": first_page_no + offset,
            "text": text,
            "chars": len(text),
        })
    return pages


def ocr_pdf_pages(data: bytes) -> "list[dict]":
    """Transcribe a scanned PDF page by page using Gemini's native PDF vision.

    School textbooks very often arrive as photographs or scans shared over
    WhatsApp, where pdfplumber finds no text layer at all. Gemini reads the
    document directly, so no page rasterisation or separate OCR engine is
    needed. Temperature is 0 and the instruction forbids summarising, because
    the transcript has to stay the book's own words to be worth citing.

    Long chapters go in page batches. A single request runs out of output budget
    at roughly 45 pages, and the resulting truncation used to surface as "no
    readable text" — sending a teacher to re-scan a book that was never the
    problem. Page numbers stay absolute across batches so citations still match
    the printed chapter.
    """
    from ai_services.core import gemini_client as _gc

    if not _gc.is_available():
        raise RuntimeError("Gemini is required to read a scanned PDF but is unavailable")

    batches = split_pdf_batches(data, _OCR_PAGES_PER_BATCH)
    pages: "list[dict]" = []
    for first_page_no, chunk in batches:
        batch_pages = _ocr_batch(chunk, first_page_no)
        pages.extend(batch_pages)
        logger.info(
            "Transcribed pages %d-%d (%d chars)",
            first_page_no, first_page_no + len(batch_pages) - 1,
            sum(p["chars"] for p in batch_pages),
        )
    return pages


def ingest_pdf(source: "str | bytes", allow_ocr: bool = True) -> dict:
    """Extract + chunk a chapter PDF. Returns passages plus a quality report.

    The report exists so a human can tell a clean digital textbook from a scan
    that produced nothing — a silently empty ingest would otherwise look
    identical to a book with no relevant content.
    """
    # Downloaded through the same guarded path as everything else. This used to
    # fetch the body directly and hand raw bytes to extract_pdf_pages, which
    # meant the size ceiling — only checked on the URL branch there — never
    # applied to the path the backend actually calls.
    data = fetch_pdf(source) if isinstance(source, str) else source

    pages = extract_pdf_pages(data)
    chunks = chunk_pages(pages)
    method = "text_layer"
    truncated = False

    # No text layer worth having — fall back to reading the pages as images.
    if allow_ocr and sum(p["chars"] for p in pages) < 200:
        try:
            ocr_pages = ocr_pdf_pages(data)
            if sum(p["chars"] for p in ocr_pages) >= 200:
                pages, method = ocr_pages, "ocr"
                chunks = chunk_pages(pages)
        except TruncatedTranscript as exc:
            # Kept separate from the catch below so this does not end up reported
            # as an unreadable scan: the book is fine, it is too long for one pass.
            truncated = True
            logger.warning("OCR transcript truncated: %s", exc)
        except Exception as exc:
            logger.warning("OCR fallback failed: %s", exc)

    total_chars = sum(p["chars"] for p in pages)
    empty_pages = sum(1 for p in pages if p["chars"] < _MIN_CHARS)
    quality = "ok"
    if truncated and not chunks:
        # Checked before the emptier diagnoses below, which would otherwise
        # blame the scan for what is really a length problem.
        quality = "too_long"
    elif not pages:
        quality = "unreadable"
    elif not chunks or total_chars < 200:
        quality = "no_text"                    # almost certainly a scanned book
    elif truncated or empty_pages > len(pages) * 0.5:
        quality = "partial"

    return {
        "pages": len(pages),
        "chunks": chunks,
        "total_chars": total_chars,
        "total_tokens": approx_tokens(" " * total_chars),
        "empty_pages": empty_pages,
        "quality": quality,
        "method": method,
        "truncated": truncated,
        "needs_ocr": quality in ("no_text", "unreadable"),
    }
