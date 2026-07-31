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
import re

import pdfplumber
import requests as _requests

logger = logging.getLogger("ai_services.textbook")

# Roughly 4 characters per token for English prose. Passages are sized so a
# handful fit in a prompt alongside the instructions without crowding them out.
_CHARS_PER_TOKEN = 4
_TARGET_CHARS = 2400          # ≈600 tokens
_MAX_CHARS = 3600             # hard ceiling before a forced split
_MIN_CHARS = 120              # below this a "page" is a header or scan artefact

_MAX_PDF_BYTES = 60 * 1024 * 1024


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


def extract_pdf_pages(source: "str | bytes") -> "list[dict]":
    """Return [{page_no, text, chars}] for a PDF given as a URL or raw bytes."""
    if isinstance(source, str):
        resp = _requests.get(source, timeout=60, stream=True)
        resp.raise_for_status()
        data = resp.content
        if len(data) > _MAX_PDF_BYTES:
            raise ValueError(f"PDF is {len(data) // 1024 // 1024}MB; limit is 60MB")
    else:
        data = source

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


def ocr_pdf_pages(data: bytes) -> "list[dict]":
    """Transcribe a scanned PDF page by page using Gemini's native PDF vision.

    School textbooks very often arrive as photographs or scans shared over
    WhatsApp, where pdfplumber finds no text layer at all. Gemini reads the
    document directly, so no page rasterisation or separate OCR engine is
    needed. Temperature is 0 and the instruction forbids summarising, because
    the transcript has to stay the book's own words to be worth citing.
    """
    from ai_services.core import gemini_client as _gc

    if not _gc.is_available():
        raise RuntimeError("Gemini is required to read a scanned PDF but is unavailable")

    import json as _json
    import os as _os
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_os.getenv("GEMINI_API_KEY"))
    result = client.models.generate_content(
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
            max_output_tokens=30000,
        ),
    )
    payload = _json.loads((getattr(result, "text", "") or "").strip() or "{}")
    pages = []
    for p in payload.get("pages", []):
        text = clean_page_text(p.get("text") or "")
        pages.append({
            "page_no": int(p.get("page_no") or len(pages) + 1),
            "text": text,
            "chars": len(text),
        })
    return pages


def ingest_pdf(source: "str | bytes", allow_ocr: bool = True) -> dict:
    """Extract + chunk a chapter PDF. Returns passages plus a quality report.

    The report exists so a human can tell a clean digital textbook from a scan
    that produced nothing — a silently empty ingest would otherwise look
    identical to a book with no relevant content.
    """
    if isinstance(source, str):
        resp = _requests.get(source, timeout=60)
        resp.raise_for_status()
        data = resp.content
    else:
        data = source

    pages = extract_pdf_pages(data)
    chunks = chunk_pages(pages)
    method = "text_layer"

    # No text layer worth having — fall back to reading the pages as images.
    if allow_ocr and sum(p["chars"] for p in pages) < 200:
        try:
            ocr_pages = ocr_pdf_pages(data)
            if sum(p["chars"] for p in ocr_pages) >= 200:
                pages, method = ocr_pages, "ocr"
                chunks = chunk_pages(pages)
        except Exception as exc:
            logger.warning("OCR fallback failed: %s", exc)

    total_chars = sum(p["chars"] for p in pages)
    empty_pages = sum(1 for p in pages if p["chars"] < _MIN_CHARS)
    quality = "ok"
    if not pages:
        quality = "unreadable"
    elif not chunks or total_chars < 200:
        quality = "no_text"                    # almost certainly a scanned book
    elif empty_pages > len(pages) * 0.5:
        quality = "partial"

    return {
        "pages": len(pages),
        "chunks": chunks,
        "total_chars": total_chars,
        "total_tokens": approx_tokens(" " * total_chars),
        "empty_pages": empty_pages,
        "quality": quality,
        "method": method,
        "needs_ocr": quality in ("no_text", "unreadable"),
    }
