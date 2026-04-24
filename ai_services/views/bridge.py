"""
Views for NestJS ai-bridge endpoints.
These endpoints match the paths called by apexiq-backend/src/modules/ai-bridge/ai-bridge.service.ts

Active endpoints:
  POST /doubt/resolve          â†’ AI #1: Doubt Clearing
  POST /tutor/session          â†’ AI #2: AI Tutor Start
  POST /tutor/continue         â†’ AI #2: AI Tutor Continue
  POST /recommend/content      â†’ AI #6: Content Recommendation
  POST /stt/notes              â†’ AI #7: Speech-to-Text Notes  (Whisper â†’ LLM)
  POST /stt/notes-from-text    â†’ AI #7b: Notes from Transcript (YouTube captions â†’ LLM, no Whisper)
  POST /feedback/generate      â†’ AI #8: Student Feedback
  POST /notes/analyze          â†’ AI #9: Notes Weak Topic Identifier
  POST /resume/analyze         â†’ AI #10: Resume Analyzer
  POST /interview/start        â†’ AI #11: Interview Prep
  POST /plan/generate          â†’ AI #12: Personalized Learning Plan
  POST /quiz/generate          â†’ AI #13: In-Video Quiz Generator
  POST /translate              â†’ AI #15: Text Translation  (Sarvam AI â€” mayura:v1)

Removed endpoints (deleted from platform):
  POST /performance/analyze    â†’ was AI #3 (performance_analysis)
  POST /grade/subjective       â†’ was AI #4 (grade_subjective)
  POST /engage/detect          â†’ was AI #5 (engagement_detect)
"""

import glob as _glob
import json
import logging
import os
import re
import tempfile
from typing import Optional

try:
    import requests as _requests
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "requests", "--quiet"])
    import requests as _requests

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.groq_keys import get_rotated_groq_keys, is_key_exhausted_error
from .base import ai_call, ai_call_text, get_llm

logger = logging.getLogger("ai_services.llm")

# -- Groq Whisper API (primary -- cloud, fast; multi-key rotation) ---------------

_GROQ_KEYS_RAW = [
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
]
GROQ_API_KEYS: list[str] = [k for k in _GROQ_KEYS_RAW if k]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # backward compat
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB Groq limit


def _parse_groq_retry_after(error_msg: str, default: float = 65.0) -> float:
    """Parse 'Please try again in 1m46.5s' from a Groq rate-limit error. Returns seconds."""
    m = re.search(r"(\d+)m(\d+\.?\d*)s", error_msg)
    if m:
        return min(int(m.group(1)) * 60 + float(m.group(2)) + 2, 360.0)
    m = re.search(r"(\d+\.?\d*)s", error_msg)
    if m:
        return min(float(m.group(1)) + 2, 360.0)
    return default


def _transcribe_with_groq(audio_path: str, language: str, prev_context: str = "") -> str:
    """Transcribe via Groq Whisper, rotating through all API keys on rate-limit (429)."""
    try:
        from groq import Groq, RateLimitError as GroqRateLimitError
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq", "--quiet"])
        from groq import Groq, RateLimitError as GroqRateLimitError

    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY configured -- set at least one in .env")

    file_size = os.path.getsize(audio_path)
    if file_size > GROQ_MAX_FILE_BYTES:
        raise RuntimeError(f"File too large for Groq ({file_size // 1024 // 1024} MB > 25 MB)")

    groq_language: str | None = None if language in ("hinglish", "auto") else language
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    kwargs: dict = dict(
        file=(filename, file_bytes),
        model=GROQ_WHISPER_MODEL,
        response_format="text",
    )
    if groq_language:
        kwargs["language"] = groq_language
    if prev_context:
        # Groq measures prompt in UTF-8 bytes (limit 896); Devanagari = 3 bytes/char.
        raw = prev_context.encode("utf-8")[-880:]
        kwargs["prompt"] = raw.decode("utf-8", errors="ignore")

    last_exc: Exception | None = None
    for round_num in range(2):  # try all keys twice before giving up
        for key_idx, api_key in enumerate(GROQ_API_KEYS):
            try:
                logger.info(
                    "Groq Whisper | key=%d/%d lang=%s",
                    key_idx + 1, len(GROQ_API_KEYS), groq_language or language,
                )
                result = Groq(api_key=api_key).audio.transcriptions.create(**kwargs)
                return result if isinstance(result, str) else result.text
            except GroqRateLimitError as exc:
                last_exc = exc
                logger.info("Groq key %d/%d rate-limited -- rotating to next key", key_idx + 1, len(GROQ_API_KEYS))
        # All keys exhausted for this round -- wait before round 2
        if round_num == 0 and last_exc is not None:
            wait = _parse_groq_retry_after(str(last_exc))
            logger.warning(
                "All %d Groq keys rate-limited -- waiting %.0fs before retry",
                len(GROQ_API_KEYS), wait,
            )
            import time as _time
            _time.sleep(wait)

    raise RuntimeError(f"All {len(GROQ_API_KEYS)} Groq keys exhausted: {last_exc}") from last_exc

# â"€â"€ faster-whisper singleton (fallback â€" local, CPU) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

_whisper_model = None

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


def _get_whisper_model():
    """Lazy singleton for faster-whisper. First call downloads the model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info(
            "Loading local Whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("Local Whisper model loaded.")
    return _whisper_model


def _transcribe_local(audio_path: str, language: str) -> str:
    """Transcribe using local faster-whisper (fallback)."""
    whisper = _get_whisper_model()
    segments, info = whisper.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    transcript = " ".join(seg.text for seg in segments).strip()
    logger.info(
        "Local Whisper done: %d chars | lang=%s", len(transcript), info.language,
    )
    return transcript


def _download_audio(audio_url: str, tmpdir: str) -> str:
    """Download a direct audio/video URL into tmpdir. Returns local file path."""
    ext = audio_url.rsplit(".", 1)[-1].split("?")[0][:8] or "mp4"
    audio_path = os.path.join(tmpdir, f"audio.{ext}")
    resp = _requests.get(audio_url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(audio_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return audio_path


def _transcribe_audio(audio_url: str, language: str = "hi") -> str:
    """
    Primary:  Groq Whisper API  (~2-3 sec, requires GROQ_API_KEY(_N), 25 MB limit)
    Fallback: local faster-whisper  (slow on CPU, no size limit)
    Supports YouTube URLs via yt-dlp.
    """
    logger.info("_transcribe_audio | url=%s | language=%s", audio_url[:80], language)
    is_youtube = "youtube.com" in audio_url or "youtu.be" in audio_url

    with tempfile.TemporaryDirectory() as tmpdir:
        if is_youtube:
            try:
                import yt_dlp
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "--quiet"])
                import yt_dlp

            ydl_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
                "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([audio_url])

            files = _glob.glob(os.path.join(tmpdir, "audio.*"))
            if not files:
                raise RuntimeError("yt-dlp downloaded nothing")
            audio_path = files[0]
        else:
            audio_path = _download_audio(audio_url, tmpdir)

        # â”€â”€ Primary: Groq â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        groq_keys = get_rotated_groq_keys()
        if groq_keys:
            try:
                import subprocess
                # Ensure ffmpeg binary is available via imageio-ffmpeg since it may not be in system PATH
                try:
                    import imageio_ffmpeg
                except ImportError:
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg", "--quiet"])
                    import imageio_ffmpeg
                
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                chunk_pattern = os.path.join(tmpdir, "chunk_%03d.mp3")
                logger.info("Chunking audio with ffmpeg to bypass 25MB Groq limit...")
                
                # Split video/audio into 10-minute MP3 chunks at 32k bitrate (mono)
                # to strictly stay within the 25MB whisper threshold
                cmd = [
                    ffmpeg_exe, "-y", "-i", audio_path,
                    "-f", "segment", "-segment_time", "600",
                    "-c:a", "libmp3lame", "-ac", "1", "-ar", "16000", "-ab", "64k",
                    "-vn", chunk_pattern
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                chunks = sorted(_glob.glob(os.path.join(tmpdir, "chunk_*.mp3")))
                if not chunks:
                    raise RuntimeError("FFMpeg generated no audio chunks.")
                
                logger.info("Chunking complete: %d segments generated.", len(chunks))
                
                full_transcript_parts = []
                prev_context = ""
                for idx, chunk_file in enumerate(chunks):
                    logger.info("Sending chunk %d/%d to Groq...", idx + 1, len(chunks))
                    try:
                        text = _transcribe_with_groq(chunk_file, language, prev_context=prev_context)
                    except Exception as exc:
                        logger.warning(
                            "Groq chunk %d/%d failed with context prompt (%s) — retrying without prompt",
                            idx + 1, len(chunks), exc,
                        )
                        text = _transcribe_with_groq(chunk_file, language, prev_context="")
                    if text:
                        full_transcript_parts.append(text)
                        prev_context = text

                transcript = " ".join(full_transcript_parts).strip()
                logger.info("Groq transcription OK — %d chars (from %d chunks)", len(transcript), len(chunks))
                return transcript
            except Exception as exc:
                raise RuntimeError(f"Groq transcription failed: {exc}") from exc

        raise RuntimeError("GROQ_API_KEY is not configured — set it in .env to enable transcription")


NON_ENGLISH_NOTES_LANGS = {"hi", "hinglish", "hi-in"}
HINGLISH_HINT_WORDS = {
    "hai", "haan", "nahi", "nahin", "kya", "kaise", "samjho", "samajh", "kyunki",
    "agar", "lekin", "wala", "wali", "isko", "usko", "karna", "karte", "hoga",
    "yahaan", "yahan", "iska", "iski", "iske", "hum", "aap", "thoda",
}

COMMON_TRANSCRIPT_GARBAGE = [
    "```",
    "<noise>",
    "</noise>",
    "<music>",
    "</music>",
    "[music]",
    "[applause]",
    "[laughter]",
]


def _looks_like_hinglish(text: str) -> bool:
    sample = " ".join(str(text or "").lower().split()[:1200])
    if not sample:
        return False
    devanagari_chars = sum(1 for ch in sample if "\u0900" <= ch <= "\u097f")
    latin_chars = sum(1 for ch in sample if "a" <= ch <= "z")
    token_hits = sum(1 for token in HINGLISH_HINT_WORDS if f" {token} " in f" {sample} ")
    return (devanagari_chars > 0 and latin_chars > 0) or token_hits >= 4


def _clean_transcript_text(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned.strip():
        return ""

    for token in COMMON_TRANSCRIPT_GARBAGE:
        cleaned = cleaned.replace(token, " ")

    replacements = [
        (r"\\text\s*\{([^}]*)\}", r"\1"),
        (r"\\gt", ">"),
        (r"\\lt", "<"),
        (r"\\geq?", ">="),
        (r"\\leq?", "<="),
        (r"\\times", " x "),
        (r"\\pi", "pi"),
        (r"\\Delta", "Delta"),
        (r"(?i)\bXB\s*=\s*1\s+XA\b", "XB = 1 - XA"),
        (r"(?i)\bXA\s*=\s*NA\s*/\s*\(\s*NA\s*\+\s*NB\s*\)", "XA = NA / (NA + NB)"),
        (r"(?i)\bXB\s*=\s*NB\s*/\s*\(\s*NA\s*\+\s*NB\s*\)", "XB = NB / (NA + NB)"),
        (r"(?i)\bpi\s*=\s*cRT\b", "pi = cRT"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned)

    cleaned = re.sub(r"\$+", " ", cleaned)
    cleaned = re.sub(r"`{3,}", " ", cleaned)
    cleaned = re.sub(r"[^\x00-\x7F\u0900-\u097F\u03B1-\u03C9\u0391-\u03A9]+", lambda m: m.group(0) if len(m.group(0).strip()) <= 3 else " ", cleaned)
    cleaned = re.sub(r"([A-Za-z])([=<>+\-/*()])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([=<>+\-/*()])([A-Za-z0-9])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b([A-Za-z])\s+\+\s+([A-Za-z])\b", r"\1 + \2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"([.!?])\s*", r"\1\n", cleaned)
    return cleaned.strip()


def _looks_like_math_or_formula_line(line: str) -> bool:
    symbols = len(re.findall(r"[=+\-/*^<>%(){}\[\]]", line))
    words = len(re.findall(r"[A-Za-z\u0900-\u097F]+", line))
    return symbols >= 3 and words <= 12


def _restore_sentence_punctuation(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    words = re.findall(r"[A-Za-z\u0900-\u097F]+", raw)
    if len(words) < 16:
        return raw

    punct_count = len(re.findall(r"[.!?]", raw))
    if punct_count >= max(2, len(words) // 25):
        return raw

    paragraphs = [p.strip() for p in re.split(r"\n+", raw) if p.strip()]
    rebuilt: list[str] = []

    for paragraph in paragraphs:
        if _looks_like_math_or_formula_line(paragraph):
            rebuilt.append(paragraph)
            continue

        tokens = paragraph.split()
        if len(tokens) < 8:
            line = paragraph
            if line and not re.search(r"[.!?]$", line):
                line = f"{line}."
            rebuilt.append(line)
            continue

        sentence_parts: list[str] = []
        current: list[str] = []
        target_len = 18

        for token in tokens:
            current.append(token)
            if len(current) >= target_len:
                sentence_parts.append(" ".join(current))
                current = []
        if current:
            sentence_parts.append(" ".join(current))

        normalized_parts: list[str] = []
        for part in sentence_parts:
            part = part.strip()
            if not part:
                continue
            part = part[0].upper() + part[1:] if part else part
            if not re.search(r"[.!?]$", part):
                part = f"{part}."
            normalized_parts.append(part)

        rebuilt.append(" ".join(normalized_parts))

    return "\n".join(rebuilt).strip()



def _strip_lecture_framing(text: str) -> str:
    """Remove teacher intro/outro, repeated greetings, and Whisper hallucinations."""
    if not text:
        return text

    text = re.sub(r"\?{2,}", "", text)
    text = re.sub(r"\[inaudible\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(inaudible\)", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"(?i)(hello|hi|hey)[,\s]+(students?|everyone|all|class|friends?|guys?)[^.!?]{0,80}[.!?]?",
        "", text,
    )

    intro, rest = text[:800], text[800:]
    intro = re.sub(
        r"(?i)^(hello|hi|good\s+(morning|afternoon|evening|day))[^.!?]{0,200}[.!?]",
        "", intro.lstrip(),
    )
    intro = re.sub(
        r"(?im)^(my name is|i am|i'm)\s+[\w\s]+[,.][^\n]*",
        "", intro,
    )
    text = (intro + rest).strip()

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _transcript_quality_flags(text: str) -> list[str]:
    sample = str(text or "")
    flags: list[str] = []
    if not sample:
        return ["empty"]

    if re.search(r"[^\x00-\x7F\u0900-\u097F\u03B1-\u03C9\u0391-\u03A9]{8,}", sample):
        flags.append("garbled_unicode")
    if sample.count("```") or sample.count("$") >= 4:
        flags.append("formatting_artifacts")
    if re.search(r"(?i)\bXB\s*=\s*1\s+XA\b", sample):
        flags.append("broken_formula")
    if re.search(r"(?i)\bpi\s*=\s*[^\n]{0,20}[^\x00-\x7F]{2,}", sample):
        flags.append("corrupted_equation")
    if re.search(r"(?i)mole fraction of b is also 0\b", sample):
        flags.append("contradictory_statement")
    return flags


def _repair_low_quality_transcript(text: str, topic_id: str, language: str, institute_id: str, flags: list[str]) -> str:
    cleaned = _clean_transcript_text(text)
    if not flags:
        return cleaned

    try:
        llm_result = get_llm().complete(
            system_prompt=(
                "You repair noisy educational lecture transcripts. Clean OCR/STT artifacts, remove garbage tokens, "
                "repair obvious equation formatting, and rewrite broken statements into clear English only when the "
                "intended meaning is scientifically obvious from context. Do not invent new topics."
            ),
            user_prompt=(
                f"Lecture topic: {topic_id or 'General'}\n"
                f"Source language: {language}\n"
                f"Detected issues: {', '.join(flags)}\n\n"
                "Clean and repair this transcript for note generation. Preserve as much original meaning as possible.\n\n"
                f"{cleaned}"
            ),
            model="edvaqwen",
            temperature=0.2,
            max_tokens=4096,
            json_mode=False,
            institute_id=institute_id,
        )
        candidate = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
        return candidate.strip() or cleaned
    except Exception as exc:
        logger.warning("Transcript repair failed (%s)", exc)
        return cleaned


def _normalize_transcript_to_english(transcript: str, language: str, institute_id: str) -> str:
    # Skip Sarvam translation entirely — the LLM (llama-3.3-70b) reads Hindi/Hinglish natively
    # and translates in full chunk context during note generation, which is far more accurate than
    # Sarvam's blind 900-char batches that mangle technical terms and lose sentence context.
    return str(transcript or "").strip()


def _prepare_transcript_for_notes(transcript: str, topic_id: str, language: str, institute_id: str) -> tuple[str, dict]:
    normalized = _normalize_transcript_to_english(transcript, language, institute_id)
    cleaned = _clean_transcript_text(normalized)
    cleaned = _strip_lecture_framing(cleaned)
    flags = _transcript_quality_flags(cleaned)
    repaired = _repair_low_quality_transcript(cleaned, topic_id, language, institute_id, flags) if flags else cleaned
    final_text = _clean_transcript_text(repaired)
    final_text = _restore_sentence_punctuation(final_text)
    return final_text, {
        "quality_flags": flags,
        "repair_applied": bool(flags),
    }


NOTES_CHUNK_CHAR_LIMIT = 7500
NOTES_CHUNK_OVERLAP_CHARS = 800
NOTES_SECTION_MAX_TOKENS = 2600
NOTES_MERGE_MAX_TOKENS = 4096


def _chunk_transcript(text: str, chunk_size: int = NOTES_CHUNK_CHAR_LIMIT, overlap: int = NOTES_CHUNK_OVERLAP_CHARS) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    current = ""
                    break
                start = max(end - overlap, start + 1)

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[max(chunk_size - overlap, 1):].strip()

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _generate_chunk_notes(chunk_text: str, topic_id: str, language: str, institute_id: str, chunk_index: int, total_chunks: int) -> str:
    is_hindi = str(language or "").lower() in ("hi", "hinglish", "hi-in")
    lang_instruction = (
        "The transcript is in Hindi or Hinglish (Hindi+English mix). "
        "READ the Hindi/Hinglish content, understand it fully, and write the notes in clear English. "
        "Translate technical terms accurately (e.g. रासायनिक बंध = chemical bond, "
        "आयनिक यौगिक = ionic compound, कक्षक = orbital). "
        "Do NOT transliterate — write proper English notes from the Hindi content.\n"
    ) if is_hindi else ""

    llm_result = get_llm().complete(
        system_prompt=(
            "You are an expert academic note-taker creating textbook-like lecture notes in English. "
            "Convert this section of a lecture transcript into rich, detailed, classroom-quality Markdown notes. "
            + lang_instruction +
            "SKIP any teacher introductions, greetings, self-introductions, roll calls, or administrative "
            "announcements (e.g. 'Hello students', 'Mere pyare bacchon', 'My name is...', 'Exams are near...'). "
            "Focus ONLY on academic and educational content: concepts, theory, formulas, examples. "
            "Preserve definitions, intuition, examples, formulas, derivations, steps, caveats, comparisons, "
            "and teacher reasoning. Do not compress aggressively. Cover every important idea in this chunk."
        ),
        user_prompt=(
            f"Lecture topic: {topic_id or 'General'}\n"
            f"Source language: {language}\n"
            f"Transcript chunk: {chunk_index} of {total_chunks}\n\n"
            "Write detailed Markdown notes for this chunk only.\n"
            "Requirements:\n"
            "- Explain concepts in a textbook-like way, not just bullets.\n"
            "- Include short definitions for important terms.\n"
            "- Include formulas or equations in plain text when relevant.\n"
            "- Preserve examples, teacher explanations, step-by-step reasoning, and cause-effect relationships.\n"
            "- Add subheadings where useful.\n"
            "- If the teacher contrasts two ideas, keep that comparison.\n"
            "- If the teacher mentions mistakes, traps, exceptions, or exam-important points, preserve them.\n"
            "- Do not add unrelated content not supported by the transcript.\n\n"
            f"{chunk_text}"
        ),
        model="edvaqwen",
        temperature=0.4,
        max_tokens=NOTES_SECTION_MAX_TOKENS,
        json_mode=False,
        institute_id=institute_id,
    )
    return llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])


def _merge_chunk_notes(chunk_notes: list[str], topic_id: str, language: str, institute_id: str) -> str:
    combined_sections = "\n\n".join(
        f"--- SECTION {idx + 1} ---\n{section.strip()}"
        for idx, section in enumerate(chunk_notes)
        if str(section).strip()
    ).strip()
    if not combined_sections:
        return ""

    llm_result = get_llm().complete(
        system_prompt=(
            "You are an expert academic editor creating final textbook-like lecture notes in English. "
            "Merge multiple chunk-level note sections into one comprehensive, coherent Markdown document. "
            "Remove any remaining teacher greetings, introductions, or non-academic content if present. "
            "Preserve coverage, remove duplication, improve structure, and keep the final notes rich and detailed. "
            "Do not shorten aggressively or flatten explanations into overly brief bullets."
        ),
        user_prompt=(
            f"Lecture topic: {topic_id or 'General'}\n"
            f"Source language: {language}\n\n"
            "Merge these section notes into one coherent Markdown note set.\n"
            "Requirements:\n"
            "- Start with a strong title and then organize into logical sections.\n"
            "- Keep all major concepts, examples, formulas, and explanations.\n"
            "- Prefer explanatory paragraphs plus bullets where helpful.\n"
            "- Preserve continuity between chunks so the final notes read like one lecture, not stitched fragments.\n"
            "- Include key distinctions, common mistakes, and exam-relevant insights when present.\n"
            "- End with a concise Summary section.\n\n"
            f"{combined_sections}"
        ),
        model="edvaqwen",
        temperature=0.3,
        max_tokens=NOTES_MERGE_MAX_TOKENS,
        json_mode=False,
        institute_id=institute_id,
    )
    return llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])


def _generate_comprehensive_notes(transcript: str, topic_id: str, language: str, institute_id: str) -> tuple[str, dict]:
    chunks = _chunk_transcript(transcript)
    if not chunks:
        return "", {"chunk_count": 0}

    if len(chunks) == 1:
        notes = _generate_chunk_notes(chunks[0], topic_id, language, institute_id, 1, 1).strip()
        return notes, {"chunk_count": 1, "merge_applied": False}

    logger.info("Generating chunked notes | chunks=%d | topic=%s | lang=%s", len(chunks), topic_id, language)
    partial_notes: list[str] = []
    for idx, chunk in enumerate(chunks):
        partial_notes.append(_generate_chunk_notes(chunk, topic_id, language, institute_id, idx + 1, len(chunks)).strip())

    merged = _merge_chunk_notes(partial_notes, topic_id, language, institute_id).strip()
    return merged, {"chunk_count": len(chunks), "merge_applied": True}


def _looks_like_unstructured_notes(notes: str) -> bool:
    text = str(notes or "").strip()
    if not text:
        return True
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    has_markdown_headings = bool(re.search(r"(?m)^#{1,4}\s+\S+", text))
    very_long_first_line = len(first_line) > 120
    title_runon = bool(re.search(r"(?i)^[A-Z][A-Za-z0-9 ,()'/-]{8,} hello\b", first_line))
    return (not has_markdown_headings) or very_long_first_line or title_runon


def _polish_notes_markdown(notes: str, topic_id: str, language: str, institute_id: str) -> tuple[str, bool]:
    cleaned = str(notes or "").strip()
    if not cleaned:
        return cleaned, False

    if not _looks_like_unstructured_notes(cleaned):
        return cleaned, False

    try:
        llm_result = get_llm().complete(
            system_prompt=(
                "You are an expert academic editor. Rewrite the provided lecture notes into clean, well-structured "
                "Markdown without changing the academic meaning. Enforce a proper title, section headings, subheadings, "
                "lists where appropriate, and a final Summary section."
            ),
            user_prompt=(
                f"Lecture topic: {topic_id or 'General'}\n"
                f"Source language: {language}\n\n"
                "Rewrite these notes into clean Markdown. Requirements:\n"
                "- Start with `# Title`\n"
                "- Use `##` for main sections\n"
                "- Break long run-on paragraphs into readable sections\n"
                "- Preserve content, formulas, and examples\n"
                "- Do not add unrelated information\n\n"
                f"{cleaned}"
            ),
            model="edvaqwen",
            temperature=0.2,
            max_tokens=4096,
            json_mode=False,
            institute_id=institute_id,
        )
        polished = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
        polished = polished.strip()
        return polished or cleaned, True
    except Exception as exc:
        logger.warning("Notes markdown polish failed (%s)", exc)
        return cleaned, False


# â”€â”€ AI #1 â€” Doubt Clearing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def resolve_doubt(request):
    data = request.data
    question_text = (data.get("questionText") or "").strip()
    question_image_url = (data.get("questionImageUrl") or "").strip()
    image_description = ""

    if question_image_url:
        # Vision API understands diagrams, equations, handwriting — use it first
        image_description = _describe_image_with_vision(question_image_url)
        if not image_description:
            # Fallback to EasyOCR if vision API is unavailable
            logger.warning("Vision API returned empty for doubt image — falling back to OCR")
            image_description = _extract_text_from_image_url(question_image_url)

    if not question_text and not image_description:
        return Response({"error": "Missing questionText or readable questionImageUrl"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("doubt_resolve")

    if image_description and question_text:
        combined_question = f"{question_text}\n\n[Image content]\n{image_description}"
    elif image_description:
        combined_question = f"[Student uploaded an image with the following content]\n{image_description}"
    else:
        combined_question = question_text

    user_prompt = template.user_template.format(
        question_text=combined_question,
        topic_id=data.get("topicId", "general"),
        mode=data.get("mode", "detailed"),
        student_context=json.dumps(data.get("studentContext", {})),
    )

    # Plain text mode â€” edvaqwen returns high-quality text but not reliable JSON
    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=get_model_for_task("doubt_resolve"),
            temperature=0.3,
            max_tokens=1024,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)

    raw_text = result["content"]
    if isinstance(raw_text, dict):
        explanation_text = raw_text.get("explanation", str(raw_text))
    else:
        explanation_text = str(raw_text).strip()

    return JsonResponse({
        "explanation": explanation_text,
        "conceptLinks": [],
        "related_topics": [],
        "difficulty_level": "medium",
        "follow_up_questions": [],
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "institute": institute_id,
        },
    })


<<<<<<< HEAD
_DOUBT_VISION_PROMPT = (
    "A student has uploaded this image as their doubt/question in an educational app. "
    "Extract and describe ALL content from the image completely: "
    "any text, questions, mathematical equations, chemical formulas, diagrams, "
    "graphs, figures, or numerical problems. "
    "Be thorough and precise. Write equations in readable plain text (e.g. x^2 + 3x = 0)."
)

_GRADING_VISION_PROMPT = (
    "The image is a student's handwritten or photographed answer for an exam or mock test. "
    "Transcribe ONLY the answer they wrote — definitions, steps, equations, and labels. "
    "Output plain text only, with line breaks only where the student's answer has separate lines. "
    "Do NOT describe the photograph, the page, notebook, desk, or background. "
    "Do NOT mention ink color, paper, margins, or whether something is 'lined paper'. "
    "Ignore date headers, calendar widgets, week numbers, and other UI unless they are clearly part of the student's written answer. "
    "Do NOT use phrases like 'The image shows', 'The note is written', or 'There are no equations'. "
    "If a word is unclear, use [illegible] for that part. If there is no readable answer, output exactly: (no readable answer)"
)


def _vision_text_from_image(image_url: str, user_prompt: str) -> str:
    """Groq Llama 4 Scout vision: shared helper for doubt vs grading prompts."""
=======
_groq_vision_clients: dict = {}

def _describe_image_with_vision(image_url: str) -> str:
    """Use Groq vision model to extract content from an image."""
>>>>>>> d7bdc312c9076cbf92a7df5ed868163c3b3824c1
    try:
        from groq import Groq
    except ImportError:
        return ""

    keys = get_rotated_groq_keys()
    if not keys:
        return ""

    for api_key in keys:
        try:
            if api_key not in _groq_vision_clients:
                _groq_vision_clients[api_key] = Groq(api_key=api_key, timeout=25.0)
            client = _groq_vision_clients[api_key]
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
<<<<<<< HEAD
                            {"type": "text", "text": user_prompt},
=======
                            {
                                "type": "text",
                                "text": (
                                    "Extract all content from this educational image: "
                                    "text, questions, equations (plain text, e.g. x^2+3x=0), "
                                    "diagrams, and numerical problems. Be concise and precise."
                                ),
                            },
>>>>>>> d7bdc312c9076cbf92a7df5ed868163c3b3824c1
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Vision API key failed for image OCR: %s", exc)
            continue

    return ""


def _describe_image_with_vision(image_url: str) -> str:
    """Doubt / general: extract and describe full content (equations, diagrams, etc.)."""
    return _vision_text_from_image(image_url, _DOUBT_VISION_PROMPT)


def _transcribe_exam_answer_with_vision(image_url: str) -> str:
    """Mock test / grading: answer text only, no photo narration."""
    return _vision_text_from_image(image_url, _GRADING_VISION_PROMPT)


def _extract_text_from_image_url(image_url: str) -> str:
    """EasyOCR fallback — used only when vision API is unavailable."""
    try:
        import numpy as _np
        from PIL import Image as _Image
        import easyocr as _easyocr
        from io import BytesIO as _BytesIO
    except Exception:
        return ""

    try:
        resp = _requests.get(image_url, timeout=20)
        if resp.status_code != 200:
            return ""
        img = _Image.open(_BytesIO(resp.content)).convert("RGB")
        arr = _np.array(img)
        gray = _np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(_np.uint8)
        bw = _np.where(gray > 165, 255, 0).astype(_np.uint8)
        p5, p95 = _np.percentile(gray, (5, 95))
        if p95 > p5:
            stretch = _np.clip((gray - p5) * (255.0 / (p95 - p5)), 0, 255).astype(_np.uint8)
        else:
            stretch = gray
        reader = _easyocr.Reader(["en", "hi"], gpu=False)
        best = ""
        for v in [arr, gray, stretch, bw]:
            try:
                parts = reader.readtext(v, detail=0, paragraph=True)
                text = " ".join([str(x).strip() for x in parts if str(x).strip()]).strip()
                if len(text) > len(best):
                    best = text
            except Exception:
                continue
        return best
    except Exception:
        return ""


@api_view(["POST"])
def ocr_doubt_image(request):
    """Transcribe handwritten / diagram content for grading and doubt flows.
    Prefers Groq **Llama 4 Scout** vision (handwriting, equations, diagrams), then EasyOCR.

    Request JSON:
      - imageUrl (required)
      - purpose: optional. ``grading`` = short transcription for mock-test answers (no 'the image shows…');
        omit or ``doubt`` = fuller extraction for doubt resolution (default).
    """
    image_url = (request.data.get("imageUrl") or "").strip()
    if not image_url:
        return Response({"error": "Missing imageUrl"}, status=400)
    purpose = (request.data.get("purpose") or "doubt").strip().lower()
    if purpose in ("grading", "mock", "assessment", "mock_test", "answer"):
        text = _transcribe_exam_answer_with_vision(image_url)
    else:
        text = _describe_image_with_vision(image_url)
    if not text:
        text = _extract_text_from_image_url(image_url)
    return JsonResponse({"text": text or ""})


# â”€â”€ AI #2 â€” AI Tutor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def start_tutor_session(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    context = data.get("context", "")

    # When a rich lesson-generation prompt is provided (long context), use it as the
    # system prompt directly so the LLM produces clean Markdown â€” not JSON-wrapped text.
    if len(context) > 300:
        system_prompt = context
        user_prompt = "Generate the complete lesson now. Write everything in full â€” do not truncate or use placeholders."
    else:
        template = get_template("tutor_session")
        system_prompt = template.system
        user_prompt = template.user_template.format(
            student_id=student_id,
            topic_id=data.get("topicId", "general"),
            context=context,
        )

    try:
        result = get_llm().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=get_model_for_task("tutor_session"),
            temperature=0.3,
            max_tokens=8192,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)

    raw_text = result["content"]
    if isinstance(raw_text, dict):
        explanation_text = raw_text.get("response", str(raw_text))
    else:
        explanation_text = str(raw_text).strip()

    return JsonResponse({
        "response": explanation_text,
        "hints": [],
        "concept_check": "",
        "encouragement": "",
        "session_notes": "",
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "institute": institute_id,
        },
    })


@api_view(["POST"])
def continue_tutor_session(request):
    data = request.data
    session_id = data.get("sessionId")
    student_message = data.get("studentMessage")
    if not session_id or not student_message:
        return Response({"error": "Missing sessionId or studentMessage"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("tutor_continue")
    user_prompt = template.user_template.format(
        session_id=session_id,
        student_message=student_message,
    )

    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=get_model_for_task("tutor_continue"),
            temperature=0.3,
            max_tokens=1024,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)

    raw_text = result["content"]
    if isinstance(raw_text, dict):
        explanation_text = raw_text.get("response", str(raw_text))
    else:
        explanation_text = str(raw_text).strip()

    return JsonResponse({
        "response": explanation_text,
        "hints": [],
        "concept_check": "",
        "progress_note": "",
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "institute": institute_id,
        },
    })


# â”€â”€ AI #6 â€” Content Recommendation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def recommend_content(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("content_recommend")
    user_prompt = template.user_template.format(
        student_id=student_id,
        context=data.get("context", "dashboard"),
        weak_topics=json.dumps(data.get("weakTopics", [])),
        recent_performance=json.dumps(data.get("recentPerformance", {})),
    )
    return ai_call_text(request, "content_recommend", user_prompt,
                        wrap_fn=lambda t: {"recommendations": t, "contentItems": []})


# â”€â”€ AI #7 â€” Speech-to-Text Notes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def generate_stt_notes(request):
    data = request.data
    audio_url = data.get("audioUrl")
    if not audio_url:
        return Response({"error": "Missing audioUrl"}, status=400)

    import time as _time
    language = data.get("language", "hi")
    logger.info("generate_stt_notes | audio_url=%s | language=%s", audio_url, language)
    _t0 = _time.perf_counter()

    # Use caller-supplied transcript if provided, otherwise transcribe
    raw_transcript = data.get("transcript", "")
    if not raw_transcript:
        try:
            raw_transcript = _transcribe_audio(audio_url, language)
            logger.info(
                "Transcription done â€” %d chars | took=%.1fs",
                len(raw_transcript), _time.perf_counter() - _t0,
            )
        except Exception as exc:
            logger.error("Transcription FAILED for %s: %s", audio_url, exc)
            return Response(
                {
                    "error": "transcription_failed",
                    "detail": str(exc),
                    "audioUrl": audio_url,
                    "hint": (
                        "Ensure the URL is publicly accessible from the server "
                        "and the file is a supported audio/video format."
                    ),
                },
                status=502,
            )

    if len(raw_transcript.strip()) < 20:
        return Response(
            {
                "error": "transcript_too_short",
                "detail": "Whisper returned almost nothing. The audio may be silent or corrupted.",
            },
            status=422,
        )

    _t1 = _time.perf_counter()
    logger.info("Sending to LLM â€” transcript=%d chars | transcription took=%.1fs", len(raw_transcript), _t1 - _t0)

    english_transcript, prep_meta = _prepare_transcript_for_notes(
        raw_transcript,
        data.get("topicId", ""),
        language,
        getattr(request, "institute_id", "default"),
    )

    institute_id = getattr(request, "institute_id", "default")
    notes_markdown, notes_meta = _generate_comprehensive_notes(
        english_transcript,
        data.get("topicId", ""),
        language,
        institute_id,
    )
    notes_markdown, markdown_polished = _polish_notes_markdown(
        notes_markdown,
        data.get("topicId", ""),
        language,
        institute_id,
    )
    logger.info(
        "STT notes generated | %d chars | chunks=%d",
        len(notes_markdown),
        notes_meta.get("chunk_count", 0),
    )

    return JsonResponse({
        "notes": notes_markdown,
        "rawTranscript": raw_transcript,
        "englishTranscript": english_transcript,
        "keyConcepts": [],
        "formulas": [],
        "summary": "",
        "_meta": {
            "source": "llm",
            "model": "edvaqwen",
            "latency_ms": 0,
            "tokens": 0,
            "institute": institute_id,
            "chunk_count": notes_meta.get("chunk_count", 0),
            "merge_applied": notes_meta.get("merge_applied", False),
            "markdown_polished": markdown_polished,
            "quality_flags": prep_meta.get("quality_flags", []),
            "repair_applied": prep_meta.get("repair_applied", False),
        },
    })


# â”€â”€ AI #8 â€” Student Feedback Engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def generate_feedback(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("feedback_generate")
    user_prompt = template.user_template.format(
        student_id=student_id,
        context=data.get("context", "post_test"),
        data_json=json.dumps(data.get("data", {})),
    )
    return ai_call_text(request, "feedback_generate", user_prompt,
                        wrap_fn=lambda t: {"feedbackText": t, "actionItems": [], "strengths": []})


# â”€â”€ AI #9 â€” Notes Weak Topic Identifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def analyze_notes(request):
    data = request.data
    student_id = data.get("studentId")
    notes_content = data.get("notesContent")
    if not student_id or not notes_content:
        return Response({"error": "Missing studentId or notesContent"}, status=400)

    template = get_template("notes_analyze")
    user_prompt = template.user_template.format(
        student_id=student_id,
        topic_id=data.get("topicId", ""),
        notes_content=notes_content,
    )
    return ai_call_text(request, "notes_analyze", user_prompt,
                        wrap_fn=lambda t: {"quality_score": 7, "weak_topics": [], "analysis": t, "suggestions": []})


# â”€â”€ AI #10 â€” Resume Analyzer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def analyze_resume(request):
    data = request.data
    resume_text = data.get("resumeText")
    if not resume_text:
        return Response({"error": "Missing resumeText"}, status=400)

    template = get_template("resume_analyze")
    user_prompt = template.user_template.format(
        resume_text=resume_text,
        target_role=data.get("targetRole", "Software Engineer"),
    )
    return ai_call_text(request, "resume_analyze", user_prompt,
                        wrap_fn=lambda t: {"score": 0, "strengths": [], "improvements": [], "feedback": t})


# â”€â”€ AI #11 â€” Interview Prep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def start_interview_prep(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("interview_prep")
    user_prompt = template.user_template.format(
        student_id=student_id,
        target_college=data.get("targetCollege", "IIT"),
    )
    return ai_call_text(request, "interview_prep", user_prompt,
                        wrap_fn=lambda t: {"questions": [t], "tips": [], "resources": []})


# â”€â”€ AI #12 â€” Personalized Learning Plan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@api_view(["POST"])
def generate_plan(request):
    import datetime
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    # Use IST date so it matches what the NestJS backend queries for "today"
    ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    today_date = datetime.datetime.now(ist_offset).strftime("%Y-%m-%d")

    academic_calendar = data.get("academicCalendar", {})
    assigned_subjects = academic_calendar.get("assignedSubjects", [])

    template = get_template("plan_generate")
    user_prompt = template.user_template.format(
        student_id=student_id,
        exam_target=data.get("examTarget", "jee"),
        exam_year=data.get("examYear", "2026"),
        daily_hours=data.get("dailyHours", 4),
        assigned_subjects=", ".join(assigned_subjects) if assigned_subjects else "all subjects",
        weak_topics=json.dumps(data.get("weakTopics", [])),
        target_college=data.get("targetCollege", ""),
        today_date=today_date,
        academic_calendar=json.dumps(academic_calendar),
    )
    return ai_call(request, "plan_generate", user_prompt)


@api_view(["POST"])
def generate_syllabus(request):
    data = request.data
    subjects = data.get("subjects", [])
    if not isinstance(subjects, list) or not any(str(subject).strip() for subject in subjects):
        return Response({"error": "Missing subjects"}, status=400)

    cleaned_subjects = [str(subject).strip() for subject in subjects if str(subject).strip()]
    template = get_template("syllabus_generate")
    user_prompt = template.user_template.format(
        exam_target=data.get("examTarget", "jee"),
        exam_year=data.get("examYear", "2026"),
        subjects=", ".join(cleaned_subjects),
    )
    return ai_call(request, "syllabus_generate", user_prompt, temperature=0.3, max_tokens=4096)


# â”€â”€ AI #13 â€” In-Video Quiz Generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_quiz_json(raw: str) -> dict:
    """Extract the JSON questions array from a potentially markdown-wrapped LLM response."""
    stripped = raw.strip()
    # Strip markdown code fences if present
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    def _loads(block: str):
        block = block.strip()
        if not block:
            return None
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            return None

    parsed = _loads(stripped)
    # LLM often adds preamble; try innermost [...] JSON array
    if parsed is None and "[" in stripped and "]" in stripped:
        l = stripped.find("[")
        r = stripped.rfind("]")
        if l != -1 and r > l:
            parsed = _loads(stripped[l : r + 1])

    if isinstance(parsed, list):
        return {"questions": parsed}
    if isinstance(parsed, dict):
        if isinstance(parsed.get("questions"), list):
            return parsed
        if isinstance(parsed.get("data"), list):
            return {"questions": parsed["data"]}
    return {"questions": []}


@api_view(["POST"])
def generate_quiz_questions(request):
    data = request.data
    transcript = data.get("transcript", "")
    if not transcript:
        return Response({"error": "Missing transcript"}, status=400)
    if len(transcript.strip()) < 50:
        return Response({"error": "Transcript too short to generate quiz questions"}, status=422)

    # Groq TPM limits â€” cap transcript at ~8 000 chars (~2 000 tokens) to stay within budget
    MAX_TRANSCRIPT_CHARS = 8000
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]
        logger.info("Quiz transcript truncated to %d chars", MAX_TRANSCRIPT_CHARS)

    template = get_template("quiz_generate")
    user_prompt = template.user_template.format(
        lecture_title=data.get("lectureTitle", "Lecture"),
        topic_id=data.get("topicId", ""),
        transcript=transcript,
    )

    institute_id = getattr(request, "institute_id", "default")
    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model="quiz",
            temperature=0.3,
            max_tokens=1200,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        logger.error("Quiz generation failed (institute=%s): %s", institute_id, e)
        return Response({"error": str(e)}, status=502)

    raw = result["content"] if isinstance(result["content"], str) else str(result["content"])
    parsed = _parse_quiz_json(raw)

    return Response({
        **parsed,
        "_meta": {
            "source": "llm",
            "model": result["model"],
            "latency_ms": round(result["latency_ms"]),
            "institute": institute_id,
        },
    })


# â”€â”€ AI #15 â€” Text Translation (Sarvam AI) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Uses Sarvam's mayura:v1 model â€” purpose-built for Indian language translation.
# Replaces the previous Groq LLM approach which had poor Indic language quality.
#
# Supports: hi, en, bn, te, mr, ta, gu, kn, ml, pa, od
# Falls back to a 502 with a clear error if SARVAM_API_KEY is not set.

@api_view(["POST"])
def translate_text(request):
    import time as _time
    from ai_services.core.sarvam_client import translate as sarvam_translate

    data = request.data
    text            = data.get("text", "")
    target_language = data.get("targetLanguage", "en")

    if not text:
        return Response({"error": "Missing text"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    logger.info(
        "translate_text (Sarvam) | target=%s | chars=%d | institute=%s",
        target_language, len(text), institute_id,
    )

    _t0 = _time.perf_counter()
    try:
        translated = sarvam_translate(text, target_language=target_language)
    except RuntimeError as exc:
        logger.error("Sarvam translation failed: %s", exc)
        return Response({"error": str(exc)}, status=502)

    latency_ms = (_time.perf_counter() - _t0) * 1000
    logger.info(
        "Sarvam translation done | %d â†’ %d chars | %.0fms",
        len(text), len(translated), latency_ms,
    )

    return Response({"translatedText": translated})


# â”€â”€ AI #16 â€” Topic Content Generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CONTENT_TYPE_PROMPTS = {
    "lesson": (
        "Generate comprehensive, well-structured lesson notes in Markdown. "
        "Include: Introduction, Key Concepts (with sub-sections), Important Points, Examples, and Common Mistakes."
    ),
    "formula": (
        "Generate a structured list of ALL key formulas for this topic in Markdown. "
        "For each formula: write it clearly, name every variable, give a one-line use-case hint. "
        "Group formulas by sub-topic. Use LaTeX notation where appropriate (e.g. $F = ma$)."
    ),
    "summary": (
        "Generate a crisp, exam-ready summary of this topic in Markdown. "
        "Use bullet points and short paragraphs. Cover every exam-important concept."
    ),
    "mindmap": (
        "Generate a hierarchical mind-map outline in Markdown. "
        "Use # for the main topic, ## for main branches, ### for sub-branches, and - for leaf nodes. "
        "Cover all sub-topics and their key points."
    ),
    "flashcard": (
        "Generate 12â€“15 flashcard pairs for this topic in Markdown. "
        "Format each as: **Q:** <question>  **A:** <answer>. "
        "Cover definitions, formulas, mechanisms, and application questions."
    ),
    "checklist": (
        "Generate a revision checklist for this topic in Markdown. "
        "Group items by sub-topic. Use - [ ] for each checkbox item. "
        "Include concepts to understand, formulas to memorise, and types of problems to practice."
    ),
    # â”€â”€ same as lesson/summary but with short label aliases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "study_guide":         "Generate a crisp, exam-ready summary of this topic in Markdown. Use bullet points and short paragraphs. Cover every exam-important concept.",
    "key_concepts":        "Generate a structured list of ALL key formulas and must-know concepts for this topic in Markdown. For each: name, definition, units (if applicable), one-line use-case.",
    "practice_questions":  (
        "Generate a Daily Practice Problem (DPP) set for this topic in Markdown. "
        "Include exactly 10 questions with a mix of difficulty (3 easy, 5 medium, 2 hard). "
        "For each question:\n"
        "- Number it (Q1, Q2 â€¦)\n"
        "- Write the question clearly (MCQ with 4 options labelled Aâ€“D, or numerical/short-answer)\n"
        "- After all questions, add a ## Answers section with: answer letter/value and a 2-3 line explanation for each.\n"
        "Ensure questions test understanding, not just recall."
    ),
    # â”€â”€ DPP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "dpp": (
        "Generate a high-quality Daily Practice Problem (DPP) sheet for this topic in Markdown, "
        "exactly as a top coaching institute would give students.\n\n"
        "Format:\n"
        "# DPP â€” {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name} | **Date:** ______\n\n"
        "## Section A â€” Multiple Choice (1 mark each)\n"
        "Generate 8 MCQ questions, each with 4 options (Aâ€“D). Mix easy and medium difficulty.\n\n"
        "## Section B â€” Assertionâ€“Reason (1 mark each)\n"
        "Generate 3 assertion-reason type questions.\n\n"
        "## Section C â€” Numericals / Short Answer (3 marks each)\n"
        "Generate 4 numerical or short-answer problems.\n\n"
        "## Answer Key\n"
        "List all correct answers and brief hints/solutions.\n\n"
        "Questions must be syllabus-aligned, conceptually varied, and gradually increasing in difficulty."
    ),
    # â”€â”€ PYQ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "pyq": (
        "Generate a Previous Year Question (PYQ) style practice set for this topic in Markdown. "
        "Simulate the style of JEE Main / NEET questions from 2018â€“2024.\n\n"
        "Format:\n"
        "# PYQ Practice Set â€” {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name}\n\n"
        "## JEE Main Style Questions\n"
        "Generate 6 questions in JEE Main MCQ style (single correct, 4 options Aâ€“D). "
        "Note the exam year pattern each question is modelled on (e.g. 'Pattern: JEE Main 2022 Jan').\n\n"
        "## NEET Style Questions\n"
        "Generate 5 questions in NEET MCQ style (single correct, 4 options).\n\n"
        "## Integer Type (JEE Advanced Style)\n"
        "Generate 3 integer-type questions where the answer is a non-negative integer.\n\n"
        "## Detailed Solutions\n"
        "Provide full step-by-step solutions for every question.\n\n"
        "Questions must be authentic in difficulty and style. Avoid trivial or textbook-definition questions."
    ),
}

_DIFFICULTY_DESC = {
    "basic":        "introductory level, simple language, suitable for beginners",
    "intermediate": "standard curriculum depth, JEE/NEET Mains level",
    "advanced":     "advanced level, JEE Advanced / NEET PG competitive exam depth",
}

_LENGTH_WORDS = {
    "brief":    "~300 words",
    "standard": "~800 words",
    "detailed": "~1500 words",
}


@api_view(["POST"])
def generate_topic_content(request):
    data = request.data
    topic_name    = data.get("topicName", "").strip()
    subject_name  = data.get("subjectName", "").strip()
    chapter_name  = data.get("chapterName", "").strip()
    content_type  = data.get("contentType", "lesson")
    difficulty    = data.get("difficulty", "intermediate")
    length        = data.get("length", "standard")
    extra_context = data.get("extraContext", "").strip()

    if not topic_name:
        return Response({"error": "Missing topicName"}, status=400)

    type_instruction = _CONTENT_TYPE_PROMPTS.get(
        content_type,
        _CONTENT_TYPE_PROMPTS["lesson"],
    ).replace("{topic_name}", topic_name).replace("{subject_name}", subject_name).replace("{chapter_name}", chapter_name)
    diff_desc  = _DIFFICULTY_DESC.get(difficulty, _DIFFICULTY_DESC["intermediate"])
    word_limit = _LENGTH_WORDS.get(length, _LENGTH_WORDS["standard"])

    system_prompt = (
        "You are an expert educational content creator specialising in Indian competitive exam preparation "
        "(JEE, NEET, CBSE, ICSE). You write accurate, engaging, curriculum-aligned educational content in Markdown."
    )

    user_prompt = (
        f"Subject: {subject_name}\n"
        f"Chapter: {chapter_name}\n"
        f"Topic: {topic_name}\n"
        f"Content type: {content_type}\n"
        f"Difficulty: {diff_desc}\n"
        f"Target length: {word_limit}\n"
    )
    if extra_context:
        user_prompt += f"Additional instructions: {extra_context}\n"
    user_prompt += (
        f"\n{type_instruction}\n\n"
        "Return ONLY the Markdown content â€” no preamble, no 'Here is your content:' prefix."
    )

    institute_id = getattr(request, "institute_id", "default")
    try:
        llm_result = get_llm().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=4096,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return Response({"error": str(e)}, status=502)

    content = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
    return Response({
        "content": content,
        "contentType": content_type,
        "topicName": topic_name,
        "_meta": {
            "model": llm_result.get("model", ""),
            "latency_ms": round(llm_result.get("latency_ms", 0)),
        },
    })


# â”€â”€ AI #7b â€” Notes from pre-existing Transcript (YouTube / manual) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Called by NestJS when the lecture videoUrl is a YouTube link.
# The NestJS backend fetches the captions via youtube-transcript and sends
# the plain-text transcript here â€” we skip Whisper entirely and go straight
# to LLM summarisation.
#
# Body:   { transcript: str, topicId: str, language: str }
# Returns the same shape as /stt/notes so NestJS needs no extra parsing.

@api_view(["POST"])
def generate_notes_from_transcript(request):
    import time as _time

    data = request.data
    transcript = data.get("transcript", "").strip()

    if not transcript:
        return Response({"error": "Missing transcript"}, status=400)

    if len(transcript) < 20:
        return Response(
            {
                "error": "transcript_too_short",
                "detail": "The transcript is too short to generate meaningful notes.",
            },
            status=422,
        )

    language = data.get("language", "en")
    topic_id = data.get("topicId", "")
    institute_id = getattr(request, "institute_id", "default")

    logger.info(
        "generate_notes_from_transcript | topic=%s | lang=%s | chars=%d | institute=%s",
        topic_id, language, len(transcript), institute_id,
    )

    _t0 = _time.perf_counter()

    english_transcript, prep_meta = _prepare_transcript_for_notes(
        transcript,
        topic_id,
        language,
        institute_id,
    )

    try:
        notes_markdown, notes_meta = _generate_comprehensive_notes(
            english_transcript,
            topic_id,
            language,
            institute_id,
        )
        notes_markdown, markdown_polished = _polish_notes_markdown(
            notes_markdown,
            topic_id,
            language,
            institute_id,
        )
    except RuntimeError as exc:
        logger.error("notes_from_transcript LLM failed (institute=%s): %s", institute_id, exc)
        return Response({"error": str(exc)}, status=502)

    logger.info(
        "notes_from_transcript done | %d chars notes | chunks=%d | took=%.1fs",
        len(notes_markdown),
        notes_meta.get("chunk_count", 0),
        _time.perf_counter() - _t0,
    )

    # Return same shape as /stt/notes so NestJS content.service.ts needs zero changes
    return Response({
        "notes": notes_markdown,
        "rawTranscript": transcript,
        "englishTranscript": english_transcript,
        "keyConcepts": [],
        "formulas": [],
        "summary": "",
        "_meta": {
            "source": "youtube_transcript",
            "model": "edvaqwen",
            "latency_ms": 0,
            "transcript_chars": len(transcript),
            "institute": institute_id,
            "chunk_count": notes_meta.get("chunk_count", 0),
            "merge_applied": notes_meta.get("merge_applied", False),
            "markdown_polished": markdown_polished,
            "quality_flags": prep_meta.get("quality_flags", []),
            "repair_applied": prep_meta.get("repair_applied", False),
        },
    })
