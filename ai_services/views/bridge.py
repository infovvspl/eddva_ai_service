"""
Views for NestJS ai-bridge endpoints.
These endpoints match the paths called by apexiq-backend/src/modules/ai-bridge/ai-bridge.service.ts

Active endpoints:
  POST /doubt/resolve          â†' AI #1: Doubt Clearing
  POST /tutor/session          â†' AI #2: AI Tutor Start
  POST /tutor/continue         â†' AI #2: AI Tutor Continue
  POST /recommend/content      â†' AI #6: Content Recommendation
  POST /stt/notes              â†' AI #7: Speech-to-Text Notes  (Whisper â†' LLM)
  POST /stt/notes-from-text    â†' AI #7b: Notes from Transcript (YouTube captions â†' LLM, no Whisper)
  POST /feedback/generate      â†' AI #8: Student Feedback
  POST /notes/analyze          â†' AI #9: Notes Weak Topic Identifier
  POST /resume/analyze         â†' AI #10: Resume Analyzer
  POST /interview/start        â†' AI #11: Interview Prep
  POST /plan/generate          â†' AI #12: Personalized Learning Plan
  POST /quiz/generate          â†' AI #13: In-Video Quiz Generator
  POST /translate              â†' AI #15: Text Translation  (Sarvam AI -- mayura:v1)

Removed endpoints (deleted from platform):
  POST /performance/analyze    â†' was AI #3 (performance_analysis)
  POST /grade/subjective       â†' was AI #4 (grade_subjective)
  POST /engage/detect          â†' was AI #5 (engagement_detect)
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
from ai_services.core.groq_keys import get_groq_api_keys, get_rotated_groq_keys, is_key_exhausted_error
from ai_services.core.llm_client import _JSON_MODE_TUTOR_SUFFIX
from .base import ai_call, ai_call_text, get_llm

logger = logging.getLogger("ai_services.llm")

# -- Groq Whisper API (primary -- cloud, fast; multi-key rotation) ---------------

GROQ_API_KEYS: list[str] = get_groq_api_keys()
_GROQ_KEYS_RAW = [
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
    os.getenv("GROQ_API_KEY_8", ""),
    os.getenv("GROQ_API_KEY_9", ""),
    os.getenv("GROQ_API_KEY_10", ""),
    os.getenv("GROQ_API_KEY_11", ""),
    os.getenv("GROQ_API_KEY_12", ""),
    os.getenv("GROQ_API_KEY_13", ""),
]
GROQ_API_KEYS: list[str] = [k for k in _GROQ_KEYS_RAW if k]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # backward compat
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB Groq limit
# Gaps between Whisper segments (seconds) come from the decoded audio; they approximate speech pauses.
# Used only to space/join text — we do not infer facial expression or intonation.
WHISPER_PAUSE_COMMA_S = 0.42
WHISPER_PAUSE_SENTENCE_S = 0.90
# Post-STT LLM pass: validate/fix punctuation + sentence breaks (incl. ! ?) without rewriting wording.
PUNCT_REFINE_CHUNK_CHARS = 6500
PUNCT_REFINE_MIN_WORDS = 8
PUNCT_REFINE_MAX_TOKENS = 8192
WORD_REPAIR_CHUNK_CHARS = 5200
WORD_REPAIR_MIN_WORDS = 20
WORD_REPAIR_MAX_TOKENS = 4096


def _parse_groq_retry_after(error_msg: str, default: float = 65.0) -> float:
    """Parse 'Please try again in 1m46.5s' from a Groq rate-limit error. Returns seconds."""
    m = re.search(r"(\d+)m(\d+\.?\d*)s", error_msg)
    if m:
        return min(int(m.group(1)) * 60 + float(m.group(2)) + 2, 360.0)
    m = re.search(r"(\d+\.?\d*)s", error_msg)
    if m:
        return min(float(m.group(1)) + 2, 360.0)
    return default


def _ends_with_any_punct(s: str) -> bool:
    t = str(s or "").rstrip()
    if not t:
        return True
    return t[-1] in ".!?;:,…।॥"


def _maybe_capitalize_english_letter(t: str) -> str:
    if not t or not t[0].islower() or not t[0].isascii():
        return t
    return t[0].upper() + t[1:]


def _snippet_has_devanagari(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", str(s or "")))


def _acoustic_sentence_boundary_punct(language: str, tail_before: str, next_segment: str) -> str:
    """
    Latin full stop (.) for English; Devanagari danda (।, U+0964 — Hindi sentence stop, often typed as "|")
    for Hindi lectures. Hinglish: danda only when Devanagari appears near the boundary.
    """
    lang = (language or "en").strip().lower()
    if lang in ("hi", "hi-in"):
        return "। "
    tail = tail_before[-120:] if tail_before else ""
    nxt = next_segment[:120] if next_segment else ""
    if lang == "hinglish":
        if _snippet_has_devanagari(tail) or _snippet_has_devanagari(nxt):
            return "। "
    return ". "


def _groq_result_flat_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(getattr(result, "text", "") or "").strip()


def _parse_groq_verbose_transcription_result(result) -> list[dict]:
    """Build [{start, end, text}, ...] from Groq verbose_json ASR (segment timestamps in seconds)."""
    d: dict | None = None
    if result is None:
        return []
    if isinstance(result, str):
        return []
    if isinstance(result, dict):
        d = result
    elif hasattr(result, "model_dump"):
        try:
            d = result.model_dump()
        except Exception:
            d = None
    if d is None:
        try:
            segs = getattr(result, "segments", None) or []
            text = getattr(result, "text", None) or ""
            d = {"segments": list(segs) if segs is not None else [], "text": text}
        except Exception:
            return []

    segs = d.get("segments") or []
    out: list[dict] = []
    for s in segs:
        if isinstance(s, dict):
            txt = str(s.get("text", "")).strip()
            if not txt:
                continue
            out.append(
                {
                    "start": float(s.get("start", 0) or 0),
                    "end": float(s.get("end", 0) or 0),
                    "text": txt,
                }
            )
        else:
            txt = str(getattr(s, "text", "") or "").strip()
            if not txt:
                continue
            out.append(
                {
                    "start": float(getattr(s, "start", 0) or 0),
                    "end": float(getattr(s, "end", 0) or 0),
                    "text": txt,
                }
            )
    return sorted(out, key=lambda x: x["start"])


def _join_timed_transcript_segments(segments: list[dict], language: str = "en") -> str:
    """
    Join STT segment texts using silence gaps between (prev.end, start).
    This uses timing from the same audio the model decoded (i.e. pauses in speech),
    not facial expression or intonation. Sentence glue respects lecture language (en / hi / hinglish).
    """
    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0].get("text", "").strip()

    out = str(segments[0].get("text", "")).strip()
    for i in range(1, len(segments)):
        t = str(segments[i].get("text", "")).strip()
        if not t:
            continue
        prev_end = float(segments[i - 1].get("end", 0) or 0)
        start = float(segments[i].get("start", 0) or 0)
        gap = max(0.0, start - prev_end)

        if _ends_with_any_punct(out):
            out = f"{out} {t}"
        elif gap >= WHISPER_PAUSE_SENTENCE_S:
            punct = _acoustic_sentence_boundary_punct(language, out, t)
            # Sentence-start capitalization applies to Latin text after a full stop, not after danda.
            t2 = _maybe_capitalize_english_letter(t) if punct.startswith(".") else t
            out = f"{out}{punct}{t2}"
        elif gap >= WHISPER_PAUSE_COMMA_S and not out.rstrip().endswith(
            (",", ";", ":", "-", "।", "?", "!", "…"),
        ):
            out = f"{out}, {t}"
        else:
            out = f"{out} {t}"
    return out.strip()


def _transcribe_with_groq_one_key(
    file_bytes: bytes,
    filename: str,
    language: str,
    prev_context: str,
    api_key: str,
) -> str:
    from groq import Groq, RateLimitError as GroqRateLimitError

    groq_language: str | None = None if language in ("hinglish", "auto") else language
    client = Groq(api_key=api_key)

    for use_verbose in (True, False):
        kwargs: dict = dict(
            file=(filename, file_bytes),
            model=GROQ_WHISPER_MODEL,
        )
        if use_verbose:
            kwargs["response_format"] = "verbose_json"
            kwargs["timestamp_granularities"] = ["segment"]
        else:
            kwargs["response_format"] = "text"
        if groq_language:
            kwargs["language"] = groq_language
        if prev_context:
            raw = prev_context.encode("utf-8")[-880:]
            kwargs["prompt"] = raw.decode("utf-8", errors="ignore")

        try:
            result = client.audio.transcriptions.create(**kwargs)
        except Exception as exc:
            if isinstance(exc, GroqRateLimitError):
                raise
            if use_verbose:
                logger.info("Groq verbose_json failed for one key: %s — trying plain text", exc)
                continue
            raise

        if use_verbose:
            segs = _parse_groq_verbose_transcription_result(result)
            if segs:
                return _join_timed_transcript_segments(segs, language)
            flat = _groq_result_flat_text(result)
            if flat:
                return flat
        else:
            t = _groq_result_flat_text(result)
            if t:
                return t

    raise RuntimeError("Groq returned an empty transcript")


def _transcribe_with_groq(audio_path: str, language: str, prev_context: str = "") -> str:
    """Transcribe via Groq Whisper; use segment timestamps to align pauses, then text fallback."""
    try:
        from groq import RateLimitError as GroqRateLimitError
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq", "--quiet"])
        from groq import RateLimitError as GroqRateLimitError

    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY configured -- set at least one in .env")

    file_size = os.path.getsize(audio_path)
    if file_size > GROQ_MAX_FILE_BYTES:
        raise RuntimeError(f"File too large for Groq ({file_size // 1024 // 1024} MB > 25 MB)")

    filename = os.path.basename(audio_path)
    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    last_exc: Exception | None = None
    for round_num in range(2):
        for key_idx, api_key in enumerate(GROQ_API_KEYS):
            try:
                logger.info(
                    "Groq Whisper | key=%d/%d lang=%s (pause-aware segments when supported)",
                    key_idx + 1, len(GROQ_API_KEYS), language,
                )
                return _transcribe_with_groq_one_key(
                    file_bytes, filename, language, prev_context, api_key,
                )
            except GroqRateLimitError as exc:
                last_exc = exc
                logger.info("Groq key %d/%d rate-limited -- rotating to next key", key_idx + 1, len(GROQ_API_KEYS))
        if round_num == 0 and last_exc is not None:
            wait = _parse_groq_retry_after(str(last_exc))
            logger.warning(
                "All %d Groq keys rate-limited -- waiting %.0fs before retry",
                len(GROQ_API_KEYS), wait,
            )
            import time as _time
            _time.sleep(wait)
            last_exc = None

    raise RuntimeError(f"All {len(GROQ_API_KEYS)} Groq keys exhausted: {last_exc}") from last_exc

# â"€â"€ faster-whisper singleton (fallback -- local, CPU) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    seg_list = [
        {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
        for s in segments
        if (s.text or "").strip()
    ]
    transcript = _join_timed_transcript_segments(seg_list, language)
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

        # â"€â"€ Primary: Groq â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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
                
                # ── Parallel Whisper: assign each chunk a dedicated key ──────────
                # Sequential Whisper = N×15s; parallel = ~15s regardless of N.
                # prev_context (cross-chunk prompt) is skipped for speed — Groq
                # Whisper handles Hindi well without it.
                from concurrent.futures import ThreadPoolExecutor, as_completed as _futures_done
                from groq import RateLimitError as _WGroqRateLimit
                import time as _wtime

                def _whisper_one_chunk(args):
                    idx, chunk_file, api_key = args
                    key_num = (GROQ_API_KEYS.index(api_key) + 1) if api_key in GROQ_API_KEYS else 0
                    logger.info(
                        "Groq Whisper | key=%d/%d lang=%s (parallel chunk %d/%d)",
                        key_num, len(GROQ_API_KEYS), language, idx + 1, len(chunks),
                    )
                    with open(chunk_file, "rb") as _f:
                        file_bytes = _f.read()
                    filename = os.path.basename(chunk_file)
                    for attempt in range(2):
                        try:
                            return idx, _transcribe_with_groq_one_key(file_bytes, filename, language, "", api_key)
                        except _WGroqRateLimit as exc:
                            if attempt == 0:
                                wait = _parse_groq_retry_after(str(exc))
                                logger.warning("Whisper key %d rate-limited — waiting %.0fs", key_num, wait)
                                _wtime.sleep(wait)
                                continue
                            raise
                    return idx, ""

                chunk_assignments = [
                    (idx, chunk_file, GROQ_API_KEYS[idx % len(GROQ_API_KEYS)])
                    for idx, chunk_file in enumerate(chunks)
                ]
                transcript_parts = [""] * len(chunks)
                max_parallel = min(len(chunks), len(GROQ_API_KEYS), 8)

                try:
                    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                        futs = {pool.submit(_whisper_one_chunk, a): a[0] for a in chunk_assignments}
                        for fut in _futures_done(futs):
                            idx, text = fut.result()
                            if text:
                                transcript_parts[idx] = text
                except Exception as exc:
                    logger.warning("Parallel Whisper failed (%s) — falling back to sequential", exc)
                    transcript_parts = []
                    prev_ctx = ""
                    for idx, chunk_file in enumerate(chunks):
                        logger.info("Sending chunk %d/%d to Groq (sequential fallback)...", idx + 1, len(chunks))
                        try:
                            text = _transcribe_with_groq(chunk_file, language, prev_context=prev_ctx)
                        except Exception as exc2:
                            logger.warning("Groq chunk %d/%d failed (%s) — skipping", idx + 1, len(chunks), exc2)
                            text = ""
                        if text:
                            transcript_parts.append(text)
                            prev_ctx = text

                transcript = " ".join(t for t in transcript_parts if t).strip()
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
    cleaned = re.sub(r"([.!?।])\s*", r"\1\n", cleaned)
    return cleaned.strip()


def _token_signature(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\u0900-\u097F]+", str(text or "").lower())


def _is_safe_punctuation_rewrite(original: str, candidate: str) -> bool:
    o = _token_signature(original)
    c = _token_signature(candidate)
    if not o or not c:
        return False
    if o == c:
        return True
    # Allow tiny drift from LLM formatting output, but reject semantic rewrites.
    overlap = sum(1 for i, tok in enumerate(o[: min(len(o), len(c))]) if c[i] == tok)
    ratio = overlap / max(len(o), len(c))
    return ratio >= 0.98 and abs(len(o) - len(c)) <= max(3, len(o) // 100)


def _split_transcript_for_punctuation_refine(
    text: str, max_chars: int = PUNCT_REFINE_CHUNK_CHARS,
) -> tuple[list[str], list[str]]:
    """
    Split transcript into chunks for LLM punctuation review.
    Returns (chunks, joiners) where joiners[i] is placed between chunks[i] and chunks[i+1]
    ('\\n' for paragraph-style breaks, ' ' for pieces of one long line).
    """
    t = str(text or "").strip()
    if not t:
        return [], []
    if len(t) <= max_chars:
        return [t], []

    chunks: list[str] = []
    joiners: list[str] = []
    buf_lines: list[str] = []

    def flush_buf() -> None:
        if not buf_lines:
            return
        block = "\n".join(buf_lines)
        if chunks:
            joiners.append("\n")
        chunks.append(block)
        buf_lines.clear()

    for line in t.split("\n"):
        if len(line) > max_chars:
            flush_buf()
            s = 0
            first_piece = True
            while s < len(line):
                e = min(s + max_chars, len(line))
                if e < len(line):
                    cut = line.rfind(" ", s + max_chars // 2, e)
                    if cut <= s:
                        cut = e
                    e = cut
                piece = line[s:e].strip()
                if piece:
                    if chunks:
                        # After a normal paragraph flush, start of a long line → newline;
                        # further splits of the same long line → space.
                        joiners.append("\n" if first_piece else " ")
                    chunks.append(piece)
                    first_piece = False
                s = e if e > s else s + max_chars
            continue

        cand = "\n".join(buf_lines + [line]) if buf_lines else line
        if len(cand) <= max_chars:
            buf_lines.append(line)
        else:
            flush_buf()
            buf_lines = [line]

    flush_buf()
    return chunks, joiners


def _normalize_hindi_sentence_punctuation(text: str, language: str) -> str:
    """
    Hindi uses the Devanagari danda (।) as the sentence full stop, not Latin '.'
    STT often emits '.' or ASCII '|'; map those to । between Hindi (Devanagari) clauses.
    Avoids digit-decimal patterns like 3.14 or ३.५.
    """
    lang = (language or "en").strip().lower()
    if lang not in ("hi", "hi-in", "hinglish"):
        return str(text or "")
    s = str(text or "")
    if lang == "hinglish" and not _snippet_has_devanagari(s):
        return s

    # ASCII '|' used like danda between Hindi words (not URL '||')
    s = re.sub(r"(?<=[\u0900-\u097F])\s*\|(?!\|)\s*(?=[\u0900-\u097F])", " । ", s)

    # Latin '.' between Devanagari sentence units (not decimal: no digit on both sides)
    period_hindi = re.compile(
        r"(?<=[\u0900-\u097F])"  # after Devanagari letter/sign
        r"(?<![\u0966-\u096F0-9])"  # not Hindi/Western digit before dot
        r"\."
        r"(?![\u0966-\u096F0-9])"  # not digit after dot
        r"(?=\s*[\u0900-\u097F]|\s*$|\s*\n)",
    )
    if lang in ("hi", "hi-in"):
        s = period_hindi.sub("।", s)
    else:
        # Hinglish: only when Hindi continues after the period
        s = re.sub(
            r"(?<=[\u0900-\u097F])(?<![\u0966-\u096F0-9])\.(?![\u0966-\u096F0-9])(?=\s*[\u0900-\u097F])",
            "।",
            s,
        )

    s = re.sub(r"।\s*।+", "।", s)
    return s


def _repair_hindi_hinglish_wording_post_stt(
    text: str, topic_id: str, language: str, institute_id: str,
) -> tuple[str, dict]:
    """
    Correct obvious ASR word errors in Hindi/Hinglish transcripts (misheard words,
    phonetic confusions, broken transliterations) while preserving speaker meaning.
    This step is intentionally language-preserving (no forced translation).
    """
    meta = {
        "word_repair_applied": False,
        "word_repair_chunks": 0,
        "word_repair_chunks_accepted": 0,
    }
    lang = (language or "").strip().lower()
    if lang not in ("hi", "hi-in", "hinglish"):
        return str(text or "").strip(), meta

    raw = str(text or "").strip()
    if not raw:
        return raw, meta

    words = re.findall(r"[A-Za-z\u0900-\u097F]+", raw)
    if len(words) < WORD_REPAIR_MIN_WORDS:
        return raw, meta

    chunks, joiners = _split_transcript_for_punctuation_refine(raw, max_chars=WORD_REPAIR_CHUNK_CHARS)
    if not chunks:
        return raw, meta
    meta["word_repair_chunks"] = len(chunks)

    system_prompt = (
        "You repair Hindi/Hinglish educational lecture transcripts produced by ASR.\n"
        "Goal: fix obvious word-level transcription mistakes so sentences make semantic sense.\n"
        "Allowed changes:\n"
        "- Correct misheard/misspelled Hindi words and Hinglish transliterations.\n"
        "- Fix small grammar glue words only when needed for coherence.\n"
        "- Preserve technical terms and formulas.\n"
        "Hard constraints:\n"
        "- Do NOT add new facts, examples, or explanations not present in source.\n"
        "- Keep sentence order and paragraph order.\n"
        "- Preserve code-mix (Hindi + English) style used by speaker.\n"
        "- Keep Hindi in Devanagari where source uses Devanagari.\n"
        "- Return plain text only."
    )

    repaired_chunks: list[str] = []
    for idx, chunk in enumerate(chunks):
        c = chunk.strip()
        if not c:
            repaired_chunks.append(chunk)
            continue
        try:
            llm_result = get_llm().complete(
                system_prompt=system_prompt,
                user_prompt=(
                    f"Lecture topic: {topic_id or 'General'}\n"
                    f"Language: {language}\n"
                    f"Section {idx + 1}/{len(chunks)}\n\n"
                    "Fix only ASR word mistakes and obvious incoherent phrasing. Keep meaning intact.\n\n"
                    f"{c}"
                ),
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=WORD_REPAIR_MAX_TOKENS,
                json_mode=False,
                institute_id=institute_id,
            )
            candidate = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
            candidate = candidate.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```\w*\s*", "", candidate, count=1)
                candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
            # Guardrail: reject wildly longer/shorter rewrites.
            if candidate and (0.65 <= (len(candidate) / max(len(c), 1)) <= 1.45):
                repaired_chunks.append(candidate)
                meta["word_repair_chunks_accepted"] += 1
                meta["word_repair_applied"] = True
            else:
                repaired_chunks.append(c)
        except Exception as exc:
            logger.warning("Hindi/Hinglish word repair failed on chunk %d/%d: %s", idx + 1, len(chunks), exc)
            repaired_chunks.append(c)

    if not repaired_chunks:
        return raw, meta

    out_parts: list[str] = [repaired_chunks[0]]
    for i in range(1, len(repaired_chunks)):
        sep = joiners[i - 1] if i - 1 < len(joiners) else "\n"
        out_parts.append(sep + repaired_chunks[i])
    return "".join(out_parts).strip(), meta


def _punctuation_refine_language_instructions(language: str) -> str:
    """Extra LLM instructions so refinement matches Hindi / Hinglish / English, not English-only."""
    lang = (language or "en").strip().lower()
    if lang in ("hi", "hi-in"):
        return (
            "\n\nLanguage (Hindi, Devanagari): Use the Hindi poorna viram / danda (Unicode U+0964, the character ।) "
            "for sentence boundaries — this is the correct Hindi full stop, not the Latin period (.). "
            "Do not use Latin '.' between Hindi words at sentence end; use ।. "
            "Use Latin '?' and '!' after Devanagari when the utterance is a question or exclamation. "
            "Use commas for pauses and lists. Fix small obvious STT spacing issues around । if needed. "
            "Never translate Hindi to English or strip Devanagari; keep every Hindi word in Devanagari as spoken."
        )
    if lang == "hinglish":
        return (
            "\n\nLanguage (Hinglish): The text mixes Devanagari Hindi and Latin English. Preserve code-switching "
            "exactly — do not replace Hindi with English or English with Hindi. Apply punctuation per phrase: "
            "English clauses follow English punctuation habits; Hindi clauses may use । or . in a way that matches "
            "that clause. Use '?' and '!' where either script would naturally require them. Keep technical terms in "
            "the script the speaker used."
        )
    if lang in ("en", "english"):
        return ""
    return (
        "\n\nLanguage: If you see Devanagari (Hindi), Latin (English), or a mix, apply appropriate punctuation for "
        "each without translating or normalizing away either script."
    )


def _refine_transcript_punctuation_post_stt(
    text: str, topic_id: str, language: str, institute_id: str,
) -> tuple[str, dict]:
    """
    After transcription (and optional pause-based joins), review whether punctuation
    and sentence boundaries make sense; fix misplaced commas, periods, question marks,
    and exclamation marks. Does not change word order or add/remove words (validated).
    """
    meta: dict = {
        "punct_refine_applied": False,
        "punct_refine_chunks": 0,
        "punct_refine_chunks_accepted": 0,
    }
    raw = str(text or "").strip()
    if not raw:
        return raw, meta

    words = re.findall(r"[A-Za-z\u0900-\u097F]+", raw)
    if len(words) < PUNCT_REFINE_MIN_WORDS:
        return _normalize_hindi_sentence_punctuation(raw, language), meta

    chunks, joiners = _split_transcript_for_punctuation_refine(raw)
    if not chunks:
        return raw, meta

    meta["punct_refine_chunks"] = len(chunks)
    refined: list[str] = []

    lang_extra = _punctuation_refine_language_instructions(language)
    system_prompt = (
        "You review lecture transcripts from speech-to-text (with optional timing hints). Transcripts may be in "
        "English, Hindi (Devanagari), Hinglish (mixed scripts), or similar — treat all of these equally.\n"
        "Your job is to ensure punctuation and sentence boundaries read naturally and match the speaker's intent "
        "(statements, questions, emphasis).\n"
        "You may use commas, the Hindi danda (।) for Hindi sentence ends, Latin full stops only for English "
        "clauses, question marks, exclamation marks, colons, and semicolons where appropriate. "
        "Use '?' for real or clearly rhetorical questions; use '!' sparingly for "
        "clear surprise or strong emphasis, not every sentence.\n"
        "You must NOT change, add, remove, or reorder words (including particles and names). "
        "You may only change spacing, line breaks, capitalization at the start of Latin sentences, and punctuation. "
        "Preserve formulas, numbers, and symbols exactly."
        f"{lang_extra}"
    )

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            refined.append(chunk)
            continue
        try:
            llm_result = get_llm().complete(
                system_prompt=system_prompt,
                user_prompt=(
                    f"Lecture topic: {topic_id or 'General'}\n"
                    f"Lecture language (respect this for punctuation rules): {language}\n"
                    f"Section {idx + 1} of {len(chunks)}.\n\n"
                    "Return ONLY the corrected transcript section (no preamble, no markdown fences). "
                    "Keep Hindi in Devanagari and English in Latin; do not translate.\n\n"
                    f"{chunk}"
                ),
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=PUNCT_REFINE_MAX_TOKENS,
                json_mode=False,
                institute_id=institute_id,
            )
            candidate = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
            candidate = candidate.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```\w*\s*", "", candidate, count=1)
                candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
            accepted = False
            if candidate and _is_safe_punctuation_rewrite(chunk, candidate):
                refined.append(candidate)
                meta["punct_refine_chunks_accepted"] += 1
                meta["punct_refine_applied"] = True
                accepted = True

            # Fallback: strict punctuation-only pass when first pass changed words too much.
            if not accepted:
                strict = get_llm().complete(
                    system_prompt=(
                        "You are a punctuation restorer.\n"
                        "Return the SAME words in the SAME order.\n"
                        "Only add/fix punctuation, spacing and line breaks.\n"
                        "Do not change, add, remove, or reorder any word."
                    ),
                    user_prompt=(
                        f"Language hint: {language}\n"
                        "Use commas, question marks, exclamation marks, and sentence stops where needed.\n"
                        "For Hindi sentence ends prefer danda (।) over Latin period.\n\n"
                        f"{chunk}"
                    ),
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=PUNCT_REFINE_MAX_TOKENS,
                    json_mode=False,
                    institute_id=institute_id,
                )
                strict_candidate = strict["content"] if isinstance(strict["content"], str) else str(strict["content"])
                strict_candidate = strict_candidate.strip()
                if strict_candidate.startswith("```"):
                    strict_candidate = re.sub(r"^```\w*\s*", "", strict_candidate, count=1)
                    strict_candidate = re.sub(r"\s*```\s*$", "", strict_candidate).strip()

                if strict_candidate and _token_signature(strict_candidate) == _token_signature(chunk):
                    refined.append(strict_candidate)
                    meta["punct_refine_chunks_accepted"] += 1
                    meta["punct_refine_applied"] = True
                else:
                    refined.append(chunk)
        except Exception as exc:
            logger.warning("Transcript punctuation refinement failed on chunk %d/%d: %s", idx + 1, len(chunks), exc)
            refined.append(chunk)

    out_parts: list[str] = [refined[0]]
    for i in range(1, len(refined)):
        sep = joiners[i - 1] if i - 1 < len(joiners) else "\n"
        out_parts.append(sep + refined[i])
    return "".join(out_parts).strip(), meta


def _restore_sentence_punctuation(text: str, topic_id: str, language: str, institute_id: str) -> str:
    """Backward-compatible name: runs post-STT punctuation + sense check (delegates to _refine_...)."""
    refined, _meta = _refine_transcript_punctuation_post_stt(text, topic_id, language, institute_id)
    return refined



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

    # Cap input to ~4500 chars (~1500 tokens for Hindi) to stay under 6000 TPM on 8b model
    _REPAIR_INPUT_CAP = 4500
    repair_input = cleaned[:_REPAIR_INPUT_CAP] if len(cleaned) > _REPAIR_INPUT_CAP else cleaned

    try:
        llm_result = get_llm().complete(
            system_prompt=(
                "You repair noisy educational lecture transcripts. Clean OCR/STT artifacts, remove garbage tokens, "
                "repair obvious equation formatting, and fix broken statements into coherent text while preserving "
                "the original source language style (Hindi/Hinglish/English as given). Do not invent new topics."
            ),
            user_prompt=(
                f"Lecture topic: {topic_id or 'General'}\n"
                f"Source language: {language}\n"
                f"Detected issues: {', '.join(flags)}\n\n"
                "Clean and repair this transcript for note generation. Preserve as much original meaning as possible.\n\n"
                f"{repair_input}"
            ),
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2048,
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

    lang = (language or "en").strip().lower()
    is_hindi_hinglish = lang in ("hi", "hi-in", "hinglish")

    if is_hindi_hinglish:
        # Skip LLM-heavy word repair and punctuation refinement for Hindi/Hinglish.
        # Chunk notes generation already reads Hindi natively and translates to English,
        # so 9+ sequential preprocessing LLM calls here add ~60s latency with no benefit.
        final_text = _normalize_hindi_sentence_punctuation(final_text, language)
        return final_text, {
            "quality_flags": flags,
            "repair_applied": bool(flags),
            "word_repair_applied": False, "word_repair_chunks": 0, "word_repair_chunks_accepted": 0,
            "punct_refine_applied": False, "punct_refine_chunks": 0, "punct_refine_chunks_accepted": 0,
        }

    final_text, word_meta = _repair_hindi_hinglish_wording_post_stt(
        final_text, topic_id, language, institute_id,
    )
    final_text, punct_meta = _refine_transcript_punctuation_post_stt(
        final_text, topic_id, language, institute_id,
    )
    final_text = _normalize_hindi_sentence_punctuation(final_text, language)
    return final_text, {
        "quality_flags": flags,
        "repair_applied": bool(flags),
        **word_meta,
        **punct_meta,
    }


NOTES_CHUNK_CHAR_LIMIT = 9000
NOTES_CHUNK_OVERLAP_CHARS = 600   # used for English only; Hindi uses 0 (see _generate_comprehensive_notes)
NOTES_SECTION_MAX_TOKENS = 700    # baseline; overridden per-call by adaptive formula below
NOTES_MERGE_MAX_TOKENS = 1800
# Adaptive formula ensures merge never overflows 6000 TPM regardless of chunk count:
#   section_tokens = max(350, min(700, 3900 // N))   → N × section_tokens ≤ 3900 + 300 + 1800 ≤ 6000 ✅
_MERGE_MAX_INPUT_CHARS = 15_500   # safety net: ~3900 English tokens × 4 chars/token


def _compress_hindi_filler(text: str) -> str:
    """Strip common Hindi/Hinglish filler words and repeated phrases to reduce token count ~10-15%."""
    import re as _re
    # Standalone filler words (word-boundary safe)
    _FILLERS = _re.compile(
        r'\b(?:um+|uh+|hmm+|haan|aur|toh|matlab|basically|actually|obviously|'
        r'theek hai|theek|dekho|dekha|suno|suniye|bolo|boliye|'
        r'samjhe|samjha|samajh gaye|samajh|'
        r'okay|ok|right|so|like|you know|i mean|'
        r'ek baar|ek bar|phir se|dobara|again)\b[,.]?\s*',
        _re.IGNORECASE,
    )
    text = _FILLERS.sub(' ', text)
    # Remove 3+ consecutive identical words ("samjhe samjhe samjhe" → "samjhe")
    text = _re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=_re.IGNORECASE)
    text = _re.sub(r'[ \t]{2,}', ' ', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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


def _generate_chunk_notes(chunk_text: str, topic_id: str, language: str, institute_id: str, chunk_index: int, total_chunks: int, max_tokens: int = NOTES_SECTION_MAX_TOKENS) -> str:
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
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=max_tokens,
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
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=NOTES_MERGE_MAX_TOKENS,
        json_mode=False,
        institute_id=institute_id,
    )
    return llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])


def _generate_comprehensive_notes(transcript: str, topic_id: str, language: str, institute_id: str) -> tuple[str, dict]:
    import time as _time

    is_hindi = str(language or "").lower() in ("hi", "hi-in", "hinglish")

    # Hindi optimizations: zero overlap (sentences are self-contained) + filler removal
    if is_hindi:
        transcript = _compress_hindi_filler(transcript)
        chunks = _chunk_transcript(transcript, overlap=0)
    else:
        chunks = _chunk_transcript(transcript)

    if not chunks:
        return "", {"chunk_count": 0}

    # Adaptive section token budget — satisfies both constraints simultaneously:
    #   TPM:   N × section_tokens + 300 prompt + 1800 merge ≤ 6000  → section_tokens ≤ 3900 // N
    #   Chars: N × section_tokens × 4 ≤ _MERGE_MAX_INPUT_CHARS      → section_tokens ≤ _MERGE_MAX_INPUT_CHARS // (N × 4)
    n = len(chunks)
    section_tokens = max(350, min(NOTES_SECTION_MAX_TOKENS, 3900 // n, _MERGE_MAX_INPUT_CHARS // max(n * 4, 1)))

    if n == 1:
        notes = _generate_chunk_notes(chunks[0], topic_id, language, institute_id, 1, 1, max_tokens=section_tokens).strip()
        return notes, {"chunk_count": 1, "merge_applied": False, "section_tokens": section_tokens}

    logger.info(
        "Generating chunked notes | chunks=%d | section_tokens=%d | topic=%s | lang=%s",
        n, section_tokens, topic_id, language,
    )

    # Sequential processing with round-robin key distribution (see llm_client.py).
    # 1.5s gap between chunks keeps each call well under 6,000 TPM.
    partial_notes: list[str] = []
    failed_chunks = 0
    for i, chunk in enumerate(chunks):
        try:
            notes = _generate_chunk_notes(
                chunk, topic_id, language, institute_id, i + 1, n, max_tokens=section_tokens,
            ).strip()
            partial_notes.append(notes)
        except Exception as exc:
            logger.warning("Chunk %d/%d notes failed (%s) — skipping", i + 1, n, exc)
            failed_chunks += 1
            partial_notes.append("")
        if i < n - 1:
            _time.sleep(1.5)

    non_empty = [p for p in partial_notes if p.strip()]
    if not non_empty:
        return "", {"chunk_count": n, "failed_chunks": failed_chunks, "error": "all_chunks_failed"}

    # Safety net: cap merge input so it never overflows 6000 TPM
    combined = "\n\n".join(non_empty)
    if len(combined) > _MERGE_MAX_INPUT_CHARS:
        logger.warning(
            "Merge input truncated %d → %d chars (adaptive formula should have prevented this)",
            len(combined), _MERGE_MAX_INPUT_CHARS,
        )
        combined = combined[:_MERGE_MAX_INPUT_CHARS]
        non_empty = [combined]

    merged = _merge_chunk_notes(non_empty, topic_id, language, institute_id).strip()
    return merged, {"chunk_count": n, "failed_chunks": failed_chunks, "merge_applied": True, "section_tokens": section_tokens}


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
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2048,  # was 4096; merged notes ≈ 1800 input tokens → 1800+2048=3848 fits under 6000 TPM
            json_mode=False,
            institute_id=institute_id,
        )
        polished = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
        polished = polished.strip()
        return polished or cleaned, True
    except Exception as exc:
        logger.warning("Notes markdown polish failed (%s)", exc)
        return cleaned, False


# â"€â"€ AI #1 -- Doubt Clearing â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

# Step-1 detector: lightweight classification (subject + question type)
_DOUBT_DETECTOR_SYSTEM = (
    "You are a JEE/NEET question classifier. "
    "Respond with ONLY a valid JSON object — no text before or after.\n"
    'Format: {"subject": "<subject>", "type": "<type>"}\n'
    "subject must be exactly one of: physics, chemistry, math, biology\n"
    "type must be exactly one of: numerical, derivation, conceptual, mcq, theory"
)

# ── Keyword-based subject & type detection (runs before LLM, free + instant) ──

PHYSICS_KEYWORDS = [
    'block', 'mass', 'incline', 'friction', 'force', 'velocity',
    'acceleration', 'momentum', 'torque', 'current', 'voltage',
    'resistance', 'lens', 'mirror', 'charge', 'electric', 'magnetic',
    'wave', 'frequency', 'pendulum', 'spring', 'collision', 'projectile',
    'gravity', 'newton', 'joule', 'watt', 'ampere', 'capacitor', 'inductor',
    'circuit', 'magnetic field', 'refraction', 'reflection', 'doppler',
    'thermodynamics', 'entropy', 'carnot', 'pressure', 'temperature',
    'radioactive', 'nuclear', 'photon', 'electron', 'proton', 'neutron',
    'kinetic energy', 'potential energy', 'power', 'work done',
]

CHEMISTRY_KEYWORDS = [
    'mole', 'molarity', 'molality', 'ka', 'kb', 'ksp', 'ph', 'reaction',
    'compound', 'element', 'titration', 'equilibrium', 'bond', 'entropy',
    'enthalpy', 'oxidation', 'reduction', 'acid', 'base', 'salt',
    'organic', 'alkane', 'alkene', 'benzene', 'ester', 'aldehyde',
    'molar mass', 'dissociation', 'buffer', 'normality', 'equivalent',
    'hybridization', 'isomer', 'polymer', 'monomer', 'catalyst',
    'activation energy', 'rate constant', 'order of reaction',
    'electrode', 'electrolysis', 'galvanic', 'cell potential',
    'freezing point', 'boiling point', 'osmotic pressure', 'vapour pressure',
]

MATH_KEYWORDS = [
    'integrate', 'differentiate', 'derivative', 'matrix', 'determinant',
    'probability', 'parabola', 'ellipse', 'hyperbola', 'complex number',
    'binomial', 'permutation', 'combination', 'limit', 'series',
    'sequence', 'polynomial', 'quadratic', 'trigonometric identity',
    'inverse trigonometric', 'definite integral', 'indefinite integral',
    'integral', 'differential equation', 'coordinate geometry', 'straight line',
    'locus', 'conic section', 'arithmetic progression', 'geometric progression',
    'value of x', 'roots of', 'zeroes of', 'solve the equation',
]

BIOLOGY_KEYWORDS = [
    'cell', 'mitosis', 'meiosis', 'photosynthesis', 'dna', 'rna',
    'enzyme', 'hormone', 'ecosystem', 'genetics', 'neuron', 'chromosome',
    'protein', 'ribosome', 'chloroplast', 'mitochondria', 'evolution',
    'respiration', 'digestion', 'excretion', 'reproduction', 'immunity',
    'biodiversity', 'food chain', 'biomolecule', 'nitrogen cycle',
    'krebs cycle', 'calvin cycle', 'glycolysis', 'atp', 'adp', 'nadh',
    'allele', 'genotype', 'phenotype', 'dominant', 'recessive', 'mendel',
]


def _detect_subject_by_keyword(question: str):
    """Returns (subject, score) where subject matches _SUBJECT_RULES keys, or (None, 0)."""
    q = question.lower()
    scores = {
        'physics':   sum(1 for kw in PHYSICS_KEYWORDS   if kw in q),
        'chemistry': sum(1 for kw in CHEMISTRY_KEYWORDS if kw in q),
        'math':      sum(1 for kw in MATH_KEYWORDS      if kw in q),
        'biology':   sum(1 for kw in BIOLOGY_KEYWORDS   if kw in q),
    }
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else (None, 0)


def _detect_type_by_keyword(question: str) -> str:
    """Returns question type string. Defaults to 'numerical' for JEE/NEET."""
    q = question.lower()
    if any(w in q for w in ['derive', 'prove', 'show that', 'establish']):
        return 'derivation'
    if any(w in q for w in ['mechanism', 'iupac', 'name the compound', 'identify the compound']):
        return 'organic'
    if any(w in q for w in ['explain', 'why does', 'what is', 'define', 'describe', 'state']):
        return 'conceptual'
    if any(w in q for w in ['sketch', 'draw the graph', 'plot']):
        return 'graph'
    if any(w in q for w in ['label', 'draw and label', 'name the parts']):
        return 'diagram'
    if any(w in q for w in ['ncert', 'according to', 'as per']):
        return 'ncert_fact'
    return 'numerical'


def _detect_subject_and_type_for_doubt(question: str, has_image: bool, institute_id: str):
    """
    Priority: keyword detection (instant) → LLM fallback (images / ambiguous text).
    Returns (subject, qtype) where subject is one of: physics, chemistry, math, biology.
    """
    if question and question.strip():
        subject, score = _detect_subject_by_keyword(question)
        if subject:
            return subject, _detect_type_by_keyword(question)

    # Fallback: LLM classifier for image-only or zero-keyword questions
    subject, qtype = "physics", "numerical"
    try:
        detect_result = get_llm().complete(
            system_prompt=_DOUBT_DETECTOR_SYSTEM,
            user_prompt=f"Classify this question:\n\n{question[:800]}",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=60,
            json_mode=True,
            institute_id=institute_id,
        )
        detect_data = detect_result.get("content", {})
        if isinstance(detect_data, str):
            import re as _re2
            m = _re2.search(r'\{[^}]+\}', detect_data)
            if m:
                try:
                    detect_data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        if isinstance(detect_data, dict):
            s = (detect_data.get("subject") or "physics").lower().strip()
            t = (detect_data.get("type") or "numerical").lower().strip()
            valid_subjects = {"physics", "chemistry", "math", "biology"}
            valid_types = {"numerical", "derivation", "conceptual", "mcq", "theory"}
            subject = s if s in valid_subjects else "physics"
            qtype = t if t in valid_types else "numerical"
    except Exception as exc:
        logger.warning("Doubt LLM detector failed (%s); defaulting to physics/numerical", exc)

    return subject, qtype


# ── Model routing ──────────────────────────────────────────────────────────────

GROQ_MODELS = {
    "reasoning": "openai/gpt-oss-120b",
    "math":      "qwen/qwen3-32b",
    "general":   "llama-3.3-70b-versatile",
    "detector":  "llama-3.1-8b-instant",
}


def _select_doubt_model(subject: str, question_type: str) -> str:
    if subject == "math":
        return GROQ_MODELS["math"]
    if question_type in ("numerical", "derivation", "graph", "organic"):
        return GROQ_MODELS["reasoning"]
    return GROQ_MODELS["general"]


def _parse_reasoning_response(raw: str) -> dict:
    """Strip DeepSeek/QwQ think blocks then parse JSON. Returns dict always."""
    cleaned = _strip_think_blocks(raw).strip()

    json_match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    else:
        json_match2 = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
        if json_match2:
            cleaned = json_match2.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "brief": {"answer": "See full solution"},
            "detailed": {
                "solution": cleaned,
                "final_answer": "",
                "verification": "",
                "key_concept": "",
            },
        }


# Per-subject rules injected into the solver system prompt at runtime.
# Kept compact (<200 tokens each) so total request stays under 6 000 TPM.
_SUBJECT_RULES: dict[str, str] = {
    "physics": (
        "PHYSICS RULES:\n"
        "1. List all Given quantities with symbols and SI units\n"
        "2. State what to Find\n"
        "3. Identify the exact law/formula — write it symbolically first\n"
        "4. Substitute values explicitly: replace each symbol with its number+unit\n"
        "5. One arithmetic operation per line; carry units at every step\n"
        "6. Pitfalls: g=9.8 or 10 m/s² (use what is given), sign conventions, degrees vs radians"
    ),
    "chemistry": (
        "CHEMISTRY RULES:\n"
        "1. Build molar mass table FIRST —\n"
        "   H=1, He=4, Li=7, C=12, N=14, O=16, Na=23, Mg=24, Al=27, S=32,\n"
        "   Cl=35.5, K=39, Ca=40, Fe=56, Cu=63.5, Zn=65, Br=80, Ag=108, I=127, Ba=137, Pb=207\n"
        "2. Balance equation before stoichiometry\n"
        "3. n = mass/M; C = n/V(L); one mole-conversion per line\n"
        "4. Equilibrium: write ICE table (Initial / Change / Equilibrium)\n"
        "5. Colligative: ΔTf = i·Kf·m; ΔTb = i·Kb·m (i = van't Hoff factor)\n"
        "6. Redox: state oxidation-number changes before balancing"
    ),
    "math": (
        "MATH RULES:\n"
        "1. State the technique/theorem before applying it\n"
        "2. Algebra: one operation per line; show every expansion/factorisation\n"
        "3. Calculus: write substitution explicitly, show du; transform limits for definite integrals\n"
        "4. Trigonometry: use exact values (sin30°=½, cos60°=½, tan45°=1, sin90°=1)\n"
        "5. Coordinate: derive from standard form; show all intermediate equations\n"
        "6. Probability: state independence/mutual-exclusivity; use P(A∪B)=P(A)+P(B)−P(A∩B)\n"
        "7. Verify: substitute final answer back into original equation"
    ),
    "biology": (
        "BIOLOGY RULES:\n"
        "1. Name all phases/stages in correct sequence\n"
        "2. Use precise scientific terminology (binomial names where relevant)\n"
        "3. Genetics: draw Punnett square; list all genotype and phenotype ratios\n"
        "4. Photosynthesis/Respiration: name molecules at each stage and the location\n"
        "5. Physiology: name the organ, tissue, and cell type involved\n"
        "6. State exceptions explicitly (C4 plants, incomplete dominance, anomalous species)"
    ),
}


def _build_solver_system_prompt(subject: str, qtype: str) -> str:
    rules = _SUBJECT_RULES.get(subject, _SUBJECT_RULES["physics"])
    is_numerical = qtype in ("numerical", "derivation")
    return (
        f"You are an expert AI doubt resolver for JEE and NEET students.\n"
        f"Subject: {subject.upper()} | Question Type: {qtype}\n\n"
        + (
            "NUMERICAL ACCURACY RULES — CRITICAL:\n"
            "A. NEVER compute anything in your head. Write EVERY number operation explicitly.\n"
            "B. NEVER skip arithmetic steps — e.g. write '9.45 / 94.5 = 0.1' not just '0.1'.\n"
            "C. NEVER round intermediate values — carry full decimal precision until final answer.\n"
            "D. After the final answer, VERIFY by substituting back into the original equation.\n"
            "E. If result seems impossible (negative mass, probability > 1, α > 1): STOP and recheck.\n\n"
            if is_numerical else ""
        )
        + "MANDATORY SETUP — DO NOT BEGIN SOLVING UNTIL DONE:\n"
        "- Physics numerical: list ALL Given quantities with symbols + SI units in one block\n"
        "- Chemistry numerical: write molar masses of ALL atoms/molecules before any calculation\n"
        "- Math: state the theorem/technique to be used in one sentence\n"
        "- Theory/Biology: identify the exact concept/process being asked\n\n"
        "UNIVERSAL RULES:\n"
        "1. Every arithmetic operation on its own separate line\n"
        "2. Carry units at EVERY step — never drop them mid-calculation\n"
        "3. One formula application per step — never chain multiple formulas in one line\n"
        "4. NEVER try multiple formulas — identify the correct one first, apply it once\n\n"
        f"{rules}\n\n"
        "OUTPUT — respond with ONLY the following JSON object. No markdown, no preamble, no trailing text.\n\n"
        '{\n'
        '  "brief": {\n'
        '    "answer": "NUMERICALS: numbered steps with explicit arithmetic + Final Answer with units. Example: Step 1: M = 12+35.5+2(1)+2(16) = 94.5 g/mol. Step 2: n = 9.45/94.5 = 0.1 mol. Final Answer: 0.372 K. THEORY: 1-3 direct sentences only, no sub-steps."\n'
        '  },\n'
        '  "detailed": {\n'
        '    "solution": "Step N — [action]: [explicit calculation]. Reason: [one phrase why]. Each arithmetic on its own line. End with Final Answer.",\n'
        '    "final_answer": "value with units",\n'
        '    "verification": "substitute answer back into original equation and confirm LHS = RHS",\n'
        '    "key_concept": "one-line formula or principle to remember"\n'
        '  }\n'
        '}'
    )


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning traces (qwen3, DeepSeek-R1).
    If the closing tag is missing (max_tokens hit mid-thinking), falls back to
    extracting any JSON block found inside the raw text."""
    import re as _re
    cleaned = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    if cleaned:
        return cleaned
    # Unclosed think block — try to salvage JSON that the model started writing inside
    m = _re.search(r'\{.*\}', text, flags=_re.DOTALL)
    return m.group() if m else ""


def _coerce_json_array_string_to_prose(t: str) -> str:
    """If the model outputs a JSON array of strings, join into plain text for the client."""
    t = (t or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        return t
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return t
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return t
    parts = [s.strip() for s in data if s.strip()]
    if not parts:
        return t
    logger.warning("LLM returned a JSON list of strings; coercing to joined prose")
    return ". ".join(parts)


def _coerce_tutor_or_doubt_text(raw) -> str:
    """Normalize doubt/tutor free-text: dict from json_mode, bare JSON arrays, or object with only `hints`."""
    if raw is None:
        return ""
    if isinstance(raw, list) and (not raw or all(isinstance(x, str) for x in raw)):
        if not raw:
            return ""
        parts = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
        if parts:
            return ". ".join(parts)
    if isinstance(raw, dict):
        r = raw.get("response", raw.get("explanation"))
        if isinstance(r, list) and r and all(isinstance(x, str) for x in r):
            r = ". ".join(s.strip() for s in r if s and s.strip())
        elif isinstance(r, str):
            r = r.strip()
        else:
            r = (str(r) if r is not None else "").strip()
        hints = raw.get("hints")
        if not r and isinstance(hints, list) and hints and all(isinstance(x, str) for x in hints):
            r = ". ".join(s.strip() for s in hints if s and s.strip())
        if r:
            raw = r
        else:
            return str(raw).strip()
    t = (raw if isinstance(raw, str) else str(raw)).strip()
    t = _coerce_json_array_string_to_prose(t)
    import re
    t = re.sub(r'<scratchpad>.*?</scratchpad>', '', t, flags=re.DOTALL).strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            return t
        if isinstance(obj, dict):
            r = (obj.get("response") or "").strip()
            h = obj.get("hints")
            if r:
                return r
            if isinstance(h, list) and h and all(isinstance(x, str) for x in h):
                logger.warning("LLM JSON had empty response; coalescing from hints")
                return ". ".join(s.strip() for s in h if s.strip())
    return t


@api_view(["POST"])
def resolve_doubt(request):
    import re as _re

    data = request.data
    question_text = (data.get("questionText") or data.get("question") or "").strip()
    raw_mode = (data.get("mode") or "detailed").strip().lower()
    mode = "brief" if raw_mode in ("brief", "short") else "detailed"

    # Image support: base64 data URL (from NestJS) or raw HTTPS URL
    image_description = ""
    image_source = (data.get("questionImageBase64") or data.get("questionImageUrl") or "").strip()
    if image_source:
        image_description = _vision_text_from_image(image_source, _DOUBT_VISION_PROMPT)
        if not image_description:
            logger.warning("Vision API returned empty for doubt image")

    if not question_text and not image_description:
        return Response({"error": "Missing questionText or readable image"}, status=400)

    institute_id = getattr(request, "institute_id", "default")

    if image_description and question_text:
        combined_question = f"{question_text}\n\n[Image content]\n{image_description}"
    elif image_description:
        combined_question = f"[Student uploaded an image]\n{image_description}"
    else:
        combined_question = question_text

    # ── Step 1: Classify subject and question type ────────────────────────────
    # Keyword detection runs first (free, instant). LLM runs only as fallback
    # for image-only questions or when zero keywords matched.
    subject, qtype = _detect_subject_and_type_for_doubt(
        combined_question, has_image=bool(image_description), institute_id=institute_id,
    )

    # ── Step 2: Route to correct model, build prompt, solve ───────────────────
    model = _select_doubt_model(subject, qtype)
    print(f"[DOUBT RESOLVER] Subject: {subject} | Type: {qtype} | Model: {model}")

    solver_system = _build_solver_system_prompt(subject, qtype)
    user_prompt = (
        f"Topic: {data.get('topicId', 'general')}\n\n"
        f"Question:\n{combined_question}"
    )

    is_reasoning_model = model in {"openai/gpt-oss-120b", "qwen/qwen3-32b"}

    try:
        if is_reasoning_model:
            # Reasoning models output <think> blocks — use text mode and parse manually.
            solve_result = get_llm().complete(
                system_prompt=solver_system,
                user_prompt=user_prompt,
                model=model,
                temperature=0.1,
                max_tokens=4096,
                json_mode=False,
                institute_id=institute_id,
            )
            raw_content = solve_result["content"]
            parsed = _parse_reasoning_response(
                raw_content if isinstance(raw_content, str) else str(raw_content)
            )
        else:
            solve_result = get_llm().complete(
                system_prompt=solver_system,
                user_prompt=user_prompt,
                model=model,
                temperature=0.1,
                max_tokens=4096,
                json_mode=True,
                json_mode_suffix="",
                institute_id=institute_id,
            )
            parsed = solve_result["content"] if isinstance(solve_result["content"], dict) else {}
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)
    brief_obj: dict = parsed.get("brief") or {}
    detailed_obj: dict = parsed.get("detailed") or {}

    # Fallback: if model returned a flat structure instead of nested brief/detailed
    if not brief_obj and not detailed_obj and parsed:
        answer_raw = parsed.get("answer") or parsed.get("solution") or ""
        brief_obj = {"answer": answer_raw}
        detailed_obj = {
            "solution": parsed.get("solution") or answer_raw,
            "final_answer": parsed.get("final_answer") or "",
            "verification": parsed.get("verification") or "",
            "key_concept": parsed.get("key_concept") or "",
        }

    # Select answer + explanation based on requested mode
    if mode == "brief":
        answer = brief_obj.get("answer") or detailed_obj.get("final_answer") or ""
        explanation = brief_obj.get("answer") or detailed_obj.get("solution") or ""
    else:
        answer = detailed_obj.get("final_answer") or brief_obj.get("answer") or ""
        explanation = detailed_obj.get("solution") or brief_obj.get("answer") or ""

    return JsonResponse({
        "subject": subject,
        "type": qtype,
        "model_used": solve_result["model"],
        "answer": answer,
        "explanation": explanation,
        "brief": brief_obj,
        "detailed": detailed_obj,
        "conceptLinks": [],
        "related_topics": [],
        "_meta": {
            "source": "llm",
            "model": solve_result["model"],
            "latency_ms": round(solve_result["latency_ms"]),
            "institute": institute_id,
        },
    })


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
    "If a word is unclear, use [illegible] for that part. If there is no readable answer, output exactly: (no readable answer) "
    "If the student wrote in English, transcribe in English only using Latin characters (A–Z, 0–9, usual math symbols). "
    "Do not output random Devanagari or other scripts unless the student clearly wrote in that language."
)

# Reused Groq client instances (keyed by API key) for vision
_groq_vision_clients: dict = {}


def _vision_text_from_image(image_url: str, user_prompt: str) -> str:
    """Groq Llama 4 Scout vision — shared for doubt and grading. Returns '' on failure (caller may OCR-fallback)."""
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq package not installed; vision OCR unavailable")
        return ""

    if not (image_url or "").strip():
        return ""

    keys = get_rotated_groq_keys()
    if not keys:
        logger.warning("No GROQ API keys in rotation; vision OCR skipped")
        return ""

    for api_key in keys:
        try:
            if api_key not in _groq_vision_clients:
                _groq_vision_clients[api_key] = Groq(api_key=api_key, timeout=45.0)
            client = _groq_vision_clients[api_key]
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            out = (response.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as exc:
            logger.warning("Vision (Groq) failed for image OCR: %s", exc)
            continue

    return ""


def _describe_image_with_vision(image_url: str) -> str:
    """Doubt / general: extract and describe full content (equations, diagrams, etc.)."""
    return _vision_text_from_image(image_url, _DOUBT_VISION_PROMPT)


def _transcribe_exam_answer_with_vision(image_url: str) -> str:
    """Mock test / grading: answer text only, no photo narration."""
    return _vision_text_from_image(image_url, _GRADING_VISION_PROMPT)


def _extract_text_from_image_url(
    image_url: str, languages: Optional[list] = None
) -> str:
    """EasyOCR fallback when Groq vision is empty or errored.
    ``languages`` default ``["en", "hi"]`` for doubt flow; use ``["en"]`` for grading
    to avoid Hindi script false positives on English handwriting."""
    try:
        import numpy as _np
        from PIL import Image as _Image
        import easyocr as _easyocr
        from io import BytesIO as _BytesIO
    except Exception:
        return ""

    lang_list = list(languages) if languages is not None else ["en", "hi"]

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
        reader = _easyocr.Reader(lang_list, gpu=False)
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
    is_grading = purpose in ("grading", "mock", "assessment", "mock_test", "answer")
    if is_grading:
        text = _transcribe_exam_answer_with_vision(image_url)
    else:
        text = _describe_image_with_vision(image_url)
    if not text:
        # English-only EasyOCR for grading — reduces garbage Devanagari on Latin handwriting
        text = _extract_text_from_image_url(
            image_url, languages=["en"] if is_grading else None
        )
    return JsonResponse({"text": text or ""})


# â"€â"€ AI #2 -- AI Tutor â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

@api_view(["POST"])
def start_tutor_session(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    context = data.get("context", "")

    # When a rich lesson-generation prompt is provided (long context), use it as the
    # system prompt directly so the LLM produces clean Markdown -- not JSON-wrapped text.
    if len(context) > 300:
        system_prompt = context
        user_prompt = "Generate the complete lesson now. Write everything in full -- do not truncate or use placeholders."
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
    explanation_text = _coerce_tutor_or_doubt_text(explanation_text)

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
            temperature=0.0,
            max_tokens=1200,
            json_mode=True,
            json_mode_suffix=_JSON_MODE_TUTOR_SUFFIX,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=502)

    raw_text = result["content"]
    explanation_text = _coerce_tutor_or_doubt_text(raw_text)

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


# â"€â"€ AI #6 -- Content Recommendation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #7 -- Speech-to-Text Notes â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
                "Transcription done -- %d chars | took=%.1fs",
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
    logger.info("Sending to LLM -- transcript=%d chars | transcription took=%.1fs", len(raw_transcript), _t1 - _t0)

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




# -- AI #7a -- Speech-to-Text Transcribe Only (Phase 1 of two-phase pipeline) --
# Accepts { audioUrl, language, topicId }
# Returns { rawTranscript, transcript } -- Whisper only, zero LLM calls.
# NestJS content.service.ts calls this first, saves transcript to DB, then
# calls /stt/notes-from-text in a second fire-and-forget pass for notes.

@api_view(["POST"])
def stt_transcribe_only(request):
    """Whisper transcription only -- no LLM. Saves transcript in ~2-5 min (vs 15+ min for full pipeline)."""
    import time as _time
    data = request.data
    audio_url = (data.get("audioUrl") or "").strip()
    if not audio_url:
        return Response({"error": "Missing audioUrl"}, status=400)

    language = data.get("language", "hi")
    topic_id = data.get("topicId", "")
    logger.info("stt_transcribe_only | url=%s | language=%s", audio_url, language)
    _t0 = _time.perf_counter()

    try:
        raw_transcript = _transcribe_audio(audio_url, language)
    except Exception as exc:
        logger.error("stt_transcribe_only FAILED for %s: %s", audio_url, exc)
        return Response(
            {
                "error": "transcription_failed",
                "detail": str(exc),
                "audioUrl": audio_url,
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

    elapsed = _time.perf_counter() - _t0
    logger.info("stt_transcribe_only done | %d chars | took=%.1fs", len(raw_transcript), elapsed)

    return JsonResponse({
        "rawTranscript": raw_transcript,
        "transcript": raw_transcript,
        "language": language,
        "topicId": topic_id,
        "_meta": {
            "source": "whisper",
            "chars": len(raw_transcript),
            "latency_s": round(elapsed, 1),
        },
    })

# â"€â"€ AI #8 -- Student Feedback Engine â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #9 -- Notes Weak Topic Identifier â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #10 -- Resume Analyzer â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #11 -- Interview Prep â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #12 -- Personalized Learning Plan â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ AI #13 -- In-Video Quiz Generator â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


def _chunk_notes(text: str, max_chars: int = 12000) -> list:
    """Split Markdown notes into chunks at H2/H3 headers, each at most max_chars."""
    import re as _re
    if len(text) <= max_chars:
        return [text]
    # Split at ## / ### headers to keep semantic boundaries
    parts = _re.split(r'\n(?=#{2,3}\s)', text)
    parts = [p.strip() for p in parts if p.strip()]
    chunks = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) + 2 <= max_chars:
            current = current + "\n\n" + part
        else:
            chunks.append(current)
            # If a single section is too large, split at paragraph boundaries
            if len(part) > max_chars:
                paras = part.split("\n\n")
                buf = ""
                for para in paras:
                    if len(buf) + len(para) + 2 <= max_chars:
                        buf = (buf + "\n\n" + para).strip() if buf else para
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = para[:max_chars]
                if buf:
                    chunks.append(buf)
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


@api_view(["POST"])
def generate_quiz_questions(request):
    data = request.data

    # Notes are the primary source; transcript is a fallback
    notes = (data.get("notes") or "").strip()
    transcript = (data.get("transcript") or "").strip()
    source_text = notes or transcript
    source_type = "notes" if notes else "transcript"

    if not source_text:
        return Response({"error": "Missing notes or transcript"}, status=400)
    if len(source_text) < 50:
        return Response({"error": "Content too short to generate quiz questions"}, status=422)

    try:
        num_questions = max(3, min(int(data.get("numQuestions", 5)), 20))
    except (TypeError, ValueError):
        num_questions = 5

    institute_id = getattr(request, "institute_id", "default")
    lecture_title = data.get("lectureTitle", "Lecture")
    topic_id = data.get("topicId", "")

    # If falling back to raw transcript, cap it — transcripts are very long and
    # tokenize at ~2 chars/token for Hindi/math, which overflows the 6K TPM limit.
    # Notes are already summarised so they stay small.
    if source_type == "transcript" and len(source_text) > 15000:
        source_text = source_text[:15000]
        logger.warning("quiz_generate: transcript truncated to 15000 chars to stay within token limits")

    # 3500 chars per chunk keeps each LLM call safely under 6000 TPM on any text type
    # (worst case Hindi text: 3500 ÷ 2 chars/token = 1750 tokens content
    #  + 400 system + 200 template + 800 output = ~3150 tokens total, well under limit)
    MAX_CHUNK_CHARS = 3500
    chunks = _chunk_notes(source_text, MAX_CHUNK_CHARS)
    n_chunks = len(chunks)

    # Distribute questions across chunks proportionally
    base_q = num_questions // n_chunks
    remainder = num_questions % n_chunks
    counts = [base_q + (1 if i < remainder else 0) for i in range(n_chunks)]

    template = get_template("quiz_generate")
    all_questions = []
    last_meta = {}
    all_latencies = []

    # Build one task per non-empty chunk; track metadata alongside each task
    chunk_meta = []  # (chunk_idx, q_count, start_pct, end_pct) for active chunks
    tasks = []
    for i, (chunk, q_count) in enumerate(zip(chunks, counts)):
        if q_count == 0:
            continue
        start_pct = max(5, int((i / n_chunks) * 90) + 5)
        end_pct = min(95, int(((i + 1) / n_chunks) * 90) + 5)
        user_prompt = template.user_template.format(
            lecture_title=lecture_title,
            topic_id=topic_id,
            num_questions=q_count,
            start_pct=start_pct,
            end_pct=end_pct,
            chunk_idx=i + 1,
            total_chunks=n_chunks,
            content=chunk,
        )
        tasks.append({
            "system_prompt": template.system,
            "user_prompt": user_prompt,
            "max_tokens": max(800, q_count * 350),
        })
        chunk_meta.append((i + 1, q_count, start_pct, end_pct))

    # Dispatch all chunks in parallel — each gets a DIFFERENT Groq API key so
    # every call fires simultaneously with its own full TPM budget.
    results = get_llm().parallel_complete_many(
        tasks=tasks,
        model="quiz",
        temperature=0.3,
        json_mode=False,
        institute_id=institute_id,
    )

    for result, (chunk_idx, q_count, start_pct, end_pct) in zip(results, chunk_meta):
        if result is None:
            logger.error("Quiz chunk %d/%d failed (institute=%s): no result", chunk_idx, n_chunks, institute_id)
            continue

        raw = result["content"] if isinstance(result["content"], str) else str(result["content"])
        parsed = _parse_quiz_json(raw)
        chunk_qs = parsed.get("questions", [])

        # Clamp triggerAtPercent to the chunk's range
        for q in chunk_qs:
            pct = q.get("triggerAtPercent", start_pct)
            try:
                q["triggerAtPercent"] = max(start_pct, min(end_pct, int(pct)))
            except (TypeError, ValueError):
                q["triggerAtPercent"] = start_pct

        all_questions.extend(chunk_qs)
        all_latencies.append(result["latency_ms"])
        last_meta = {"model": result["model"]}
        logger.info(
            "Quiz chunk %d/%d | source=%s | q_count=%d | got=%d",
            chunk_idx, n_chunks, source_type, q_count, len(chunk_qs),
        )

    if not all_questions:
        return Response({"error": "Quiz generation produced no questions. Try again."}, status=502)

    # Renumber IDs sequentially
    for idx, q in enumerate(all_questions):
        q["id"] = f"q{idx + 1}"

    return Response({
        "questions": all_questions,
        "_meta": {
            "source": source_type,
            "chunks": n_chunks,
            "requested": num_questions,
            "generated": len(all_questions),
            "institute": institute_id,
            "parallel": True,
            "wall_time_ms": round(max(all_latencies)) if all_latencies else 0,
            **last_meta,
        },
    })


# â"€â"€ AI #15 -- Text Translation (Sarvam AI) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#
# Uses Sarvam's mayura:v1 model -- purpose-built for Indian language translation.
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
        "Sarvam translation done | %d â†' %d chars | %.0fms",
        len(text), len(translated), latency_ms,
    )

    return Response({"translatedText": translated})


# â"€â"€ AI #16 -- Topic Content Generator â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        "Generate 12-15 flashcard pairs for this topic in Markdown. "
        "Format each as: **Q:** <question>  **A:** <answer>. "
        "Cover definitions, formulas, mechanisms, and application questions."
    ),
    "checklist": (
        "Generate a revision checklist for this topic in Markdown. "
        "Group items by sub-topic. Use - [ ] for each checkbox item. "
        "Include concepts to understand, formulas to memorise, and types of problems to practice."
    ),
    # â"€â"€ same as lesson/summary but with short label aliases â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    "study_guide":         "Generate a crisp, exam-ready summary of this topic in Markdown. Use bullet points and short paragraphs. Cover every exam-important concept.",
    "key_concepts":        "Generate a structured list of ALL key formulas and must-know concepts for this topic in Markdown. For each: name, definition, units (if applicable), one-line use-case.",
    "practice_questions":  (
        "Generate a Daily Practice Problem (DPP) set for this topic in Markdown. "
        "Include exactly 10 questions with a mix of difficulty (3 easy, 5 medium, 2 hard). "
        "For each question:\n"
        "- Number it (Q1, Q2 â€¦)\n"
        "- Write the question clearly (MCQ with 4 options labelled A-D, or numerical/short-answer)\n"
        "- After all questions, add a ## Answers section with: answer letter/value and a 2-3 line explanation for each.\n"
        "Ensure questions test understanding, not just recall."
    ),
    # â"€â"€ DPP â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    "dpp": (
        "Generate a high-quality Daily Practice Problem (DPP) sheet for this topic in Markdown, "
        "exactly as a top coaching institute would give students.\n\n"
        "Format:\n"
        "# DPP -- {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name} | **Date:** ______\n\n"
        "## Section A -- Multiple Choice (1 mark each)\n"
        "Generate 8 MCQ questions, each with 4 options (A-D). Mix easy and medium difficulty.\n\n"
        "## Section B -- Assertion-Reason (1 mark each)\n"
        "Generate 3 assertion-reason type questions.\n\n"
        "## Section C -- Numericals / Short Answer (3 marks each)\n"
        "Generate 4 numerical or short-answer problems.\n\n"
        "## Answer Key\n"
        "List all correct answers and brief hints/solutions.\n\n"
        "Questions must be syllabus-aligned, conceptually varied, and gradually increasing in difficulty."
    ),
    # â"€â"€ PYQ â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    "pyq": (
        "Generate a Previous Year Question (PYQ) style practice set for this topic in Markdown. "
        "Simulate the style of JEE Main / NEET questions from 2018-2024.\n\n"
        "Format:\n"
        "# PYQ Practice Set -- {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name}\n\n"
        "## JEE Main Style Questions\n"
        "Generate 6 questions in JEE Main MCQ style (single correct, 4 options A-D). "
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
        "Return ONLY the Markdown content -- no preamble, no 'Here is your content:' prefix."
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


# â"€â"€ AI #7b -- Notes from pre-existing Transcript (YouTube / manual) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#
# Called by NestJS when the lecture videoUrl is a YouTube link.
# The NestJS backend fetches the captions via youtube-transcript and sends
# the plain-text transcript here -- we skip Whisper entirely and go straight
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
        # Skip polish for Hindi/Hinglish: the 70b merge already outputs clean structured markdown.
        # Polish saves ~3,800 tokens per video × 8 keys = meaningful daily capacity gain.
        _is_hindi_lang = str(language or "").lower() in ("hi", "hi-in", "hinglish")
        if _is_hindi_lang:
            markdown_polished = False
        else:
            notes_markdown, markdown_polished = _polish_notes_markdown(
                notes_markdown, topic_id, language, institute_id,
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


# ── AI #7c – Full YouTube → Notes pipeline (captions fetched server-side) ─────
#
# NestJS sends { videoId, topicId, language }.  This endpoint fetches captions
# via the Python youtube-transcript-api library (more reliable on VPS/cloud IPs
# than the npm youtube-transcript package) and pipes them to the LLM pipeline.
#
# Body:   { videoId: str, topicId: str, language: str }
# Returns same shape as /stt/notes.

def _fetch_yt_captions_python(video_id: str) -> str:
    """
    Fetch YouTube captions for a video ID.
    Primary:  youtube-transcript-api  (pure Python, fast, no binary)
    Fallback: yt-dlp --write-auto-subs --skip-download  (more robust on server IPs,
              bypasses YouTube bot detection via regular yt-dlp user-agent rotation)
    """
    _noise = {"[music]", "[applause]", "[laughter]", "[noise]", "[inaudible]", "[ __ ]"}

    # ── Primary: youtube-transcript-api ───────────────────────────────────────
    try:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            import subprocess as _sp, sys as _sys
            _sp.check_call([_sys.executable, "-m", "pip", "install", "youtube-transcript-api", "--quiet"])
            from youtube_transcript_api import YouTubeTranscriptApi

        segments = None
        for langs in (["en", "en-US", "en-GB", "en-IN"], None):
            try:
                segments = (
                    YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
                    if langs
                    else YouTubeTranscriptApi.get_transcript(video_id)
                )
                if segments:
                    break
            except Exception:
                continue

        if segments:
            text = " ".join(
                s["text"].strip()
                for s in segments
                if s.get("text", "").strip() and s["text"].strip().lower() not in _noise
            ).strip()
            if len(text) > 20:
                logger.info("_fetch_yt_captions_python | transcript-api OK | videoId=%s | %d chars", video_id, len(text))
                return text
    except Exception as _e:
        logger.debug("youtube-transcript-api failed for %s: %s", video_id, _e)

    # ── Fallback: yt-dlp subtitle download (no video, captions only) ─────────
    logger.info("_fetch_yt_captions_python | falling back to yt-dlp for videoId=%s", video_id)
    try:
        import yt_dlp
    except ImportError:
        import subprocess as _sp2, sys as _sys2
        _sp2.check_call([_sys2.executable, "-m", "pip", "install", "yt-dlp", "--quiet"])
        import yt_dlp

    import json as _json
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _tmpdir:
        ydl_opts = {
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en", "en-US", "en-IN", "hi"],
            "subtitlesformat": "json3/srv3/ttml/vtt/best",
            "skip_download": True,
            "outtmpl": os.path.join(_tmpdir, "%(id)s"),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as _ydl:
                _ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as _dl_exc:
            raise ValueError(f"yt-dlp subtitle download failed for {video_id}: {_dl_exc}") from _dl_exc

        # Find the downloaded subtitle file
        sub_files = (
            _glob.glob(os.path.join(_tmpdir, "*.json3")) or
            _glob.glob(os.path.join(_tmpdir, "*.srv3")) or
            _glob.glob(os.path.join(_tmpdir, "*.vtt")) or
            _glob.glob(os.path.join(_tmpdir, "*"))
        )
        if not sub_files:
            raise ValueError(f"yt-dlp found no subtitle files for {video_id} — captions may be disabled")

        sub_file = sub_files[0]
        with open(sub_file, "r", encoding="utf-8") as _f:
            raw = _f.read()

        # Parse JSON3 (YouTube's native caption format)
        if sub_file.endswith(".json3"):
            data = _json.loads(raw)
            parts = []
            for event in data.get("events", []):
                for seg in event.get("segs", []):
                    t = seg.get("utf8", "").replace("\n", " ").strip()
                    if t and t.lower() not in _noise:
                        parts.append(t)
            text = " ".join(parts).strip()
        else:
            # VTT / SRV3 — strip timestamps and headers
            text = re.sub(r"\d{2}:\d{2}[:\.,]\d{2,3}\s*-->\s*\d{2}:\d{2}[:\.,]\d{2,3}[^\n]*", "", raw)
            text = re.sub(r"^WEBVTT.*$", "", text, flags=re.MULTILINE)
            text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
            text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags
            text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 20:
            raise ValueError(f"yt-dlp subtitle file empty after parsing for {video_id}")

        logger.info("_fetch_yt_captions_python | yt-dlp fallback OK | videoId=%s | %d chars", video_id, len(text))
        return text


@api_view(["POST"])
def generate_notes_from_youtube(request):
    import time as _time

    data = request.data
    video_id = (data.get("videoId") or data.get("video_id") or "").strip()
    topic_id = data.get("topicId", "")
    language = data.get("language", "en")
    institute_id = getattr(request, "institute_id", "default")

    if not video_id:
        return Response({"error": "Missing videoId"}, status=400)

    _t0 = _time.perf_counter()

    # ── Step 1: Fetch captions ──────────────────────────────────────────────────
    try:
        transcript = _fetch_yt_captions_python(video_id)
        logger.info(
            "generate_notes_from_youtube | videoId=%s | captions=%d chars | topic=%s",
            video_id, len(transcript), topic_id,
        )
    except Exception as cap_exc:
        logger.warning(
            "generate_notes_from_youtube | captions unavailable for %s: %s",
            video_id, cap_exc,
        )
        return Response(
            {
                "error": "captions_unavailable",
                "detail": (
                    f"Could not fetch captions for video {video_id}. "
                    "Ensure captions are enabled on the YouTube video, "
                    "or re-upload the lecture as a file."
                ),
            },
            status=422,
        )

    if len(transcript) < 20:
        return Response(
            {
                "error": "transcript_too_short",
                "detail": "The YouTube captions are too short to generate meaningful notes.",
            },
            status=422,
        )

    # ── Step 2: LLM summarisation ───────────────────────────────────────────────
    english_transcript, prep_meta = _prepare_transcript_for_notes(
        transcript, topic_id, language, institute_id
    )

    try:
        notes_markdown, notes_meta = _generate_comprehensive_notes(
            english_transcript, topic_id, language, institute_id
        )
        notes_markdown, markdown_polished = _polish_notes_markdown(
            notes_markdown, topic_id, language, institute_id
        )
    except RuntimeError as exc:
        logger.error("generate_notes_from_youtube LLM failed for %s: %s", video_id, exc)
        return Response({"error": str(exc)}, status=502)

    logger.info(
        "generate_notes_from_youtube done | videoId=%s | notes=%d chars | took=%.1fs",
        video_id, len(notes_markdown), _time.perf_counter() - _t0,
    )

    return Response({
        "notes": notes_markdown,
        "rawTranscript": transcript,
        "englishTranscript": english_transcript,
        "keyConcepts": [],
        "formulas": [],
        "summary": "",
        "_meta": {
            "source": "youtube_captions_python",
            "video_id": video_id,
            "latency_ms": round((_time.perf_counter() - _t0) * 1000),
            "transcript_chars": len(transcript),
            "institute": institute_id,
            "chunk_count": notes_meta.get("chunk_count", 0),
            "merge_applied": notes_meta.get("merge_applied", False),
            "markdown_polished": markdown_polished,
            "quality_flags": prep_meta.get("quality_flags", []),
            "repair_applied": prep_meta.get("repair_applied", False),
        },
    })


# ── AI Engine Health Check ────────────────────────────────────────────────────
#
# Returns status of all configured AI language model keys.
# Uses in-memory state for instant response (no live API calls by default).
# Pass ?refresh=true to re-probe all keys (~2-5s, one call per key).

@api_view(["GET"])
def ai_engine_health(request):
    from ai_services.core.llm_client import (
        GROQ_API_KEYS, _DISABLED_GROQ_KEYS, _KEY_STATE_LOCK, check_groq_keys,
    )

    refresh = request.query_params.get("refresh", "false").lower() == "true"

    if refresh:
        summary = check_groq_keys()
    else:
        with _KEY_STATE_LOCK:
            dead_keys = set(_DISABLED_GROQ_KEYS)
        total = len(GROQ_API_KEYS)
        dead = len(dead_keys)
        usable = total - dead
        summary = {"total": total, "ok": usable, "rate_limited": 0, "dead": dead, "error": 0, "usable": usable}

    with _KEY_STATE_LOCK:
        dead_keys = set(_DISABLED_GROQ_KEYS)

    keys_status = []
    for i, key in enumerate(GROQ_API_KEYS):
        hint = f"{key[:4]}…{key[-3:]}" if len(key) > 8 else "****"
        status = "dead" if key in dead_keys else "ok"
        keys_status.append({"index": i + 1, "hint": hint, "status": status})

    if summary.get("usable", 0) == 0:
        overall = "critical"
    elif summary.get("dead", 0) > 0 or summary.get("error", 0) > 0:
        overall = "degraded"
    else:
        overall = "operational"

    return Response({
        "overall": overall,
        "summary": summary,
        "keys": keys_status,
        "cached": not refresh,
    })
