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
import time
from typing import Optional



import requests as _requests



from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response



from ai_services.core.model_tier import get_model_for_task
from ai_services.core.prompt_templates import get_template
from ai_services.core.groq_keys import get_groq_api_keys, get_rotated_groq_keys, is_key_exhausted_error
from ai_services.core.gemini_keys import (
    gemini_key_count,
    get_rotated_gemini_keys,
    has_gemini_api_key,
    is_gemini_permanent_key_error,
    is_gemini_retryable_error,
    mark_gemini_key_disabled,
)
from ai_services.core.llm_client import _JSON_MODE_TUTOR_SUFFIX
from ai_services.core.usage_logger import log_usage
from ai_services.core.serpapi_images import search_google_images
from .base import ai_call, ai_call_text, get_llm



logger = logging.getLogger("ai_services.llm")

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

def _resolve_institute_id(request) -> str:
    """
    Return the school's UUID for usage attribution.

    Precedence:
      1. request.institute_id when it already looks like a UUID
         (middleware correctly resolved it — e.g. is_service_account=True)
      2. X-Tenant-ID header (always the NestJS school UUID; safe fallback
         when middleware fell back to slug because is_service_account=False)
      3. raw request.institute_id as-is (could be a slug — NestJS will reject
         invalid UUID, but at least the event is logged)
    """
    mid = getattr(request, "institute_id", "") or ""
    if _UUID_RE.match(mid):
        return mid
    xtid = request.headers.get("X-Tenant-ID", "").strip()
    if xtid and _UUID_RE.match(xtid):
        logger.warning(
            "institute_id from middleware (%r) is not a UUID; using X-Tenant-ID (%s) instead",
            mid, xtid,
        )
        return xtid
    return mid or "unknown"


@api_view(["POST"])
def search_educational_images(request):
    """Search Google Images through SerpApi without exposing its API key."""
    query = str(request.data.get("query") or "").strip()
    if not query:
        return Response({"error": "Missing query"}, status=400)
    try:
        limit = int(request.data.get("limit") or 5)
        images = search_google_images(query, limit, request.data.get("language") or "")
        return Response({"images": images, "provider": "serpapi"})
    except RuntimeError as error:
        logger.warning("SerpApi image search unavailable: %s", error)
        return Response({"error": str(error)}, status=503)
    except Exception as error:
        logger.warning("SerpApi image search failed: %s", error)
        return Response({"error": "Image search failed"}, status=502)



# -- Groq Whisper API (primary -- cloud, fast; multi-key rotation) ---------------




# ISSUE-3 FIX: Use the single canonical key loader from groq_keys.py.
# The old inline key list only loaded keys 0–13, while groq_keys.py loads all keys 1–32.
# All Whisper transcription calls now use the same complete key pool as LLM calls.
GROQ_API_KEYS: list[str] = get_groq_api_keys()
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



    # Hinglish → send "hi" so Whisper outputs Devanagari+Latin (code-switching)
    # instead of auto-detecting and picking Urdu (acoustically identical to Hindi).
    # "auto" still means no hint (let Whisper detect freely).
    groq_language: str | None = "hi" if language == "hinglish" else (None if language == "auto" else language)

    # For Hinglish, prime Whisper with a bilingual prompt so it code-switches
    # to Latin for English words instead of forcing everything into Devanagari.
    hinglish_primer = (
        "यह एक हिंदी-अंग्रेजी मिश्रित (Hinglish) व्याख्यान है। "
        "Hindi words should be in Devanagari, English words in Latin script."
    ) if language == "hinglish" else ""

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
        # Merge hinglish primer with any rolling context window
        combined_prompt = hinglish_primer
        if prev_context:
            raw = prev_context.encode("utf-8")[-880:]
            ctx = raw.decode("utf-8", errors="ignore")
            combined_prompt = (hinglish_primer + " " + ctx).strip() if hinglish_primer else ctx
        if combined_prompt:
            kwargs["prompt"] = combined_prompt



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
    # faster-whisper doesn't know "hinglish" — use "hi" so it outputs
    # Devanagari+Latin (code-switched) instead of auto-detecting Urdu.
    fw_language = "hi" if language == "hinglish" else language
    segments, info = whisper.transcribe(
        audio_path,
        beam_size=5,
        language=fw_language,
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





def _normalize_lecture_language(language: str | None) -> str:
    """Map language aliases to a canonical short code (od/hi/hinglish/en)."""
    lang = str(language or "en").strip().lower()
    if lang in ("odia", "od-in", "or", "or-in"):
        return "od"
    if lang in ("hindi", "hi-in"):
        return "hi"
    return lang or "en"


def _is_odia_language(language: str | None) -> bool:
    return _normalize_lecture_language(language) == "od"


def _transcribe_audio(audio_url: str, language: str = "hi") -> str:
    """
    Odia:     Sarvam Speech-to-Text  (Whisper cannot transcribe Odia)
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

        # ── Odia: Sarvam Speech-to-Text (Whisper cannot transcribe Odia) ──────
        if _is_odia_language(language):
            try:
                import subprocess
                try:
                    import imageio_ffmpeg
                except ImportError:
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg", "--quiet"])
                    import imageio_ffmpeg
                from ai_services.core.sarvam_client import transcribe_file as _sarvam_transcribe_file

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                chunk_seconds = int(os.getenv("SARVAM_STT_CHUNK_SECONDS", "25"))
                chunk_seconds = max(10, min(chunk_seconds, 25))
                chunk_pattern = os.path.join(tmpdir, "sarvam_chunk_%04d.mp3")
                logger.info(
                    "Chunking Odia audio for Sarvam STT | segment_time=%ss | lang=od-IN",
                    chunk_seconds,
                )
                cmd = [
                    ffmpeg_exe, "-y", "-i", audio_path,
                    "-f", "segment", "-segment_time", str(chunk_seconds),
                    "-c:a", "libmp3lame", "-ac", "1", "-ar", "16000", "-ab", "64k",
                    "-vn", chunk_pattern,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                chunks = sorted(_glob.glob(os.path.join(tmpdir, "sarvam_chunk_*.mp3")))
                if not chunks:
                    raise RuntimeError("FFMpeg generated no Sarvam audio chunks.")

                def _split_sarvam_chunk(source_file: str, chunk_idx: int) -> list:
                    sub_pattern = os.path.join(tmpdir, f"sarvam_chunk_{chunk_idx:04d}_part_%03d.mp3")
                    cmd2 = [
                        ffmpeg_exe, "-y", "-i", source_file,
                        "-f", "segment", "-segment_time", "15", "-reset_timestamps", "1",
                        "-c:a", "libmp3lame", "-ac", "1", "-ar", "16000", "-ab", "64k",
                        "-vn", sub_pattern,
                    ]
                    subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return sorted(_glob.glob(os.path.join(tmpdir, f"sarvam_chunk_{chunk_idx:04d}_part_*.mp3")))

                transcript_parts: list = []
                for idx, chunk_file in enumerate(chunks):
                    logger.info("Sarvam Odia STT | chunk %d/%d", idx + 1, len(chunks))
                    try:
                        part = _sarvam_transcribe_file(chunk_file, language="od")
                    except Exception as exc:
                        msg = str(exc)
                        if "duration exceeds the maximum limit" in msg.lower():
                            logger.warning(
                                "Sarvam Odia STT chunk %d/%d exceeded 30s; splitting into 15s subchunks",
                                idx + 1, len(chunks),
                            )
                            sub_parts: list = []
                            for sub_idx, sub_file in enumerate(_split_sarvam_chunk(chunk_file, idx + 1)):
                                try:
                                    sub_text = _sarvam_transcribe_file(sub_file, language="od")
                                except Exception as sub_exc:
                                    logger.warning(
                                        "Sarvam Odia STT chunk %d.%d failed: %s",
                                        idx + 1, sub_idx + 1, sub_exc,
                                    )
                                    sub_text = ""
                                if sub_text:
                                    sub_parts.append(sub_text)
                            part = " ".join(sub_parts).strip()
                        else:
                            logger.warning("Sarvam Odia STT chunk %d/%d failed: %s", idx + 1, len(chunks), exc)
                            part = ""
                    if part:
                        transcript_parts.append(part)

                transcript = " ".join(transcript_parts).strip()
                if not transcript:
                    raise RuntimeError("Sarvam returned no Odia transcript text")
                logger.info("Sarvam Odia transcription OK -- %d chars (from %d chunks)", len(transcript), len(chunks))
                return transcript
            except Exception as exc:
                raise RuntimeError(f"Sarvam Odia transcription failed: {exc}") from exc



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
    cleaned = re.sub(r"[^\x00-\x7F\u0900-\u097F\u0B00-\u0B7F\u03B1-\u03C9\u0391-\u03A9]+", lambda m: m.group(0) if len(m.group(0).strip()) <= 3 else " ", cleaned)
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



    if re.search(r"[^\x00-\x7F\u0900-\u097F\u0B00-\u0B7F\u03B1-\u03C9\u0391-\u03A9]{8,}", sample):
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
    lang = _normalize_lecture_language(language)
    is_odia = lang == "od"
    is_hindi_hinglish = lang in ("hi", "hi-in", "hinglish")

    normalized = _normalize_transcript_to_english(transcript, language, institute_id)
    cleaned = _clean_transcript_text(normalized)
    cleaned = _strip_lecture_framing(cleaned)
    flags = _transcript_quality_flags(cleaned)
    # Odia must NOT go through the English/Groq repair path — it can misclassify valid
    # Odia script as noise and truncate a full transcript to a tiny repair input.
    repaired = cleaned if is_odia else (_repair_low_quality_transcript(cleaned, topic_id, language, institute_id, flags) if flags else cleaned)
    final_text = _clean_transcript_text(repaired)



    if is_odia:
        # Gemini reads Odia natively; skip English word-repair and punctuation refinement.
        return final_text, {
            "quality_flags": flags,
            "repair_applied": False,
            "word_repair_applied": False, "word_repair_chunks": 0, "word_repair_chunks_accepted": 0,
            "punct_refine_applied": False, "punct_refine_chunks": 0, "punct_refine_chunks_accepted": 0,
        }



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

# ── Odia notes via Gemini (Groq/llama are weak at Odia script) ───────────────
GEMINI_ODIA_NOTES_ENABLED = os.getenv("GEMINI_ODIA_NOTES_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
GEMINI_ODIA_NOTES_MODEL = os.getenv("GEMINI_ODIA_NOTES_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
GEMINI_ODIA_NOTES_CHUNK_CHARS = int(os.getenv("GEMINI_ODIA_NOTES_CHUNK_CHARS", "5000"))
GEMINI_ODIA_NOTES_MAX_CHUNK_CHARS = int(os.getenv("GEMINI_ODIA_NOTES_MAX_CHUNK_CHARS", "5000"))
GEMINI_ODIA_SECTION_MAX_TOKENS = int(os.getenv("GEMINI_ODIA_SECTION_MAX_TOKENS", "2200"))
GEMINI_ODIA_MERGE_MAX_TOKENS = int(os.getenv("GEMINI_ODIA_MERGE_MAX_TOKENS", "3000"))
GEMINI_ODIA_MERGE_INPUT_CHARS = int(os.getenv("GEMINI_ODIA_MERGE_INPUT_CHARS", "14000"))
GEMINI_ODIA_DETERMINISTIC_MERGE = os.getenv("GEMINI_ODIA_DETERMINISTIC_MERGE", "true").strip().lower() not in {"0", "false", "no", "off"}
GEMINI_ODIA_REQUEST_SPACING_SECONDS = float(os.getenv("GEMINI_ODIA_REQUEST_SPACING_SECONDS", "13"))





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





def _hard_split_text(text: str, max_chars: int) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind("\n", start, end),
                cleaned.rfind(". ", start, end),
                cleaned.rfind("। ", start, end),
                cleaned.rfind(" ", start, end),
            )
            if boundary > start + int(max_chars * 0.55):
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _cap_chunks(chunks: list[str], max_chars: int) -> list[str]:
    capped: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            capped.append(chunk)
        else:
            capped.extend(_hard_split_text(chunk, max_chars))
    return [chunk for chunk in capped if chunk.strip()]


def _odia_script_ratio(text: str) -> float:
    letters = [ch for ch in str(text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    odia_letters = sum(1 for ch in letters if "଀" <= ch <= "୿")
    return odia_letters / max(len(letters), 1)


def _looks_like_bad_odia_notes(notes: str) -> bool:
    text = str(notes or "").strip()
    if len(text) < 300:
        return True
    if _odia_script_ratio(text) < 0.35:
        return True
    malformed_bullets = re.findall(r"(?m)^\s*[-*•]\s*(?:[*()/\\|&A-Z]{1,12})\s*$", text)
    if len(malformed_bullets) >= 2:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        weak_lines = sum(1 for line in lines if len(line) <= 8 and not line.startswith("#"))
        if weak_lines >= max(4, len(lines) // 4):
            return True
    repeated = re.search(r"(\b[\w଀-୿]{1,20}\b)(?:\s+\1){4,}", text, re.IGNORECASE)
    return bool(repeated)


def _gemini_retry_delay_seconds(error: Exception, fallback: float) -> float:
    text = str(error or "")
    matches = re.findall(r"retry(?:\s+in|Delay['\"]?:\s*)\s*'?(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
    if matches:
        try:
            return min(float(matches[-1]) + 2.0, 90.0)
        except ValueError:
            pass
    matches = re.findall(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", text, flags=re.IGNORECASE)
    if matches:
        try:
            return min(float(matches[-1]) + 2.0, 90.0)
        except ValueError:
            pass
    return fallback


def _gemini_odia_generate(system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:
    if not has_gemini_api_key():
        raise RuntimeError("GEMINI_API_KEY is not set -- add it to .env to enable Gemini Odia notes")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(f"google-genai is unavailable: {exc}") from exc

    last_exc: Exception | None = None
    response = None
    for key_index, api_key in get_rotated_gemini_keys():
        try:
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model=GEMINI_ODIA_NOTES_MODEL,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except TypeError:
                response = client.models.generate_content(
                    model=GEMINI_ODIA_NOTES_MODEL,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        max_output_tokens=max_tokens,
                    ),
                )
            logger.info(
                "Gemini Odia notes call OK | key=%d/%d | model=%s",
                key_index, gemini_key_count(), GEMINI_ODIA_NOTES_MODEL,
            )
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if is_gemini_permanent_key_error(msg):
                mark_gemini_key_disabled(api_key)
                logger.warning("Gemini key %d/%d disabled for permanent error: %s", key_index, gemini_key_count(), msg[:180])
                continue
            if is_gemini_retryable_error(msg):
                logger.warning("Gemini key %d/%d retryable error; rotating to next key: %s", key_index, gemini_key_count(), msg[:220])
                continue
            raise RuntimeError(f"Gemini Odia notes failed: {exc}") from exc
    else:
        raise RuntimeError(f"Gemini Odia notes failed across all {gemini_key_count()} key(s): {last_exc}") from last_exc

    content = str(getattr(response, "text", "") or "").strip()
    if not content:
        raise RuntimeError("Gemini returned empty Odia notes")
    content = re.sub(r"^```(?:markdown)?\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()
    if _looks_like_bad_odia_notes(content):
        raise RuntimeError("Gemini returned malformed or low-quality Odia notes")
    content = _cleanup_odia_notes_markdown(content)
    logger.info("Gemini Odia notes OK | model=%s | chars=%d", GEMINI_ODIA_NOTES_MODEL, len(content))
    return content


def _generate_gemini_odia_chunk_notes(chunk_text: str, topic_id: str, chunk_index: int, total_chunks: int) -> str:
    system_prompt = (
        "Write transcript-grounded Odia Markdown lecture notes only. Do not show reasoning, planning, analysis, or steps. "
        "Use only the supplied transcript chunk. Do not use outside knowledge, even if correct. "
        "If a term, fact, count, or number is not explicitly present in this chunk, omit it. "
        "Ignore greetings, filler, repeated words, ASR noise, and non-academic chatter. "
        "Keep formulas, abbreviations, and standard scientific names in English where natural. "
        "Preserve each numeric value with its exact subject and unit; never transfer a number from one concept to another."
    )
    user_prompt = (
        f"Lecture topic: {topic_id or 'General'}\n"
        f"Transcript section: {chunk_index} of {total_chunks}\n\n"
        "Return only final notes in Odia script. Use headings, short paragraphs, and useful bullets. "
        "Preserve definitions, examples, comparisons, formulas, and exam points when present. "
        "Do not include isolated tokens like AND, OR, *, /, or brackets. Do not add outside facts. "
        "Do not infer standard textbook details that the teacher did not say in this chunk.\n\n"
        f"Transcript:\n{chunk_text}"
    )
    return _gemini_odia_generate(system_prompt, user_prompt, max_tokens=GEMINI_ODIA_SECTION_MAX_TOKENS)


def _normalize_odia_note_line_for_dedupe(line: str) -> str:
    text = re.sub(r"^[#*\-\s•]+", "", str(line or "").strip())
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[।.!:：\-–—]+$", "", text)
    return text.strip().lower()


def _cleanup_odia_notes_markdown(markdown: str) -> str:
    text = str(markdown or "").replace("\\n", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"```(?:markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"(?m)^\s*(?:---\s*)?SECTION\s+\d+\s*(?:---)?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*<!--.*?-->\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s*(?:AND|OR|[*/()\\|&]+)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^([^\n#*\-•].{0,80})\s*:\s*$", r"## \1", text)
    text = re.sub(r"(?m)^#{3,}\s+", "## ", text)

    cleaned_lines: list[str] = []
    seen_headings: set[str] = set()
    recent_content: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        line = re.sub(r"^\s*[•]\s*", "- ", line)
        line = re.sub(r"^\s*[*]\s+(?=\S)", "- ", line)
        line = re.sub(r"^-{2,}\s*$", "", line).strip()
        if not line:
            continue
        norm = _normalize_odia_note_line_for_dedupe(line)
        if not norm or len(norm) <= 2:
            continue
        if line.startswith("#"):
            if norm in seen_headings:
                continue
            seen_headings.add(norm)
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            cleaned_lines.append(line)
            cleaned_lines.append("")
            continue
        if len(norm) > 24 and norm in recent_content:
            continue
        recent_content.append(norm)
        if len(recent_content) > 80:
            recent_content = recent_content[-80:]
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?m)^##\s+##\s+", "## ", cleaned)
    cleaned = re.sub(r"(?m)^-\s*-\s+", "- ", cleaned)
    return cleaned.strip()


def _assemble_gemini_odia_notes(chunk_notes: list[str], topic_id: str) -> str:
    sections = [_cleanup_odia_notes_markdown(section) for section in chunk_notes if str(section or "").strip()]
    sections = [section for section in sections if section]
    body = "\n\n".join(sections).strip()
    title = "# ବ୍ୟାଖ୍ୟାନ ନୋଟ୍ସ"
    if topic_id:
        title = f"# ବ୍ୟାଖ୍ୟାନ ନୋଟ୍ସ\n\n<!-- topic: {topic_id} -->"
    return _cleanup_odia_notes_markdown(f"{title}\n\n{body}")


def _generate_gemini_odia_chunk_notes_with_retry(chunk_text: str, topic_id: str, chunk_index: int, total_chunks: int) -> str:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return _generate_gemini_odia_chunk_notes(chunk_text, topic_id, chunk_index, total_chunks)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if attempt < 2 and any(token in msg for token in ("503", "unavailable", "high demand", "rate", "timeout", "temporarily")):
                wait = _gemini_retry_delay_seconds(exc, 2.0 * (attempt + 1))
                logger.warning("Gemini Odia chunk %d/%d transient failure; retrying in %.1fs (%s)", chunk_index, total_chunks, wait, exc)
                time.sleep(wait)
                continue
            break
    raise RuntimeError(f"Gemini Odia chunk {chunk_index}/{total_chunks} failed after retries: {last_exc}") from last_exc


def _merge_gemini_odia_notes(chunk_notes: list[str], topic_id: str) -> str:
    combined_sections = "\n\n".join(
        f"--- SECTION {idx + 1} ---\n{section.strip()}"
        for idx, section in enumerate(chunk_notes)
        if str(section).strip()
    ).strip()
    if not combined_sections:
        return ""
    if len(combined_sections) > GEMINI_ODIA_MERGE_INPUT_CHARS:
        logger.warning(
            "Gemini Odia merge input truncated %d -> %d chars",
            len(combined_sections), GEMINI_ODIA_MERGE_INPUT_CHARS,
        )
        combined_sections = combined_sections[:GEMINI_ODIA_MERGE_INPUT_CHARS]

    system_prompt = (
        "Merge Odia section notes into final Odia Markdown lecture notes only. "
        "Do not show reasoning, planning, analysis, or steps. Remove duplication and malformed bullets. "
        "Preserve numeric values with their exact subject and unit; do not move a number from one concept to another."
    )
    user_prompt = (
        f"Lecture topic: {topic_id or 'General'}\n\n"
        "Return only the final notes in Odia script. Start with a title, organize logical sections, "
        "keep transcript-grounded examples/formulas, and end with a concise summary.\n\n"
        f"{combined_sections}"
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return _gemini_odia_generate(system_prompt, user_prompt, max_tokens=GEMINI_ODIA_MERGE_MAX_TOKENS)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if attempt < 2 and any(token in msg for token in ("503", "unavailable", "high demand", "rate", "timeout", "temporarily")):
                wait = _gemini_retry_delay_seconds(exc, 2.0 * (attempt + 1))
                logger.warning("Gemini Odia merge transient failure; retrying in %.1fs (%s)", wait, exc)
                time.sleep(wait)
                continue
            break
    raise RuntimeError(f"Gemini Odia merge failed after retries: {last_exc}") from last_exc


def _generate_gemini_odia_comprehensive_notes(transcript: str, topic_id: str) -> tuple[str, dict]:
    if not GEMINI_ODIA_NOTES_ENABLED:
        raise RuntimeError("Gemini Odia notes are disabled by GEMINI_ODIA_NOTES_ENABLED=false")

    chunk_chars = max(1800, min(GEMINI_ODIA_NOTES_CHUNK_CHARS, GEMINI_ODIA_NOTES_MAX_CHUNK_CHARS))
    chunks = _cap_chunks(
        _chunk_transcript(transcript, chunk_size=chunk_chars, overlap=0),
        chunk_chars,
    )
    if not chunks:
        return "", {"chunk_count": 0, "provider": "gemini", "model": GEMINI_ODIA_NOTES_MODEL}

    logger.info(
        "Generating Gemini Odia notes | chunks=%d | chunk_chars=%d | transcript_chars=%d | model=%s | topic=%s",
        len(chunks), chunk_chars, len(transcript), GEMINI_ODIA_NOTES_MODEL, topic_id,
    )

    if len(chunks) == 1:
        notes = _generate_gemini_odia_chunk_notes_with_retry(chunks[0], topic_id, 1, 1)
        return notes, {
            "chunk_count": 1,
            "merge_applied": False,
            "provider": "gemini",
            "model": GEMINI_ODIA_NOTES_MODEL,
        }

    partial_notes: list[str] = []
    failed_chunks = 0
    for i, chunk in enumerate(chunks):
        try:
            partial_notes.append(_generate_gemini_odia_chunk_notes_with_retry(chunk, topic_id, i + 1, len(chunks)))
        except Exception as exc:
            logger.warning("Gemini Odia chunk %d/%d failed (%s)", i + 1, len(chunks), exc)
            failed_chunks += 1
            partial_notes.append("")
        if i < len(chunks) - 1 and GEMINI_ODIA_REQUEST_SPACING_SECONDS > 0:
            logger.info("Waiting %.1fs before next Gemini Odia chunk to respect rate limits", GEMINI_ODIA_REQUEST_SPACING_SECONDS)
            time.sleep(GEMINI_ODIA_REQUEST_SPACING_SECONDS)

    non_empty = [p for p in partial_notes if p.strip()]
    if failed_chunks:
        raise RuntimeError(f"Gemini Odia notes incomplete: {failed_chunks}/{len(chunks)} transcript chunks failed")
    if not non_empty:
        return "", {
            "chunk_count": len(chunks),
            "failed_chunks": failed_chunks,
            "provider": "gemini",
            "model": GEMINI_ODIA_NOTES_MODEL,
            "error": "all_chunks_failed",
        }

    if GEMINI_ODIA_DETERMINISTIC_MERGE:
        merged = _assemble_gemini_odia_notes(non_empty, topic_id)
        return merged, {
            "chunk_count": len(chunks),
            "failed_chunks": failed_chunks,
            "merge_applied": False,
            "merge_strategy": "deterministic_concat",
            "provider": "gemini",
            "model": GEMINI_ODIA_NOTES_MODEL,
        }

    merged = _merge_gemini_odia_notes(non_empty, topic_id)
    return merged, {
        "chunk_count": len(chunks),
        "failed_chunks": failed_chunks,
        "merge_applied": True,
        "provider": "gemini",
        "model": GEMINI_ODIA_NOTES_MODEL,
    }


def _generate_comprehensive_notes(transcript: str, topic_id: str, language: str, institute_id: str) -> tuple[str, dict]:
    import time as _time



    if _is_odia_language(language):
        return _generate_gemini_odia_comprehensive_notes(transcript, topic_id)



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
    _t0_generate = _time.perf_counter()
    _MAX_GEN_TIME = 600  # 10 minutes absolute max for the chunk phase

    

    for i, chunk in enumerate(chunks):
        if _time.perf_counter() - _t0_generate > _MAX_GEN_TIME:
            logger.warning("Chunk generation timed out after %.1f seconds. Using %d/%d partial chunks.", _time.perf_counter() - _t0_generate, len(partial_notes), n)
            break

            

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





# ── AI #1 -- Doubt Clearing ──────────────────────────────────────────────────



# Step-1 detector: lightweight classification (subject + question type)
_DOUBT_DETECTOR_SYSTEM = (
    "You are a high-precision academic classifier for JEE/NEET. Categorize the question into Subject and Type.\\n"
    "Subjects:\\n"
    "- biology: Human health, diseases, immune system, allergy, genetics, cell biology, plant/animal physiology.\\n"
    "- chemistry: Reactions, bonding, molarity, organic/inorganic compounds, thermodynamics.\\n"
    "- physics: Mechanics, optics, electricity, magnetism, modern physics.\\n"
    "- math: Calculus, algebra, geometry, probability.\\n"
    "Types: numerical, derivation, conceptual, mcq, theory\\n"
    "Respond with ONLY a valid JSON object.\\n"
    'Format: {"subject": "<subject>", "type": "<type>"}\\n'
)



# ── Keyword-based subject & type detection (runs before LLM, free + instant) ──

PHYSICS_KEYWORDS = ['block', 'mass', 'incline', 'friction', 'force', 'velocity', 'acceleration', 'momentum', 'torque', 'current', 'voltage', 'resistance', 'lens', 'mirror', 'charge', 'electric', 'magnetic', 'wave', 'frequency', 'pendulum', 'spring', 'collision', 'projectile', 'gravity', 'newton', 'joule', 'watt', 'ampere', 'capacitor', 'inductor', 'circuit', 'magnetic field', 'refraction', 'reflection', 'doppler', 'thermodynamics', 'entropy', 'carnot', 'pressure', 'temperature', 'radioactive', 'nuclear', 'photon', 'electron', 'proton', 'neutron', 'kinetic energy', 'potential energy', 'power', 'work done', 'semiconductor', 'diode', 'transistor', 'logic gate', 'photoelectric', 'work function', 'de broglie', 'bohr model', 'spectroscopy', 'interference', 'diffraction', 'polarization', 'youngs double slit', 'viscosity', 'surface tension', 'bernoulli', 'elasticity', 'stress', 'strain', 'hookes law', 'shm', 'oscillation', 'resonance', 'doppler effect', 'superposition', 'gravitation', 'escape velocity', 'orbital', 'kepler', 'angular momentum', 'centripetal', 'centrifugal', 'moment of inertia', 'rotational', 'torque', 'rolling', 'rigid body', 'vector', 'scalar', 'kinematics', 'displacement', 'capacitance', 'inductance', 'gauss law', 'ampere law', 'faraday law', 'lenz law', 'ac current', 'rms value', 'transformer', 'prism', 'magnification', 'resolution', 'focal length', 'diopter']

CHEMISTRY_KEYWORDS = ['mole', 'molarity', 'molality', 'ka', 'kb', 'ksp', 'ph', 'reaction', 'compound', 'element', 'titration', 'equilibrium', 'bond', 'entropy', 'enthalpy', 'oxidation', 'reduction', 'acid', 'base', 'salt', 'organic', 'alkane', 'alkene', 'alkyne', 'benzene', 'ester', 'aldehyde', 'ketone', 'alcohol', 'phenol', 'ether', 'molar mass', 'dissociation', 'buffer', 'normality', 'equivalent', 'hybridization', 'isomer', 'polymer', 'monomer', 'catalyst', 'activation energy', 'rate constant', 'order of reaction', 'electrode', 'electrolysis', 'galvanic', 'cell potential', 'freezing point', 'boiling point', 'osmotic pressure', 'vapour pressure', 'raoult', 'henry law', 'colligative', 'solid state', 'lattice', 'unit cell', 'bragg', 'surface chemistry', 'adsorption', 'colloid', 'coordination', 'ligand', 'nomenclature', 'iupac', 'mechanism', 'sn1', 'sn2', 'e1', 'e2', 'electrophile', 'nucleophile', 'resonance', 'inductive effect', 'mesomeric', 'hyperconjugation', 'carbocation', 'carbanion', 'free radical', 'metallurgy', 'ore', 'refining', 's-block', 'p-block', 'd-block', 'f-block', 'lanthanide', 'actinide', 'periodic table', 'electronegativity', 'ionization enthalpy', 'electron gain enthalpy', 'biomolecule', 'carbohydrate', 'protein', 'vitamin', 'nucleic acid', 'amino acid', 'polymerization']

MATH_KEYWORDS = ['integrate', 'differentiate', 'derivative', 'matrix', 'determinant', 'probability', 'parabola', 'ellipse', 'hyperbola', 'complex number', 'binomial', 'permutation', 'combination', 'limit', 'series', 'sequence', 'polynomial', 'quadratic', 'trigonometric identity', 'inverse trigonometric', 'definite integral', 'indefinite integral', 'integral', 'differential equation', 'coordinate geometry', 'straight line', 'locus', 'conic section', 'arithmetic progression', 'geometric progression', 'harmonic progression', 'logarithm', 'exponential', 'inequality', 'modulus', 'function', 'domain', 'range', 'one-to-one', 'onto', 'composite function', 'continuity', 'differentiability', 'mean value theorem', 'maxima', 'minima', 'tangent', 'normal', 'area under curve', 'vector algebra', 'dot product', 'cross product', 'scalar triple product', '3d geometry', 'plane', 'direction cosines', 'statistics', 'mean', 'median', 'variance', 'standard deviation', 'bayes theorem', 'random variable', 'bernoulli trials', 'mathematical induction', 'set theory', 'relation', 'logic', 'truth table']

BIOLOGY_KEYWORDS = ['cell', 'mitosis', 'meiosis', 'photosynthesis', 'dna', 'rna', 'transcription', 'translation', 'replication', 'eukaryote', 'prokaryote', 'enzyme', 'hormone', 'ecosystem', 'genetics', 'neuron', 'chromosome', 'protein', 'ribosome', 'chloroplast', 'mitochondria', 'evolution', 'respiration', 'digestion', 'excretion', 'reproduction', 'immunity', 'biodiversity', 'food chain', 'biomolecule', 'nitrogen cycle', 'krebs cycle', 'calvin cycle', 'glycolysis', 'atp', 'adp', 'nadh', 'allele', 'genotype', 'phenotype', 'dominant', 'recessive', 'mendel', 'plant', 'animal', 'kingdom', 'phylum', 'species', 'genus', 'class', 'order', 'family', 'taxonomy', 'biological classification', 'monera', 'protista', 'fungi', 'plantae', 'animalia', 'virus', 'viroid', 'lichen', 'morphology', 'anatomy', 'tissue', 'circulatory', 'nervous', 'endocrine', 'excretory', 'skeletal', 'muscular', 'neural', 'chemical coordination', 'photosystem', 'c3 cycle', 'c4 cycle', 'cam plant', 'photorespiration', 'growth regulator', 'auxin', 'gibberellin', 'cytokinin', 'abscisic acid', 'ethylene', 'ecology', 'population', 'community', 'succession', 'environment', 'pollution', 'biotechnology', 'cloning', 'pcr', 'restriction enzyme', 'plasmid', 'recombinant dna', 'allergy', 'allergen', 'allergic', 'mast cell', 'histamine', 'antibody', 'antigen', 'pathogen', 'health', 'disease', 'infection', 'immune system', 'response']





def _detect_subject_by_keyword(question: str):
    """Returns (subject, score, all_scores) where subject is the best match."""
    q = question.lower()
    scores = {
        'physics':   sum(1 for kw in PHYSICS_KEYWORDS   if kw in q),
        'chemistry': sum(1 for kw in CHEMISTRY_KEYWORDS if kw in q),
        'math':      sum(1 for kw in MATH_KEYWORDS      if kw in q),
        'biology':   sum(1 for kw in BIOLOGY_KEYWORDS   if kw in q),
    }
    best_subject = max(scores, key=scores.get)
    max_score = scores[best_subject]
    return best_subject, max_score, scores





def _detect_type_by_keyword(question: str) -> str:
    """Returns question type string. Defaults to 'numerical' for JEE/NEET."""
    q = question.lower()

    # 0. MCQ / Objective - Checked FIRST as per user priority
    if any(w in q for w in [
        'correct option', 'correct answer', 'which of the following', 'choose the correct',
        'select the correct', 'identify the correct', 'pick the correct', 'best answer',
        'most appropriate', 'incorrect statement', 'true/false', 'match the following',
        'assertion and reason', 'statement i and ii', 'sequence of events', 'arrange in order',
        'fill in the blanks', '____'
    ]):
        return 'mcq'

    # 1. Derivation / proof
    if any(w in q for w in [
        'derive', 'prove', 'show that', 'establish', 'deduce',
        'verify that', 'demonstrate that',
    ]):
        return 'derivation'



    # 2. Organic chemistry mechanisms
    if any(w in q for w in ['mechanism', 'iupac', 'name the compound', 'identify the compound']):
        return 'organic'



    # 3. Numerical / calculation - checked BEFORE conceptual so 'find', 'determine',
    #    'evaluate', 'check', 'compute' do not fall through to 'conceptual'.
    if any(w in q for w in [
        # English calculation verbs
        'find', 'calculate', 'evaluate', 'compute', 'determine',
        'solve', 'obtain', 'check the', 'verify', 'simplify',
        # Continuity / limits / calculus
        'continuity', 'continuous', 'differentiable', 'limit', 'lim',
        'integral', 'integrate', 'differentiate', 'derivative',
        # Algebra / arithmetic
        'value of', 'sum of', 'product of', 'roots of',
        'expand', 'factorise', 'factorize',
        # Physics / chemistry numerics
        'how much', 'how many', 'how far', 'how fast',
        'rate of', 'time taken', 'velocity', 'acceleration',
        'pressure', 'temperature', 'concentration', 'molarity',
        'converges', 'diverges', 'sequence', 'series',
    ]):
        return 'numerical'



    # 4. Graph / diagram
    if any(w in q for w in ['sketch', 'draw the graph', 'plot']):
        return 'graph'
    if any(w in q for w in ['label', 'draw and label', 'name the parts']):
        return 'diagram'



    # 5. NCERT / fact recall
    if any(w in q for w in ['ncert', 'according to', 'as per']):
        return 'ncert_fact'



    # 6. Conceptual / theory - catch-all
    if any(w in q for w in ['explain', 'why does', 'why is', 'what is', 'define', 'describe', 'state', 'write', 'suggest', 'list', 'mention']):
        return 'theory'

    # 7. Default for JEE/NEET
    return 'numerical'





def _detect_subject_and_type_for_doubt(question: str, has_image: bool, institute_id: str):
    """
    Hybrid Logic:
    - High Confidence (score >= 2): Instant keyword match.
    - Ambiguous (score == 1) or None: LLM verification.
    """
    if question and question.strip():
        subject, score, all_scores = _detect_subject_by_keyword(question)
        # Trust keyword ONLY if score is high (2+) and significantly better than others
        if score >= 2:
            # Check for ties or near-ties
            other_scores = [v for k, v in all_scores.items() if k != subject]
            if score > max(other_scores):
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





def _select_doubt_model(subject: str, question_type: str, vertical: str = "coaching") -> str:
    # Coaching math routes to Qwen — strong on JEE/NEET-level symbolic work, but the
    # priciest model (~$3/M output, 4-5x the others). School math (Classes 1-10) is
    # far simpler, so it routes to GPT-OSS-120B instead: a capable reasoning model at
    # a fraction of the output cost ($0.60/M vs $3.00/M). GPT-OSS is already handled
    # as a reasoning model downstream, so nothing else changes.
    if subject == "math":
        return GROQ_MODELS["reasoning"] if vertical == "school" else GROQ_MODELS["math"]
    if question_type in ("numerical", "derivation", "graph", "organic"):
        return GROQ_MODELS["reasoning"]
    return GROQ_MODELS["general"]





def _parse_reasoning_response(raw: str) -> dict:
    """Extracts brief/detailed fields using regex to handle malformed LLM JSON output."""
    import re as _re
    import json as _json
    # 1. Aggressively remove <think> blocks (including unclosed ones)
    cleaned = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL | _re.IGNORECASE).strip()
    cleaned = _re.sub(r'<think>.*', '', cleaned, flags=_re.DOTALL | _re.IGNORECASE).strip()
    
    def get_field(target, field):
        # Find "field": "content" (handles escaped quotes and newlines)
        pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
        match = _re.search(pattern, target, _re.DOTALL)
        if match:
            try:
                # Use json.loads to handle escaped characters like \n
                return _json.loads(f'"{match.group(1)}"')
            except:
                return match.group(1).strip()
        # Fallback for unquoted or weirdly quoted
        pattern_lazy = rf'"{field}"\s*:\s*(.*?)(?=\s*[,}}])'
        match_lazy = _re.search(pattern_lazy, target, _re.DOTALL)
        if match_lazy:
            return match_lazy.group(1).strip().strip('"')
        return ""

    # Try to extract the brief answer
    brief_ans = get_field(cleaned, "answer")
    if not brief_ans:
        # Maybe it's outside the JSON? Try to find any text before the JSON starts
        brief_ans = cleaned.split('{')[0].strip()[:300]

    # Try to extract detailed fields
    sol = get_field(cleaned, "solution")
    if not sol: 
        # Check if detailed block exists but solution field failed
        detailed_match = _re.search(r'"detailed"\s*:\s*\{([\s\S]*?)\}', cleaned)
        if detailed_match:
            sol = get_field(detailed_match.group(1), "solution")
        
    if not sol: sol = cleaned # Final fallback
    
    return {
        "brief": {
            "answer": brief_ans or "Calculation in progress...",
            "question_nature": "numerical" if ("Step" in sol or "=" in sol) else "theory"
        },
        "detailed": {
            "solution": sol,
            "final_answer": get_field(cleaned, "final_answer"),
            "verification": get_field(cleaned, "verification"),
            "key_concept": get_field(cleaned, "key_concept"),
        }
    }





# Per-subject rules injected into the solver system prompt at runtime.
# Kept compact (<200 tokens each) so total request stays under 6 000 TPM.
_SUBJECT_RULES: dict[str, str] = {
    "physics": (
        "PHYSICS RULES:\n"
        "1. MECHANICS: For multiple blocks/pulleys, MANDATORY: State the Constraint Relation ($a_1, a_2$) and draw FBD for EACH block separately.\n"
        "2. ROTATIONAL: List all torques about a fixed axis before writing $\\tau = I\\alpha$.\n"
        "3. OPTICS: Use Cartesian Sign Convention strictly ($u$ is usually negative).\n"
        "4. PRECISION: Carry 4 decimal places for all intermediate calculations."
    ),
    "chemistry": (
        "CHEMISTRY RULES:\n"
        "1. MIXING & DILUTION: For any mixture of two solutions (e.g., Acid + Salt), MANDATORY: Final Concentration = (Initial Moles) / (Total Volume). If volumes are equal, molarity is HALVED ($0.1M \to 0.05M$).\n"
        "2. REACTION STEPS: For Buffer/Titration, MANDATORY: Write the chemical reaction and a Stoichiometry Table (Reaction -> Initial Moles -> Final Moles) before Henderson-Hasselbalch.\n"
        "3. ORGANIC: MANDATORY: Check for Carbocation Rearrangement (1,2-shift) before any attack.\n"
        "4. PRECISION: Carry 4 decimal places ($pKa$, $pH$). NO shortcut arithmetic."
    ),
    "math": (
        "MATH RULES:\n"
        "1. MATRICES: For $A^{-1}/A^n$ equations, MANDATORY: Use Cayley-Hamilton ($|A - \\lambda I| = 0$). NO direct inversion.\n"
        "2. CALCULUS: For Area, MANDATORY: Find intersection points ($f(x)=g(x)$) and use absolute values for area below x-axis.\n"
        "3. DOMAIN: State the domain restrictions ($x>0$ for log, etc.) as the very first step.\n"
        "4. VERIFY: Substitute result into original condition. NO hallucinations."
    ),
    "biology": (
        "BIOLOGY RULES:\n"
        "2. Name all phases/stages in correct sequence with locations (nucleus, cytoplasm, stroma)\n"
        "3. Genetics: draw Punnett square; list all genotype and phenotype ratios\n"
        "4. Photosynthesis/Respiration: name molecules at each stage and the exact enzyme involved\n"
        "5. Physiology: name the organ, tissue, and cell type involved\n"
        "6. State exceptions explicitly (C4 plants, incomplete dominance, anomalous species)"
    ),
}





# Per-vertical framing for the doubt solver prompt. The scientific rigor rules
# are shared; only the academic *context* (competitive vs school) changes.
_DOUBT_VERTICAL_FRAMING = {
    "coaching": {
        "theory_role":     "JEE/NEET Subject Matter Expert",
        "numerical_rigor": "MANDATORY JEE/NEET GOLD RULES",
    },
    # NOTE: the school entries are templates — {board} / {textbooks} are filled from
    # the request's board (CBSE/ICSE/State) by _doubt_framing(). They used to hardcode
    # "NCERT", which is CBSE's textbook body and simply wrong for an ICSE school.
    "school": {
        "theory_role":     "{board} School Teacher for Classes 1-10",
        "numerical_rigor": "MANDATORY SCHOOL-LEVEL ({textbooks}) CONCEPT RULES",
    },
}


def _doubt_framing(vertical: str, board: str = "") -> dict:
    """
    Resolve doubt-prompt framing for a vertical, filling in the board for school.

    School framing is board-specific: an ICSE school follows CISCE and its prescribed
    books, not NCERT (which is CBSE's). Coaching framing is board-independent.
    """
    framing = _DOUBT_VERTICAL_FRAMING.get(vertical or "coaching", _DOUBT_VERTICAL_FRAMING["coaching"])
    if (vertical or "").lower() != "school":
        return framing

    from ai_services.core.boards import get_board
    profile = get_board(board)
    return {
        k: v.replace("{board}", profile.display_name).replace("{textbooks}", profile.textbooks)
        for k, v in framing.items()
    }


def _build_solver_system_prompt(subject: str, qtype: str, mode: str = "detailed", vertical: str = "coaching", board: str = "") -> str:
    """
    CLEANED & RE-PRIORITIZED SOLVER PROMPT.

    `vertical` selects the academic framing (competitive exam vs school); the
    underlying scientific/format rules are identical across verticals.
    """
    framing = _doubt_framing(vertical, board)
    subject_rules = _SUBJECT_RULES.get(subject, "")
    is_numerical = qtype.lower() in ("numerical", "derivation")
    is_mcq = qtype.lower() == "mcq"
    
    if is_mcq:
        return (
            f"SYSTEM ROLE: You are a Senior Subject Expert. Subject: {subject.upper()}. Respond strictly in JSON format.\n"
            "TASK: Solve the MCQ/Objective question.\n\n"
            "RULES:\n"
            "1. FORMAT: Your answer must follow this exact structure inside the 'solution' field:\n"
            "   Correct Answer: <Option/Value>\n"
            "   Justification: <2-3 lines explanation using **bold** NCERT keywords>\n"
            "2. NO PREAMBLE: Respond ONLY with JSON.\n"
            "3. BOLDING: Use **bold** for all key academic terms. ALWAYS double escape backslashes in JSON (e.g., \\\\frac, \\\\sqrt) so they parse correctly.\n\n"
            "OUTPUT SCHEMA (JSON):\n"
            '{\n'
            '  "brief": {\n'
            '    "answer": "Correct Answer: <Option>\\nJustification: <1-2 lines>.",\n'
            '    "question_nature": "mcq"\n'
            '  },\n'
            '  "detailed": {\n'
            '    "solution": "Correct Answer: <Option>\\nJustification: <2-3 lines with **bold** keywords>.",\n'
            '    "final_answer": "Correct Option: <Option>",\n'
            '    "verification": "None",\n'
            '    "key_concept": "None"\n'
            '  }\n'
            '}'
        )
    elif is_numerical:
        return (
            f"You are EDVA AI (Logic v3.0). Subject: {subject.upper()}. Type: {qtype}.\n\n"
            f"{framing['numerical_rigor']}:\n"
            f"{subject_rules}\n\n"
            "UNIVERSAL RIGOR RULES:\n"
            "0. DO NOT THINK: Do NOT use <think> tags. Start response directly with '{'.\n"
            "1. NO PREAMBLE: Respond ONLY with JSON.\n"
            "2. PERFORM ACTUAL MATH: Step 1, Step 2... format.\n"
            "3. PRECISION: Carry 4 decimal places.\n"
            "4. MATH FORMATTING: Wrap ONLY mathematical variables, numbers, and equations in '$' (e.g., $x = 2$, $H_2O$). ALWAYS double escape backslashes in JSON (e.g., \\\\frac, \\\\sqrt) so they parse correctly. Do NOT wrap plain English sentences or step headers in '$'.\n"
            "5. FINAL ANSWER: End with 'Final Answer: [summary]'.\n\n"
            "SCIENTIFIC TRAP DETECTION (MANDATORY):\n"
            "1. INTEGRAL EQUATIONS: Check for constant solutions (f(t)=k). Verify domains/singularities.\n"
            "2. EXTREMA TRAPS: Use Leibniz Rule for differentiation under the integral sign—DO NOT brute force integrate if (x-t) is present. Perform sign analysis.\n"
            "3. PHYSICS CONCEPTS: a=-kx+c is shifted SHM. If a=kt, DO NOT use SUVAT.\n"
            "4. DOMAIN VALIDITY: Check for extraneous roots in radicals and log domains.\n\n"
            "OUTPUT SCHEMA:\n"
            '{\n'
            '  "brief": {\n'
            '    "answer": "Step 1: [Plain Text Explanation].\\n$Math Equation$\\nFinal Answer: [Summary].",\n'
            '    "question_nature": "numerical"\n'
            '  },\n'
            '  "detailed": {\n'
            '    "solution": "Step 1: [Plain Text Header]\\n$Detailed Equation$\\nStep 2: ...",\n'
            '    "final_answer": "Final result with units.",\n'
            '    "verification": "Cross-check logic.",\n'
            '    "key_concept": "Primary principle."\n'
            '  }\n'
            '}'
        )
    else:
        return (
            f"SYSTEM ROLE: You are a Senior {framing['theory_role']}. Subject: {subject.upper()}. Respond strictly in JSON format.\n"
            "TASK: Generate structured, high-depth academic answers.\n\n"
            "RULES:\n"
            "1. MATCH QUESTION STRUCTURE: You MUST match the number of sub-parts in the question exactly. If the question has 4 parts (a, b, c, d) or (i, ii, iii, iv), you MUST provide 4 corresponding bold headers.\n"
            "2. HEADERS: Use **(i) [Title]**, **(ii) [Title]**, etc., as headers. If no labels exist, create your own **Bold Categorical Headers**.\n"
            "3. NO PARAGRAPHS: Use bullet points (•) for all content. 3-4 points per part for Brief, 4-5 deep points for Detailed.\n"
            "4. EXPLAIN HOW: In 'Detailed' mode, each point MUST be a 2-3 sentence explanation.\n"
            "5. NO MARKDOWN BLOCKS: Do NOT use triple backticks (```).\n"
            "6. BOLDING: Use **bold** for all NCERT keywords. ALWAYS double escape backslashes in JSON (e.g., \\\\frac, \\\\sqrt) so they parse correctly.\n\n"
            "OUTPUT SCHEMA (JSON):\n"
            '{\n'
            '  "brief": {\n'
            '    "answer": "**(i) Header**\\n• Point...\\n\\n**(ii) Header**\\n• Point... (Continue for all sub-parts i, ii, iii, iv...)",\n'
            '    "question_nature": "theory"\n'
            '  },\n'
            '  "detailed": {\n'
            '    "solution": "**(i) Header**\\n• Deep point...\\n\\n**(ii) Header**\\n• Deep point... (Provide all sub-parts requested in the question)",\n'
            '    "final_answer": "Summary sentence.",\n'
            '    "verification": "None",\n'
            '    "key_concept": "None"\n'
            '  }\n'
            '}'
        )





def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning traces and extract ONLY the JSON object.
    Prioritizes content between the first '{' and the last '}' to ignore preambles/trailing text."""
    import re as _re
    # Remove reasoning blocks (including unclosed ones)
    cleaned = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    cleaned = _re.sub(r'<think>.*', '', cleaned, flags=_re.DOTALL | _re.IGNORECASE).strip()
    
    # Aggressively find the outermost JSON object
    m = _re.search(r'(\{[\s\S]*\})', cleaned, flags=_re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # Fallback to searching the raw text if cleaned was somehow empty
    m = _re.search(r'(\{[\s\S]*\})', text, flags=_re.DOTALL)
    return m.group(1).strip() if m else cleaned





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
    _start_time = time.time()


    data = request.data
    question_text = (data.get("questionText") or data.get("question") or "").strip()
    raw_mode = (data.get("mode") or "detailed").strip().lower()
    # Normalize mode: BEGINNER/detailed vs ADVANCED/brief/short
    if raw_mode in ("beginner", "detailed", "teach"):
        mode = "detailed"
    elif raw_mode in ("advanced", "brief", "short", "solve"):
        mode = "brief"
    else:
        mode = "detailed"



    # Image support: base64 data URL (from NestJS) or raw HTTPS URL
    image_description = ""
    image_source = (data.get("questionImageBase64") or data.get("questionImageUrl") or "").strip()
    has_image = bool(image_source)



    if image_source:
        logger.info("[DOUBT] Image detected, attempting Groq vision... url_len=%d", len(image_source))
        # Primary: Groq Llama 4 Scout vision
        image_description = _vision_text_from_image(image_source, _DOUBT_VISION_PROMPT)
        if image_description:
            logger.info("[DOUBT] Groq vision succeeded: %d chars", len(image_description))
        else:
            logger.warning("[DOUBT] Groq vision returned empty. Trying EasyOCR fallback...")
            # Fallback: EasyOCR (local, no network dependency)
            try:
                image_description = _extract_text_from_image_url(image_source)
                if image_description:
                    logger.info("[DOUBT] EasyOCR extracted: %d chars", len(image_description))
                else:
                    logger.warning("[DOUBT] EasyOCR also returned empty.")
            except Exception as ocr_err:
                logger.warning("[DOUBT] EasyOCR fallback crashed: %s", ocr_err)



        # Last resort: if all extraction failed, use a placeholder so the LLM
        # still attempts to answer based on any question_text provided alongside the image.
        if not image_description:
            image_description = "[Student uploaded an image of their question. The image content could not be automatically extracted. Please answer based on any text description provided, or ask the student to type their question.]"
            logger.warning("[DOUBT] All image extraction failed — using placeholder text.")



    if not question_text and not has_image:
        return Response({"error": "Missing questionText or readable image"}, status=400)



    institute_id = _resolve_institute_id(request)
    user_id = data.get('userId') or data.get('user_id') or data.get('studentId') or ''



    if image_description and question_text:
        combined_question = f"{question_text}\n\n[Image content]\n{image_description}"
    elif image_description:
        combined_question = f"[Student uploaded an image of their question]\n{image_description}"
    else:
        combined_question = question_text



    # ── Step 1: Classify subject and question type ────────────────────────────
    # Keyword detection runs first (free, instant). LLM runs only as fallback
    # for image-only questions or when zero keywords matched.
    subject, qtype = _detect_subject_and_type_for_doubt(
        combined_question, has_image=bool(image_description), institute_id=institute_id,
    )



    # ── Step 2: Route to correct model, build prompt, solve ───────────────────
    vertical = getattr(request, "vertical", "coaching")
    model = _select_doubt_model(subject, qtype, vertical)
    print(f"[DOUBT RESOLVER] Subject: {subject} | Type: {qtype} | Model: {model} | Vertical: {vertical}")
    board = getattr(request, "board", "")
    solver_system = _build_solver_system_prompt(subject, qtype, mode, vertical, board)
    user_prompt = (
        f"Topic: {data.get('topicId', 'general')}\n\n"
        f"Question:\n{combined_question}"
    )



    is_reasoning_model = model in {"openai/gpt-oss-120b", "qwen/qwen3-32b"}



    # Step 2a: Try the scientific solver (symbolic compute) for science/math doubts.
    # It is exact when it works; on any failure we transparently fall back to the LLM.
    try:
        from asgiref.sync import async_to_sync
        from ai_services.solver.scientific_solver import scientific_solver

        if subject in ("physics", "chemistry", "mathematics", "math", "science"):
            logger.info("[DOUBT RESOLVER] Routing to scientific solver for %s/%s (vertical=%s, board=%s)", subject, qtype, vertical, board or "—")
            # Pass the vertical: the solver's formula knowledge base is built from
            # JEE/NEET formula sheets, so it is used for coaching only — a Class 1-10
            # answer must not be grounded with IIT-JEE formulae.
            scientific_res = async_to_sync(scientific_solver.solve)(combined_question, mode, vertical)
            if scientific_res and ("brief" in scientific_res or "detailed" in scientific_res):
                parsed = scientific_res
                solve_result = {"model": "scientific_solver"}
            else:
                raise RuntimeError("scientific_solver returned empty/invalid response")
        else:
            raise NotImplementedError("Subject not mapped to scientific solver")
    except Exception as solver_err:
        logger.warning("[DOUBT RESOLVER] Scientific solver bypassed/failed (%s). Using LLM.", solver_err)
        try:
            if is_reasoning_model:
                # Reasoning models output <think> blocks — use text mode and parse manually.
                solve_result = get_llm().complete(
                    system_prompt=solver_system,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=0.1,
                    max_tokens=3500,
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
                    max_tokens=3500,
                    json_mode=True,
                    json_mode_suffix="",
                    institute_id=institute_id,
                )
                parsed = solve_result["content"] if isinstance(solve_result["content"], dict) else {}
        except RuntimeError as llm_err:
            try:
                log_usage(
                    institute_id=institute_id,
                    institute_type='school',
                    feature_id='doubt_resolver',
                    feature_category='student',
                    model_used='unknown',
                    latency_ms=int((time.time() - _start_time) * 1000),
                    success=False,
                    error_message=str(llm_err)[:500],
                    user_id=user_id,
                )
            except Exception:
                pass
            return JsonResponse({"error": str(llm_err)}, status=502)
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

    # Defensive: Ensure all returned values are strings and stripped of formatting artifacts
    def _safe_str(v):
        if v is None: return ""
        if isinstance(v, (list, dict, tuple)):
            import json as _json2
            s = _json2.dumps(v) if isinstance(v, dict) else "\n".join(map(str, v))
        else:
            s = str(v)
        
        # Aggressively remove code blocks and leading/trailing whitespace per line
        import re as _re3
        s = _re3.sub(r'```[a-z]*', '', s) # Remove opening backticks
        s = s.replace('```', '')         # Remove closing backticks
        lines = [l.strip() for l in s.split('\n')]
        return "\n".join(lines).strip()

    answer = _safe_str(answer)
    explanation = _safe_str(explanation)
    # Also clean the deep objects
    for k, v in brief_obj.items(): brief_obj[k] = _safe_str(v)
    for k, v in detailed_obj.items(): detailed_obj[k] = _safe_str(v)

    try:
        _doubt_model = solve_result.get('model', 'unknown')
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='doubt_resolver',
            feature_category='student',
            model_used=_doubt_model if _doubt_model != 'scientific_solver' else 'llama-3.3-70b-versatile',
            tokens_input=solve_result.get('tokens_input', 0),
            tokens_output=solve_result.get('tokens_output', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id,
        )
    except Exception:
        pass

    return JsonResponse({
        "subject": subject,
        "type": qtype,
        "model_used": solve_result.get("model", "unknown"),
        "answer": answer,
        "explanation": explanation,
        "brief": brief_obj,
        "detailed": detailed_obj,
        "conceptLinks": [],
        "related_topics": [],
        "_meta": {
            "source": "solver" if solve_result.get("model") == "scientific_solver" else "llm",
            "model": solve_result.get("model", "unknown"),
            "latency_ms": round(solve_result.get("latency_ms", 0)),
            "institute": institute_id,
            "vertical": vertical,
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





def _url_to_base64_data_uri(image_url: str) -> str:
    """
    Download an image from an HTTP(S) URL and return it as a base64 data URI.
    Resizes image to max 1024px and compresses as JPEG to ensure it fits in LLM context limits.
    """
    import base64 as _b64
    from io import BytesIO as _BytesIO
    try:
        from PIL import Image as _PILImage
    except ImportError:
        logger.warning("[VISION] PIL not installed; skipping image compression.")
        _PILImage = None

    try:
        resp = _requests.get(image_url, timeout=20)
        resp.raise_for_status()
        
        image_data = resp.content
        content_type = "image/jpeg" # Default to JPEG for compression benefits

        # Compress if PIL is available
        if _PILImage:
            with _PILImage.open(_BytesIO(image_data)) as img:
                # Convert to RGB if necessary (e.g. for PNG with alpha or GIF)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Resize if larger than 1024px
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, _PILImage.Resampling.LANCZOS)
                
                # Save to buffer with compression
                buffer = _BytesIO()
                img.save(buffer, format="JPEG", quality=80, optimize=True)
                image_data = buffer.getvalue()

        encoded = _b64.b64encode(image_data).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except Exception as exc:
        logger.warning("[VISION] Could not convert URL to base64 (%s) — using original URL", exc)
        return image_url





def _vision_text_from_image(image_url: str, user_prompt: str) -> str:
    """Groq vision — tries Llama 4 Scout first, then Llama 3.2 Vision as fallback.
    Returns '' on complete failure (caller adds EasyOCR or placeholder fallback)."""
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



    # Download HTTP(S) images to base64 so Groq doesn't need to reach S3/presigned URLs.
    # Data URIs and already-base64 strings are passed as-is.
    if image_url.startswith(("http://", "https://")):
        effective_image = _url_to_base64_data_uri(image_url)
        logger.info("[VISION] Converted URL to base64 data URI (%d bytes)", len(effective_image))
    else:
        effective_image = image_url



    # Vision models to try in order
    vision_models = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ]



    for api_key in keys:
        for model_name in vision_models:
            try:
                if api_key not in _groq_vision_clients:
                    _groq_vision_clients[api_key] = Groq(api_key=api_key, timeout=45.0)
                client = _groq_vision_clients[api_key]
                logger.info("[VISION] Trying model=%s (base64=%s)", model_name, str(effective_image).startswith("data:"))
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": effective_image},
                                },
                            ],
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.0,
                )
                out = (response.choices[0].message.content or "").strip()
                if out:
                    logger.info("[VISION] Model=%s succeeded: %d chars", model_name, len(out))
                    return out
                else:
                    logger.warning("[VISION] Model=%s returned empty content", model_name)
            except Exception as exc:
                exc_str = str(exc)
                # If it's a model-not-found / unsupported error, try next model immediately
                if any(kw in exc_str.lower() for kw in ["not found", "not supported", "invalid model", "does not exist", "404"]):
                    logger.warning("[VISION] Model=%s not available: %s — trying next model", model_name, exc_str[:120])
                    break  # break inner loop (models), move to next key
                # Rate limit — try next key
                elif "rate" in exc_str.lower() or "429" in exc_str:
                    logger.warning("[VISION] Model=%s rate-limited, rotating key", model_name)
                    break
                else:
                    logger.warning("[VISION] Model=%s failed: %s", model_name, exc_str[:200])
                    continue  # try next model with same key



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
    _start_time = time.time()
    image_url = (request.data.get("imageUrl") or "").strip()
    if not image_url:
        return Response({"error": "Missing imageUrl"}, status=400)
    purpose = (request.data.get("purpose") or "doubt").strip().lower()
    is_grading = purpose in ("grading", "mock", "assessment", "mock_test", "answer")
    institute_id_ocr = _resolve_institute_id(request)
    user_id_ocr = request.data.get('userId') or request.data.get('user_id') or ''
    if is_grading:
        text = _transcribe_exam_answer_with_vision(image_url)
    else:
        text = _describe_image_with_vision(image_url)
    if not text:
        # English-only EasyOCR for grading — reduces garbage Devanagari on Latin handwriting
        text = _extract_text_from_image_url(
            image_url, languages=["en"] if is_grading else None
        )
    try:
        _ocr_model = 'llama-4-scout-17b-16e-instruct' if text else 'easyocr-local'
        log_usage(
            institute_id=institute_id_ocr,
            institute_type='school',
            feature_id='image_ocr_handwriting',
            feature_category='shared',
            model_used=_ocr_model,
            tokens_input=int(len(image_url) / 4),
            tokens_output=int(len(text or '') / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=bool(text),
            user_id=user_id_ocr,
        )
    except Exception:
        pass
    return JsonResponse({"text": text or ""})





# â"€â"€ AI #2 -- AI Tutor â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



@api_view(["POST"])
def start_tutor_session(request):
    _start_time = time.time()
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)



    institute_id = _resolve_institute_id(request)
    user_id = data.get('userId') or data.get('user_id') or student_id or ''
    context = data.get("context", "")
    vertical = getattr(request, "vertical", "coaching")



    # When a rich lesson-generation prompt is provided (long context), use it as the
    # system prompt directly so the LLM produces clean Markdown -- not JSON-wrapped text.
    if len(context) > 300:
        system_prompt = context
        user_prompt = "Generate the complete lesson now. Write everything in full -- do not truncate or use placeholders."
    else:
        template = get_template("tutor_session", vertical)
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
            model=get_model_for_task("tutor_session", vertical),
            temperature=0.3,
            max_tokens=8192,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        try:
            log_usage(
                institute_id=institute_id,
                institute_type=vertical,
                feature_id='ai_lecture_notes',
                feature_category='teacher',
                model_used='unknown',
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(e)[:500],
                user_id=user_id,
            )
        except Exception:
            pass
        return JsonResponse({"error": str(e)}, status=502)



    raw_text = result["content"]
    if isinstance(raw_text, dict):
        explanation_text = raw_text.get("response", str(raw_text))
    else:
        explanation_text = str(raw_text).strip()
    explanation_text = _coerce_tutor_or_doubt_text(explanation_text)

    try:
        log_usage(
            institute_id=institute_id,
            institute_type=vertical,
            feature_id='ai_lecture_notes',
            feature_category='teacher',
            model_used=result.get('model', 'llama-3.3-70b-versatile'),
            tokens_input=result.get('tokens_input', 0),
            tokens_output=result.get('tokens_output', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id,
        )
    except Exception:
        pass

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
    _start_time = time.time()
    data = request.data
    session_id = data.get("sessionId")
    student_message = data.get("studentMessage")
    if not session_id or not student_message:
        return Response({"error": "Missing sessionId or studentMessage"}, status=400)



    institute_id = _resolve_institute_id(request)
    user_id = data.get('userId') or data.get('user_id') or data.get('studentId') or ''
    vertical = getattr(request, "vertical", "coaching")
    template = get_template("tutor_continue", vertical)
    user_prompt = template.user_template.format(
        session_id=session_id,
        student_message=student_message,
    )



    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=get_model_for_task("tutor_continue", vertical),
            temperature=0.0,
            max_tokens=1200,
            json_mode=True,
            json_mode_suffix=_JSON_MODE_TUTOR_SUFFIX,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        try:
            log_usage(
                institute_id=institute_id,
                institute_type=vertical,
                feature_id='doubt_resolver',
                feature_category='student',
                model_used='unknown',
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(e)[:500],
                user_id=user_id,
            )
        except Exception:
            pass
        return JsonResponse({"error": str(e)}, status=502)



    raw_text = result["content"]
    explanation_text = _coerce_tutor_or_doubt_text(raw_text)

    try:
        log_usage(
            institute_id=institute_id,
            institute_type=vertical,
            feature_id='doubt_resolver',
            feature_category='student',
            model_used=result.get('model', 'llama-3.3-70b-versatile'),
            tokens_input=result.get('tokens_input', 0),
            tokens_output=result.get('tokens_output', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id,
        )
    except Exception:
        pass

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
    _start_time = time.time()
    data = request.data
    audio_url = data.get("audioUrl")
    if not audio_url:
        return Response({"error": "Missing audioUrl"}, status=400)



    import time as _time
    language = data.get("language", "hi")
    logger.info("generate_stt_notes | audio_url=%s | language=%s", audio_url, language)
    _t0 = _time.perf_counter()
    institute_id_stt = _resolve_institute_id(request)
    user_id_stt = data.get('userId') or data.get('user_id') or data.get('studentId') or ''



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
            try:
                _stt_model = 'sarvam-stt' if _is_odia_language(language) else 'whisper-large-v3-turbo'
                log_usage(
                    institute_id=institute_id_stt,
                    institute_type='school',
                    feature_id='lecture_transcription',
                    feature_category='teacher',
                    model_used=_stt_model,
                    tokens_input=int(len(audio_url) / 4),
                    tokens_output=0,
                    latency_ms=int((time.time() - _start_time) * 1000),
                    success=False,
                    error_message=str(exc)[:500],
                    user_id=user_id_stt,
                )
            except Exception:
                pass
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
        _resolve_institute_id(request),
    )



    institute_id = _resolve_institute_id(request)
    notes_markdown, notes_meta = _generate_comprehensive_notes(
        english_transcript,
        data.get("topicId", ""),
        language,
        institute_id,
    )
    if _normalize_lecture_language(language) in ("hi", "hi-in", "hinglish", "od"):
        markdown_polished = False
    else:
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

    try:
        _stt_model_used = 'sarvam-stt' if _is_odia_language(language) else 'whisper-large-v3-turbo'
        _notes_provider = notes_meta.get('provider', 'groq')
        _notes_model = 'gemini-2.5-flash' if _notes_provider == 'gemini' else 'llama-3.3-70b-versatile'
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='lecture_transcription',
            feature_category='teacher',
            model_used=_stt_model_used,
            tokens_input=int(len(audio_url) / 4),
            tokens_output=int(len(raw_transcript) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_stt,
        )
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='ai_lecture_notes',
            feature_category='teacher',
            model_used=_notes_model,
            tokens_input=int(len(english_transcript) / 4),
            tokens_output=int(len(notes_markdown) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_stt,
        )
    except Exception:
        pass

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
    _start_time = time.time()
    data = request.data
    audio_url = (data.get("audioUrl") or "").strip()
    if not audio_url:
        return Response({"error": "Missing audioUrl"}, status=400)



    language = data.get("language", "hi")
    topic_id = data.get("topicId", "")
    institute_id_stt2 = _resolve_institute_id(request)
    user_id_stt2 = data.get('userId') or data.get('user_id') or data.get('studentId') or ''
    logger.info("stt_transcribe_only | url=%s | language=%s", audio_url, language)
    _t0 = _time.perf_counter()



    try:
        raw_transcript = _transcribe_audio(audio_url, language)
    except Exception as exc:
        logger.error("stt_transcribe_only FAILED for %s: %s", audio_url, exc)
        try:
            _stt2_model = 'sarvam-stt' if _is_odia_language(language) else 'whisper-large-v3-turbo'
            log_usage(
                institute_id=institute_id_stt2,
                institute_type='school',
                feature_id='lecture_transcription',
                feature_category='teacher',
                model_used=_stt2_model,
                tokens_input=int(len(audio_url) / 4),
                tokens_output=0,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(exc)[:500],
                user_id=user_id_stt2,
            )
        except Exception:
            pass
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

    try:
        _stt2_model_ok = 'sarvam-stt' if _is_odia_language(language) else 'whisper-large-v3-turbo'
        log_usage(
            institute_id=institute_id_stt2,
            institute_type='school',
            feature_id='lecture_transcription',
            feature_category='teacher',
            model_used=_stt2_model_ok,
            tokens_input=int(len(audio_url) / 4),
            tokens_output=int(len(raw_transcript) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_stt2,
        )
    except Exception:
        pass

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


def _quiz_language_instruction(language: str) -> str:
    normalized = _normalize_lecture_language(language)
    if normalized == "od":
        return (
            "LANGUAGE REQUIREMENT: Write every question, every option text, every segmentTitle, and every "
            "explanation in natural Odia (ଓଡ଼ିଆ) script. Keep only JSON field names, IDs, and option labels "
            "A/B/C/D in English. Do not translate technical symbols or formulas. Do not output Hindi or English prose."
        )
    return "LANGUAGE REQUIREMENT: Write the quiz content in English."


def _gemini_complete(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float = 0.3) -> dict:
    from google import genai
    from google.genai import types
    import os
    import time
    
    if not has_gemini_api_key():
        raise RuntimeError("GEMINI_API_KEY is not set -- add it to .env to enable Gemini generation")

    model_name = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

    last_exc = None
    response = None
    start_time = time.perf_counter()
    for key_index, api_key in get_rotated_gemini_keys():
        try:
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except TypeError:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
            logger.info(
                "Gemini Parallel Complete OK | key=%d/%d | model=%s",
                key_index, gemini_key_count(), model_name,
            )
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if is_gemini_permanent_key_error(msg):
                mark_gemini_key_disabled(api_key)
                logger.warning("Gemini key %d/%d disabled for permanent error: %s", key_index, gemini_key_count(), msg[:180])
                continue
            if is_gemini_retryable_error(msg):
                logger.warning("Gemini key %d/%d retryable error; rotating to next key: %s", key_index, gemini_key_count(), msg[:220])
                continue
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc
    else:
        raise RuntimeError(f"Gemini generation failed across all {gemini_key_count()} key(s): {last_exc}") from last_exc

    latency_ms = (time.perf_counter() - start_time) * 1000
    content = str(getattr(response, "text", "") or "").strip()
    
    # Strip markdown block formatting if any
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()
    
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if response and hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
        }

    return {
        "content": content,
        "usage": usage,
        "model": model_name,
        "latency_ms": latency_ms,
    }


def _gemini_parallel_complete_many(tasks: list[dict], temperature: float = 0.3) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    n_tasks = len(tasks)
    if n_tasks == 0:
        return []

    def _worker(task_idx):
        task = tasks[task_idx]
        requested_max = task.get("max_tokens", 3500)
        max_tokens = max(2000, requested_max * 2)
        return _gemini_complete(
            system_prompt=task["system_prompt"],
            user_prompt=task["user_prompt"],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        results = list(pool.map(_worker, range(n_tasks)))

    return results


@api_view(["POST"])
def generate_quiz_questions(request):
    _start_time = time.time()
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



    logger.info("generate_quiz_questions: input numQuestions=%r -> parsed as %d", data.get("numQuestions"), num_questions)



    institute_id = _resolve_institute_id(request)
    user_id_quiz = data.get('userId') or data.get('user_id') or data.get('studentId') or ''
    lecture_title = data.get("lectureTitle", "Lecture")
    topic_id = data.get("topicId", "")
    course_level = data.get("courseLevel", "General")
    language = _normalize_lecture_language(data.get("language") or "en")
    language_instruction = _quiz_language_instruction(language)



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



    # Vertical-aware: quiz_generate builds the system prompt directly from the
    # template (it does not go through ai_call), so the vertical must be passed
    # here or school users would silently get the coaching prompt.
    template = get_template("quiz_generate", getattr(request, "vertical", "coaching"))
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
            course_level=course_level,
        )
        user_prompt = f"{language_instruction}\n\n{user_prompt}\n\n{language_instruction}"
        tasks.append({
            "system_prompt": f"{template.system}\n\n{language_instruction}",
            "user_prompt": user_prompt,
            "max_tokens": max(800, q_count * 350),
        })
        chunk_meta.append((i + 1, q_count, start_pct, end_pct))



    # Dispatch all chunks in parallel — if Odia, use Gemini API key rotation;
    # otherwise use Groq API key rotation.
    if language == "od":
        results = _gemini_parallel_complete_many(
            tasks=tasks,
            temperature=0.3,
        )
    else:
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
        if not chunk_qs:
            logger.error("Quiz chunk %d/%d parsing failed. Raw response: %r", chunk_idx, n_chunks, raw)



        logger.info("generate_quiz_questions: chunk %d/%d requested %d questions, received %d questions", chunk_idx, n_chunks, q_count, len(chunk_qs))



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
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='in_video_quiz_generator',
                feature_category='teacher',
                model_used='llama-3.3-70b-versatile',
                tokens_input=int(len(source_text) / 4),
                tokens_output=0,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message='no_questions_generated',
                user_id=user_id_quiz,
            )
        except Exception:
            pass
        return Response({"error": "Quiz generation produced no questions. Try again."}, status=502)



    # Renumber IDs sequentially
    for idx, q in enumerate(all_questions):
        q["id"] = f"q{idx + 1}"

    try:
        _quiz_total_input = sum(len(t.get('user_prompt', '')) for t in tasks)
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='in_video_quiz_generator',
            feature_category='teacher',
            model_used=last_meta.get('model', 'llama-3.3-70b-versatile'),
            tokens_input=int(_quiz_total_input / 4),
            tokens_output=int(len(str(all_questions)) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_quiz,
        )
    except Exception:
        pass

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
    _start_time = time.time()
    from ai_services.core.sarvam_client import translate as sarvam_translate



    data = request.data
    text            = data.get("text", "")
    target_language = data.get("targetLanguage", "en")



    if not text:
        return Response({"error": "Missing text"}, status=400)



    institute_id = _resolve_institute_id(request)
    user_id_tr = data.get('userId') or data.get('user_id') or data.get('studentId') or ''
    logger.info(
        "translate_text (Sarvam) | target=%s | chars=%d | institute=%s",
        target_language, len(text), institute_id,
    )



    _t0 = _time.perf_counter()
    try:
        translated = sarvam_translate(text, target_language=target_language)
    except RuntimeError as exc:
        logger.error("Sarvam translation failed: %s", exc)
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='multilingual_translation',
                feature_category='shared',
                model_used='mayura:v1',
                tokens_input=int(len(text) / 4),
                tokens_output=0,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(exc)[:500],
                user_id=user_id_tr,
            )
        except Exception:
            pass
        return Response({"error": str(exc)}, status=502)



    latency_ms = (_time.perf_counter() - _t0) * 1000
    logger.info(
        "Sarvam translation done | %d -> %d chars | %.0fms",
        len(text), len(translated), latency_ms,
    )

    try:
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='multilingual_translation',
            feature_category='shared',
            model_used='mayura:v1',
            tokens_input=int(len(text) / 4),
            tokens_output=int(len(translated) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_tr,
        )
    except Exception:
        pass

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
    "faq": (
        "Generate a Frequently Asked Questions (FAQ) sheet for this topic in Markdown. "
        "Do NOT write lesson notes, summaries, or long explanatory sections. "
        "FAQ means questions that are repeatedly asked in target exams. "
        "For every question, you must specify the actual past exam years it was asked (e.g., JEE Main 2019, 2022). "
        "Format exactly as question-answer pairs grouped under clear sub-topic headings:\n"
        "## <Sub-topic>\n"
        "**Q1. [EXAMTAG: {exam_target} <comma-separated years>] <common student question?>**\n\n"
        "**A.** <clear, concise answer in 2-5 sentences>\n\n"
        "Include 12-15 genuinely common questions students ask, covering definitions, misconceptions, "
        "formula use, conceptual doubts, and application confusions. Each item must be a real FAQ, not a note bullet.\n\n"
        "For numerical questions, the answer must provide a detailed step-by-step solution where each new step is on a new line (never in paragraph format). "
        "For theory questions, the answer must provide a total, complete solution explaining the concept. "
        "Do not just give the final answer; provide the full, comprehensive explanation.\n\n"
        "CRITICAL MATH NOTATION: For all mathematics, equations, exponents, and variables, always use valid KaTeX/LaTeX Markdown. Exponents must use carets (e.g., $x^2$, $x^3$), and all mathematical expressions must be wrapped in single dollar signs (e.g. $3\\sqrt{5}$, $f(3) = 0$). Never output raw math or variables without dollar signs, and never use raw exponents like x2 or x3.\n\n"
        "Math formatting rules: wrap every inline expression in single dollar signs, "
        "for example $x = \\frac{6}{3 + \\sqrt{2}}$. Use LaTeX commands such as \\frac and \\sqrt, "
        "never raw fractions outside math delimiters and never the Unicode square-root symbol."
    ),
    "checklist": (
        "Generate a revision checklist for this topic in Markdown. "
        "Group items by sub-topic. Use - [ ] for each checkbox item. "
        "Include concepts to understand, formulas to memorise, and types of problems to practice."
    ),
    "revision_checklist": (
        "Generate a revision checklist for this topic in Markdown. "
        "Group items by sub-topic. Use - [ ] for each checkbox item. "
        "Include concepts to understand, formulas to memorise, and types of problems to practice. "
        "Do NOT write normal notes; every actionable item must be a checkbox."
    ),
    # â"€â"€ same as lesson/summary but with short label aliases â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    "study_guide":         "Generate a crisp, exam-ready summary of this topic in Markdown. Use bullet points and short paragraphs. Cover every exam-important concept.",
    "key_concepts":        "Generate a structured list of ALL key formulas and must-know concepts for this topic in Markdown. For each: name, definition, units (if applicable), one-line use-case.",
    "practice_questions":  (
        "Generate a practice problem set for this topic in Markdown. "
        "Include exactly 10 questions with a mix of difficulty (3 easy, 5 medium, 2 hard). "
        "For each question:\n"
        "- Number it (Q1, Q2 ...)\n"
        "- Write the question clearly\n"
        "- After all questions, add a ## Answers section with answer + 2-3 line explanation.\n"
        "Ensure questions test understanding, not just recall. Keep everything syllabus-appropriate."
    ),
    "dpp": (
        "Generate a high-quality Daily Practice Problem (DPP) sheet for this topic in Markdown.\n\n"
        "Format:\n"
        "# DPP -- {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name} | **Date:** ______\n\n"
        "## Section A -- Multiple Choice (1 mark each)\n"
        "Generate 8 MCQ questions, each with 4 options (A-D). Mix easy and medium difficulty.\n"
        "CRITICAL MCQ OPTION FORMATTING: Write each option (A-D) on a new line, never inline on a single line.\n\n"
        "## Section B -- Numericals / Short Answer (3 marks each)\n"
        "Generate 4 numerical or short-answer problems.\n\n"
        "## Answer Key\n"
        "List all correct answers and brief hints/solutions.\n\n"
        "Math formatting rules: wrap every inline expression in single dollar signs, "
        "for example $x = \\frac{6}{3 + \\sqrt{2}}$. Use LaTeX commands such as \\frac and \\sqrt, "
        "never raw fractions outside math delimiters and never the Unicode square-root symbol.\n\n"
        "Questions must be syllabus-aligned, conceptually varied, and gradually increasing in difficulty. "
        "Do NOT mention any class, grade, board, or exam name anywhere in the output."
    ),
    "pyq": (
        "Generate a school Previous Year Question (PYQ) style practice set on this topic in Markdown. "
        "Simulate authentic class/board exam questions for the selected class only.\n\n"
        "Format:\n"
        "# PYQ Practice Set -- {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name}\n\n"
        "## Practice Questions\n"
        "Generate 10 syllabus-appropriate questions covering this topic using school board patterns "
        "(MCQ, short answer, long answer, case/source-based where suitable).\n"
        "Format every question exactly as: `1. [EXAMTAG: CBSE Class 10 2021] <question text>`. "
        "The exam name, class, and year must appear ONLY inside EXAMTAG and must never be repeated in the visible question text. "
        "It MUST be a real, authentic past year of the exam, never a dummy year or empty placeholder like '____' or 'Year' or '20XX'.\n\n"
        "## Detailed Solutions\n"
        "Provide full step-by-step solutions for every question. For all numerical questions, provide a detailed step-by-step solution showing calculations and working, where each new mathematical step is written on a new line, never combined into a single paragraph. For all MCQ and theory questions, provide the complete explanation/reasoning along with the correct option, not just the option letter alone.\n\n"
        "CRITICAL MATH NOTATION: For all mathematics, equations, exponents, and variables, always use valid KaTeX/LaTeX Markdown. Exponents must use carets (e.g., $x^2$, $x^3$), and all mathematical expressions must be wrapped in single dollar signs (e.g. $3\\sqrt{5}$, $f(3) = 0$). Never output raw math or variables without dollar signs, and never use raw exponents like x2 or x3.\n\n"
        "CRITICAL MCQ OPTION FORMATTING: Write each option (A-D) on a new line, never inline on a single line.\n\n"
        "Math formatting rules: wrap only mathematical expressions in single dollar signs, "
        "for example Determine whether $3\\sqrt{5}$ is rational. Do not wrap complete English sentences "
        "inside math delimiters.\n\n"
        "Questions must match the selected class syllabus and board-question difficulty. "
        "Do NOT generate JEE, NEET, Olympiad, integer-type, multi-correct, match-the-column, or competitive exam PYQs. "
        "Do NOT mention any class, grade, board, or exam name anywhere in the output."
    ),
}



_COACHING_CONTENT_TYPE_PROMPTS: dict[str, str] = {
    # ── DPP (coaching/competitive) ────────────────────────────────────────
    "dpp": (
        "Generate a high-quality {exam_target} Daily Practice Problem (DPP) "
        "sheet for competitive exam preparation in Markdown.\n\n"
        "Format:\n"
        "# DPP — {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name} | "
        "**Exam:** {exam_target} | **Date:** ______\n\n"
        "---\n\n"
        "## Section A — Single Correct MCQ (4 marks each, –1 for wrong)\n"
        "Generate {mcq_count} single-correct MCQs (including Assertion-Reason and Statement-Based questions where relevant). Each must have exactly 4 options (A–D). "
        "Questions should be multi-step and conceptually challenging at the "
        "{exam_target} level. Do NOT make questions trivially easy.\n"
        "CRITICAL MCQ OPTION FORMATTING: Write each option (A-D) on a new line, never inline on a single line.\n\n"
        "## Section B — Integer Type Numericals (4 marks each) [include for JEE]\n"
        "Generate {integer_count} integer-answer numericals. The answer must be a valid "
        "integer (can be any positive or negative integer). Show the numerical value, not a letter option.\n\n"
        "## Section C — Multi-Correct MCQ (4 marks, –2 for partial) [include for JEE Advanced]\n"
        "Generate {multicorrect_count} multi-correct MCQs where one or more options may be correct. "
        "Mark correct options clearly in the Answer Key — NOT inline with questions. "
        "If a section's count is 0, do NOT generate that section header or any questions for it.\n\n"
        "---\n\n"
        "## Answer Key\n"
        "List ONLY the correct answers (e.g. 'A1. B', 'B1. 42'). "
        "Do NOT include solutions or explanations here.\n\n"
        "---\n\n"
        "## Detailed Solutions\n"
        "Provide complete step-by-step solutions for EVERY question. For all numerical questions, provide a detailed step-by-step solution showing calculations and working, where each new mathematical step is written on a new line, never combined into a single paragraph. For all MCQ and theory questions, provide the complete explanation/reasoning along with the correct option, not just the option letter alone. Show all working, apply relevant formulas explicitly, and point out common traps/mistakes where relevant.\n\n"
        "CRITICAL MATH NOTATION: For all mathematics, equations, exponents, and variables, always use valid KaTeX/LaTeX Markdown. Exponents must use carets (e.g., $x^2$, $x^3$), and all mathematical expressions must be wrapped in single dollar signs (e.g. $3\\sqrt{5}$, $f(3) = 0$). Never output raw math or variables without dollar signs, and never use raw exponents like x2 or x3.\n\n"
        "Math formatting rules: wrap every inline expression in single dollar signs, "
        "e.g. $F = ma$ or $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$. "
        "Use LaTeX commands (\\frac, \\sqrt, \\int, \\sum, etc.). "
        "Never write raw fractions or the Unicode √ symbol outside math delimiters.\n\n"
        "Questions must be {exam_target}-syllabus-aligned, conceptually varied, "
        "and gradually increasing in difficulty. "
        "Do NOT generate school-board questions, NCERT recall questions, or "
        "trivial definition-based MCQs."
    ),

    # ── PYQ (coaching/competitive) ────────────────────────────────────────
    "pyq": (
        "Generate a {exam_target} Previous Year Question (PYQ) style practice set "
        "for this topic in Markdown. Simulate authentic {exam_target} exam questions "
        "at genuine difficulty — NOT school board level.\n\n"
        "Format:\n"
        "# {exam_target} PYQ Practice — {topic_name}\n"
        "**Subject:** {subject_name} | **Chapter:** {chapter_name}\n\n"
        "---\n\n"
        "## Practice Questions\n"
        "Generate {question_count} questions in authentic {exam_target} pattern:\n"
        "- For JEE: Single-correct MCQ ({mcq_count}) + Integer-type numericals ({integer_count}) + Multi-correct MCQ ({multicorrect_count})\n"
        "- For NEET: Single-correct MCQ ({mcq_count}) + Assertion-Reason ({assertion_reason_count})\n"
        "- For JEE/NEET combined: Single-correct MCQ ({mcq_count}) + Integer-type numericals ({integer_count}) + Assertion-Reason ({assertion_reason_count})\n"
        "Number questions clearly (Q1, Q2, ...). "
        "Format every question exactly as: `1. [EXAMTAG: JEE Main 2019] <question text>` or `1. [EXAMTAG: NEET 2021] <question text>`. "
        "The exam name and year must appear ONLY inside EXAMTAG and must never be repeated in the visible question text. "
        "It MUST be a real, authentic past year of the exam, never a dummy year or empty placeholder like '____' or 'Year' or '20XX'. "
        "Do NOT include solutions or answers inline.\n\n"
        "CRITICAL MCQ OPTION FORMATTING: Write each option (A-D) on a new line, never inline on a single line.\n\n"
        "---\n\n"
        "## Detailed Solutions\n"
        "Provide full step-by-step solutions for every question. For all numerical questions, provide a detailed step-by-step solution showing calculations and working, where each new mathematical step is written on a new line, never combined into a single paragraph. For all MCQ and theory questions, provide the complete explanation/reasoning along with the correct option, not just the option letter alone. Explain the concept, show all working, cite relevant formulas, and highlight common mistakes.\n\n"
        "CRITICAL MATH NOTATION: For all mathematics, equations, exponents, and variables, always use valid KaTeX/LaTeX Markdown. Exponents must use carets (e.g., $x^2$, $x^3$), and all mathematical expressions must be wrapped in single dollar signs (e.g. $3\\sqrt{5}$, $f(3) = 0$). Never output raw math or variables without dollar signs, and never use raw exponents like x2 or x3.\n\n"
        "Math formatting rules: wrap every inline expression in single dollar signs, "
        "e.g. $v = u + at$. Use LaTeX commands for fractions, roots, integrals, etc. "
        "Never write raw fractions or the Unicode √ symbol outside dollar signs.\n\n"
        "Questions must be at genuine {exam_target} difficulty and pattern. "
        "Do NOT generate school-board questions, CBSE-style long answers, "
        "NCERT recall questions, or trivial 1-mark definitions."
    ),
}


def _get_content_prompts(vertical: str) -> dict[str, str]:
    """Return the prompt dict appropriate for the given product vertical."""
    if vertical == "school":
        return _CONTENT_TYPE_PROMPTS
    return {**_CONTENT_TYPE_PROMPTS, **_COACHING_CONTENT_TYPE_PROMPTS}



_DIFFICULTY_DESC = {
    "basic":        "introductory level, simple language, suitable for beginners",
    "intermediate": "standard curriculum depth, JEE/NEET Mains level",
    "advanced":     "advanced level, JEE Advanced / NEET PG competitive exam depth",
}



# Hard exam-specific constraints injected as the FIRST block in the system prompt.
# These prevent the model from mixing question styles across exam targets.
_EXAM_TARGET_RULES: dict[str, str] = {
    "jee": (
        "TARGET EXAM: JEE (Joint Entrance Examination)\n"
        "STRICT CONSTRAINTS — NEVER VIOLATE:\n"
        "- Allowed question styles: Single-correct MCQ, Multi-correct MCQ, Integer-type numericals, Match-the-column\n"
        "- Subjects in scope: Physics, Chemistry, Mathematics ONLY\n"
        "- ZERO biology content. ZERO NEET-style assertion-reason questions.\n"
        "- Difficulty: JEE Main = moderate multi-step, JEE Advanced = deep conceptual + multi-step\n"
        "- Numericals must require formula application and explicit multi-step calculation\n"
        "- Wrong-answer traps must be physics/chemistry/math conceptual errors, not biology facts"
    ),
    "neet": (
        "TARGET EXAM: NEET (National Eligibility cum Entrance Test)\n"
        "STRICT CONSTRAINTS — NEVER VIOLATE:\n"
        "- Allowed question styles: Single-correct MCQ, Assertion-Reason, Statement-based (True/False combos)\n"
        "- Subjects in scope: Physics, Chemistry, Botany, Zoology\n"
        "- ZERO integer-type numericals. ZERO multi-correct MCQs. ZERO JEE-style matrix match.\n"
        "- Every question must be directly derivable from NCERT Class 11-12 textbooks\n"
        "- Biology questions must use correct scientific/taxonomic names (italicised)\n"
        "- Physics/Chemistry: NEET pattern — conceptual recall over heavy derivation"
    ),
    # The class_* rules are TEMPLATES: {board}/{textbooks} are substituted from the
    # tenant's board by _resolve_exam_rule(). They previously hardcoded CBSE/NCERT,
    # which is wrong for an ICSE school (ICSE follows CISCE, not NCERT).
    "class_12": (
        "TARGET: Class 12 Board Exam ({board})\n"
        "STRICT CONSTRAINTS — NEVER VIOLATE:\n"
        "- Question styles: 1-mark MCQ/assertion, 2-mark short answer, 3-mark derivation, 5-mark long answer, case-study\n"
        "- Strictly the {board} Class 12 syllabus — every definition and example from {textbooks}\n"
        "- DO NOT generate JEE advanced multi-step or NEET trap questions\n"
        "- Difficulty: Board exam moderate level — test conceptual understanding, not tricks"
    ),
    "class_11": (
        "TARGET: Class 11 Board Exam ({board})\n"
        "STRICT CONSTRAINTS — NEVER VIOLATE:\n"
        "- Same board pattern as Class 12 but strictly Class 11 syllabus only\n"
        "- Aligned to {textbooks} for Class 11 — definitions, basic numericals, conceptual recall\n"
        "- Do not use Class 12 topics. Do not use competitive exam pattern."
    ),
    "class_10": (
        "TARGET: Class 10 Board Exam ({board})\n"
        "STRICT CONSTRAINTS — NEVER VIOLATE:\n"
        "- Question styles: MCQ, 2-mark, 3-mark, 5-mark\n"
        "- Aligned to {textbooks} for Class 10 — simple foundational concepts\n"
        "- Simple language. No advanced derivations. No competitive exam traps."
    ),
}





def _resolve_exam_rule(exam_target: str, board: str = "CBSE") -> str:
    """Normalise raw examTarget string and return the matching rule block."""
def _resolve_exam_rule(exam_target: str, board: str = "") -> str:
    """
    Normalise raw examTarget and return the matching rule block.

    The class_* (school board) rules are templates: {board}/{textbooks} are filled
    from the tenant's board, so an ICSE school gets CISCE framing instead of the
    NCERT/CBSE wording these rules used to hardcode. JEE/NEET rules are unaffected.
    """
    def _fill(rule: str) -> str:
        if "{board}" not in rule and "{textbooks}" not in rule:
            return rule
        from ai_services.core.boards import get_board
        p = get_board(board)
        return rule.replace("{board}", p.display_name).replace("{textbooks}", p.textbooks)

    t = (exam_target or "").lower().strip()
    if not t:
        return ""
    if "jee" in t:
        return _EXAM_TARGET_RULES["jee"]
    if "neet" in t:
        return _EXAM_TARGET_RULES["neet"]
    if "12" in t:
        return _EXAM_TARGET_RULES["class_12"].replace("CBSE", board)
    if "11" in t:
        return _EXAM_TARGET_RULES["class_11"].replace("CBSE", board)
    if "10" in t:
        return _EXAM_TARGET_RULES["class_10"].replace("CBSE", board)
        return _fill(_EXAM_TARGET_RULES["class_12"])
    if "11" in t:
        return _fill(_EXAM_TARGET_RULES["class_11"])
    if "10" in t:
        return _fill(_EXAM_TARGET_RULES["class_10"])
    return ""



_LENGTH_WORDS = {
    "brief":    "~300 words",
    "standard": "~800 words",
    "detailed": "~1500 words",
}


_CONTENT_LATEX_COMMAND_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|int|sum|prod|lim|sin|cos|tan|log|ln|"
    r"theta|alpha|beta|gamma|delta|pi|phi|psi|omega|lambda|sigma|mu|"
    r"times|cdot|div|pm|leq|geq|neq|to)(?:\b|(?=[_^{]))"
)
_CONTENT_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
})


def _normalize_generated_math_markdown(markdown: str) -> str:
    """Repair common LLM math output so remark-math can reliably invoke KaTeX.

    This is deliberately conservative: existing dollar-delimited math is left
    intact, while standalone calculation lines and raw LaTeX commands are
    wrapped. It runs before generated content leaves the AI service, keeping
    school and coaching previews/storage consistent.
    """
    text = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    # Models occasionally wrap the entire Markdown response in a code fence;
    # that makes remark-math treat every formula as literal code.
    outer_fence = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*", text, re.IGNORECASE)
    if outer_fence:
        text = outer_fence.group(1).strip()

    # A fenced LaTeX block is display math, not source code.
    text = re.sub(
        r"```(?:latex|math|katex)\s*\n([\s\S]*?)\n```",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        text,
        flags=re.IGNORECASE,
    )

    text = (
        text.replace(r"\[", "$$").replace(r"\]", "$$")
        .replace(r"\(", "$").replace(r"\)", "$")
    )

    # Convert presentation-oriented Unicode math to valid LaTeX first.
    text = re.sub(
        r"(?<=[A-Za-z0-9})\]])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾]+)",
        lambda match: "^{" + match.group(1).translate(_CONTENT_SUPERSCRIPT_MAP) + "}",
        text,
    )
    text = re.sub(r"√\s*\(([^()]+)\)", r"\\sqrt{\1}", text)
    text = re.sub(r"√\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", text)

    lines = text.split("\n")
    in_fence = False
    in_display_math = False
    repaired: list[str] = []

    for original_line in lines:
        line = original_line
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            repaired.append(line)
            continue
        if in_fence or not stripped:
            repaired.append(line)
            continue

        display_count = line.count("$$")
        if in_display_math:
            repaired.append(line)
            if display_count % 2 == 1:
                in_display_math = False
            continue
        if display_count:
            repaired.append(line)
            if display_count % 2 == 1:
                in_display_math = True
            continue

        single_dollars = len(re.findall(r"(?<!\$)\$(?!\$)", line))
        if single_dollars % 2 == 1:
            # An unmatched delimiter on a calculation line otherwise consumes
            # the following Markdown and prevents KaTeX rendering.
            if stripped.endswith("$") and not stripped.startswith("$"):
                leading = line[:len(line) - len(line.lstrip())]
                line = leading + "$" + line.lstrip()
            else:
                line = line.rstrip() + "$"
            repaired.append(line)
            continue
        if single_dollars:
            repaired.append(line)
            continue

        body_match = re.match(r"^(\s*(?:(?:Step\s+\d+|\d+[.)]|[-*+])\s*[:.-]?\s*)?)(.*)$", line, re.IGNORECASE)
        prefix, body = body_match.groups() if body_match else ("", line)
        body = body.strip()
        has_latex = bool(_CONTENT_LATEX_COMMAND_RE.search(body))
        has_equation = bool(re.search(r"(?:[A-Za-z0-9})\]])\s*(?:=|≤|≥|≠|≈|→)\s*", body))
        has_math_structure = has_latex or has_equation or bool(re.search(r"[A-Za-z0-9})\]]\s*[+*/^_]\s*[A-Za-z0-9({\[]", body))
        prose_words = re.findall(r"\b[A-Za-z]{3,}\b", _CONTENT_LATEX_COMMAND_RE.sub("", body))

        if has_math_structure and not prose_words:
            math_body = (body.replace("×", r"\times ").replace("÷", r"\div ")
                         .replace("≤", r"\leq ").replace("≥", r"\geq ")
                         .replace("≠", r"\neq ").replace("→", r"\to "))
            repaired.append(f"{prefix}${math_body.strip()}$")
        elif has_equation:
            # Preserve explanatory prose while wrapping a trailing equation,
            # e.g. "Therefore, x = \\frac{1}{2}".
            equation_match = re.search(r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9_]*(?:\^\{[^}]+\})?\s*=.+)$", body)
            equation = equation_match.group(1).strip() if equation_match else ""
            equation_words = re.findall(
                r"\b[A-Za-z]{3,}\b",
                _CONTENT_LATEX_COMMAND_RE.sub("", equation),
            )
            if equation_match and not equation_words:
                math_equation = (equation.replace("×", r"\times ").replace("÷", r"\div ")
                                 .replace("≤", r"\leq ").replace("≥", r"\geq ")
                                 .replace("≠", r"\neq ").replace("→", r"\to "))
                repaired.append(f"{prefix}{body[:equation_match.start(1)]}${math_equation}$")
            else:
                repaired.append(line)
        else:
            repaired.append(line)

    result = "\n".join(repaired)
    # Close a genuinely unfinished display block at EOF rather than allowing it
    # to swallow the remainder of the document.
    if in_display_math:
        result = result.rstrip() + "\n$$"
    return result.strip()


_EXAM_YEAR_TAG = (
    r"(?:[A-Za-z\s]+(?:\s+Class\s+\d+)?\s+\d{4}|CLASS\s+\d+\s+\d{4}|"
    r"NEET(?:\s+UG)?\s+\d{4}|JEE(?:\s+(?:Main|Advanced))?\s+\d{4})"
)


def _normalize_pyq_exam_tags(markdown: str) -> str:
    """Keep a PYQ's exam/year solely in the renderer's badge marker."""
    normalized: list[str] = []
    leading_tag = re.compile(
        rf"^(\s*)(?:Q\s*)?(\d+)[.)]\s*(?:\*\*)?(?:\[|\()?\s*({_EXAM_YEAR_TAG})\s*(?:\]|\))?"
        rf"(?:\*\*)?\s*[:—–-]?\s*",
        re.IGNORECASE,
    )
    trailing_tag = re.compile(
        rf"^(\s*)(?:Q\s*)?(\d+)[.)]\s*(.*?)[\[(]\s*({_EXAM_YEAR_TAG})\s*[\])]\s*$",
        re.IGNORECASE,
    )

    for line in str(markdown or "").split("\n"):
        if "[EXAMTAG:" in line.upper():
            marker = re.match(r"^(\s*\d+\.\s*\[EXAMTAG:\s*([^\]]+)\]\s*)(.*)$", line, re.IGNORECASE)
            if marker:
                prefix, tag, question = marker.groups()
                question = re.sub(
                    rf"\s*(?:\*\*)?(?:\[|\()?\s*{re.escape(tag)}\s*(?:\]|\))?(?:\*\*)?\s*[:—–-]?\s*",
                    " ",
                    question,
                    flags=re.IGNORECASE,
                ).strip()
                normalized.append(f"{prefix}{question}")
            else:
                normalized.append(line)
            continue
        match = leading_tag.match(line)
        if match:
            indent, number, tag = match.group(1), match.group(2), match.group(3)
            normalized.append(f"{indent}{number}. [EXAMTAG: {tag}] {line[match.end():].lstrip()}")
            continue
        match = trailing_tag.match(line)
        if match:
            indent, number, question, tag = match.groups()
            normalized.append(f"{indent}{number}. [EXAMTAG: {tag}] {question.rstrip()}")
            continue
        normalized.append(line)
    return "\n".join(normalized)


def _has_incomplete_mcq_options(markdown: str) -> bool:
    """Detect generated option labels that have no option text.

    Only inspect substantial question-sheet output so short mocked responses and
    non-question content are not treated as malformed.
    """
    text = str(markdown or "")
    if len(text) < 200 or not re.search(r"(?im)^#{1,3}\s+.*(?:MCQ|Practice Questions)", text):
        return False
    question_section = re.split(
        r"(?im)^#{1,3}\s+(?:Answer Key|Detailed Solutions?|Solutions?)\s*$",
        text,
        maxsplit=1,
    )[0]
    option_line = re.compile(
        r"^\s*(?:[-*+]\s*)?(?:\*\*)?([A-D])(?:\*\*)?\s*(?:[.):—–-]\s*)?(?:\*\*)?\s*(.*?)\s*$",
        re.IGNORECASE,
    )
    option_count = 0
    for line in question_section.splitlines():
        match = option_line.match(line)
        if not match:
            continue
        option_count += 1
        value = match.group(2).strip().strip("*").strip()
        if not value or re.match(r"^(?:Q\s*)?\d+[.)]\s+", value, re.IGNORECASE):
            return True
    return option_count == 0





@api_view(["POST"])
def generate_topic_content(request):
    _start_time = time.time()
    data = request.data
    vertical      = getattr(request, "vertical", "coaching")
    topic_name    = data.get("topicName", "").strip()
    subject_name  = data.get("subjectName", "").strip()
    chapter_name  = data.get("chapterName", "").strip()
    content_type  = str(data.get("contentType", "lesson")).strip().lower()
    difficulty    = data.get("difficulty", "intermediate")
    length        = data.get("length", "standard")
    exam_target   = (data.get("examTarget") or data.get("exam_target") or "").strip()
    course_name   = (data.get("courseName") or data.get("course_name") or "").strip()
    extra_context = data.get("extraContext", "").strip()
    board         = str(data.get("board") or "CBSE").strip()
    # Language: 'english' (default/None) → Groq/llama; 'hindi' → Groq with Devanagari instruction;
    # 'odia' → Gemini (Groq/llama are weak at Odia script, same as for STT notes).
    language = str(data.get("language") or "").strip().lower()

    question_count_val = data.get("questionCount") or data.get("question_count")
    try:
        q_count = int(question_count_val)
    except (TypeError, ValueError):
        q_count = 10 if content_type == "dpp" else 12

    mcq_count = q_count
    integer_count = 0
    multicorrect_count = 0
    assertion_reason_count = 0



    # If exam_target not supplied, try to infer from course name
    if not exam_target and course_name:
        cn = course_name.lower()
        if "neet" in cn:
            exam_target = "NEET"
        elif "jee" in cn:
            exam_target = "JEE"
        elif "class 12" in cn or "12th" in cn:
            exam_target = "Class 12"
        elif "class 11" in cn or "11th" in cn:
            exam_target = "Class 11"
        elif "class 10" in cn or "10th" in cn:
            exam_target = "Class 10"

    # Vertical-aware fallback: school content must not default to JEE framing.
    # School with no explicit target → school-level (Class 10) rules.
    if not exam_target and vertical == "school":
        exam_target = "Class 10"



    if not topic_name:
        return Response({"error": "Missing topicName"}, status=400)



    active_prompts = _get_content_prompts(vertical)
    exam_upper = exam_target.upper() if exam_target else "JEE"

    if "JEE" in exam_upper:
        if q_count >= 3:
            multicorrect_count = max(1, round(q_count * 0.20))
            integer_count = max(1, round(q_count * 0.20))
            mcq_count = q_count - multicorrect_count - integer_count
        else:
            mcq_count = q_count
            integer_count = 0
            multicorrect_count = 0
    elif "NEET" in exam_upper:
        if q_count >= 5:
            assertion_reason_count = max(1, round(q_count * 0.20))
            mcq_count = q_count - assertion_reason_count
        else:
            mcq_count = q_count
            assertion_reason_count = 0
    else:
        if q_count >= 5:
            assertion_reason_count = max(1, round(q_count * 0.20))
            mcq_count = q_count - assertion_reason_count
        else:
            mcq_count = q_count
            assertion_reason_count = 0

    type_instruction = active_prompts.get(
        content_type,
        active_prompts["lesson"],
    ).replace("{topic_name}", topic_name).replace("{subject_name}", subject_name).replace("{chapter_name}", chapter_name).replace("CBSE", board)

    type_instruction = (
        type_instruction
        .replace("{question_count}", str(q_count))
        .replace("{mcq_count}", str(mcq_count))
        .replace("{integer_count}", str(integer_count))
        .replace("{multicorrect_count}", str(multicorrect_count))
        .replace("{assertion_reason_count}", str(assertion_reason_count))
    )

    type_instruction = type_instruction.replace("{exam_target}", exam_upper)
    if vertical == "school":
        school_difficulty_desc = {
            "basic": "introductory school-level depth with simple language",
            "intermediate": "standard school curriculum and board exam depth",
            "advanced": "higher-order school board questions, still within the selected class syllabus",
        }
        diff_desc = school_difficulty_desc.get(difficulty, school_difficulty_desc["intermediate"])
    else:
        diff_desc = _DIFFICULTY_DESC.get(difficulty, _DIFFICULTY_DESC["intermediate"])
    word_limit = _LENGTH_WORDS.get(length, _LENGTH_WORDS["standard"])



    # Build exam-specific constraint block — injected first in system prompt
    exam_rule = _resolve_exam_rule(exam_target, board)
    exam_rule = _resolve_exam_rule(exam_target, getattr(request, "board", ""))



    system_prompt = (
        (
            f"EXAM CONSTRAINT — READ AND FOLLOW STRICTLY:\n{exam_rule}\n\n"
            if exam_rule else ""
        )
        + "You are an expert educational content creator for Indian exams. "
        "Write accurate, engaging, curriculum-aligned educational content in Markdown. "
        + (
            "Strictly follow the exam constraint above. Never mix question styles or content patterns from other exams. "
            if exam_rule else ""
        )
    )
    # Language-aware instruction appended to system prompt
    if language == "hindi":
        system_prompt += (
            "\n\nLANGUAGE REQUIREMENT — MANDATORY: Write ALL content entirely in Hindi using Devanagari script. "
            "This includes all headings, explanations, questions, answer choices, and solutions. "
            "Do NOT mix English prose into the output. Technical terms (e.g. formula names, chemical symbols, "
            "mathematical symbols) may remain in their standard notation."
        )
    elif language == "odia":
        system_prompt += (
            "\n\nLANGUAGE REQUIREMENT — MANDATORY: Write ALL content entirely in Odia using Odia script (\u0b13\u0b21\u0b3c\u0b3f\u0b06). "
            "This includes all headings, explanations, questions, answer choices, and solutions. "
            "Do NOT mix English or Hindi prose into the output. Technical terms and mathematical symbols "
            "may remain in their standard notation."
        )


    # Build user prompt with course + exam prominently at the top
    user_prompt_parts = []
    if course_name:
        user_prompt_parts.append(f"Course: {course_name}")
    if exam_target:
        user_prompt_parts.append(f"Exam Target: {exam_target.upper()}")
    user_prompt_parts += [
        f"Subject: {subject_name}",
        f"Chapter: {chapter_name}",
        f"Topic: {topic_name}",
        f"Content type: {content_type}",
        f"Difficulty: {diff_desc}",
        f"Target length: {word_limit}",
    ]
    user_prompt = "\n".join(user_prompt_parts) + "\n"
    if extra_context:
        user_prompt += f"Additional instructions: {extra_context}\n"
    user_prompt += (
        f"\n{type_instruction}\n\n"
        "Return ONLY the Markdown content -- no preamble, no 'Here is your content:' prefix."
    )
    if content_type in {"dpp", "pyq"}:
        user_prompt += (
            "\n\nMANDATORY OUTPUT VALIDATION BEFORE RETURNING:\n"
            "- Do not place the Markdown response or any solution inside a code fence.\n"
            "- Every MCQ must have four non-empty options formatted exactly as `A. <text>`, `B. <text>`, `C. <text>`, and `D. <text>`, one option per line. Never output a bare option letter.\n"
            "- Every calculation must use valid LaTeX inside math delimiters.\n"
            "- Use $...$ for inline math and put multi-step equations in separate $$...$$ blocks.\n"
            "- Never emit Unicode superscripts, the Unicode square-root symbol, or raw LaTeX outside delimiters.\n"
            "- Ensure every opening $ or $$ has a matching closing delimiter.\n"
            "- Finish the Detailed Solutions section for every generated question."
        )
    if content_type == "pyq":
        user_prompt += (
            "\n- Format every question as `1. [EXAMTAG: <exact exam/class and year>] <question text>`."
            "\n- Put the exam name and year only inside EXAMTAG; do not repeat either in the question text."
        )



    institute_id = _resolve_institute_id(request)
    user_id_tc = data.get('userId') or data.get('user_id') or data.get('studentId') or ''
    logger.info(
        # vertical/board are logged so the resolved personalisation is observable in
        # prod: without them there is no way to confirm an ICSE school actually got
        # ICSE framing rather than silently falling back to the CBSE default.
        "generate_topic_content | vertical=%s | board=%s | course=%s | exam=%s | subject=%s | topic=%s | type=%s | language=%s",
        getattr(request, "vertical", "—"), getattr(request, "board", "—"),
        course_name or "—", exam_target or "—", subject_name or "—", topic_name[:40], content_type, language or "english",
    )

    # ── Odia: route to Gemini (Groq/llama are weak at Odia script) ───────────
    if language == "odia" and GEMINI_ODIA_NOTES_ENABLED and has_gemini_api_key():
        try:
            try:
                from google import genai
                from google.genai import types as _gtypes
            except Exception as _genai_exc:
                raise RuntimeError(f"google-genai unavailable for Odia content: {_genai_exc}") from _genai_exc

            _odia_content = None
            _odia_exc_last = None
            for _key_idx, _api_key in get_rotated_gemini_keys():
                try:
                    _client = genai.Client(api_key=_api_key)
                    _max_tok_odia = 8192 if content_type in {"dpp", "pyq"} else 4096
                    try:
                        _resp = _client.models.generate_content(
                            model=GEMINI_ODIA_NOTES_MODEL,
                            contents=[user_prompt],
                            config=_gtypes.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.7,
                                max_output_tokens=_max_tok_odia,
                                thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                    except TypeError:
                        _resp = _client.models.generate_content(
                            model=GEMINI_ODIA_NOTES_MODEL,
                            contents=[user_prompt],
                            config=_gtypes.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.7,
                                max_output_tokens=_max_tok_odia,
                            ),
                        )
                    _odia_content = str(getattr(_resp, "text", "") or "").strip()
                    _odia_content = re.sub(r"^```(?:markdown)?\s*", "", _odia_content, flags=re.IGNORECASE).strip()
                    _odia_content = re.sub(r"\s*```$", "", _odia_content).strip()
                    logger.info(
                        "generate_topic_content Odia/Gemini OK | key=%d/%d | model=%s | chars=%d",
                        _key_idx, gemini_key_count(), GEMINI_ODIA_NOTES_MODEL, len(_odia_content),
                    )
                    break
                except Exception as _key_exc:
                    _odia_exc_last = _key_exc
                    msg = str(_key_exc)
                    if is_gemini_permanent_key_error(msg):
                        mark_gemini_key_disabled(_api_key)
                        logger.warning("Gemini key %d/%d disabled for permanent error: %s", _key_idx, gemini_key_count(), msg[:180])
                        continue
                    if is_gemini_retryable_error(msg):
                        logger.warning("Gemini key %d/%d retryable; rotating: %s", _key_idx, gemini_key_count(), msg[:180])
                        continue
                    raise RuntimeError(f"Gemini Odia content failed: {_key_exc}") from _key_exc

            if not _odia_content:
                raise RuntimeError(f"Gemini Odia content empty or all keys exhausted: {_odia_exc_last}")

            # Apply the same post-processing as the Groq path
            _odia_content = _normalize_generated_math_markdown(_odia_content)
            if content_type == "pyq":
                _odia_content = _normalize_pyq_exam_tags(_odia_content)

            # DPP/PYQ: retry once via Gemini if MCQ options are incomplete
            if content_type in {"dpp", "pyq"} and _has_incomplete_mcq_options(_odia_content):
                logger.warning("generate_topic_content Odia: incomplete MCQ options; regenerating once via Gemini")
                _retry_prompt_odia = (
                    user_prompt
                    + "\n\nYOUR PREVIOUS OUTPUT HAD EMPTY OR MISSING MCQ OPTION TEXT. Regenerate the complete document. "
                      "Every MCQ must contain four non-empty choices. Never output a bare option letter. Keep each option on one line."
                )
                _retry_odia = None
                for _key_r, _api_key_r in get_rotated_gemini_keys():
                    try:
                        _client_r = genai.Client(api_key=_api_key_r)
                        try:
                            _resp_r = _client_r.models.generate_content(
                                model=GEMINI_ODIA_NOTES_MODEL,
                                contents=[_retry_prompt_odia],
                                config=_gtypes.GenerateContentConfig(
                                    system_instruction=system_prompt,
                                    temperature=0.35,
                                    max_output_tokens=8192,
                                    thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
                                ),
                            )
                        except TypeError:
                            _resp_r = _client_r.models.generate_content(
                                model=GEMINI_ODIA_NOTES_MODEL,
                                contents=[_retry_prompt_odia],
                                config=_gtypes.GenerateContentConfig(
                                    system_instruction=system_prompt,
                                    temperature=0.35,
                                    max_output_tokens=8192,
                                ),
                            )
                        _retry_odia = str(getattr(_resp_r, "text", "") or "").strip()
                        _retry_odia = re.sub(r"^```(?:markdown)?\s*", "", _retry_odia, flags=re.IGNORECASE).strip()
                        _retry_odia = re.sub(r"\s*```$", "", _retry_odia).strip()
                        break
                    except Exception as _r_exc:
                        logger.warning("Gemini Odia MCQ retry key error: %s", _r_exc)
                        continue
                if _retry_odia:
                    _odia_content = _normalize_generated_math_markdown(_retry_odia)
                    if content_type == "pyq":
                        _odia_content = _normalize_pyq_exam_tags(_odia_content)
                    if _has_incomplete_mcq_options(_odia_content):
                        raise RuntimeError("Gemini Odia MCQ retry still incomplete; falling back to Groq")

            try:
                log_usage(
                    institute_id=institute_id,
                    institute_type=vertical if vertical in ('school', 'coaching') else 'coaching',
                    feature_id=f'content_{content_type}_odia' if content_type else 'content_generate_odia',
                    feature_category='content',
                    model_used=GEMINI_ODIA_NOTES_MODEL,
                    latency_ms=int((time.time() - _start_time) * 1000),
                    success=True,
                    user_id=user_id_tc,
                )
            except Exception:
                pass
            return Response({
                "content": _odia_content,
                "contentType": content_type,
                "topicName": topic_name,
                "_meta": {"model": GEMINI_ODIA_NOTES_MODEL, "latency_ms": round((time.time() - _start_time) * 1000)},
            })
        except Exception as odia_exc:
            logger.warning(
                "generate_topic_content Odia/Gemini failed (%s) — falling back to Groq", odia_exc
            )
            # Fall through to Groq below


    try:
        llm_result = get_llm().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            # PYQ/DPP put complete worked solutions at the end. A 4096-token cap
            # routinely truncated that section and left math delimiters open.
            max_tokens=8192 if content_type in {"dpp", "pyq"} else 4096,
            json_mode=False,
            institute_id=institute_id,
        )
    except RuntimeError as e:
        try:
            log_usage(
                institute_id=institute_id,
                institute_type=vertical if vertical in ('school', 'coaching') else 'coaching',
                feature_id=f'content_{content_type}' if content_type else 'content_generate',
                feature_category='content',
                model_used='llama-3.3-70b-versatile',
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(e)[:500],
                user_id=user_id_tc,
            )
        except Exception:
            pass
        return Response({"error": str(e)}, status=502)

    first_content = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
    if content_type in {"dpp", "pyq"} and _has_incomplete_mcq_options(first_content):
        logger.warning("generate_topic_content: incomplete MCQ options detected; regenerating once")
        retry_prompt = (
            user_prompt
            + "\n\nYOUR PREVIOUS OUTPUT HAD EMPTY OR MISSING MCQ OPTION TEXT. Regenerate the complete document. "
              "Every MCQ must contain four non-empty choices using exactly `A. <text>`, `B. <text>`, "
              "`C. <text>`, and `D. <text>`. Never output a bare option letter. Keep each option on one line."
        )
        try:
            llm_result = get_llm().complete(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.35,
                max_tokens=8192,
                json_mode=False,
                institute_id=institute_id,
            )
        except RuntimeError as e:
            try:
                log_usage(
                    institute_id=institute_id,
                    institute_type=vertical if vertical in ('school', 'coaching') else 'coaching',
                    feature_id=f'content_{content_type}' if content_type else 'content_generate',
                    feature_category='content',
                    model_used='llama-3.3-70b-versatile',
                    latency_ms=int((time.time() - _start_time) * 1000),
                    success=False,
                    error_message=f"MCQ retry failed: {str(e)[:500]}",
                    user_id=user_id_tc,
                )
            except Exception:
                pass
            return Response({"error": f"AI regeneration failed: {e}"}, status=502)
        retried_content = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
        if _has_incomplete_mcq_options(retried_content):
            try:
                log_usage(
                    institute_id=institute_id,
                    institute_type=vertical if vertical in ('school', 'coaching') else 'coaching',
                    feature_id=f'content_{content_type}' if content_type else 'content_generate',
                    feature_category='content',
                    model_used=llm_result.get('model', 'llama-3.3-70b-versatile'),
                    tokens_input=llm_result.get('usage', {}).get('prompt_tokens', 0),
                    tokens_output=llm_result.get('usage', {}).get('completion_tokens', 0),
                    latency_ms=int((time.time() - _start_time) * 1000),
                    success=False,
                    error_message="Incomplete MCQ options after retry",
                    user_id=user_id_tc,
                )
            except Exception:
                pass
            return Response(
                {"error": "AI returned incomplete MCQ options. Please generate again."},
                status=502,
            )

    try:
        log_usage(
            institute_id=institute_id,
            institute_type=vertical if vertical in ('school', 'coaching') else 'coaching',
            feature_id=f'content_{content_type}' if content_type else 'content_generate',
            feature_category='content',
            model_used=llm_result.get('model', 'llama-3.3-70b-versatile'),
            tokens_input=llm_result.get('usage', {}).get('prompt_tokens', 0),
            tokens_output=llm_result.get('usage', {}).get('completion_tokens', 0),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_tc,
        )
    except Exception:
        pass

    content = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
    content = _normalize_generated_math_markdown(content)
    if content_type == "pyq":
        content = _normalize_pyq_exam_tags(content)
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
    _start_time = time.time()


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
    skip_image_generation = bool(
        data.get("skipImageGeneration", False) or data.get("skip_image_generation", False)
    )
    institute_id = _resolve_institute_id(request)
    user_id_nft = data.get('userId') or data.get('user_id') or data.get('studentId') or ''



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
        # Skip polish for Hindi/Hinglish/Odia: the merge already outputs clean structured markdown,
        # and the English-oriented polish would corrupt Odia script.
        _skip_polish_lang = _normalize_lecture_language(language) in ("hi", "hi-in", "hinglish", "od")
        if _skip_polish_lang:
            markdown_polished = False
        else:
            notes_markdown, markdown_polished = _polish_notes_markdown(
                notes_markdown, topic_id, language, institute_id,
            )
    except RuntimeError as exc:
        logger.error("notes_from_transcript LLM failed (institute=%s): %s", institute_id, exc)
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='ai_lecture_notes',
                feature_category='teacher',
                model_used='llama-3.3-70b-versatile',
                tokens_input=int(len(transcript) / 4),
                tokens_output=0,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(exc)[:500],
                user_id=user_id_nft,
            )
        except Exception:
            pass
        return Response({"error": str(exc)}, status=502)

    images = []
    image_meta = {"enabled": False, "count": 0, "errors": []}
    if skip_image_generation:
        image_meta = {"enabled": False, "count": 0, "errors": [], "skip_reason": "requested_by_client"}
    else:
        try:
            from ai_services.core.llm_client import LLMClient
            from ai_services.core.note_images import enrich_notes_with_images

            notes_markdown, images, image_meta = enrich_notes_with_images(
                notes_markdown,
                topic_id,
                language,
                institute_id,
                LLMClient(),
            )
        except Exception as exc:
            logger.warning("notes_from_transcript image enrichment skipped: %s", exc)
            image_meta = {"enabled": False, "count": 0, "errors": [str(exc)[:160]]}



    logger.info(
        "notes_from_transcript done | %d chars notes | images=%d | chunks=%d | took=%.1fs",
        len(notes_markdown),
        len(images),
        notes_meta.get("chunk_count", 0),
        _time.perf_counter() - _t0,
    )

    try:
        _nft_provider = notes_meta.get('provider', 'groq')
        _nft_model = 'gemini-2.5-flash' if _nft_provider == 'gemini' else 'llama-3.3-70b-versatile'
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='ai_lecture_notes',
            feature_category='teacher',
            model_used=_nft_model,
            tokens_input=int(len(transcript) / 4),
            tokens_output=int(len(notes_markdown) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_nft,
        )
    except Exception:
        pass

    # Return same shape as /stt/notes so NestJS content.service.ts needs zero changes
    return Response({
        "notes": notes_markdown,
        "rawTranscript": transcript,
        "englishTranscript": english_transcript,
        "keyConcepts": [],
        "formulas": [],
        "summary": "",
        "images": images,
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
            "image_generation": image_meta,
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
    _start_time = time.time()


    data = request.data
    video_id = (data.get("videoId") or data.get("video_id") or "").strip()
    topic_id = data.get("topicId", "")
    language = data.get("language", "en")
    institute_id = _resolve_institute_id(request)
    user_id_nfy = data.get('userId') or data.get('user_id') or data.get('studentId') or ''



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
        if _normalize_lecture_language(language) in ("hi", "hi-in", "hinglish", "od"):
            markdown_polished = False
        else:
            notes_markdown, markdown_polished = _polish_notes_markdown(
                notes_markdown, topic_id, language, institute_id
            )
    except RuntimeError as exc:
        logger.error("generate_notes_from_youtube LLM failed for %s: %s", video_id, exc)
        try:
            log_usage(
                institute_id=institute_id,
                institute_type='school',
                feature_id='ai_lecture_notes',
                feature_category='teacher',
                model_used='llama-3.3-70b-versatile',
                tokens_input=int(len(transcript) / 4),
                tokens_output=0,
                latency_ms=int((time.time() - _start_time) * 1000),
                success=False,
                error_message=str(exc)[:500],
                user_id=user_id_nfy,
            )
        except Exception:
            pass
        return Response({"error": str(exc)}, status=502)



    logger.info(
        "generate_notes_from_youtube done | videoId=%s | notes=%d chars | took=%.1fs",
        video_id, len(notes_markdown), _time.perf_counter() - _t0,
    )

    try:
        _nfy_provider = notes_meta.get('provider', 'groq')
        _nfy_model = 'gemini-2.5-flash' if _nfy_provider == 'gemini' else 'llama-3.3-70b-versatile'
        log_usage(
            institute_id=institute_id,
            institute_type='school',
            feature_id='ai_lecture_notes',
            feature_category='teacher',
            model_used=_nfy_model,
            tokens_input=int(len(transcript) / 4),
            tokens_output=int(len(notes_markdown) / 4),
            latency_ms=int((time.time() - _start_time) * 1000),
            success=True,
            user_id=user_id_nfy,
        )
    except Exception:
        pass

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


@api_view(["POST"])
def regenerate_single_note_image(request):
    data = request.data
    topic_id = data.get("topicId", "General")
    caption = data.get("caption", "")
    visual_description = data.get("visualDescription", "")
    evidence_quote = data.get("evidenceQuote", "")
    section_heading = data.get("sectionHeading", "")
    notes = data.get("notes", "")
    language = data.get("language", "en")

    if not caption or not visual_description:
        return Response({"error": "Missing caption or visualDescription"}, status=400)

    candidate = {
        "caption": caption,
        "visual_description": visual_description,
        "evidence_quote": evidence_quote,
        "section_heading": section_heading,
    }

    from ai_services.core.note_images import (
        _build_image_prompt,
        label_generated_note_image,
    )
    from ai_services.core.image_generation import (
        generate_note_image,
        can_generate_note_images,
    )

    if not can_generate_note_images():
        return Response({"error": "Image generation is not enabled or configured"}, status=400)

    image_prompt = _build_image_prompt(candidate, topic_id)
    image_result = generate_note_image(image_prompt)
    if not image_result.get("ok"):
        return Response({"error": image_result.get("error") or "image_generation_failed"}, status=502)

    overlay_labels, overlay_error = label_generated_note_image(image_result, candidate, notes, language)

    return JsonResponse({
        "url": image_result["url"],
        "caption": candidate["caption"],
        "visual_description": candidate["visual_description"],
        "overlay_labels": overlay_labels,
        "section_heading": candidate["section_heading"],
        "evidence_quote": candidate["evidence_quote"],
        "prompt": image_prompt,
        "provider": image_result["provider"],
        "model": image_result["model"],
        "image_size": image_result["image_size"],
        "aspect_ratio": image_result.get("aspect_ratio"),
        "embedded_text_removed": image_result.get("embedded_text_removed", 0),
        "embedded_text_strip_error": image_result.get("embedded_text_strip_error"),
        "overlay_error": overlay_error,
    })


@api_view(["POST"])
def extract_image_search_terms(request):
    data = request.data
    notes = data.get("notes", "").strip()
    language = data.get("language", "en")

    if not notes:
        return Response({"error": "Missing notes"}, status=400)

    truncated = notes[:4000]

    system_prompt = 'You are a JSON-only API that returns {"sections": [...]}.'
    user_prompt = f"""Given these lecture notes (Markdown), identify 3–4 major section headings (## or ###) that would benefit from an illustrative educational image.

For each section:
- "heading": copy the exact heading line from the notes (include the ## or ### prefix).
- "searchTerm": 4–7 words, be SPECIFIC to that sub-topic, include a visual type hint (diagram, photograph, chart, illustration, map, microscope, experiment).
- "caption": one sentence describing exactly what the image shows and how it helps students understand this section.

Return ONLY: {{"sections": [{{"heading": "## Exact Heading", "searchTerm": "...", "caption": "..."}}]}}

NOTES:
{truncated}"""

    norm_lang = _normalize_lecture_language(language)
    if norm_lang == "od":
        logger.info("extract_image_search_terms: routing through Gemini for language 'od'")
        try:
            res = _gemini_complete(system_prompt, user_prompt, max_tokens=1024, temperature=0.3)
            content = res.get("content", "").strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            sections = parsed.get("sections", parsed)
            if not isinstance(sections, list):
                sections = []
            return Response({"sections": sections})
        except Exception as exc:
            logger.warning("Gemini extract headings failed: %s", exc)
            return Response({"error": f"Gemini failed: {str(exc)}"}, status=502)
    else:
        logger.info("extract_image_search_terms: routing through Groq for language %r", language)
        try:
            llm_result = get_llm("notes_analyze")(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.3,
                json_mode=True,
            )
            content = llm_result.get("content", "").strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            sections = parsed.get("sections", parsed)
            if not isinstance(sections, list):
                sections = []
            return Response({"sections": sections})
        except Exception as exc:
            logger.warning("Groq extract headings failed: %s", exc)
            return Response({"error": f"Groq failed: {str(exc)}"}, status=502)
