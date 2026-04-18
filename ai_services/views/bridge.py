"""
Views for NestJS ai-bridge endpoints.
These endpoints match the paths called by apexiq-backend/src/modules/ai-bridge/ai-bridge.service.ts

Active endpoints:
  POST /doubt/resolve          → AI #1: Doubt Clearing
  POST /tutor/session          → AI #2: AI Tutor Start
  POST /tutor/continue         → AI #2: AI Tutor Continue
  POST /recommend/content      → AI #6: Content Recommendation
  POST /stt/notes              → AI #7: Speech-to-Text Notes  (faster-whisper)
  POST /feedback/generate      → AI #8: Student Feedback
  POST /notes/analyze          → AI #9: Notes Weak Topic Identifier
  POST /resume/analyze         → AI #10: Resume Analyzer
  POST /interview/start        → AI #11: Interview Prep
  POST /plan/generate          → AI #12: Personalized Learning Plan
  POST /quiz/generate          → AI #13: In-Video Quiz Generator

Removed endpoints (deleted from platform):
  POST /performance/analyze    → was AI #3 (performance_analysis)
  POST /grade/subjective       → was AI #4 (grade_subjective)
  POST /engage/detect          → was AI #5 (engagement_detect)
"""

import glob as _glob
import json
import logging
import os
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
from .base import ai_call, ai_call_text, get_llm

logger = logging.getLogger("ai_services.llm")

# ── Groq Whisper API (primary — cloud, fast) ─────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB Groq limit


def _transcribe_with_groq(audio_path: str, language: str) -> str:
    """Transcribe a local audio file via Groq Whisper API. Returns transcript text."""
    try:
        from groq import Groq
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq", "--quiet"])
        from groq import Groq

    file_size = os.path.getsize(audio_path)
    if file_size > GROQ_MAX_FILE_BYTES:
        raise RuntimeError(
            f"File too large for Groq ({file_size // 1024 // 1024} MB > 25 MB) — use local Whisper"
        )

    client = Groq(api_key=GROQ_API_KEY)
    filename = os.path.basename(audio_path)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(filename, f.read()),
            model=GROQ_WHISPER_MODEL,
            language=language,
            response_format="text",
        )
    # Groq returns a plain string when response_format="text"
    return result if isinstance(result, str) else result.text


# ── faster-whisper singleton (fallback — local, CPU) ─────────────────────────

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
    Primary:  Groq Whisper API  (~2-3 sec, requires GROQ_API_KEY, 25 MB limit)
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

        # ── Primary: Groq ────────────────────────────────────────────────────
        if GROQ_API_KEY:
            try:
                transcript = _transcribe_with_groq(audio_path, language)
                logger.info("Groq transcription OK — %d chars", len(transcript))
                return transcript
            except Exception as exc:
                logger.warning(
                    "Groq transcription failed (%s) — falling back to local Whisper", exc
                )

        # ── Fallback: local Whisper ──────────────────────────────────────────
        logger.info("Using local Whisper (GROQ_API_KEY not set or Groq failed)")
        return _transcribe_local(audio_path, language)


# ── AI #1 — Doubt Clearing ──────────────────────────────────────────────────

@api_view(["POST"])
def resolve_doubt(request):
    data = request.data
    question_text = data.get("questionText")
    if not question_text:
        return Response({"error": "Missing questionText"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("doubt_resolve")
    user_prompt = template.user_template.format(
        question_text=question_text,
        topic_id=data.get("topicId", "general"),
        mode=data.get("mode", "detailed"),
        student_context=json.dumps(data.get("studentContext", {})),
    )

    # Plain text mode — edvaqwen returns high-quality text but not reliable JSON
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


# ── AI #2 — AI Tutor ────────────────────────────────────────────────────────

@api_view(["POST"])
def start_tutor_session(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    institute_id = getattr(request, "institute_id", "default")
    template = get_template("tutor_session")
    user_prompt = template.user_template.format(
        student_id=student_id,
        topic_id=data.get("topicId", "general"),
        context=data.get("context", ""),
    )

    try:
        result = get_llm().complete(
            system_prompt=template.system,
            user_prompt=user_prompt,
            model=get_model_for_task("tutor_session"),
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


# ── AI #6 — Content Recommendation ──────────────────────────────────────────

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


# ── AI #7 — Speech-to-Text Notes ────────────────────────────────────────────

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
                "Transcription done — %d chars | took=%.1fs",
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
    logger.info("Sending to LLM — transcript=%d chars | transcription took=%.1fs", len(raw_transcript), _t1 - _t0)

    template = get_template("stt_notes")
    user_prompt = template.user_template.format(
        topic_id=data.get("topicId", ""),
        language=language,
        transcript=raw_transcript,
    )

    institute_id = getattr(request, "institute_id", "default")

    # Use plain-text mode — edvaqwen generates good markdown notes but fails strict JSON
    llm_result = get_llm().complete(
        system_prompt=template.system,
        user_prompt=user_prompt,
        model="edvaqwen",
        temperature=0.5,
        max_tokens=1024,
        json_mode=False,
        institute_id=institute_id,
    )

    notes_markdown = llm_result["content"] if isinstance(llm_result["content"], str) else str(llm_result["content"])
    logger.info(
        "STT notes generated | %d chars | latency=%.0fms",
        len(notes_markdown), llm_result["latency_ms"],
    )

    return JsonResponse({
        "notes": notes_markdown,
        "rawTranscript": raw_transcript,
        "keyConcepts": [],
        "formulas": [],
        "summary": "",
        "_meta": {
            "source": "llm",
            "model": llm_result["model"],
            "latency_ms": round(llm_result["latency_ms"]),
            "tokens": 0,
            "institute": institute_id,
        },
    })


# ── AI #8 — Student Feedback Engine ─────────────────────────────────────────

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


# ── AI #9 — Notes Weak Topic Identifier ─────────────────────────────────────

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


# ── AI #10 — Resume Analyzer ────────────────────────────────────────────────

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


# ── AI #11 — Interview Prep ─────────────────────────────────────────────────

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


# ── AI #12 — Personalized Learning Plan ─────────────────────────────────────

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
    return ai_call_text(request, "plan_generate", user_prompt,
                        wrap_fn=lambda t: {"planItems": [{"activity": t, "duration": "flexible", "priority": "high"}], "summary": t})


# ── AI #13 — In-Video Quiz Generator ────────────────────────────────────────

@api_view(["POST"])
def generate_quiz_questions(request):
    data = request.data
    transcript = data.get("transcript")
    if not transcript:
        return Response({"error": "Missing transcript"}, status=400)
    if len(transcript.strip()) < 50:
        return Response({"error": "Transcript too short to generate quiz questions"}, status=422)

    template = get_template("quiz_generate")
    user_prompt = template.user_template.format(
        lecture_title=data.get("lectureTitle", "Lecture"),
        topic_id=data.get("topicId", ""),
        transcript=transcript,
    )
    return ai_call_text(
        request, "quiz_generate", user_prompt,
        wrap_fn=lambda t: {"questions": [{"question": t, "type": "ai_generated"}]},
        skip_cache=True, max_tokens=4096,
    )
