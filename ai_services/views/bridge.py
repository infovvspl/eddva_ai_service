"""
Views for all 12 NestJS ai-bridge endpoints.
These endpoints match the paths called by apexiq-backend/src/modules/ai-bridge/ai-bridge.service.ts

All endpoints:
  POST /doubt/resolve          → AI #1: Doubt Clearing
  POST /tutor/session          → AI #2: AI Tutor Start
  POST /tutor/continue         → AI #2: AI Tutor Continue
  POST /performance/analyze    → AI #3: Performance Analysis
  POST /grade/subjective       → AI #4: Assessment Grading
  POST /engage/detect          → AI #5: Engagement Monitoring
  POST /recommend/content      → AI #6: Content Recommendation
  POST /stt/notes              → AI #7: Speech-to-Text Notes
  POST /feedback/generate      → AI #8: Student Feedback
  POST /notes/analyze          → AI #9: Notes Weak Topic Identifier
  POST /resume/analyze         → AI #10: Resume Analyzer
  POST /interview/start        → AI #11: Interview Prep
  POST /plan/generate          → AI #12: Personalized Learning Plan
"""

import glob as _glob
import json
import logging
import os
import tempfile

try:
    import requests as _requests
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "requests", "--quiet"])
    import requests as _requests
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ai_services.core.llm_client import get_groq_client
from ai_services.core.prompt_templates import get_template
from .base import ai_call

logger = logging.getLogger("ai_services.llm")


# ── Transcription helper ─────────────────────────────────────────────────────

def _transcribe_audio(audio_url: str, language: str = "en") -> str:
    """
    Download audio from a URL and transcribe with Groq Whisper.
    Supports YouTube URLs (via yt-dlp) and direct audio/video file URLs.
    Returns the transcript text.
    """
    import sys
    logger.info("_transcribe_audio | python=%s | url=%s", sys.executable, audio_url[:80])
    is_youtube = "youtube.com" in audio_url or "youtu.be" in audio_url
    logger.info("_transcribe_audio | is_youtube=%s", is_youtube)

    with tempfile.TemporaryDirectory() as tmpdir:
        if is_youtube:
            try:
                import yt_dlp
            except ImportError:
                import sys, subprocess
                logger.warning("yt-dlp not found — installing to %s", sys.executable)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "--quiet"])
                import yt_dlp
            ydl_opts = {
                # Prefer m4a — Groq Whisper accepts it natively (no ffmpeg needed)
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
                "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([audio_url])
            # Find whichever file was downloaded
            files = _glob.glob(os.path.join(tmpdir, "audio.*"))
            if not files:
                raise RuntimeError("yt-dlp downloaded nothing")
            audio_path = files[0]
        else:
            # Direct audio/video URL — download it
            ext = audio_url.rsplit(".", 1)[-1].split("?")[0][:8] or "mp3"
            audio_path = os.path.join(tmpdir, f"audio.{ext}")
            resp = _requests.get(audio_url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

        client = get_groq_client()
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3-turbo",
                language=language,
                response_format="text",
            )
        transcript = result if isinstance(result, str) else getattr(result, "text", str(result))
        logger.info("Whisper transcription done: %d chars for %s", len(transcript), audio_url[:60])
        return transcript


# ── AI #1 — Doubt Clearing ──────────────────────────────────────────────────
@api_view(["POST"])
def resolve_doubt(request):
    data = request.data
    question_text = data.get("questionText")
    if not question_text:
        return Response({"error": "Missing questionText"}, status=400)

    template = get_template("doubt_resolve")
    user_prompt = template.user_template.format(
        question_text=question_text,
        topic_id=data.get("topicId", "general"),
        mode=data.get("mode", "detailed"),
        student_context=json.dumps(data.get("studentContext", {})),
    )
    return ai_call(request, feature="doubt_resolve", user_prompt=user_prompt)


# ── AI #2 — AI Tutor ────────────────────────────────────────────────────────
@api_view(["POST"])
def start_tutor_session(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("tutor_session")
    user_prompt = template.user_template.format(
        student_id=student_id,
        topic_id=data.get("topicId", "general"),
        context=data.get("context", ""),
    )
    return ai_call(request, feature="tutor_session", user_prompt=user_prompt)


@api_view(["POST"])
def continue_tutor_session(request):
    data = request.data
    session_id = data.get("sessionId")
    student_message = data.get("studentMessage")
    if not session_id or not student_message:
        return Response({"error": "Missing sessionId or studentMessage"}, status=400)

    template = get_template("tutor_continue")
    user_prompt = template.user_template.format(
        session_id=session_id,
        student_message=student_message,
    )
    return ai_call(request, feature="tutor_continue", user_prompt=user_prompt)


# ── AI #3 — Performance Analysis ────────────────────────────────────────────
@api_view(["POST"])
def analyze_performance_v2(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("performance_analysis")
    user_prompt = template.user_template.format(
        student_id=student_id,
        test_session_id=data.get("testSessionId", ""),
        exam_target=data.get("examTarget", "jee"),
        attempts_json=json.dumps(data.get("attempts", [])),
    )
    return ai_call(request, feature="performance_analysis", user_prompt=user_prompt)


# ── AI #4 — Assessment Grading ──────────────────────────────────────────────
@api_view(["POST"])
def grade_subjective(request):
    data = request.data
    question_text = data.get("questionText")
    student_answer = data.get("studentAnswer")
    if not question_text or not student_answer:
        return Response({"error": "Missing questionText or studentAnswer"}, status=400)

    template = get_template("grade_subjective")
    user_prompt = template.user_template.format(
        question_text=question_text,
        expected_answer=data.get("expectedAnswer", ""),
        student_answer=student_answer,
        max_marks=data.get("maxMarks", 10),
    )
    return ai_call(request, feature="grade_subjective", user_prompt=user_prompt)


# ── AI #5 — Engagement Monitoring ───────────────────────────────────────────
@api_view(["POST"])
def detect_engagement(request):
    data = request.data
    student_id = data.get("studentId")
    if not student_id:
        return Response({"error": "Missing studentId"}, status=400)

    template = get_template("engagement_detect")
    user_prompt = template.user_template.format(
        student_id=student_id,
        context=data.get("context", "practice"),
        signals_json=json.dumps(data.get("signals", {})),
    )
    return ai_call(request, feature="engagement_detect", user_prompt=user_prompt)


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
    return ai_call(request, feature="content_recommend", user_prompt=user_prompt)


# ── AI #7 — Speech-to-Text Notes ────────────────────────────────────────────
@api_view(["POST"])
def generate_stt_notes(request):
    data = request.data
    audio_url = data.get("audioUrl")
    if not audio_url:
        return Response({"error": "Missing audioUrl"}, status=400)

    language = data.get("language", "en")
    logger.info("generate_stt_notes | audio_url=%s | language=%s", audio_url, language)

    # Use caller-supplied transcript if provided, otherwise transcribe via Whisper
    raw_transcript = data.get("transcript", "")
    if not raw_transcript:
        try:
            raw_transcript = _transcribe_audio(audio_url, language)
            logger.info("Whisper OK — %d chars | preview: %s", len(raw_transcript), raw_transcript[:300])
        except Exception as exc:
            # Return a clear error — do NOT call LLM with a bad transcript
            logger.error("Transcription FAILED for %s: %s", audio_url, exc)
            return Response(
                {
                    "error": "transcription_failed",
                    "detail": str(exc),
                    "audioUrl": audio_url,
                    "hint": "Ensure the URL is publicly accessible from the server and the file is a supported audio/video format.",
                },
                status=502,
            )

    if len(raw_transcript.strip()) < 20:
        return Response(
            {"error": "transcript_too_short", "detail": "Whisper returned almost nothing. The audio may be silent or corrupted."},
            status=422,
        )

    logger.info("Sending to LLM — transcript length: %d chars", len(raw_transcript))

    template = get_template("stt_notes")
    user_prompt = template.user_template.format(
        audio_url=audio_url,
        topic_id=data.get("topicId", ""),
        language=language,
        transcript=raw_transcript,
    )
    result = ai_call(request, feature="stt_notes", user_prompt=user_prompt, skip_cache=True, max_tokens=8192)

    # Inject the raw Whisper transcript so the frontend can show it in the Transcript tab
    # (the LLM's "transcript" field is the cleaned-up version — we keep the raw one too)
    if hasattr(result, "data") and isinstance(result.data, dict):
        result.data["rawTranscript"] = raw_transcript

    return result


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
    return ai_call(request, feature="quiz_generate", user_prompt=user_prompt, skip_cache=True, max_tokens=4096)


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
    return ai_call(request, feature="feedback_generate", user_prompt=user_prompt)


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
    return ai_call(request, feature="notes_analyze", user_prompt=user_prompt)


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
    return ai_call(request, feature="resume_analyze", user_prompt=user_prompt)


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
    return ai_call(request, feature="interview_prep", user_prompt=user_prompt)


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
    return ai_call(request, feature="plan_generate", user_prompt=user_prompt)
