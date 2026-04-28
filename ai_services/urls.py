from django.urls import path
from .views import (
    feedback, notes, content, test, career,
    personalization, admin_api, bridge,
)

urlpatterns = [
    # ══════════════════════════════════════════════════════════════════════════
    #  NestJS ai-bridge endpoints
    #  These paths match apexiq-backend/src/modules/ai-bridge/ai-bridge.service.ts
    # ══════════════════════════════════════════════════════════════════════════

    # AI #1 — Doubt Clearing
    path("doubt/resolve", bridge.resolve_doubt),
    path("doubt/ocr-image", bridge.ocr_doubt_image),

    # AI #2 — AI Tutor
    path("tutor/session", bridge.start_tutor_session),
    path("tutor/continue", bridge.continue_tutor_session),

    # AI #6 — Content Recommendation
    path("recommend/content", bridge.recommend_content),

    # AI #7 — Speech-to-Text Notes
    path("stt/notes", bridge.generate_stt_notes),

    # AI #7a -- Transcribe-only (Phase 1 of two-phase pipeline)
    path("stt/transcribe", bridge.stt_transcribe_only),

    # AI #7b — Notes from pre-existing Transcript (YouTube captions, no Whisper)
    path("stt/notes-from-text", bridge.generate_notes_from_transcript),

    # AI #7c — YouTube video ID → fetch captions server-side → notes (production-safe)
    path("stt/notes-from-youtube", bridge.generate_notes_from_youtube),

    # AI #8 — Student Feedback Engine
    path("feedback/generate", bridge.generate_feedback),

    # AI #9 — Notes Weak Topic Identifier
    path("notes/analyze", bridge.analyze_notes),

    # AI #10 — Resume Analyzer
    path("resume/analyze", bridge.analyze_resume),

    # AI #11 — Interview Prep
    path("interview/start", bridge.start_interview_prep),

    # AI #12 — Personalized Learning Plan
    path("plan/generate", bridge.generate_plan),
    path("syllabus/generate", bridge.generate_syllabus),

    # AI #13 — In-Video Quiz Generator
    path("quiz/generate", bridge.generate_quiz_questions),

    # AI #15 — Text Translation
    path("translate", bridge.translate_text),

    # AI #16 — Topic Content Generator
    path("content/generate", bridge.generate_topic_content),

    # AI Engine Health (teacher/admin panel)
    path("ai/health", bridge.ai_engine_health),

    # ══════════════════════════════════════════════════════════════════════════
    #  Legacy Django endpoints (original ai_services views)
    # ══════════════════════════════════════════════════════════════════════════

    # Feedback
    path("feedback/analyze/", feedback.analyze_feedback),
    path("feedback/health/", feedback.health),

    # Notes
    path("notes/upload/", notes.upload_and_generate_notes),
    path("notes/list/", notes.list_saved_notes),
    path("notes/health/", notes.health),

    # Content
    path("content/suggest/", content.suggest_resources),
    path("content/health/", content.health),

    # Test (single + batch)
    path("test/generate/", test.generate_practice_test),
    path("test/batch/", test.batch_generate_tests),
    path("test/batch/status/", test.batch_status),
    path("test/health/", test.health),

    # Career
    path("career/generate/", career.generate_career_plan),
    path("career/health/", career.health),

    # Personalization
    path("personalization/generate/", personalization.generate_study_plan),
    path("personalization/health/", personalization.health),

    # Tenant Admin API
    path("admin-api/usage/", admin_api.usage_dashboard),
    path("admin-api/cache/flush/", admin_api.flush_cache),
    path("admin-api/info/", admin_api.institute_info),

    # Root
    path("", feedback.root),
]
