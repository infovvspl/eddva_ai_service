"""
Audited catalog of every AI call site in the Django AI service.

This is the routing matrix as data. ``ai_benchmark --plan`` resolves each routed
entry through the live router, and tests assert two properties over it:

  * with Together fully configured, every routed entry is served by a Together
    model chosen by capability;
  * with Together NOT configured, every routed entry keeps its current provider
    and model exactly (source=legacy).

Entries with status "specialized" or "blocked" are NOT routed. They either are
not chat completions (speech, OCR, image generation) or depend on a capability
that has not been verified on Together (Odia-script quality, image and PDF
input). They keep their current providers and are listed so the audit is
complete rather than implied.

Keep this file in step with the code: a new LLMClient call site needs an entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ROUTED, SPECIALIZED, BLOCKED = "routed", "specialized", "blocked"

_120B = "openai/gpt-oss-120b"
_20B = "openai/gpt-oss-20b"
_GEMINI = "gemini-2.5-flash"


@dataclass(frozen=True)
class CallSite:
    id: str
    feature: str
    location: str
    current_provider: str
    current_model: str
    status: str = ROUTED
    capability: Optional[str] = None      # hint the call site passes (None = router infers from model)
    route_feature: Optional[str] = None   # feature kwarg the call site passes
    json_mode: bool = False
    grounding: bool = False
    vision: bool = False
    risk: str = "low"
    notes: str = ""


def _r(id, feature, location, model, *, provider="groq", **kw) -> CallSite:
    return CallSite(id, feature, location, provider, model, ROUTED, **kw)


CALL_SITES: tuple = (
    # ── student doubts ───────────────────────────────────────────────────────
    _r("doubt.subject_detect", "Student doubt: subject/type classifier", "bridge._detect_subject_and_type_for_doubt",
       _20B, risk="medium", notes="lightweight -> DeepSeek; text latency higher than Groq 20B in probes"),
    _r("doubt.solve_text", "Student doubt: LLM solve (text)", "bridge.resolve_doubt",
       _120B, risk="low", notes="coaching math pins qwen/qwen3-32b; also reasoning"),
    _r("doubt.solve_json", "Student doubt: LLM solve (JSON)", "bridge.resolve_doubt", _120B, json_mode=True),
    _r("doubt.scientific_code", "Scientific solver: code generation", "solver.scientific_solver.solve", _120B),
    _r("doubt.scientific_explain", "Scientific solver: explanation", "solver.scientific_solver.solve", _120B),
    _r("doubt.scientific_parallel", "Scientific solver: parallel JSON", "solver.scientific_solver.solve",
       _120B, json_mode=True),
    # ── tutor ────────────────────────────────────────────────────────────────
    _r("tutor.session", "AI tutor: start session", "bridge.start_tutor_session", _120B),
    _r("tutor.continue", "AI tutor: continue session", "bridge.continue_tutor_session", _120B, json_mode=True),
    # ── quiz / assessment / grading ──────────────────────────────────────────
    _r("quiz.in_video", "In-video quiz generation (parallel chunks)", "bridge.generate_quiz_questions", "quiz"),
    _r("assessment.practice_test", "Practice test / assessment questions", "test.generate_practice_test",
       _120B, json_mode=True),
    _r("assessment.coverage", "Assessment topic-coverage decomposition", "bridge._decompose_topic_coverage",
       _120B, json_mode=True),
    _r("grading.rubrics", "Subjective rubric generation", "bridge.generate_subjective_rubrics",
       _120B, json_mode=True),
    _r("grading.answer", "Subjective answer grading", "bridge.grade_subjective_answer", _120B, json_mode=True),
    _r("teacher.recording_analysis", "Teacher recording analysis", "bridge.analyze_teacher_recording (ai_call)",
       _120B, route_feature="teacher_recording_analysis", json_mode=True),
    # ── content ──────────────────────────────────────────────────────────────
    _r("content.topic", "Textbook/chapter content, DPP, PYQ, assessment paper (ungrounded)",
       "bridge.generate_topic_content", _120B, capability="content", route_feature="content_generate",
       risk="high", notes="Qwen3.8 Flash reasons before answering; output budget and latency must be checked"),
    _r("content.topic_retry", "Content MCQ-repair regeneration", "bridge.generate_topic_content",
       _120B, capability="content", route_feature="content_generate", risk="high"),
    _r("content.grounded", "Textbook-grounded content", "bridge.generate_topic_content (grounded)",
       _GEMINI, provider="gemini", capability="grounded", route_feature="content_generate", grounding=True,
       risk="high", notes="migrated from Gemini; retrieval/citations/source rules unchanged; "
                          "Together only when grounding + context are verified"),
    # ── PPT ──────────────────────────────────────────────────────────────────
    _r("ppt.coverage", "PPT coverage decomposition", "ppt._decompose_ppt_coverage", _120B, json_mode=True),
    _r("ppt.deck", "PPT deck (ungrounded)", "ppt.generate_presentation", _120B, capability="content",
       route_feature="ppt_generate", json_mode=True, risk="high"),
    _r("ppt.regenerate_slide", "PPT single-slide regeneration", "ppt.regenerate_slide", _120B,
       capability="content", route_feature="ppt_generate", json_mode=True, risk="high"),
    _r("ppt.grounded_deck", "Textbook-grounded PPT deck", "ppt._generate_grounded", _GEMINI, provider="gemini",
       capability="grounded", route_feature="ppt_generate", json_mode=True, grounding=True, risk="high",
       notes="migrated from Gemini; badge reason codes preserved (non-Gemini failures -> 'unavailable')"),
    # ── lecture notes ────────────────────────────────────────────────────────
    _r("notes.chunk", "Lecture notes: per-chunk writing", "bridge._generate_chunk_notes", _20B,
       capability="content", route_feature="ai_lecture_notes", risk="high",
       notes="was Groq 20B; content policy moves it to Qwen3.8 Flash"),
    _r("notes.merge", "Lecture notes: merge chunks", "bridge._merge_chunk_notes", _120B,
       capability="content", route_feature="ai_lecture_notes", risk="high"),
    _r("notes.polish", "Lecture notes: polish markdown", "bridge._polish_notes_markdown", _20B,
       capability="content", route_feature="ai_lecture_notes", risk="high"),
    _r("notes.image_terms", "Notes image search terms", "bridge.extract_image_search_terms", _20B,
       json_mode=True, risk="medium"),
    _r("notes.image_planner", "Notes image planner", "note_images.plan_note_images", _20B,
       json_mode=True, risk="medium"),
    _r("stt.hinglish_repair", "Transcript Hindi/Hinglish repair", "bridge._repair_hindi_hinglish_wording_post_stt",
       _20B, risk="medium"),
    _r("stt.punctuation", "Transcript punctuation refinement", "bridge._refine_transcript_punctuation_post_stt",
       _20B, risk="medium"),
    _r("stt.low_quality_repair", "Low-quality transcript repair", "bridge._repair_single_chunk", _20B,
       risk="medium"),
    # ── ai_call / ai_call_text tier features ─────────────────────────────────
    _r("feature.content_recommend", "Content recommendations", "bridge.recommend_content (ai_call_text)", _20B,
       route_feature="content_recommend", risk="medium"),
    _r("feature.feedback_generate", "Student feedback", "bridge.generate_feedback (ai_call_text)", _120B,
       route_feature="feedback_generate"),
    _r("feature.notes_analyze", "Notes analysis", "bridge.analyze_notes (ai_call_text)", _120B,
       route_feature="notes_analyze"),
    _r("feature.resume_analyze", "Resume analysis", "bridge.analyze_resume (ai_call_text)", _120B,
       route_feature="resume_analyze"),
    _r("feature.interview_prep", "Interview prep", "bridge.start_interview_prep (ai_call_text)", _120B,
       route_feature="interview_prep"),
    _r("feature.plan_generate", "Study plan", "bridge.generate_plan (ai_call)", _120B,
       route_feature="plan_generate", json_mode=True),
    _r("feature.syllabus_generate", "Syllabus", "bridge.generate_syllabus (ai_call)", _120B,
       route_feature="syllabus_generate", json_mode=True),
    _r("feature.memorization", "Memorization aids", "bridge.generate_memorization_items (ai_call)", _120B,
       route_feature="ai_memorization_retention", json_mode=True),
    _r("feature.career_guidance", "Career guidance", "career.career_guidance", _120B),
    _r("feature.career_roadmap", "Career roadmap", "career.generate_career_plan (ai_call_text)", _120B,
       route_feature="career_roadmap"),
    _r("feature.content_suggest", "Resource suggestions", "content.suggest_resources (ai_call_text)", _20B,
       route_feature="content_suggest", risk="medium"),
    _r("feature.evaluate_batch", "QA evaluation batch", "evaluate.evaluate_batch", _120B, json_mode=True),
    _r("feature.feedback_analyze", "Feedback analysis", "feedback.analyze_feedback (ai_call_text)", _120B,
       route_feature="feedback_analyze"),
    _r("feature.notes_generate", "Notes upload generation", "notes.upload_and_generate_notes (ai_call)", _120B,
       route_feature="notes_generate", json_mode=True),
    _r("feature.study_plan", "Personalized study plan", "personalization.generate_study_plan (ai_call_text)",
       _120B, route_feature="study_plan"),
    _r("feature.batch_jobs", "Batch pre-generation jobs", "batch_processor._process_item", _120B,
       json_mode=True, notes="model from the job's feature tier"),

    # ── not routed: Odia (quality on Together unverified) ────────────────────
    CallSite("odia.notes", "Lecture notes in Odia", "bridge._gemini_odia_generate", "gemini", _GEMINI, BLOCKED,
             risk="-", notes="Odia-script output quality on Together is unverified; kept on Gemini"),
    CallSite("odia.doubt", "Student doubt in Odia", "bridge.resolve_doubt (_gemini_complete)", "gemini", _GEMINI,
             BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    CallSite("odia.quiz", "In-video quiz in Odia", "bridge._gemini_parallel_complete_many", "gemini", _GEMINI,
             BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    CallSite("odia.topic_content", "Content in Odia", "bridge.generate_topic_content (gemini_generate)",
             "gemini", _GEMINI, BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    CallSite("odia.image_terms", "Notes image terms in Odia", "bridge.extract_image_search_terms (_gemini_complete)",
             "gemini", _GEMINI, BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    CallSite("odia.practice_test", "Practice test in Odia", "test._gemini_test_generate", "gemini", _GEMINI,
             BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    CallSite("odia.image_planner", "Notes image planner (Odia)", "note_images._gemini_image_plan", "gemini",
             _GEMINI, BLOCKED, risk="-", notes="Odia-script quality unverified on Together"),
    # ── not routed: vision / OCR ─────────────────────────────────────────────
    CallSite("vision.image_doubt", "Image doubts / answer-sheet transcription",
             "bridge._vision_text_from_image", "groq",
             "meta-llama/llama-4-scout-17b-16e-instruct -> gemini-2.5-flash", BLOCKED, vision=True, risk="-",
             notes="image input; Together multimodal not verified against this contract"),
    CallSite("vision.ocr_fallback", "OCR fallback", "bridge._extract_text_from_image_url", "easyocr",
             "easyocr-local", SPECIALIZED, vision=True, risk="-", notes="local OCR, not an LLM"),
    CallSite("vision.note_image_label", "Generated note-image labelling", "note_images.label_generated_note_image",
             "groq", "meta-llama/llama-4-maverick-17b-128e-instruct -> gemini-2.5-flash", BLOCKED, vision=True,
             risk="-", notes="image input; not verified on Together"),
    CallSite("textbook.pdf_ocr", "Textbook PDF transcription (ingest)", "textbook._ocr_batch", "gemini", _GEMINI,
             BLOCKED, vision=True, risk="-",
             notes="sends raw PDF bytes (application/pdf part); Together chat API has no PDF input"),
    # ── not routed: speech / translation / images ────────────────────────────
    CallSite("speech.transcribe", "Lecture transcription", "bridge._transcribe_with_groq_one_key / faster-whisper",
             "groq", "whisper-large-v3-turbo", SPECIALIZED, risk="-", notes="speech-to-text"),
    CallSite("speech.odia_stt", "Odia transcription", "sarvam_client.transcribe_file", "sarvam", "saaras:v3",
             SPECIALIZED, risk="-", notes="speech-to-text"),
    CallSite("translate.text", "Translation", "bridge.translate_text (sarvam_client.translate)", "sarvam",
             "mayura:v1", SPECIALIZED, risk="-", notes="dedicated translation model"),
    CallSite("image.generate", "Slide / note image generation", "core.image_generation", "huggingface",
             "FLUX", SPECIALIZED, risk="-", notes="image generation"),
    CallSite("image.search", "Educational image search", "core.serpapi_images", "serper", "-", SPECIALIZED,
             risk="-", notes="search API, not an LLM"),
)


def routed_sites() -> list:
    return [c for c in CALL_SITES if c.status == ROUTED]
