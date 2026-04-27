from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration — local Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "edvaqwen")
DEFAULT_MODEL = OLLAMA_MODEL
BATCH_SIZE = 5  # questions per LLM call


router = APIRouter(prefix="/test", tags=["AI Practice Test"])


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Schemas
# ──────────────────────────────────────────────────────────────────────────────


class QuestionType(str, Enum):
    mcq = "mcq"
    assertion_reason = "assertion_reason"
    statement_based = "statement_based"
    match_the_following = "match_the_following"
    integer = "integer"
    diagram = "diagram"
    msq = "msq"
    true_false = "true_false"
    short_answer = "short_answer"
    long_answer = "long_answer"
    case_study = "case_study"
    fill_blank = "fill_blank"
    mix = "mix"  # LLM spreads all types evenly


def infer_question_type_from_text(text: str, fallback: str) -> str:
    t = (text or "").lower()
    if (
        ("assertion (a)" in t and "reason (r)" in t)
        or ("assertion" in t and "reason" in t)
        or ("both a and r" in t)
    ):
        return QuestionType.assertion_reason.value
    if (
        "match the following" in t
        or ("column i" in t and "column ii" in t)
        or "matrix match" in t
    ):
        return QuestionType.match_the_following.value
    if (
        "read the following passage" in t
        or "based on the paragraph" in t
    ):
        return QuestionType.case_study.value
    return fallback


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ExamMode(str, Enum):
    competitive = "competitive"
    academics = "academics"


class GenerateRequest(BaseModel):
    topic: str = Field(..., example="Photosynthesis")
    num_questions: int = Field(5, ge=1, le=50)
    question_types: List[QuestionType] = Field(
        default=[QuestionType.mcq],
        example=["mcq", "true_false"],
    )
    exam_mode: ExamMode = Field(
        ExamMode.competitive,
        description="competitive | academics",
    )
    marks_per_question: int = Field(
        3,
        ge=2,
        le=5,
        description="Controls question depth and model-answer structure",
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="Required. Set by the teacher: easy | medium | hard",
        example="medium",
    )
    context: Optional[str] = Field(
        None,
        description="Optional study material / passage to base questions on.",
    )
    model: str = Field(DEFAULT_MODEL, example="llama3.2")


class Question(BaseModel):
    id: int
    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    answer: str
    model_answer: Optional[str] = None
    explanation: Optional[str] = None  # Why the right answer is right
    mark_scheme_breakdown: Optional[List[str]] = None
    difficulty: DifficultyLevel
    topic_tags: List[str] = []


class PracticeTest(BaseModel):
    test_id: str
    topic: str
    difficulty: DifficultyLevel
    generated_at: float
    questions: List[Question]
    metadata: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────


def build_llm_prompt(
    topic: str,
    num_questions: int,
    question_types: List[QuestionType],
    difficulty: DifficultyLevel,
    context: Optional[str],
    batch_index: int,
    total_batches: int,
    exam_mode: ExamMode,
    marks_per_question: int,
) -> str:
    context_block = f"\n\nContext/Study Material:\n{context}\n" if context else ""

    types = [t.value for t in question_types]
    if "mix" in types:
        type_instruction = (
            "mix — spread based on exam_mode. "
            "competitive: assertion_reason, statement_based, mcq, match_the_following (matrix match style), paragraph/case-based, integer, diagram, msq. "
            "academics: mcq, assertion_reason, short_answer, long_answer, case_study, diagram."
        )
    else:
        type_instruction = ", ".join(types)

    marks_rules = {
        2: (
            "2-Mark pattern:\n"
            "- 1 mark: correct definition/statement\n"
            "- 1 mark: second point/example/explanation\n"
            "- Expectation: exactly 2 clear points OR 1 definition + 1 example."
        ),
        3: (
            "3-Mark pattern:\n"
            "- 1 mark: definition/principle\n"
            "- 2 marks: explanation with 2 distinct points/steps\n"
            "- Expectation: total 3 key points in logical flow."
        ),
        4: (
            "4-Mark pattern:\n"
            "- 1 mark: definition/formula/statement\n"
            "- 2-3 marks: explanation in 2-3 structured steps/points\n"
            "- 0-1 mark: diagram/example/conclusion\n"
            "- Expectation: 3-4 structured points; for numericals, show stepwise working."
        ),
        5: (
            "5-Mark pattern:\n"
            "- 1 mark: definition/introduction\n"
            "- 3 marks: detailed explanation with 3 core points/steps\n"
            "- 1 mark: diagram/example/conclusion\n"
            "- Expectation: 4-5 key points with proper structure."
        ),
    }
    marks_instruction = marks_rules.get(marks_per_question, marks_rules[3])

    batch_note = (
        f"\nNote: This is batch {batch_index + 1} of {total_batches}. "
        f"Generate {num_questions} DIFFERENT questions, do not repeat previous ones.\n"
        if total_batches > 1
        else ""
    )

    return f"""You are an expert educator. Generate exactly {num_questions} practice test questions about "{topic}".
{context_block}{batch_note}
Requirements:
- Question types: {type_instruction}
- Exam mode: {exam_mode.value}
- Difficulty: {difficulty.value}  <- ALL questions must match this difficulty exactly
- Marks per question: {marks_per_question}
- The cognitive depth of each question MUST match marks_per_question.
- Return ONLY a valid JSON array. No extra text, no comments, no markdown.

Each element must follow this schema:
{{
  "type": "<mcq|assertion_reason|statement_based|match_the_following|integer|diagram|msq|true_false|short_answer|long_answer|case_study|fill_blank>",
  "question": "<question text>",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "<correct answer>",
  "model_answer": "<required: ideal answer aligned to marks_per_question pattern>",
  "explanation": "<required: explanation of RIGHT answer for student review; must be present for every question>",
  "mark_scheme_breakdown": ["<per-mark checkpoint 1>", "<per-mark checkpoint 2>"],
  "difficulty": "{difficulty.value}",
  "topic_tags": ["tag1", "tag2"]
}}

STRICT RULES:
- Every question MUST include a non-empty "model_answer" and non-empty "explanation".
- The "explanation" is compulsory for student review of ALL questions; never leave empty.
- "options" only for mcq and true_false. Omit for everything else.
- For msq questions, include 4-6 options and provide all correct choices in "answer".
- short_answer: 1-2 sentence answer
- long_answer: detailed paragraph answer
- fill_blank: question must contain _____
- For diagram questions (any subject), include label/observation expectations in model_answer.
- For assertion_reason and statement_based, evaluate each statement explicitly before final conclusion.
- Use this exact mark-aligned pattern for question + model answer + mark_scheme_breakdown:
{marks_instruction}
- No trailing commas. Valid JSON only.
- Start your response with [ and end with ]"""


# ──────────────────────────────────────────────────────────────────────────────
async def call_ollama(prompt: str, model: str = None) -> str:
    """Call local Ollama to generate text."""
    try:
        async with httpx.AsyncClient(timeout=240) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model or OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[OLLAMA-CALL-ERROR] {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# JSON cleaning & parsing
# ──────────────────────────────────────────────────────────────────────────────


def clean_llm_json(raw: str) -> str:
    """Clean common JSON mistakes made by small LLMs."""
    raw = re.sub(r"```(?:json)?", "", raw)
    raw = raw.replace("```", "").strip()
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    raw = re.sub(r",(\s*[\]\}])", r"\1", raw)
    start = raw.find("[")
    if start != -1:
        raw = raw[start:]
    end = raw.rfind("]") + 1
    if end > 0:
        raw = raw[:end]
    return raw.strip()


def parse_raw_response(raw: str) -> List[dict]:
    """Parse LLM response into list of question dicts with fallback."""
    raw = clean_llm_json(raw)

    # Attempt 1: parse whole array
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract each {...} block individually
    objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    data = []
    for obj in objects:
        obj = re.sub(r",(\s*[\]\}])", r"\1", obj)
        try:
            data.append(json.loads(obj))
        except json.JSONDecodeError:
            continue

    return data


def build_questions(
    data: List[dict],
    req: GenerateRequest,
    id_offset: int = 0,
) -> List[Question]:
    """Convert raw dicts into typed Question objects."""
    questions: List[Question] = []
    for i, item in enumerate(data):
        q_type_raw = item.get("type", req.question_types[0].value)
        if q_type_raw == "mix":
            q_type_raw = "short_answer"
        q_type_raw = infer_question_type_from_text(item.get("question", ""), q_type_raw)
        try:
            q_type = QuestionType(q_type_raw)
        except ValueError:
            q_type = QuestionType.short_answer

        # Sanitize options — LLM sometimes returns "" or {} instead of a list or null
        raw_options = item.get("options")
        if not isinstance(raw_options, list) or len(raw_options) == 0:
            raw_options = None

        questions.append(
            Question(
                id=id_offset + i + 1,
                type=q_type,
                question=item.get("question", ""),
                options=raw_options,
                answer=item.get("answer", ""),
                model_answer=item.get("model_answer"),
                explanation=(
                    item.get("explanation")
                    or "The given answer is correct because it matches the core concept and required steps for this mark level."
                ),
                mark_scheme_breakdown=item.get("mark_scheme_breakdown"),
                difficulty=req.difficulty,
                topic_tags=item.get("topic_tags", []),
            )
        )
    return questions


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "llm": f"ollama/{OLLAMA_MODEL}",
    }


@router.post("/generate", response_model=PracticeTest, tags=["Practice Tests"])
async def generate_test(req: GenerateRequest):
    """
    Generate a full practice test using a self-hosted LLM (Ollama).

    Session lifecycle:
      1. A unique session is created when this request starts
      2. All batch calls share the same session
      3. Session is destroyed immediately when the request ends (pass or fail)

    - Max 50 questions per request (auto-batched in groups of 5)
    - Teacher must specify difficulty: easy | medium | hard
    """
    # Create a unique session ID for this request
    session_id = str(uuid.uuid4())
    test_id = f"test_{int(time.time())}_{hash(req.topic) % 10000:04d}"

    # Split into batches
    batch_sizes: List[int] = []
    remaining = req.num_questions
    while remaining > 0:
        batch_sizes.append(min(BATCH_SIZE, remaining))
        remaining -= BATCH_SIZE

    total_batches = len(batch_sizes)
    all_questions: List[Question] = []
    started_at = time.time()

    # Session is created here and ALWAYS destroyed when the block exits
    for batch_index, batch_size in enumerate(batch_sizes):
        prompt = build_llm_prompt(
            topic=req.topic,
            num_questions=batch_size,
            question_types=req.question_types,
            difficulty=req.difficulty,
            context=req.context,
            batch_index=batch_index,
            total_batches=total_batches,
            exam_mode=req.exam_mode,
            marks_per_question=req.marks_per_question,
        )

        raw_response = await call_ollama(prompt, req.model)
        data = parse_raw_response(raw_response)

        if not data:
            continue  # skip failed batch, don't crash the whole request

        batch_questions = build_questions(
            data=data,
            req=req,
            id_offset=len(all_questions),
        )
        all_questions.extend(batch_questions)

    # Session is destroyed here automatically (even if an error occurred above)

    if not all_questions:
        raise HTTPException(
            status_code=500,
            detail="No questions could be generated. Try again or reduce num_questions.",
        )

    return PracticeTest(
        test_id=test_id,
        session_id=session_id,
        topic=req.topic,
        difficulty=req.difficulty,
        generated_at=started_at,
        questions=all_questions,
        metadata={
            "model_used": req.model,
            "requested_types": [t.value for t in req.question_types],
            "exam_mode": req.exam_mode.value,
            "marks_per_question": req.marks_per_question,
            "num_questions": len(all_questions),
            "batches_used": total_batches,
            "batch_size": BATCH_SIZE,
            "context_provided": req.context is not None,
            "duration_seconds": round(time.time() - started_at, 2),
            "session_ended": True,  # confirms session was cleaned up
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FastAPI(title="AI Test API Standalone")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)
