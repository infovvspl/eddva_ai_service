from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3.2"
BATCH_SIZE      = 5        # questions per LLM call
OLLAMA_TIMEOUT  = 600.0    # 10 minutes per batch call


app = FastAPI(
    title="AI Practice Test Generator",
    description="Generate practice tests using a self-hosted LLM. Difficulty is set by the teacher.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Session Manager
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages per-request HTTP sessions.

    Each generate request gets its own httpx.AsyncClient (session).
    The session is registered on creation and automatically cleaned up
    (closed + removed) after the request finishes — success or failure.

    This ensures:
    - No lingering connections holding RAM after a request ends
    - No context/cookie bleed between separate requests
    - Clean resource usage — one session = one test generation
    """

    def __init__(self):
        self._sessions: Dict[str, httpx.AsyncClient] = {}

    def create(self, session_id: str) -> httpx.AsyncClient:
        """Create and register a new HTTP session."""
        client = httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        self._sessions[session_id] = client
        return client

    async def destroy(self, session_id: str) -> None:
        """Close and remove a session, freeing its resources."""
        client = self._sessions.pop(session_id, None)
        if client:
            await client.aclose()

    def active_count(self) -> int:
        return len(self._sessions)

    def active_ids(self) -> List[str]:
        return list(self._sessions.keys())


# Single global session manager
session_manager = SessionManager()


@asynccontextmanager
async def get_session(session_id: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Context manager that creates a session, yields it for use,
    then destroys it no matter what happens (even on error/timeout).

    Usage:
        async with get_session(session_id) as client:
            resp = await client.post(...)
    """
    client = session_manager.create(session_id)
    try:
        yield client
    finally:
        await session_manager.destroy(session_id)


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Schemas
# ──────────────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    mcq          = "mcq"
    true_false   = "true_false"
    short_answer = "short_answer"
    long_answer  = "long_answer"
    fill_blank   = "fill_blank"
    mix          = "mix"   # LLM spreads all types evenly


class DifficultyLevel(str, Enum):
    easy   = "easy"
    medium = "medium"
    hard   = "hard"


class GenerateRequest(BaseModel):
    topic: str = Field(..., example="Photosynthesis")
    num_questions: int = Field(5, ge=1, le=50)
    question_types: List[QuestionType] = Field(
        default=[QuestionType.mcq],
        example=["mcq", "true_false"],
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
    explanation: Optional[str] = None
    difficulty: DifficultyLevel
    topic_tags: List[str] = []


class PracticeTest(BaseModel):
    test_id: str
    session_id: str             # which session generated this test
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
) -> str:
    context_block = f"\n\nContext/Study Material:\n{context}\n" if context else ""

    types = [t.value for t in question_types]
    if "mix" in types:
        type_instruction = (
            "mix — spread evenly across: mcq, true_false, short_answer, long_answer, fill_blank"
        )
    else:
        type_instruction = ", ".join(types)

    batch_note = (
        f"\nNote: This is batch {batch_index + 1} of {total_batches}. "
        f"Generate {num_questions} DIFFERENT questions, do not repeat previous ones.\n"
        if total_batches > 1 else ""
    )

    return f"""You are an expert educator. Generate exactly {num_questions} practice test questions about "{topic}".
{context_block}{batch_note}
Requirements:
- Question types: {type_instruction}
- Difficulty: {difficulty.value}  <- ALL questions must match this difficulty exactly
- Return ONLY a valid JSON array. No extra text, no comments, no markdown.

Each element must follow this schema:
{{
  "type": "<mcq|true_false|short_answer|long_answer|fill_blank>",
  "question": "<question text>",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "<correct answer>",
  "explanation": "<brief explanation>",
  "difficulty": "{difficulty.value}",
  "topic_tags": ["tag1", "tag2"]
}}

STRICT RULES:
- "options" only for mcq and true_false. Omit for everything else.
- short_answer: 1-2 sentence answer
- long_answer: detailed paragraph answer
- fill_blank: question must contain _____
- No trailing commas. Valid JSON only.
- Start your response with [ and end with ]"""


# ──────────────────────────────────────────────────────────────────────────────
# Ollama caller — uses injected session client
# ──────────────────────────────────────────────────────────────────────────────

async def call_ollama(client: httpx.AsyncClient, prompt: str, model: str) -> str:
    """Send a prompt to Ollama using the provided session client."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 4096,
        },
    }
    try:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to Ollama. "
                "Make sure it is running: `ollama serve` "
                "and the model is pulled: `ollama pull llama3.2`"
            ),
        )
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Ollama timed out after {int(OLLAMA_TIMEOUT)}s. "
                "Try reducing num_questions or restart Ollama."
            ),
        )


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
                explanation=item.get("explanation"),
                difficulty=req.difficulty,
                topic_tags=item.get("topic_tags", []),
            )
        )
    return questions


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Check if Ollama is reachable and show active sessions."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status"         : "ok",
        "ollama"         : "online" if ollama_ok else "offline — run `ollama serve`",
        "active_sessions": session_manager.active_count(),
    }


@app.post("/generate", response_model=PracticeTest, tags=["Practice Tests"])
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
    test_id    = f"test_{int(time.time())}_{hash(req.topic) % 10000:04d}"

    # Split into batches
    batch_sizes: List[int] = []
    remaining = req.num_questions
    while remaining > 0:
        batch_sizes.append(min(BATCH_SIZE, remaining))
        remaining -= BATCH_SIZE

    total_batches   = len(batch_sizes)
    all_questions: List[Question] = []
    started_at      = time.time()

    # Session is created here and ALWAYS destroyed when the block exits
    async with get_session(session_id) as client:
        for batch_index, batch_size in enumerate(batch_sizes):
            prompt = build_llm_prompt(
                topic          = req.topic,
                num_questions  = batch_size,
                question_types = req.question_types,
                difficulty     = req.difficulty,
                context        = req.context,
                batch_index    = batch_index,
                total_batches  = total_batches,
            )

            raw_response    = await call_ollama(client, prompt, req.model)
            data            = parse_raw_response(raw_response)

            if not data:
                continue  # skip failed batch, don't crash the whole request

            batch_questions = build_questions(
                data      = data,
                req       = req,
                id_offset = len(all_questions),
            )
            all_questions.extend(batch_questions)

            # Pause between batches to let Ollama free memory
            if batch_index < total_batches - 1:
                await asyncio.sleep(1)

    # Session is destroyed here automatically (even if an error occurred above)

    if not all_questions:
        raise HTTPException(
            status_code=500,
            detail="No questions could be generated. Try again or reduce num_questions.",
        )

    return PracticeTest(
        test_id        = test_id,
        session_id     = session_id,
        topic          = req.topic,
        difficulty     = req.difficulty,
        generated_at   = started_at,
        questions      = all_questions,
        metadata       = {
            "model_used"       : req.model,
            "requested_types"  : [t.value for t in req.question_types],
            "num_questions"    : len(all_questions),
            "batches_used"     : total_batches,
            "batch_size"       : BATCH_SIZE,
            "context_provided" : req.context is not None,
            "duration_seconds" : round(time.time() - started_at, 2),
            "session_ended"    : True,   # confirms session was cleaned up
        },
    )


@app.get("/models/list", tags=["System"])
async def list_ollama_models():
    """List all locally available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {"available_models": models}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Ollama not reachable. Run `ollama serve` first.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("ai_test:app", host="0.0.0.0", port=8000, reload=True)