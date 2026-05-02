from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import os
import httpx
import json
from io import BytesIO
from ai_services.core.groq_keys import get_rotated_groq_keys
from PIL import Image
import easyocr
import numpy as np
from app.scientific_solver import scientific_solver
import logging

logger = logging.getLogger("ai_services.doubt_resolve")

router = APIRouter(prefix="/doubt", tags=["Doubt Resolution"], redirect_slashes=False)


class StudentContext(BaseModel):
    source: Optional[str] = None
    sourceRefId: Optional[str] = None
    doubtId: Optional[str] = None
    batchId: Optional[str] = None


class ResolveDoubtRequest(BaseModel):
    questionText: str
    questionImageUrl: Optional[str] = None
    topicId: Optional[str] = None
    mode: str = "detailed"  # 'short' | 'detailed'
    studentContext: Optional[Any] = None


class ResolveDoubtResponse(BaseModel):
    explanation: Optional[str] = None
    brief: Optional[Dict[str, Any]] = None
    detailed: Optional[Dict[str, Any]] = None
    conceptLinks: list[str] = []

class OcrImageRequest(BaseModel):
    imageUrl: str


class OcrImageResponse(BaseModel):
    text: str


def _build_theory_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    
    system = (
        "SYSTEM ROLE: You are an expert CBSE/NEET answer generator v3 specialized in THEORY questions.\n"
        "TASK: Generate exam-ready answers strictly following CBSE/NEET marking schemes.\n\n"
        "STEP 1: STRUCTURE (MANDATORY)\n"
        "- Follow exact subparts from the question: (i), (ii), (iii), etc.\n"
        "- Use ONLY bullet points (•) or numbered points.\n"
        "- DO NOT use sub-headers like 'Definition', 'Mechanism', 'Example', 'Explanation'.\n"
        "- NO extra sections (No intro, no conclusion, no exam tips, no misconceptions).\n\n"
        "STEP 2: KEYWORD ENFORCEMENT (CRITICAL)\n"
        "- Include ALL important NCERT keywords and **bold** them.\n"
        "- Use textbook terminology. Prioritize scoring terms.\n\n"
        "STEP 3: MODE HANDLING\n"
        "- If MODE = 'brief': Give 3-4 high-quality scoring points per subpart.\n"
        "- If MODE = 'detailed': Give 6-8 comprehensive points per subpart.\n\n"
        "STEP 4: STYLE\n"
        "- Clean, exam-ready format. Each point = 1 mark style. No emojis. No long paragraphs.\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "Theory", '
        '"brief": {"question_nature": "theoretical", "steps": [{"text": "bullet"}], "final_answer": "(i) ... (ii) ..."}, '
        '"detailed": {"explanation": "Labeled bullet points with **keywords**", "final_answer": "(i) ... (ii) ...", "verification": "...", "key_concept": "..."}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user

def _build_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert scientific tutor. Provide structured responses in JSON.\n"
        "1. ADDRESS ALL SUB-PARTS: Use labels (a), (b) or (i), (ii) explicitly.\n"
        "2. MANDATORY LABELING: The 'final_answer' MUST be a labeled list.\n"
        "3. NO TRAILING BACKSLASHES.\n"
        "JSON STRUCTURE:\n"
        '{"subject": "...", "type": "...", '
        '"brief": {"question_nature": "numerical", "steps": [{"text": "..."}], "final_answer": "..."}, '
        '"detailed": {"explanation": "...", "final_answer": "...", "verification": "...", "key_concept": "..."}, '
        '"key_concepts": ["..."]}'
    )
    user = f"Question: {question_text}\nMODE: {req.mode}"
    return system, user
    length = "concise (2-3 sentences)" if req.mode == "short" else "detailed and step-by-step"
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    if not question_text:
        question_text = "No typed question provided."
    image_hint = (
        f"\nImage URL (for reference): {req.questionImageUrl}\n"
        if req.questionImageUrl
        else ""
    )
    return (
        f"You are an expert scientific tutor. A student has asked the following question.{topic_hint}\n\n"
        f"Question text: {question_text}\n"
        f"{image_hint}\n"
        f"If this came from a screenshot or book photo, rely on extracted text and solve the exact asked problem.\n"
        f"CONTENT RULES:\n"
        f"1. ADDRESS ALL SUB-PARTS: Use labels like (a), (b), (i), (ii) explicitly in explanation AND final_answer.\n"
        f"2. MANDATORY LABELING: The 'final_answer' MUST be a labeled list, e.g., '(a)(i) ... (a)(ii) ... (b) ...'. NEVER omit labels.\n"
        f"3. BRIEF VIEW (Quick Steps): Math/results ONLY. No prose.\n"
        f"4. DETAILED VIEW: Full prose explanations + math.\n"
        f"5. NO TRAILING BACKSLASHES: Do not end lines with '\\'.\n\n"
        f"JSON STRUCTURE:\n"
        f'{{"subject": "...", "type": "...", '
        f'"brief": {{"question_nature": "numerical", "steps": [{{"text": "Math only step"}}], "final_answer": "(a) ... (b) ..."}}, '
        f'"detailed": {{"explanation": "Full prose", "final_answer": "(a) ... (b) ...", "verification": "...", "key_concept": "..."}}, '
        f'"key_concepts": ["concept1", "concept2"]}}'
    )


def _build_true_false_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert CBSE/NEET answer generator specialized in TRUE/FALSE questions.\n"
        "TASK: Determine whether the statement is True or False and justify it.\n\n"
        "RULES:\n"
        "- answer: exactly 'True' or 'False'\n"
        "- justification: 2-3 lines max using bold **NCERT keywords**. No extra sections.\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "true_false", '
        '"brief": {"answer": "True/False", "justification": "2-3 line justification with **keywords**"}, '
        '"detailed": {"answer": "True/False", "justification": "2-3 line justification with **keywords**"}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user


def _build_fitb_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert CBSE/NEET answer generator specialized in FILL IN THE BLANKS questions.\n"
        "TASK: Identify the correct word/phrase for the blank and justify it.\n\n"
        "RULES:\n"
        "- answer: the exact word or phrase that fills the blank\n"
        "- justification: 2-3 lines max using bold **NCERT keywords**. No extra sections.\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "fill_in_blank", '
        '"brief": {"answer": "<exact answer>", "justification": "2-3 line justification with **keywords**"}, '
        '"detailed": {"answer": "<exact answer>", "justification": "2-3 line justification with **keywords**"}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user


def _build_assertion_reason_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert CBSE/NEET answer generator specialized in ASSERTION-REASON questions.\n"
        "Standard options: (A) Both A and R true, R explains A. (B) Both A and R true, R does NOT explain A. "
        "(C) A true, R false. (D) A false, R true.\n\n"
        "TASK: Evaluate assertion and reason, pick the correct option, and explain concisely.\n\n"
        "RULES:\n"
        "- answer: correct option letter only (A/B/C/D)\n"
        "- assertion_status: 'Correct' or 'Incorrect'\n"
        "- reason_status: 'Correct' or 'Incorrect'\n"
        "- explanation: 1-2 lines stating whether Reason correctly explains Assertion. Use **NCERT keywords**.\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "assertion_reason", '
        '"brief": {"answer": "Option X", "assertion_status": "Correct/Incorrect", "reason_status": "Correct/Incorrect", "explanation": "1-2 line explanation"}, '
        '"detailed": {"answer": "Option X", "assertion_status": "Correct/Incorrect", "reason_status": "Correct/Incorrect", "explanation": "1-2 line explanation with **keywords**"}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user


def _build_sequence_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert CBSE/NEET answer generator specialized in SEQUENCE-BASED MCQ questions.\n"
        "TASK: Identify the correct order/sequence and select the matching option.\n\n"
        "RULES:\n"
        "- answer: correct option letter only (A/B/C/D)\n"
        "- sequence: steps joined by → arrows (e.g. Step 1 → Step 2 → Step 3 → Step 4)\n"
        "- key_logic: 1-2 lines explaining the ordering principle using **NCERT keywords**\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "sequence_mcq", '
        '"brief": {"answer": "Option X", "sequence": "Step 1 → Step 2 → Step 3 → Step 4", "key_logic": "1-2 line logic"}, '
        '"detailed": {"answer": "Option X", "sequence": "Step 1 → Step 2 → Step 3 → Step 4", "key_logic": "1-2 line logic with **keywords**"}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user


def _build_multi_correct_prompt(req: ResolveDoubtRequest) -> tuple[str, str]:
    topic_hint = f" The topic is: {req.topicId}." if req.topicId else ""
    question_text = (req.questionText or "").strip()
    system = (
        "You are an expert CBSE/NEET answer generator specialized in MULTI-CORRECT MCQ questions.\n"
        "TASK: Identify ALL correct options and justify each one with a keyword-based reason.\n\n"
        "RULES:\n"
        "- answers: array of all correct option letters (e.g. ['A', 'C', 'D'])\n"
        "- reasons: array of objects, one per correct option, each with 'option' and 'reason' (1 line, **bold keywords**)\n"
        "- incorrect_note: optional 1 line explaining why wrong options fail\n\n"
        "JSON STRUCTURE:\n"
        '{"subject": "Biology|Chemistry|Physics|Maths", "type": "multi_correct", '
        '"brief": {"answers": ["A", "C"], "reasons": [{"option": "A", "reason": "correct because **keyword**"}, {"option": "C", "reason": "correct because **keyword**"}], "incorrect_note": ""}, '
        '"detailed": {"answers": ["A", "C"], "reasons": [{"option": "A", "reason": "correct because **keyword**"}, {"option": "C", "reason": "correct because **keyword**"}], "incorrect_note": "B and D are incorrect because ..."}, '
        '"key_concepts": ["concept1"]}'
    )
    user = f"Question: {question_text}\n{topic_hint}\nMODE: {req.mode}"
    return system, user


_ocr_reader: Optional[easyocr.Reader] = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en", "hi"], gpu=False)
    return _ocr_reader


def _prepare_ocr_variants(img_rgb: np.ndarray) -> list[np.ndarray]:
    # Multiple variants improve OCR across screenshots and camera-captured book pages.
    gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    bw = np.where(gray > 165, 255, 0).astype(np.uint8)
    p5, p95 = np.percentile(gray, (5, 95))
    if p95 > p5:
        stretch = np.clip((gray - p5) * (255.0 / (p95 - p5)), 0, 255).astype(np.uint8)
    else:
        stretch = gray
    return [img_rgb, gray, stretch, bw]


async def _extract_text_from_image_url(image_url: str) -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(image_url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    arr = np.array(img)
    reader = _get_ocr_reader()
    best = ""
    for v in _prepare_ocr_variants(arr):
        try:
            parts = reader.readtext(v, detail=0, paragraph=True)
            text = " ".join([str(x).strip() for x in parts if str(x).strip()]).strip()
            if len(text) > len(best):
                best = text
        except Exception:
            continue
    return best


@router.post("/ocr-image", response_model=OcrImageResponse)
async def ocr_image(req: OcrImageRequest):
    try:
        text = await _extract_text_from_image_url(req.imageUrl)
        return {"text": text or ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


async def _call_groq(system_prompt: str, user_prompt: str, api_key: str) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq error: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _call_gemini(system_prompt: str, user_prompt: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json={
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Gemini error: {resp.text}")
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


@router.post("/resolve", response_model=ResolveDoubtResponse)
async def resolve_doubt(req: ResolveDoubtRequest):
    groq_keys = get_rotated_groq_keys()
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_keys and not gemini_key:
        raise HTTPException(status_code=500, detail="No AI API key configured")

    ocr_text = ""
    if req.questionImageUrl:
        try:
            ocr_text = await _extract_text_from_image_url(req.questionImageUrl)
        except Exception:
            ocr_text = ""

    req_for_prompt = req
    if ocr_text:
        merged_text = (req.questionText or "").strip()
        merged_text = (merged_text + "\n\nExtracted from image:\n" + ocr_text).strip()
        req_for_prompt = ResolveDoubtRequest(
            questionText=merged_text,
            questionImageUrl=req.questionImageUrl,
            topicId=req.topicId,
            mode=req.mode,
            studentContext=req.studentContext,
        )

    # Detection logic
    q_lower = (req_for_prompt.questionText or "").lower()
    q_raw = req_for_prompt.questionText or ""

    assertion_keywords = ["assertion", "statement a", "statement r", "assertion and reason", "(a) and (r)"]
    sequence_keywords = ["correct sequence", "arrange in order", "correct order", "sequence of", "in the correct order", "in correct sequence"]
    multi_correct_keywords = ["more than one correct", "select all that apply", "one or more correct", "multiple correct", "more than one option"]
    true_false_keywords = ["true or false", "true/false", "state true or false", "is it true", "correct or incorrect"]
    fitb_keywords = ["fill in the blank", "fill the blank", "complete the following", "complete the sentence"]

    is_assertion_reason = any(kw in q_lower for kw in assertion_keywords)
    is_sequence = any(kw in q_lower for kw in sequence_keywords)
    is_multi_correct = any(kw in q_lower for kw in multi_correct_keywords)
    is_true_false = any(kw in q_lower for kw in true_false_keywords)
    is_fitb = any(kw in q_lower for kw in fitb_keywords) or "___" in q_raw

    theory_keywords = ["define", "explain", "why", "how", "difference", "process", "mechanism", "steps", "reason", "theory", "concept", "principle", "state", "list", "mention", "describe", "distinguish", "compare", "what is"]
    numerical_keywords = ["solve", "calculate", "plot", "graph", "formula", "integral", "derivative", "value of", "compute", "evaluate"]
    symbols = ["+", "=", "^", "\\"] # Reduced symbols to avoid OCR noise

    is_theory = any(kw in q_lower for kw in theory_keywords)
    is_numerical = (any(kw in q_lower for kw in numerical_keywords) or any(sym in q_raw for sym in symbols)) and not is_theory

    if is_assertion_reason:
        system, user = _build_assertion_reason_prompt(req_for_prompt)
        logger.info(f"Routing to AssertionReason: {q_lower[:50]}...")
    elif is_sequence:
        system, user = _build_sequence_prompt(req_for_prompt)
        logger.info(f"Routing to SequenceMCQ: {q_lower[:50]}...")
    elif is_multi_correct:
        system, user = _build_multi_correct_prompt(req_for_prompt)
        logger.info(f"Routing to MultiCorrect: {q_lower[:50]}...")
    elif is_true_false:
        system, user = _build_true_false_prompt(req_for_prompt)
        logger.info(f"Routing to TrueFalse: {q_lower[:50]}...")
    elif is_fitb:
        system, user = _build_fitb_prompt(req_for_prompt)
        logger.info(f"Routing to FillInBlank: {q_lower[:50]}...")
    elif is_theory:
        system, user = _build_theory_prompt(req_for_prompt)
        logger.info(f"Routing to TheoryMode v3: {q_lower[:50]}...")
    elif is_numerical:
        try:
            logger.info(f"Routing to ScientificSolver: {q_lower[:50]}...")
            scientific_res = await scientific_solver.solve(req_for_prompt.questionText, req.mode)
            return scientific_res
        except Exception as e:
            logger.error(f"ScientificSolver failed: {str(e)}. Falling back to standard LLM.")
            system, user = _build_prompt(req_for_prompt)
    else:
        system, user = _build_prompt(req_for_prompt)

    try:
        if groq_keys:
            last_groq_error = None
            result = None
            for i, groq_key in enumerate(groq_keys):
                try:
                    result = await _call_groq(system, user, groq_key)
                    break
                except HTTPException as e:
                    last_groq_error = e
                    text = str(e.detail).lower()
                    key_like_error = any(x in text for x in ("rate limit", "quota", "too many requests", "invalid api key", "authentication"))
                    if key_like_error and i < len(groq_keys) - 1:
                        continue
                    raise
            if result is None and last_groq_error:
                raise last_groq_error
        else:
            result = await _call_gemini(system, user, gemini_key)

        # Ensure conceptLinks is populated from either key_concepts or conceptLinks
        if "key_concepts" in result and "conceptLinks" not in result:
            result["conceptLinks"] = result["key_concepts"]
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
