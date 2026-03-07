import os, re, io, json, asyncio, math, string
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# ── Download NLTK data immediately (before any corpus is referenced) ─────────
import nltk
for _corpus in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(_corpus, quiet=True)
# ─────────────────────────────────────────────────────────────────────────────

from google import genai
from google.genai import types
from rake_nltk import Rake

import numpy as np
import torch
import requests
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from transformers import (
    pipeline,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModel,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import faiss
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MIN_WORDS_AI        = 10
PROFILE_FILE        = "profiles.json"
EMBEDDING_DIM       = 384
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"   # or local path

SERPAPI_KEY         = os.getenv("SERPAPI_KEY", "")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = "gemini-2.5-flash-lite"
POPPLER_PATH        = os.getenv("POPPLER_PATH", r"C:\poppler-25.12.0\Library\bin")

TYPED_TEXT_MIN_CHARS = 200
AUTO_KEYWORDS_N      = 8          # how many keywords RAKE extracts per answer

TROCR_HW_MODEL      = "microsoft/trocr-large-handwritten"
TROCR_PRINTED_MODEL = "microsoft/trocr-large-printed"

LINE_DARK_PIXEL_THRESHOLD = 5
LINE_MIN_HEIGHT_PX        = 15
LINE_PADDING_PX           = 8
TROCR_MIN_CHARS           = 2
OCR_DPI                   = 300

# Grading weights (must sum to 1.0)
WEIGHT_CORRECTNESS  = 0.50   # semantic match to model answer
WEIGHT_KEYWORDS     = 0.20   # key-term coverage
WEIGHT_WRITING      = 0.15   # quality / style
WEIGHT_COMPLETENESS = 0.15   # length / depth relative to model answer

# Penalty multipliers applied to the final score
PLAGIARISM_PENALTY_THRESHOLD = 25    # % above which penalty kicks in
AI_PENALTY_THRESHOLD         = 0.55  # probability above which penalty kicks in
MAX_PLAGIARISM_PENALTY       = 0.60  # max deduction factor (40 % score retained)
MAX_AI_PENALTY               = 0.50

# Pass/Fail threshold
PASS_THRESHOLD = 50.0   # overall percentage required to pass

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL MODEL HANDLES
# ══════════════════════════════════════════════════════════════════════════════

embed_tokenizer:          AutoTokenizer            = None
embed_model:              AutoModel                = None
openai_pipe                                        = None
chatgpt_pipe                                       = None
gpt2_model:               GPT2LMHeadModel          = None
gpt2_tok:                 GPT2TokenizerFast        = None
_faiss_index:             faiss.IndexFlatIP        = None

trocr_hw_processor:       TrOCRProcessor           = None
trocr_hw_model:           VisionEncoderDecoderModel= None
trocr_printed_processor:  TrOCRProcessor           = None
trocr_printed_model:      VisionEncoderDecoderModel= None

# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_tokenizer, embed_model, openai_pipe, chatgpt_pipe
    global gpt2_model, gpt2_tok, _faiss_index
    global trocr_hw_processor, trocr_hw_model
    global trocr_printed_processor, trocr_printed_model

    print("[STARTUP] Loading embedding model …")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    embed_model.eval()
    _faiss_index    = faiss.IndexFlatIP(EMBEDDING_DIM)

    print("[STARTUP] Loading AI-detection pipelines …")
    openai_pipe  = pipeline("text-classification",
                            model="roberta-base-openai-detector")
    chatgpt_pipe = pipeline("text-classification",
                            model="Hello-SimpleAI/chatgpt-detector-roberta")

    print("[STARTUP] Loading GPT-2 perplexity model …")
    gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()

    print("[STARTUP] Loading TrOCR — handwritten …")
    trocr_hw_processor = TrOCRProcessor.from_pretrained(TROCR_HW_MODEL)
    trocr_hw_model     = VisionEncoderDecoderModel.from_pretrained(TROCR_HW_MODEL)
    trocr_hw_model.eval()

    print("[STARTUP] Loading TrOCR — printed …")
    trocr_printed_processor = TrOCRProcessor.from_pretrained(TROCR_PRINTED_MODEL)
    trocr_printed_model     = VisionEncoderDecoderModel.from_pretrained(TROCR_PRINTED_MODEL)
    trocr_printed_model.eval()

    print("[STARTUP] All models ready ✓")
    yield


app = FastAPI(title="AI Exam Grading API", version="1.0.0", lifespan=lifespan)

# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PrepareRequest(BaseModel):
    questions:  list[str]
    max_marks:  list[float] = []    # one per question; defaults to 10 each


class Question(BaseModel):
    question_number: int
    question_text:   str
    model_answer:    str  = ""      # leave blank → auto-generated by Gemini
    max_marks:       float    = 10.0
    keywords:        list[str] = [] # leave empty → auto-extracted by RAKE


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def preprocess(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def sentences(t: str) -> list[str]:
    return nltk.sent_tokenize(t)

def words(t: str) -> list[str]:
    return re.findall(r"\b\w+\b", t.lower())

def content_words(t: str) -> list[str]:
    return [w for w in words(t) if w not in STOPWORDS and len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO KEYWORD EXTRACTION  (RAKE)
# ══════════════════════════════════════════════════════════════════════════════

def auto_keywords(text: str, top_n: int = AUTO_KEYWORDS_N) -> list[str]:
    """
    Extract important keywords/phrases from text using RAKE.
    Falls back to top TF-IDF content words if RAKE returns nothing.
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    rake_phrases = r.get_ranked_phrases()[:top_n]

    # Supplement with high-value single content words
    cw = [w for w in content_words(text) if len(w) > 4]
    # Deduplicate while preserving order
    seen, combined = set(), []
    for item in rake_phrases + cw:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            combined.append(item)
        if len(combined) >= top_n:
            break

    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI — MODEL ANSWER + KEYWORD GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_client() -> genai.Client:
    """Return a configured google.genai Client, raising clearly if key missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file:  GEMINI_API_KEY=AIza..."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def gemini_generate_model_answer(question_text: str) -> tuple[str, list[str]]:
    """
    Ask Gemini (Gemini 2.5 Flash Lite) to produce:
      • a concise, factually accurate model answer (3-6 sentences)
      • 6-8 important keywords a student's answer must contain

    Returns (model_answer, keywords).
    Falls back to ("", []) if GEMINI_API_KEY is not configured or call fails.
    """
    if not GEMINI_API_KEY:
        print("[GEMINI] WARNING: GEMINI_API_KEY not set — model answer will be empty.")
        return "", []

    prompt = f"""You are an experienced examiner. For the exam question below, provide:
1. A concise model answer (3-6 sentences, factually accurate, suitable as a marking guide).
2. Exactly 6-8 important keywords or key phrases that a student's answer MUST contain to score full marks.

Question: {question_text}

Respond with valid JSON only — no markdown fences, no explanation, no extra text:
{{"model_answer": "...", "keywords": ["...", "..."]}}"""

    try:
        client   = _gemini_client()
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature       = 0.2,
                max_output_tokens = 600,
            ),
        )
        raw = response.text.strip()
        # Strip any accidental markdown code fences Gemini sometimes adds
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        data         = json.loads(raw)
        model_answer = data.get("model_answer", "").strip()
        kws          = [str(k).strip() for k in data.get("keywords", [])]
        print(f"[GEMINI] Generated model answer ({len(model_answer)} chars), "
              f"{len(kws)} keywords: {kws}")
        return model_answer, kws
    except json.JSONDecodeError as e:
        print(f"[GEMINI] JSON parse error: {e}\nRaw response: {raw[:300]}")
        return "", []
    except Exception as e:
        print(f"[GEMINI] ERROR: {e}")
        return "", []

# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS  (MiniLM via FAISS)
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> np.ndarray:
    tokens = embed_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        out = embed_model(**tokens)
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return F.normalize(emb, p=2, dim=1).squeeze(0).numpy().astype("float32")


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity in embedding space, 0-1."""
    if not a.strip() or not b.strip():
        return 0.0
    ea, eb = get_embedding(a), get_embedding(b)
    return float(np.clip(np.dot(ea, eb), 0.0, 1.0))


def tfidf_similarity(a: str, b: str) -> float:
    try:
        tf  = TfidfVectorizer(stop_words="english").fit_transform(
                  [preprocess(a), preprocess(b)])
        return float(sk_cosine(tf[0], tf[1])[0][0])
    except ValueError:
        return 0.0


def blended_similarity(a: str, b: str) -> float:
    return (semantic_similarity(a, b) + tfidf_similarity(a, b)) / 2

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. ANSWER CORRECTNESS ────────────────────────────────────────────────────

def score_correctness(student_ans: str, model_ans: str) -> float:
    """Semantic + lexical similarity to model answer, 0-100."""
    return round(blended_similarity(student_ans, model_ans) * 100, 2)


# ── 2. KEYWORD COVERAGE ──────────────────────────────────────────────────────

def score_keyword_coverage(student_ans: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer, 0-100."""
    if not keywords:
        return 100.0                        # no keywords specified → full marks
    ans_lower = student_ans.lower()
    found = sum(1 for kw in keywords if kw.lower() in ans_lower)
    return round(found / len(keywords) * 100, 2)


# ── 3. WRITING QUALITY ───────────────────────────────────────────────────────

def score_writing_quality(text: str) -> dict:
    """
    Returns a quality score 0-100 and sub-metrics.

    Sub-metrics:
      • type_token_ratio  — vocabulary richness (unique/total words)
      • avg_sentence_len  — target band: 12-22 words
      • sentence_variety  — coefficient of variation of sentence lengths
      • content_density   — content-words / total words
    """
    w = words(text)
    s = sentences(text)
    if not w or not s:
        return {"score": 0.0, "type_token_ratio": 0.0,
                "avg_sentence_len": 0.0, "sentence_variety": 0.0,
                "content_density": 0.0}

    sent_lens  = [len(ww.split()) for ww in s]
    avg_len    = float(np.mean(sent_lens))
    std_len    = float(np.std(sent_lens))
    cv         = std_len / avg_len if avg_len > 0 else 0.0   # coefficient of variation

    ttr        = len(set(w)) / len(w)
    cw_ratio   = len(content_words(text)) / len(w)

    # Score each dimension 0-1
    ttr_score  = min(ttr * 2, 1.0)                           # ttr ≥ 0.5 → full marks
    len_score  = 1.0 - min(abs(avg_len - 17) / 17, 1.0)     # optimal ≈ 17 words/sentence
    var_score  = min(cv, 1.0)                                 # more variety = better
    cd_score   = min(cw_ratio * 2, 1.0)                      # cw_ratio ≥ 0.5 → full

    overall    = (ttr_score * 0.35 + len_score * 0.25 +
                  var_score * 0.20 + cd_score  * 0.20) * 100

    return {
        "score":              round(overall, 2),
        "type_token_ratio":   round(ttr, 3),
        "avg_sentence_len":   round(avg_len, 1),
        "sentence_variety":   round(cv, 3),
        "content_density":    round(cw_ratio, 3),
    }


# ── 4. COMPLETENESS / DEPTH ──────────────────────────────────────────────────

def score_completeness(student_ans: str, model_ans: str) -> float:
    """
    Ratio of content-word count (student vs model), capped at 1.0.
    Very short answers relative to the model answer are penalised.
    """
    if not model_ans.strip():
        return 100.0
    sw = len(content_words(student_ans))
    mw = len(content_words(model_ans))
    if mw == 0:
        return 100.0
    # Sigmoid-smooth so partial credit rises gracefully
    ratio = sw / mw
    score = (1 / (1 + math.exp(-10 * (ratio - 0.4)))) * 100
    return round(min(score, 100.0), 2)


# ── 5. WRITING FEATURES (for authorship profiling) ───────────────────────────

def writing_features(text: str) -> dict:
    w = words(text)
    s = sentences(text)
    if not w:
        return {}
    lens = [len(x.split()) for x in s]
    return {
        "avg_sentence_length": float(np.mean(lens)),
        "sentence_var":        float(np.var(lens)),
        "type_token_ratio":    len(set(w)) / len(w),
    }


def authorship_shift(old: dict, new: dict) -> float:
    if not old or not new:
        return 0.0
    common = [k for k in old if k in new]
    if not common:
        return 0.0
    return sum(abs(old[k] - new[k]) for k in common) / len(common)


# ── 6. AI DETECTION ──────────────────────────────────────────────────────────

def gpt2_prob(text: str) -> float:
    enc = gpt2_tok(text[:1000], return_tensors="pt", truncation=True)
    with torch.no_grad():
        loss = gpt2_model(**enc, labels=enc["input_ids"]).loss.item()
    return max(0.0, min(1.0, (4.5 - loss) / 3))


def classify_pipe(pipe, text: str) -> float:
    r   = pipe(text[:1200])[0]
    lbl = r["label"].upper()
    s   = float(r["score"])
    return s if lbl != "REAL" else 1 - s


def ai_probability(text: str) -> float:
    if len(text.split()) < MIN_WORDS_AI:
        return 0.0
    s1 = classify_pipe(openai_pipe,  text)
    s2 = classify_pipe(chatgpt_pipe, text)
    s3 = gpt2_prob(text)
    print(f"[AI] openai={s1:.3f}  chatgpt={s2:.3f}  gpt2={s3:.3f}")
    return float(np.mean([s1, s2, s3]))


# ── 7. PLAGIARISM ─────────────────────────────────────────────────────────────

def google_snippets(query: str, n: int = 5) -> list[dict]:
    if not SERPAPI_KEY:
        print("[SERPAPI] WARNING: SERPAPI_KEY not set — plagiarism score will be 0.")
        return []
    params = {"engine": "google", "q": query[:500], "num": n, "api_key": SERPAPI_KEY}
    results: list[dict] = []
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        ab = data.get("answer_box", {})
        for field in ("answer", "snippet"):
            t = ab.get(field, "").strip()
            if t:
                results.append({"snippet": t,
                                 "link": ab.get("link", ""),
                                 "title": ab.get("title", "Featured Snippet")})
                break

        for item in data.get("organic_results", []):
            sn = item.get("snippet", "").strip()
            if sn:
                results.append({"snippet": sn,
                                 "link": item.get("link", ""),
                                 "title": item.get("title", "")})
            if len(results) >= n:
                break
    except Exception as e:
        print(f"[SERPAPI] FAILED: {e}")
    return results


def cosine_similarity_faiss_batch(query: str, corpus: list[str]) -> list[float]:
    if not corpus:
        return []
    q_emb  = get_embedding(query)
    c_embs = np.stack([get_embedding(c) for c in corpus]).astype("float32")
    idx    = faiss.IndexFlatIP(EMBEDDING_DIM)
    idx.add(c_embs)
    dists, idxs = idx.search(np.expand_dims(q_emb, 0), k=len(corpus))
    scores = np.clip(dists[0], 0.0, 1.0)
    aligned = np.empty(len(corpus), dtype="float32")
    for rank, i in enumerate(idxs[0]):
        aligned[i] = scores[rank]
    return aligned.tolist()


async def plagiarism(answer: str, question: str) -> dict:
    results = google_snippets(question + " " + answer[:80])
    if not results:
        return {"score": 0.0, "matched_sentences": []}

    valid = [r for r in results if len(r["snippet"]) > 20]
    if not valid:
        return {"score": 0.0, "matched_sentences": []}

    ans_sents = [s.strip() for s in sentences(answer) if len(s.strip().split()) >= 4]
    if not ans_sents:
        return {"score": 0.0, "matched_sentences": []}

    snippets_text = [r["snippet"] for r in valid]
    matched: list[dict]  = []
    sent_scores: list[float] = []

    for sent in ans_sents:
        faiss_sc  = cosine_similarity_faiss_batch(sent, snippets_text)
        try:
            tf = TfidfVectorizer(stop_words="english").fit_transform(
                     [preprocess(sent)] + [preprocess(s) for s in snippets_text])
            tfidf_sc = [float(sk_cosine(tf[0], tf[i+1])[0][0])
                        for i in range(len(snippets_text))]
        except ValueError:
            tfidf_sc = [0.0] * len(snippets_text)

        blended   = [(f + t) / 2 for f, t in zip(faiss_sc, tfidf_sc)]
        best_idx  = int(np.argmax(blended))
        best_sc   = blended[best_idx]
        sent_scores.append(best_sc)

        if best_sc >= 0.25:
            matched.append({
                "sentence":       sent,
                "similarity":     round(best_sc * 100, 2),
                "matched_source": snippets_text[best_idx],
                "source_title":   valid[best_idx]["title"],
                "source_link":    valid[best_idx]["link"],
            })

    overall = round(float(np.mean(sent_scores)) * 100, 2) if sent_scores else 0.0
    matched.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"[PLAGIARISM] overall={overall}  matched={len(matched)}/{len(ans_sents)}")
    return {"score": overall, "matched_sentences": matched}

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL GRADE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_final_grade(
    correctness:  float,
    keywords:     float,
    writing:      float,
    completeness: float,
    plag_score:   float,
    ai_prob:      float,
    max_marks:    float,
) -> dict:
    """
    Blend sub-scores → raw_score (0-100).
    Apply plagiarism & AI penalties → penalised_score.
    Scale to max_marks.
    """
    raw_score = (
        correctness  * WEIGHT_CORRECTNESS  +
        keywords     * WEIGHT_KEYWORDS     +
        writing      * WEIGHT_WRITING      +
        completeness * WEIGHT_COMPLETENESS
    )

    # Plagiarism penalty: linear from threshold → max penalty
    plag_penalty = 0.0
    if plag_score > PLAGIARISM_PENALTY_THRESHOLD:
        excess       = (plag_score - PLAGIARISM_PENALTY_THRESHOLD) / \
                       (100 - PLAGIARISM_PENALTY_THRESHOLD)
        plag_penalty = excess * MAX_PLAGIARISM_PENALTY

    # AI penalty: linear from threshold → max penalty
    ai_penalty = 0.0
    if ai_prob > AI_PENALTY_THRESHOLD:
        excess     = (ai_prob - AI_PENALTY_THRESHOLD) / (1 - AI_PENALTY_THRESHOLD)
        ai_penalty = excess * MAX_AI_PENALTY

    total_penalty      = min(plag_penalty + ai_penalty, 0.90)   # cap at 90 % deduction
    penalised_score    = raw_score * (1 - total_penalty)
    marks_awarded      = round(penalised_score / 100 * max_marks, 2)

    return {
        "raw_score":         round(raw_score, 2),
        "plagiarism_penalty": round(plag_penalty * 100, 1),
        "ai_penalty":         round(ai_penalty  * 100, 1),
        "penalised_score":    round(penalised_score, 2),
        "marks_awarded":      marks_awarded,
        "max_marks":          max_marks,
    }


def grade_verdict(penalised_score: float, plag_score: float, ai_prob: float) -> str:
    flags = []
    if plag_score > 50:
        flags.append("HIGH PLAGIARISM")
    elif plag_score > 25:
        flags.append("MODERATE PLAGIARISM")
    if ai_prob > AI_PENALTY_THRESHOLD:
        flags.append("LIKELY AI-GENERATED")

    if penalised_score >= 80:
        grade = "EXCELLENT"
    elif penalised_score >= 65:
        grade = "GOOD"
    elif penalised_score >= 50:
        grade = "SATISFACTORY"
    elif penalised_score >= 35:
        grade = "POOR"
    else:
        grade = "FAIL"

    return f"{grade}" + (f" | ⚠ {', '.join(flags)}" if flags else "")


# ══════════════════════════════════════════════════════════════════════════════
#  AI FEEDBACK — explanations, mistake patterns, difficulty, skill scores
# ══════════════════════════════════════════════════════════════════════════════

def _default_feedback() -> dict:
    """Return a safe fallback when Gemini is unavailable or fails."""
    return {
        "marking_reason": "",
        "strong_parts": [],
        "weak_parts": [],
        "what_to_improve": [],
        "how_to_improve": [],
        "mistake_patterns": {
            "conceptual_mistakes": 0,
            "calculation_mistakes": 0,
            "time_management_issues": 0,
            "guessing_behavior": 0,
            "details": "Feedback unavailable — Gemini API key not set or call failed.",
        },
        "difficulty_analysis": {
            "ai_assigned_difficulty": "Unknown",
            "student_accuracy_at_level": 0.0,
            "reasoning": "",
        },
        "skill_scores": {
            "problem_solving": 0,
            "concept_clarity": 0,
            "accuracy": 0,
            "overall_learning_index": 0.0,
        },
    }


def generate_ai_feedback(
    question_text: str,
    model_answer: str,
    student_answer: str,
    correctness: float,
    keyword_coverage: float,
    writing_score: float,
    completeness: float,
    grade_info: dict,
    is_wrong: bool = False,
) -> dict:
    """
    Use Gemini to produce rich AI feedback for a single question:
      1. Wrong-answer explanation — ONLY when is_wrong=True
      2. Mistake pattern detection (conceptual / calculation / time / guessing)
      3. Difficulty analysis (AI-assigned Easy/Medium/Hard + accuracy)
      4. Skill scores / learning index (problem solving, clarity, speed, accuracy)

    Falls back to defaults if GEMINI_API_KEY is unset or the call fails.
    """
    if not GEMINI_API_KEY:
        print("[FEEDBACK] WARNING: GEMINI_API_KEY not set — returning default feedback.")
        return _default_feedback()

    # Build the wrong-answer section of the prompt only for wrong answers
    if is_wrong:
        wrong_ans_block = """  "wrong_answer_explanation": {{
    "correct_answer": "brief correct answer",
    "step_by_step_explanation": "step-by-step walkthrough of the correct solution",
    "concept_explanation": "underlying concept the student should revise",
    "suggested_resources": "short video or notes recommendation for the topic"
  }},"""
        wrong_ans_instruction = (
            "- The student answered this question INCORRECTLY. "
            "Provide a helpful wrong_answer_explanation with the correct answer, "
            "step-by-step solution, concept review, and resource suggestions."
        )
    else:
        wrong_ans_block = ""
        wrong_ans_instruction = (
            "- The student answered this question correctly. "
            "Do NOT include a wrong_answer_explanation field."
        )

    prompt = f"""You are an expert educational AI. Analyse the student's exam answer below and produce structured feedback.

Question: {question_text}

Model Answer (correct): {model_answer}

Student Answer: {student_answer}

Scoring context (0-100 unless noted):
  Correctness:      {correctness}
  Keyword Coverage: {keyword_coverage}
  Writing Quality:  {writing_score}
  Completeness:     {completeness}
  Marks Awarded:    {grade_info.get('marks_awarded', 0)} / {grade_info.get('max_marks', 0)}
  Penalised Score:  {grade_info.get('penalised_score', 0)}

Return ONLY valid JSON (no markdown fences, no extra text) with exactly this structure:
{{
{wrong_ans_block}
  "marking_reason": "clear explanation of why marks were awarded or deducted — reference specific strengths and weaknesses in the answer",
  "strong_parts": ["list of things the student did well in this answer"],
  "weak_parts": ["list of areas where the answer fell short or was incorrect"],
  "what_to_improve": ["specific topics or skills the student should work on"],
  "how_to_improve": ["actionable study tips, practice suggestions, or resources to address each weakness"],
  "mistake_patterns": {{
    "conceptual_mistakes": integer 0-10 (0=none, 1-3=minor, 4-6=moderate, 7-10=severe \u2014 use the FULL range based on how much the answer shows misunderstanding),
    "calculation_mistakes": integer 0-10 (0=none, 1-3=minor arithmetic slips, 4-6=moderate errors, 7-10=fundamentally wrong calculations),
    "time_management_issues": integer 0-10 (0=well-paced, 1-3=slightly rushed/verbose, 4-6=noticeably unbalanced, 7-10=extremely rushed or incomplete),
    "guessing_behavior": integer 0-10 (0=clearly understood, 1-3=slight uncertainty, 4-6=partially guessed, 7-10=mostly random guessing),
    "details": "brief explanation of what patterns you detected and why you gave these specific scores"
  }},
  "difficulty_analysis": {{
    "ai_assigned_difficulty": "Easy" or "Medium" or "Hard",
    "student_accuracy_at_level": a number 0-100 representing how well the student performed relative to the difficulty,
    "reasoning": "why you assigned this difficulty level"
  }},
  "skill_scores": {{
    "problem_solving": integer 1-10,
    "concept_clarity": integer 1-10,
    "accuracy": integer 1-10,
    "overall_learning_index": float average of the three scores above
  }}
}}

IMPORTANT:
{wrong_ans_instruction}
- Difficulty MUST be decided by YOU based on the question content — it is NOT pre-set by the teacher.
- Be constructive and educational in your feedback."""

    try:
        client   = _gemini_client()
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature       = 0.3,
                max_output_tokens = 1200,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        # ── validate & sanitise the skill scores ────────────────────────────
        sk = data.get("skill_scores", {})
        sk.pop("speed", None)  # remove speed if Gemini included it
        for key in ("problem_solving", "concept_clarity", "accuracy"):
            sk[key] = max(1, min(10, int(sk.get(key, 5))))
        sk["overall_learning_index"] = round(
            sum(sk[k] for k in ("problem_solving", "concept_clarity", "accuracy")) / 3, 1
        )
        data["skill_scores"] = sk

        # ── validate & clamp mistake pattern scores ───────────────────────
        mp = data.get("mistake_patterns", {})
        for key in ("conceptual_mistakes", "calculation_mistakes",
                    "time_management_issues", "guessing_behavior"):
            mp[key] = max(0, min(10, int(mp.get(key, 0))))
        data["mistake_patterns"] = mp

        # Strip wrong_answer_explanation if it crept in for correct answers
        if not is_wrong:
            data.pop("wrong_answer_explanation", None)

        print(f"[FEEDBACK] Generated AI feedback for: {question_text[:60]}")
        return data

    except json.JSONDecodeError as e:
        print(f"[FEEDBACK] JSON parse error: {e}")
        return _default_feedback()
    except Exception as e:
        print(f"[FEEDBACK] ERROR: {e}")
        return _default_feedback()


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION  (typed + TrOCR fallback — copied from reference code)
# ══════════════════════════════════════════════════════════════════════════════

def extract_typed(data: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _detect_line_regions(img: Image.Image) -> list[tuple[int, int]]:
    gray    = np.array(img.convert("L"))
    binary  = (gray < 180).astype(np.uint8)
    row_sum = binary.sum(axis=1)
    in_line, regions, start = False, [], 0
    for y, count in enumerate(row_sum):
        if not in_line and count >= LINE_DARK_PIXEL_THRESHOLD:
            in_line, start = True, y
        elif in_line and count < LINE_DARK_PIXEL_THRESHOLD:
            in_line = False
            top, bottom = max(0, start - LINE_PADDING_PX), min(img.height, y + LINE_PADDING_PX)
            if bottom - top >= LINE_MIN_HEIGHT_PX:
                regions.append((top, bottom))
    if in_line:
        top, bottom = max(0, start - LINE_PADDING_PX), min(img.height, len(row_sum) + LINE_PADDING_PX)
        if bottom - top >= LINE_MIN_HEIGHT_PX:
            regions.append((top, bottom))
    return regions


def _is_blank_region(img: Image.Image, threshold: int = 180, min_dark_ratio: float = 0.005) -> bool:
    gray  = np.array(img.convert("L"))
    return ((gray < threshold).sum() / gray.size) < min_dark_ratio


def _deduplicate_runs(lines: list[str], max_run: int = 2) -> list[str]:
    if not lines:
        return lines
    out, run = [lines[0]], 1
    for line in lines[1:]:
        if line == out[-1]:
            run += 1
            if run <= max_run:
                out.append(line)
        else:
            run = 1
            out.append(line)
    return out


def _trocr_decode_line(img: Image.Image, processor, model) -> str:
    pv = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pv, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _trocr_page(img: Image.Image, processor, model, page_num: int = 0, label: str = "?") -> str:
    regions = _detect_line_regions(img)
    if not regions:
        if _is_blank_region(img):
            return ""
        return _trocr_decode_line(img, processor, model)

    raw_lines = []
    for i, (top, bottom) in enumerate(regions):
        crop = img.crop((0, top, img.width, bottom))
        if _is_blank_region(crop):
            continue
        text = _trocr_decode_line(crop, processor, model)
        if len(text.replace(" ", "")) >= TROCR_MIN_CHARS:
            raw_lines.append(text)
    return "\n".join(_deduplicate_runs(raw_lines))


def extract_ocr(data: bytes, handwritten: bool = True) -> str:
    processor = trocr_hw_processor      if handwritten else trocr_printed_processor
    model     = trocr_hw_model          if handwritten else trocr_printed_model
    label     = "handwritten"           if handwritten else "printed"
    images    = convert_from_bytes(data, dpi=OCR_DPI, poppler_path=POPPLER_PATH)
    parts     = []
    for page_num, img in enumerate(images, 1):
        text = _trocr_page(img, processor, model, page_num=page_num, label=label)
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def extract_pdf(data: bytes) -> tuple[str, str]:
    typed      = extract_typed(data)
    unique_w   = set(re.findall(r"[a-zA-Z]{3,}", typed))
    gc_ratio   = sum(1 for c in typed if ord(c) > 127) / max(len(typed), 1)

    if (len(typed.strip()) >= TYPED_TEXT_MIN_CHARS
            and len(unique_w) >= 10
            and gc_ratio <= 0.15):
        return typed, "typed"

    print("[PDF] Switching to TrOCR …")
    hw_text   = extract_ocr(data, handwritten=True)
    pr_text   = extract_ocr(data, handwritten=False)
    if len(pr_text.strip()) > len(hw_text.strip()):
        return pr_text, "printed-trocr"
    return hw_text, "handwritten-trocr"


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_answers(text: str, questions: list) -> dict[int, str]:
    """
    Split extracted/OCR text into per-question answer blocks.

    Detects ALL common question numbering formats:
      Arabic    : 1. 2. 3)  (1)  1 .
      Alphabet  : a. B. c)  (A)
      Roman     : i. II. iv) (III)
      Prefixed  : Question 1 / Question 1. / Q1. / Q.1
    Maps every detected marker to a 1-based sequential index so the
    result dict always uses plain integers matching question_number.
    """

    # ── Roman numeral helpers ────────────────────────────────────────────────
    _ROMAN = {
        "i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,
        "ix":9,"x":10,"xi":11,"xii":12,"xiii":13,"xiv":14,"xv":15,
        "xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20,
    }

    def _marker_to_int(marker: str) -> int | None:
        """Convert any detected marker string to a 1-based integer."""
        m = marker.strip(" \t.)(").lower()
        if not m:
            return None
        # Arabic digit
        if m.isdigit():
            return int(m)
        # Single alphabet  a→1, b→2 …
        if len(m) == 1 and m.isalpha() and m not in _ROMAN:
            return ord(m) - ord("a") + 1
        # Roman numeral
        if m in _ROMAN:
            return _ROMAN[m]
        return None

    # ── Unified pattern — matches ALL formats at line start ─────────────────
    PATTERN = re.compile(
        r"(?:^|\n)"
        r"[ \t]*"
        r"(?:"
          r"(?:question|q(?:\.|\s*))\s*"
          r"(?P<prefixed>[ivxlcdmIVXLCDM]{1,8}|\d+|[a-zA-Z])"
          r"(?:\s*[\.\):]|\s|$)"
        r"|"
          r"\((?P<bracketed>[ivxlcdmIVXLCDM]{1,8}|\d+|[a-zA-Z])\)"
          r"[ \t]"
        r"|"
          r"(?P<marker>[ivxlcdmIVXLCDM]{1,8}|\d+|[a-zA-Z])"
          r"\s*[\.\)]"
          r"[ \t]"
        r")",
        re.MULTILINE | re.IGNORECASE,
    )

    hits = list(PATTERN.finditer(text))

    if not hits:
        fixed = re.sub(r"(?:^|\n)[ \t]*[lI][ \t]*\.", "\n1. ", text)
        fixed = re.sub(r"(?:^|\n)[ \t]*[zZ][ \t]*\.", "\n2. ", fixed)
        hits  = list(PATTERN.finditer(fixed))
        if hits:
            text = fixed

    if not hits:
        print("[SEGMENT] WARNING: No question markers found in text.")
        print(f"[SEGMENT] Text sample: {repr(text[:400])}")
        return {questions[0].question_number: text.strip()} if questions else {}

    resolved: list[tuple[int, int, int]] = []

    for h in hits:
        raw = (
            h.group("prefixed")
            or h.group("bracketed")
            or h.group("marker")
            or ""
        )
        idx = _marker_to_int(raw)
        if idx is not None:
            resolved.append((h.start(), h.end(), idx))

    if not resolved:
        print("[SEGMENT] WARNING: Markers found but none resolved to integers.")
        return {questions[0].question_number: text.strip()} if questions else {}

    resolved.sort(key=lambda x: x[0])

    index_map = {}
    for seq, (_, _, idx) in enumerate(resolved, start=1):
        if idx not in index_map:
            index_map[idx] = seq

    segs: dict[int, str] = {}
    for i, (start, end, raw_idx) in enumerate(resolved):
        seq_idx  = index_map.get(raw_idx, i + 1)
        seg_start = end
        seg_end   = resolved[i + 1][0] if i + 1 < len(resolved) else len(text)
        ans       = text[seg_start:seg_end].strip()
        if ans:
            segs[seq_idx] = ans
            print(f"[SEGMENT] Q{seq_idx} (marker '{raw_idx}'): {len(ans)} chars")
        else:
            print(f"[SEGMENT] Q{seq_idx} (marker '{raw_idx}'): empty — skipped")

    return segs

# ══════════════════════════════════════════════════════════════════════════════
#  AUTHORSHIP PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def load_profiles() -> dict:
    if not os.path.exists(PROFILE_FILE):
        return {}
    try:
        return json.load(open(PROFILE_FILE))
    except Exception:
        return {}

def save_profiles(p: dict):
    json.dump(p, open(PROFILE_FILE, "w"), indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.getenv("RESULTS_DIR", "grade_results")   # folder next to grading.py

def _save_result(output: dict) -> None:
    """
    Save every grading result to two files inside RESULTS_DIR/:

    1. JSON  — full detail, one file per student per timestamp
               grade_results/<student_id>_<timestamp>.json

    2. CSV   — flat summary row appended to a single master file
               grade_results/summary.csv
               Columns: timestamp, student_id, pdf_mode, letter_grade,
                        pass_fail_status, overall_pct, total_marks, max_marks,
                        q<N>_marks, q<N>_correctness, q<N>_keywords,
                        q<N>_writing, q<N>_completeness,
                        q<N>_plagiarism, q<N>_ai_prob, q<N>_verdict
    """
    import csv
    from datetime import datetime

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    student   = re.sub(r"[^\w\-]", "_", output["student_id"])

    # ── 1. Full JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, f"{student}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON → {json_path}")

    # ── 2. CSV summary row ────────────────────────────────────────────────────
    csv_path  = os.path.join(RESULTS_DIR, "summary.csv")
    file_exists = os.path.exists(csv_path)

    # Build per-question flat columns
    q_cols: dict[str, str] = {}
    for r in output["results"]:
        n   = r["question_number"]
        g   = r["grade"]
        ss  = r["sub_scores"]
        wq  = ss["writing_quality"]
        ing = r["integrity"]
        q_cols[f"q{n}_marks"]        = g["marks_awarded"]
        q_cols[f"q{n}_max_marks"]    = g["max_marks"]
        q_cols[f"q{n}_correctness"]  = ss["correctness"]
        q_cols[f"q{n}_keywords"]     = ss["keyword_coverage"]
        q_cols[f"q{n}_writing"]      = wq["score"]
        q_cols[f"q{n}_completeness"] = ss["completeness"]
        q_cols[f"q{n}_plagiarism"]   = ing["plagiarism_score"]
        q_cols[f"q{n}_ai_prob"]      = ing["ai_probability"]
        q_cols[f"q{n}_auth_shift"]   = ing["authorship_shift"]
        q_cols[f"q{n}_verdict"]      = g["verdict"]

    base_cols = {
        "timestamp":        ts,
        "student_id":       output["student_id"],
        "pdf_mode":         output["pdf_mode"],
        "letter_grade":     output["letter_grade"],
        "pass_fail_status": output.get("pass_fail_status", ""),   # ← NEW
        "overall_pct":      output["overall_percentage"],
        "total_marks":      output["total_marks_awarded"],
        "max_marks":        output["total_max_marks"],
    }
    row = {**base_cols, **q_cols}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[SAVE] CSV  → {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  /prepare  — generate model answers + keywords via Gemini
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/prepare")
async def prepare(body: PrepareRequest):
    """
    Given a list of plain question strings, return a fully populated
    questions JSON ready to pass directly into POST /grade.

    Each entry will contain:
      • question_number  (1-based index)
      • question_text    (your original question)
      • model_answer     (Gemini-generated marking guide)
      • keywords         (Gemini-suggested key terms, or RAKE fallback)
      • max_marks        (from request, default 10)

    Requires GEMINI_API_KEY in .env.
    """
    if not body.questions:
        return {"error": "No questions provided.", "questions": []}

    prepared = []
    for i, q_text in enumerate(body.questions, start=1):
        max_m = body.max_marks[i - 1] if i - 1 < len(body.max_marks) else 10.0
        print(f"[PREPARE] Q{i}: '{q_text[:80]}' …")
        model_answer, kws = gemini_generate_model_answer(q_text)

        # If Gemini returned no keywords, RAKE the model answer as fallback
        if not kws and model_answer:
            kws = auto_keywords(model_answer)
            print(f"[PREPARE] Q{i}: Gemini gave no keywords — RAKE extracted: {kws}")

        prepared.append({
            "question_number": i,
            "question_text":   q_text,
            "model_answer":    model_answer,
            "keywords":        kws,
            "max_marks":       max_m,
        })
        await asyncio.sleep(0.3)   # stay within Gemini free-tier rate limits

    return {
        "total_questions": len(prepared),
        "questions":       prepared,
        "note": (
            "Pass the 'questions' list directly as the 'questions' form field "
            "in POST /grade."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GRADING ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/grade")
async def grade(
    questions: str       = Form(...),
    pdf_file:  UploadFile= File(...),
    student_id: str      = Form(default="anonymous"),
):
    """
    Grade a student's PDF exam submission.

    Parameters (multipart/form-data)
    ---------------------------------
    questions  : JSON list of Question objects (see schema above)
    pdf_file   : uploaded PDF (typed or handwritten)
    student_id : optional identifier used for authorship-shift tracking

    Returns
    -------
    JSON with per-question breakdown and overall summary.
    """
    # ── Robust JSON parsing ──────────────────────────────────────────────────
    try:
        raw_qs = json.loads(questions)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", questions, re.DOTALL)
        if not match:
            raise ValueError(
                "Could not parse 'questions' field as JSON. "
                "Make sure you paste only the [...] array, not the full /prepare response."
            )
        raw_qs = json.loads(match.group())

    if isinstance(raw_qs, str):
        raw_qs = json.loads(raw_qs)

    if isinstance(raw_qs, dict) and "questions" in raw_qs:
        raw_qs = raw_qs["questions"]

    parsed_qs = []
    for item in raw_qs:
        if isinstance(item, str):
            item = json.loads(item)
        parsed_qs.append(item)

    qs: list[Question] = [Question(**q) for q in parsed_qs]
    # ─────────────────────────────────────────────────────────────────────────
    raw: bytes          = await pdf_file.read()
    text, mode          = extract_pdf(raw)

    print("=" * 60)
    print(f"[GRADE] PDF mode : {mode}  |  chars : {len(text)}")
    print("=" * 60)

    # ── Auto-fill missing model answers and keywords ─────────────────────────
    for q in qs:
        if not q.model_answer.strip():
            print(f"[GRADE] Q{q.question_number}: no model answer — generating via Gemini …")
            q.model_answer, generated_kws = gemini_generate_model_answer(q.question_text)
            if not q.keywords and generated_kws:
                q.keywords = generated_kws
                print(f"[GRADE] Q{q.question_number}: using Gemini keywords: {q.keywords}")

        if not q.keywords and q.model_answer.strip():
            q.keywords = auto_keywords(q.model_answer)
            print(f"[GRADE] Q{q.question_number}: auto-extracted keywords via RAKE: {q.keywords}")
    # ─────────────────────────────────────────────────────────────────────────

    answers  = segment_answers(text, qs)
    profiles = load_profiles()
    results      = []
    all_feedback = []           # collected separately, appended after results

    total_marks_awarded = 0.0
    total_max_marks     = 0.0

    for q in qs:
        ans = answers.get(q.question_number, "")

        # ── Sub-scores ──────────────────────────────────────────────────────
        correctness  = score_correctness(ans, q.model_answer)
        kw_coverage  = score_keyword_coverage(ans, q.keywords)
        wq           = score_writing_quality(ans)
        completeness = score_completeness(ans, q.model_answer)

        # ── Integrity checks ────────────────────────────────────────────────
        plag_result  = await plagiarism(ans, q.question_text)
        plag_score   = plag_result["score"]
        ai_prob      = ai_probability(ans)

        # ── Authorship shift ────────────────────────────────────────────────
        feat  = writing_features(ans)
        shift = authorship_shift(profiles.get(student_id, {}), feat)
        profiles[student_id] = feat        # update running profile

        # ── Final grade ─────────────────────────────────────────────────────
        grade_info = compute_final_grade(
            correctness=correctness,
            keywords=kw_coverage,
            writing=wq["score"],
            completeness=completeness,
            plag_score=plag_score,
            ai_prob=ai_prob,
            max_marks=q.max_marks,
        )

        total_marks_awarded += grade_info["marks_awarded"]
        total_max_marks     += q.max_marks

        verdict = grade_verdict(
            grade_info["penalised_score"], plag_score, ai_prob
        )

        # ── AI Feedback (Gemini-powered) ── collected separately ─────────
        is_wrong = grade_info["penalised_score"] < 60.0
        try:
            fb = generate_ai_feedback(
                question_text   = q.question_text,
                model_answer    = q.model_answer,
                student_answer  = ans,
                correctness     = correctness,
                keyword_coverage= kw_coverage,
                writing_score   = wq["score"],
                completeness    = completeness,
                grade_info      = grade_info,
                is_wrong        = is_wrong,
            )
        except Exception as fb_err:
            print(f"[FEEDBACK] Unexpected error for Q{q.question_number}: {fb_err}")
            fb = _default_feedback()
        all_feedback.append({"question_number": q.question_number, **fb})

        results.append({
            "question_number":    q.question_number,
            "question_text":      q.question_text,
            "answer_preview":     ans[:300] + ("…" if len(ans) > 300 else ""),
            "model_answer_used":  q.model_answer[:200] + ("…" if len(q.model_answer) > 200 else ""),
            "keywords_used":      q.keywords,

            # Sub-scores (0-100 unless noted)
            "sub_scores": {
                "correctness":        correctness,
                "keyword_coverage":   kw_coverage,
                "writing_quality":    wq,
                "completeness":       completeness,
            },

            # Integrity
            "integrity": {
                "plagiarism_score":   plag_score,
                "ai_probability":     round(ai_prob, 3),
                "authorship_shift":   round(shift, 3),
                "matched_sentences":  plag_result["matched_sentences"],
            },

            # Grade
            "grade": {
                **grade_info,
                "verdict": verdict,
            },
        })

        await asyncio.sleep(0.5)   # be kind to SerpApi rate limits

    save_profiles(profiles)

    overall_pct = round(total_marks_awarded / total_max_marks * 100, 2) \
                  if total_max_marks > 0 else 0.0

    final_output = {
        "student_id":           student_id,
        "pdf_mode":             mode,
        "total_questions":      len(qs),
        "total_marks_awarded":  round(total_marks_awarded, 2),
        "total_max_marks":      total_max_marks,
        "overall_percentage":   overall_pct,
        "letter_grade":         _letter_grade(overall_pct),
        "pass_fail_status":     _pass_fail(overall_pct),        # ← NEW
        "results":              results,
        "ai_feedback":          all_feedback,
    }

    # ── Persist result to disk for further analysis ───────────────────────────
    _save_result(final_output)
    # ─────────────────────────────────────────────────────────────────────────

    return final_output


def _letter_grade(pct: float) -> str:
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 60: return "C"
    if pct >= 50: return "D"
    return "F"


def _pass_fail(pct: float) -> str:
    """Return 'PASS' if overall percentage meets or exceeds PASS_THRESHOLD, else 'FAIL'."""
    return "PASS" if pct >= PASS_THRESHOLD else "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "serpapi_key_set": bool(SERPAPI_KEY),
        "models_loaded": all([
            embed_model is not None,
            gpt2_model  is not None,
            openai_pipe is not None,
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS — list and retrieve saved grade outputs
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/results")
def list_results(student_id: str = None):
    """
    List all saved grading results.
    Optionally filter by ?student_id=student_001

    Returns summary of every saved JSON result file.
    """
    if not os.path.exists(RESULTS_DIR):
        return {"results": [], "note": "No results saved yet."}

    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
        reverse=True   # newest first
    )

    summaries = []
    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            if student_id and data.get("student_id") != student_id:
                continue
            summaries.append({
                "file":               fname,
                "student_id":         data.get("student_id"),
                "timestamp":          fname.replace(".json", "").split("_", 1)[-1],
                "letter_grade":       data.get("letter_grade"),
                "pass_fail_status":   data.get("pass_fail_status", ""),   # ← NEW
                "overall_percentage": data.get("overall_percentage"),
                "total_marks_awarded":data.get("total_marks_awarded"),
                "total_max_marks":    data.get("total_max_marks"),
                "pdf_mode":           data.get("pdf_mode"),
            })
        except Exception as e:
            summaries.append({"file": fname, "error": str(e)})

    return {
        "total":   len(summaries),
        "results": summaries,
        "csv_summary": os.path.join(RESULTS_DIR, "summary.csv")
                       if os.path.exists(os.path.join(RESULTS_DIR, "summary.csv"))
                       else "not generated yet",
    }


@app.get("/results/{filename}")
def get_result(filename: str):
    """
    Retrieve the full JSON detail of a specific saved result.
    GET /results/student_001_20250305_143000.json
    """
    if not filename.endswith(".json"):
        filename += ".json"
    fpath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(fpath):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Result file '{filename}' not found.")
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("feedback:app", host="0.0.0.0", port=8000, reload=True)