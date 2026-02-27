import os, re, io, json, hashlib, asyncio, logging, textwrap
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

# ── Set cache dir BEFORE importing transformers ────────────────────────────
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "D:/hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]       = "0"
os.environ["HUGGINGFACE_HUB_CACHE"]            = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]               = HF_CACHE_DIR

import numpy as np
import nltk, spacy, torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from transformers import (
    AutoTokenizer, AutoModel,
    GPT2LMHeadModel, GPT2TokenizerFast,
    pipeline,
)
from googlesearch import search as google_search
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.6
MINHASH_THRESHOLD    = 0.65
NUM_PERM             = 128
TYPED_MIN_CHARS      = 50
GOOGLE_SLEEP         = 1.5

DISTILBERT_MODEL   = "distilbert-base-uncased"
OPENAI_DETECTOR    = "roberta-base-openai-detector"
CHATGPT_DETECTOR   = "Hello-SimpleAI/chatgpt-detector-roberta"
GPT2_MODEL         = "gpt2"

# ── Globals ────────────────────────────────────────────────────────────────
nlp                   = None
lsh                   = None
distilbert_tok        = None
distilbert_mdl        = None
tfidf_fallback        = False
openai_pipe           = None
chatgpt_pipe          = None
gpt2_mdl              = None
gpt2_tok              = None
ocr_available         = False


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP — load every model once
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp, lsh, distilbert_tok, distilbert_mdl, tfidf_fallback
    global openai_pipe, chatgpt_pipe, gpt2_mdl, gpt2_tok, ocr_available

    log.info(f"Loading models from {HF_CACHE_DIR} ...")
    device = 0 if torch.cuda.is_available() else -1

    for pkg in ("punkt", "stopwords", "punkt_tab"):
        try: nltk.download(pkg, quiet=True)
        except: pass

    try:
        nlp = spacy.load("en_core_web_sm")
        log.info("spaCy ✓")
    except Exception as e:
        log.error(f"spaCy FAILED: {e} → run: python -m spacy download en_core_web_sm")

    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERM)

    # DistilBERT
    try:
        distilbert_tok = AutoTokenizer.from_pretrained(DISTILBERT_MODEL, cache_dir=HF_CACHE_DIR)
        distilbert_mdl = AutoModel.from_pretrained(DISTILBERT_MODEL, cache_dir=HF_CACHE_DIR)
        distilbert_mdl.eval()
        log.info("DistilBERT ✓")
    except Exception as e:
        log.warning(f"DistilBERT failed ({e}) → TF-IDF fallback")
        tfidf_fallback = True

    # OpenAI RoBERTa
    try:
        openai_pipe = pipeline(
            "text-classification", model=OPENAI_DETECTOR, device=device,
            model_kwargs={"cache_dir": HF_CACHE_DIR},
        )
        log.info("OpenAI RoBERTa ✓")
    except Exception as e:
        log.error(f"OpenAI RoBERTa failed: {e}")

    # ChatGPT RoBERTa
    try:
        chatgpt_pipe = pipeline(
            "text-classification", model=CHATGPT_DETECTOR, device=device,
            model_kwargs={"cache_dir": HF_CACHE_DIR},
        )
        log.info("ChatGPT RoBERTa ✓")
    except Exception as e:
        log.error(f"ChatGPT RoBERTa failed: {e}")

    # GPT-2
    try:
        gpt2_tok = GPT2TokenizerFast.from_pretrained(GPT2_MODEL, cache_dir=HF_CACHE_DIR)
        gpt2_mdl = GPT2LMHeadModel.from_pretrained(GPT2_MODEL, cache_dir=HF_CACHE_DIR)
        gpt2_mdl.eval()
        log.info("GPT-2 ✓")
    except Exception as e:
        log.error(f"GPT-2 failed: {e}")

    # Tesseract OCR
    try:
        import pytesseract
        # Try common Windows install paths automatically
        if os.name == "nt":
            _tess_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\karak\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            ]
            for _p in _tess_paths:
                if os.path.exists(_p):
                    pytesseract.pytesseract.tesseract_cmd = _p
                    log.info(f"Tesseract found at {_p}")
                    break
            else:
                log.warning("Tesseract not found at common paths — trying PATH")
        pytesseract.get_tesseract_version()
        ocr_available = True
        log.info("Tesseract OCR ✓")
    except Exception as e:
        log.warning(f"Tesseract not found ({e}) — handwritten PDF disabled")

    log.info("🚀 API ready!")
    yield
    log.info("Shutdown.")


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Plagiarism & AI Detection API",
    description="One endpoint: POST /check — upload questions JSON + student PDF",
    version="5.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
class ExamQuestion(BaseModel):
    question_number: int           = Field(..., description="1-based question number")
    question_text:   str           = Field(..., description="The question")
    max_marks:       Optional[int] = Field(None)

class SimilarityMatch(BaseModel):
    source_url:        str
    source_title:      str
    snippet:           str
    similarity_score:  float
    similarity_method: str   # "distilbert" | "tfidf"
    minhash_score:     float
    is_plagiarised:    bool

class AIResult(BaseModel):
    provider:       str
    ai_probability: float
    label:          str   # AI | HUMAN | MIXED | UNKNOWN
    details:        Dict[str, Any]

class QuestionResult(BaseModel):
    question_number:    int
    question_text:      str
    max_marks:          Optional[int]
    extracted_answer:   str
    pdf_mode:           str   # typed | ocr_handwritten
    similarity_engine:  str
    plagiarism_score:   float
    plagiarism_matches: List[SimilarityMatch]
    ai_detection:       List[AIResult]
    verdict:            str

class CheckResponse(BaseModel):
    filename:        str
    pdf_mode:        str
    total_questions: int
    results:         List[QuestionResult]
    overall_summary: str


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())

def sentence_split(t: str) -> List[str]:
    try:    return nltk.sent_tokenize(t)
    except: return [s.strip() for s in t.split(".") if s.strip()]

def _shingle(t: str, k=3) -> List[str]:
    w = t.split()
    return [" ".join(w[i:i+k]) for i in range(len(w)-k+1)]

def minhash_sim(a: str, b: str) -> float:
    def mh(t):
        m = MinHash(num_perm=NUM_PERM)
        for s in _shingle(preprocess(t)): m.update(s.encode())
        return m
    return mh(a).jaccard(mh(b))

def ai_label(p: float) -> str:
    return "AI" if p >= 0.70 else ("MIXED" if p >= 0.40 else "HUMAN")

def sim_engine() -> str:
    return "tfidf" if (tfidf_fallback or distilbert_mdl is None) else "distilbert"


# ══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY — DistilBERT → TF-IDF fallback
# ══════════════════════════════════════════════════════════════════════════════
def _mean_pool(emb, mask):
    m = mask.unsqueeze(-1).expand(emb.size()).float()
    return (torch.sum(emb * m, 1) / torch.clamp(m.sum(1), min=1e-9)).detach().numpy()

def _distilbert_sim(a: str, b: str) -> float:
    enc = distilbert_tok([a[:512], b[:512]], padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = distilbert_mdl(**enc)
    embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    return float(sk_cosine([embs[0]], [embs[1]])[0][0])

def _tfidf_sim(a: str, b: str) -> float:
    try:
        v = TfidfVectorizer(stop_words="english", min_df=1)
        t = v.fit_transform([preprocess(a), preprocess(b)])
        return float(sk_cosine(t[0], t[1])[0][0])
    except: return 0.0

def compute_sim(a: str, b: str) -> Tuple[float, str]:
    global tfidf_fallback
    if not tfidf_fallback and distilbert_mdl is not None:
        try:    return _distilbert_sim(a, b), "distilbert"
        except Exception as e:
            log.warning(f"DistilBERT sim failed ({e}) → TF-IDF")
            tfidf_fallback = True
    return _tfidf_sim(a, b), "tfidf"


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION — typed + handwritten OCR
# ══════════════════════════════════════════════════════════════════════════════
def _extract_typed(data: bytes) -> Tuple[str, str]:
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text()
            if t: parts.append(t)
    return "\n\n".join(parts).strip(), "typed"

def _extract_ocr(data: bytes) -> Tuple[str, str]:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        if os.name == "nt":
            _tess_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\karak\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            ]
            for _p in _tess_paths:
                if os.path.exists(_p):
                    pytesseract.pytesseract.tesseract_cmd = _p
                    break
    except ImportError as e:
        raise HTTPException(500, f"OCR packages missing: {e}. Run: pip install pdf2image pytesseract")

    try:
        images = convert_from_bytes(data, dpi=300)
    except Exception as e:
        raise HTTPException(500,
            f"PDF→image conversion failed: {e}\n"
            "Windows: Download Poppler from https://github.com/oschwartz10612/poppler-windows/releases "
            "then add its bin/ folder to your system PATH and restart the terminal."
        )

    parts = []
    for i, img in enumerate(images):
        try:
            t = pytesseract.image_to_string(img, lang="eng").strip()
            if t:
                parts.append(t)
                log.info(f"OCR page {i+1}: {len(t)} chars")
        except Exception as e:
            log.warning(f"OCR failed on page {i+1}: {e}")

    text = "\n\n".join(parts).strip()
    if not text:
        raise HTTPException(422,
            "OCR ran but extracted no text. "
            "Make sure the PDF is a scanned image (not a blank or corrupt file). "
            "Try opening the PDF in a viewer to confirm it has visible content."
        )
    return text, "ocr_handwritten"

def extract_pdf(data: bytes) -> Tuple[str, str]:
    """
    Auto-detect typed vs handwritten.
    1. Try pdfplumber (fast, lossless for digital PDFs).
    2. If text is sparse or empty → run Tesseract OCR regardless of ocr_available flag,
       so we always give a useful error instead of a silent empty result.
    """
    typed_text = ""
    try:
        typed_text, mode = _extract_typed(data)
        if len(typed_text) >= TYPED_MIN_CHARS:
            log.info(f"PDF mode: TYPED ({len(typed_text)} chars)")
            return typed_text, mode
        log.info(f"Typed extraction sparse ({len(typed_text)} chars) → trying OCR")
    except Exception as e:
        log.warning(f"pdfplumber error ({e}) → trying OCR")

    # Always attempt OCR — give a helpful error if Tesseract isn't installed
    if not ocr_available:
        raise HTTPException(422,
            "This PDF appears to be handwritten or scanned (no embedded text found). "
            "Tesseract OCR is required to read it but was not found on startup.\n\n"
            "Fix for Windows:\n"
            "  1. Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  2. Install and ADD to PATH during setup\n"
            "  3. Restart your terminal and re-run the API\n\n"
            "Fix for Linux: sudo apt install tesseract-ocr poppler-utils\n"
            "Fix for macOS: brew install tesseract poppler"
        )
    return _extract_ocr(data)


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER SEGMENTATION — split PDF text into per-question blocks
# ══════════════════════════════════════════════════════════════════════════════
def segment_answers(text: str, questions: List[ExamQuestion]) -> Dict[int, str]:
    """
    Looks for markers like:  1.  Q1.  Q1:  Question 1  Ans 1  Answer 1  (1)
    Falls back to equal splitting if no markers found.
    """
    q_nums  = sorted(q.question_number for q in questions)
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?|Ans(?:wer)?|A(?:ns)?\.?)?\s*"
        r"(?P<num>\d+)\s*[.:\)]\s*",
        re.IGNORECASE,
    )
    hits = list(pattern.finditer(text))
    segs: Dict[int, str] = {}

    if len(hits) >= len(q_nums):
        for i, h in enumerate(hits):
            n = int(h.group("num"))
            if n not in q_nums: continue
            start = h.end()
            end   = hits[i+1].start() if i+1 < len(hits) else len(text)
            segs[n] = text[start:end].strip()
    else:
        # fallback: divide lines equally
        log.warning("No question markers detected — splitting text equally among questions")
        lines = [l for l in text.splitlines() if l.strip()]
        chunk = max(1, len(lines) // len(q_nums))
        for idx, n in enumerate(q_nums):
            segs[n] = "\n".join(lines[idx*chunk:(idx+1)*chunk]).strip()

    return segs


# ══════════════════════════════════════════════════════════════════════════════
#  WEB SEARCH — Google (free, no key)
# ══════════════════════════════════════════════════════════════════════════════
def _search_sync(query: str, n: int) -> List[Dict]:
    results = []
    try:
        for r in google_search(query, num_results=n, advanced=True,
                               lang="en", sleep_interval=GOOGLE_SLEEP):
            results.append({
                "title":   r.title       or "",
                "link":    r.url         or "",
                "snippet": r.description or "",
            })
    except Exception as e:
        log.warning(f"Google search error: {e}")
    return results

async def web_search(query: str, n: int) -> List[Dict]:
    return await asyncio.get_event_loop().run_in_executor(None, _search_sync, query, n)


# ══════════════════════════════════════════════════════════════════════════════
#  PLAGIARISM CHECK
# ══════════════════════════════════════════════════════════════════════════════
async def check_plagiarism(answer: str, n: int, query: str) -> Tuple[float, List[SimilarityMatch]]:
    log.info(f"Google: {query[:70]}…")
    items   = await web_search(query, n)
    matches = []

    for item in items:
        snippet = item.get("snippet", "").strip()
        if len(snippet) < 20: continue
        score, method = compute_sim(answer, snippet)
        mh            = minhash_sim(answer, snippet)
        matches.append(SimilarityMatch(
            source_url       =item.get("link", ""),
            source_title     =item.get("title", ""),
            snippet          =snippet[:300],
            similarity_score =round(score, 4),
            similarity_method=method,
            minhash_score    =round(mh, 4),
            is_plagiarised   =score >= SIMILARITY_THRESHOLD or mh >= MINHASH_THRESHOLD,
        ))

    if not matches: return 0.0, []
    pct = (sum(1 for m in matches if m.is_plagiarised) / len(matches)) * 100
    return round(pct, 2), sorted(matches, key=lambda x: x.similarity_score, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AI DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _classify(pipe, text: str, name: str) -> AIResult:
    if pipe is None:
        return AIResult(provider=name, ai_probability=0.0,
                        label="UNKNOWN", details={"error": "model not loaded"})
    try:
        r         = pipe(text[:2000], truncation=True, max_length=512)[0]
        raw       = r["label"].upper()
        score     = float(r["score"])
        # "Real" → human,  "Fake" / "ChatGPT" / "LABEL_1" → AI
        # "Real"/"Human" = human-written → ai_prob = 1-score
        # "Fake"/"ChatGPT"/"LABEL_1" = AI-written → ai_prob = score
        ai_prob   = score if raw in ("LABEL_1","FAKE","AI","GENERATED","CHATGPT") else 1.0 - score
        return AIResult(provider=name, ai_probability=round(ai_prob,4),
                        label=ai_label(ai_prob),
                        details={"raw_label": r["label"], "raw_score": round(score,4)})
    except Exception as e:
        return AIResult(provider=name, ai_probability=0.0,
                        label="UNKNOWN", details={"error": str(e)})

def _gpt2_perplexity(text: str) -> AIResult:
    name = "Binoculars/GPT-2 Perplexity"
    if gpt2_mdl is None or gpt2_tok is None:
        return AIResult(provider=name, ai_probability=0.0,
                        label="UNKNOWN", details={"error": "GPT-2 not loaded"})
    try:
        enc = gpt2_tok(text[:1500], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = gpt2_mdl(**enc, labels=enc["input_ids"]).loss.item()
        prob = max(0.0, min(1.0, (4.5 - loss) / 3.0))
        return AIResult(provider=name, ai_probability=round(prob,4),
                        label=ai_label(prob),
                        details={"nll_loss": round(loss,4),
                                 "guide": "loss<2 → AI likely | loss>3.5 → Human likely"})
    except Exception as e:
        return AIResult(provider=name, ai_probability=0.0,
                        label="UNKNOWN", details={"error": str(e)})

async def detect_ai(text: str) -> List[AIResult]:
    loop = asyncio.get_event_loop()
    return list(await asyncio.gather(
        loop.run_in_executor(None, _classify, openai_pipe,  text, "OpenAI RoBERTa"),
        loop.run_in_executor(None, _classify, chatgpt_pipe, text, "ChatGPT RoBERTa"),
        loop.run_in_executor(None, _gpt2_perplexity, text),
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  VERDICT
# ══════════════════════════════════════════════════════════════════════════════
def verdict(plag: float, ai: List[AIResult]) -> str:
    probs  = [r.ai_probability for r in ai if r.label != "UNKNOWN"]
    avg_ai = sum(probs) / len(probs) if probs else 0.0
    if   plag >= 50 and avg_ai >= 0.60: return "⚠️ HIGH RISK: Plagiarised + AI-generated"
    elif plag >= 50:                    return "⚠️ HIGH RISK: Likely plagiarised"
    elif avg_ai >= 0.60:                return "⚠️ HIGH RISK: Likely AI-generated"
    elif plag >= 25 or avg_ai >= 0.40:  return "⚡ MODERATE RISK: Some concerns"
    else:                               return "✅ LOW RISK: Appears original and human-written"


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "cache_dir":        HF_CACHE_DIR,
        "similarity":       sim_engine(),
        "distilbert":       distilbert_mdl is not None,
        "openai_roberta":   openai_pipe    is not None,
        "chatgpt_roberta":  chatgpt_pipe   is not None,
        "gpt2":             gpt2_mdl       is not None,
        "ocr":              ocr_available,
        "spacy":            nlp            is not None,
        "gpu":              torch.cuda.is_available(),
    }


@app.post("/check", response_model=CheckResponse)
async def check(
    questions: str = Form(
        ...,
        description=textwrap.dedent("""\
            JSON array of questions. Example:
            [
              {"question_number": 1, "question_text": "Explain photosynthesis.", "max_marks": 5},
              {"question_number": 2, "question_text": "What is Newton's second law?", "max_marks": 3}
            ]
        """),
    ),
    pdf_file: UploadFile = File(
        ...,
        description="Student answer sheet — typed PDF or handwritten/scanned PDF"
    ),
    max_web_results: int  = Form(5,    ge=1,  le=10,  description="Google results per question"),
    check_ai:        bool = Form(True,                description="Run AI detection"),
    min_answer_len:  int  = Form(30,   ge=10,         description="Min chars to run AI detection"),
):
    """
    ## Single endpoint for exam answer checking

    **What it does for each question:**
    1. Parses the questions from the JSON form field
    2. Extracts the student's answer from the uploaded PDF
       - Auto-detects **typed** (pdfplumber) or **handwritten** (Tesseract OCR)
    3. Searches **Google** using the *question text* to find reference web answers
    4. Compares student answer vs web results → **plagiarism score**
    5. Runs **3 local AI detectors** on the student's answer
    6. Returns a verdict per question + overall summary

    ---

    ### questions JSON format
    ```json
    [
      {"question_number": 1, "question_text": "Explain osmosis.", "max_marks": 5},
      {"question_number": 2, "question_text": "Describe the water cycle.", "max_marks": 3}
    ]
    ```

    ### PDF format
    - Typed/digital: any normal PDF
    - Handwritten: scanned PDF (requires Tesseract + Poppler installed)
    - Answers should ideally be labelled as `1.` / `Q1.` / `Answer 1:` etc.
      so the system can match them to questions. If no labels are found,
      the text is split equally among questions.

    ---

    ### Postman setup
    - Method: `POST`
    - URL: `http://localhost:8000/check`
    - Body → **form-data**

    | Key             | Type | Value                        |
    |-----------------|------|------------------------------|
    | questions       | Text | (paste your JSON array)      |
    | pdf_file        | File | (select your PDF)            |
    | max_web_results | Text | 5                            |
    | check_ai        | Text | true                         |
    """

    # ── 1. Parse questions ────────────────────────────────────────────────
    try:
        qs = [ExamQuestion(**q) for q in json.loads(questions)]
    except Exception as e:
        raise HTTPException(422, f"Invalid questions JSON: {e}")
    if not qs:
        raise HTTPException(422, "questions array is empty.")

    # ── 2. Extract PDF ────────────────────────────────────────────────────
    raw = await pdf_file.read()
    if not raw:
        raise HTTPException(422, "Uploaded PDF is empty.")

    full_text, pdf_mode = extract_pdf(raw)
    log.info(f"PDF '{pdf_file.filename}' → {pdf_mode}, {len(full_text)} chars")

    # ── 3. Segment answers ────────────────────────────────────────────────
    answer_map = segment_answers(full_text, qs)

    # ── 4. Process each question ──────────────────────────────────────────
    results: List[QuestionResult] = []

    for q in qs:
        answer = answer_map.get(q.question_number, "").strip()
        if not answer:
            answer = "[No answer found for this question in the PDF]"

        # Build Google query from the question (not the answer)
        # — this finds what the "correct" answer looks like online
        google_q = q.question_text + " " + answer[:120]
        if nlp:
            chunks = list({c.text for c in nlp(q.question_text).noun_chunks})[:5]
            if chunks:
                google_q = " ".join(chunks)

        log.info(f"Q{q.question_number}: {q.question_text[:60]}")

        # Run plagiarism + AI concurrently
        plag_coro = check_plagiarism(answer, max_web_results, google_q)
        ai_coro   = detect_ai(answer) if check_ai and len(answer) >= min_answer_len else None

        gathered          = await asyncio.gather(*([plag_coro] + ([ai_coro] if ai_coro else [])))
        plag_score, plag_matches = gathered[0]
        ai_results        = gathered[1] if ai_coro else []

        results.append(QuestionResult(
            question_number   = q.question_number,
            question_text     = q.question_text,
            max_marks         = q.max_marks,
            extracted_answer  = answer[:1000],
            pdf_mode          = pdf_mode,
            similarity_engine = sim_engine(),
            plagiarism_score  = plag_score,
            plagiarism_matches= plag_matches,
            ai_detection      = ai_results,
            verdict           = verdict(plag_score, ai_results),
        ))

        await asyncio.sleep(0.3)  # small pause between questions

    # ── 5. Summary ────────────────────────────────────────────────────────
    high     = sum(1 for r in results if "HIGH"     in r.verdict)
    moderate = sum(1 for r in results if "MODERATE" in r.verdict)
    avg_plag = sum(r.plagiarism_score for r in results) / len(results)
    modes    = list({r.pdf_mode for r in results})

    return CheckResponse(
        filename        = pdf_file.filename or "unknown.pdf",
        pdf_mode        = modes[0] if len(modes) == 1 else "mixed",
        total_questions = len(qs),
        results         = results,
        overall_summary = (
            f"{len(qs)} question(s) checked from '{pdf_file.filename}' [{pdf_mode}]. "
            f"High-risk: {high}/{len(qs)}.  "
            f"Moderate: {moderate}/{len(qs)}.  "
            f"Avg plagiarism: {round(avg_plag,1)}%."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("plagiarism:app", host="0.0.0.0", port=8000, reload=True)