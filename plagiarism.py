import os, re, io, json, asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()   # reads .env from the current working directory into os.environ

import numpy as np
import nltk
import torch
import requests
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from transformers import pipeline, GPT2TokenizerFast, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel
import faiss
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────── CONFIG ─────────────────

MIN_WORDS_AI    = 10
PROFILE_FILE    = "profiles.json"
EMBEDDING_DIM   = 384                       # all-MiniLM-L6-v2 output dim
EMBEDDING_MODEL = "D:/hf_cache/minilm"

# ── SerpApi credentials ───────────────────────────────────────────────────────
# 1. Sign up free at https://serpapi.com (100 searches/month free)
# 2. Copy your API key from https://serpapi.com/manage-api-key
# 3. Set as env var before running:
#       Windows:  set SERPAPI_KEY=your_key_here
#    Or hard-code below for quick local testing:
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")  # ← paste key here if needed

embed_tokenizer: AutoTokenizer = None
embed_model:     AutoModel     = None
openai_pipe  = None
chatgpt_pipe = None
gpt2_model   = None
gpt2_tok     = None

_faiss_index: faiss.IndexFlatIP = None

# ───────────────── STARTUP ─────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_tokenizer, embed_model, openai_pipe, chatgpt_pipe
    global gpt2_model, gpt2_tok, _faiss_index

    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("[STARTUP] Loading embedding model...")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    embed_model.eval()

    _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)

    print("[STARTUP] Loading AI-detection pipelines...")
    openai_pipe  = pipeline("text-classification",
                            model="roberta-base-openai-detector")
    chatgpt_pipe = pipeline("text-classification",
                            model="Hello-SimpleAI/chatgpt-detector-roberta")

    print("[STARTUP] Loading GPT-2 perplexity model...")
    gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()

    print("[STARTUP] All models ready ✓")
    yield

app = FastAPI(title="Exam AI & Plagiarism API", lifespan=lifespan)

# ───────────────── SCHEMAS ─────────────────

class Question(BaseModel):
    question_number: int
    question_text:   str

# ───────────────── UTILITIES ─────────────────

STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def preprocess(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def sentences(t: str) -> list:
    return nltk.sent_tokenize(t)

def words(t: str) -> list:
    return re.findall(r"\b\w+\b", t.lower())

# ───────────────── WRITING FEATURES ─────────────────

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

# ───────────────── AUTHORSHIP PROFILE ─────────────────

def load_profiles() -> dict:
    if not os.path.exists(PROFILE_FILE):
        return {}
    return json.load(open(PROFILE_FILE))

def save_profiles(p: dict):
    json.dump(p, open(PROFILE_FILE, "w"), indent=2)

def authorship_shift(old: dict, new: dict) -> float:
    if not old:
        return 0.0
    return sum(abs(old[k] - new[k]) for k in old) / len(old)

# ───────────────── PDF EXTRACTION ─────────────────

def extract_typed(data: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)

def extract_ocr(data: bytes) -> str:
    images = convert_from_bytes(data, dpi=300)
    parts  = []
    for img in images:
        t = pytesseract.image_to_string(img)
        if t.strip():
            parts.append(t)
    return "\n".join(parts)

def extract_pdf(data: bytes) -> tuple[str, str]:
    txt = extract_typed(data)
    if len(txt) > 80:
        return txt, "typed"
    return extract_ocr(data), "handwritten"

# ───────────────── SEGMENT ANSWERS ─────────────────

def segment_answers(text: str, questions: list) -> dict:
    pattern = re.compile(r"(?:^|\n)\s*(\d+)[\.\)]")
    hits    = list(pattern.finditer(text))
    segs: dict = {}
    for i, h in enumerate(hits):
        n     = int(h.group(1))
        start = h.end()
        end   = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        segs[n] = text[start:end].strip()
    return segs

# ───────────────── SERPAPI GOOGLE SEARCH ─────────────────────────────────────

def google_snippets(query: str, n: int = 5) -> list[dict]:
    """
    Fetch Google search results via SerpApi.

    Endpoint : GET https://serpapi.com/search
    Docs     : https://serpapi.com/search-api
    Free tier: 100 searches / month

    Returns a list of dicts:
        { "snippet": str, "link": str, "title": str }

    Pulls from:
      • answer_box      → featured snippet at the top
      • organic_results → standard blue-link results
    """
    if not SERPAPI_KEY:
        print(
            "[SERPAPI] WARNING: SERPAPI_KEY is not set.\n"
            "          Plagiarism score will be 0 until you configure it.\n"
            "          Set env var:  set SERPAPI_KEY=your_key_here\n"
            "          Or hard-code in the CONFIG section at the top of this file."
        )
        return []

    params = {
        "engine":  "google",
        "q":       query[:500],
        "num":     n,
        "api_key": SERPAPI_KEY,
    }

    results: list[dict] = []

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # ── 1. Answer box (featured snippet) ────────────────────────────────
        answer_box = data.get("answer_box", {})
        for field in ("answer", "snippet"):
            text = answer_box.get(field, "").strip()
            if text:
                results.append({
                    "snippet": text,
                    "link":    answer_box.get("link", answer_box.get("displayed_link", "")),
                    "title":   answer_box.get("title", "Featured Snippet"),
                })
                break

        # ── 2. Organic results ───────────────────────────────────────────────
        for item in data.get("organic_results", []):
            snippet = item.get("snippet", "").strip()
            if snippet:
                results.append({
                    "snippet": snippet,
                    "link":    item.get("link", ""),
                    "title":   item.get("title", ""),
                })
            if len(results) >= n:
                break

        print(f"[SERPAPI] '{query[:60]}' → {len(results)} results")

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        body   = e.response.text[:300]
        if status == 401:
            print(f"[SERPAPI] ERROR 401: Invalid API key. Detail: {body}")
        elif status == 429:
            print("[SERPAPI] ERROR 429: Monthly quota exhausted (100/month on free tier).")
        else:
            print(f"[SERPAPI] HTTP error {status}: {body}")
    except requests.exceptions.Timeout:
        print("[SERPAPI] Request timed out after 15 s.")
    except Exception as e:
        print(f"[SERPAPI] FAILED: {e}")

    return results

# ───────────────── EMBEDDINGS ─────────────────

def get_embedding(text: str) -> np.ndarray:
    """
    Encode text with the HuggingFace model loaded at startup.
    Mean-pools last hidden states (masked), then L2-normalises.
    Returns float32 ndarray of shape (EMBEDDING_DIM,).
    """
    tokens = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    with torch.no_grad():
        output = embed_model(**tokens)

    mask = tokens["attention_mask"].unsqueeze(-1).float()
    emb  = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
    emb  = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).numpy().astype("float32")


def cosine_similarity_faiss(a: str, b: str) -> float:
    """Pairwise cosine similarity via shared FAISS IndexFlatIP."""
    global _faiss_index
    emb_a = get_embedding(a)
    emb_b = get_embedding(b)
    _faiss_index.reset()
    _faiss_index.add(np.expand_dims(emb_a, 0))
    distances, _ = _faiss_index.search(np.expand_dims(emb_b, 0), k=1)
    return float(np.clip(distances[0][0], 0.0, 1.0))


def cosine_similarity_faiss_batch(query: str, corpus: list[str]) -> list[float]:
    """
    Cosine similarity between query and every item in corpus —
    one FAISS search instead of N pairwise calls.
    Returns scores aligned to the original corpus order.
    """
    global _faiss_index
    if not corpus:
        return []

    query_emb  = get_embedding(query)
    corpus_emb = np.stack([get_embedding(c) for c in corpus]).astype("float32")

    _faiss_index.reset()
    _faiss_index.add(corpus_emb)

    distances, indices = _faiss_index.search(
        np.expand_dims(query_emb, 0), k=len(corpus)
    )
    scores = np.clip(distances[0], 0.0, 1.0)

    aligned = np.empty(len(corpus), dtype="float32")
    for rank, idx in enumerate(indices[0]):
        aligned[idx] = scores[rank]
    return aligned.tolist()

# ───────────────── SIMILARITY (TF-IDF + FAISS blend) ─────────────────

def similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer(stop_words="english")
    try:
        tf = vec.fit_transform([preprocess(a), preprocess(b)])
        tfidf_score = float(cosine_similarity(tf[0], tf[1])[0][0])
    except ValueError:
        tfidf_score = 0.0
    faiss_score = cosine_similarity_faiss(a, b)
    return (tfidf_score + faiss_score) / 2

# ───────────────── PLAGIARISM ─────────────────

async def plagiarism(answer: str, question: str) -> dict:
    """
    Returns:
        {
            "score": float,           # 0–100 overall plagiarism score
            "matched_sentences": [    # sentences from the answer that matched
                {
                    "sentence":       str,   # the student's sentence
                    "similarity":     float, # 0–100
                    "matched_source": str,   # snippet from the web result
                    "source_title":   str,
                    "source_link":    str,
                }
            ]
        }
    """
    search_results = google_snippets(question + " " + answer[:80])
    if not search_results:
        return {"score": 0.0, "matched_sentences": []}

    valid = [r for r in search_results if len(r["snippet"]) > 20]
    if not valid:
        return {"score": 0.0, "matched_sentences": []}

    # Split the answer into individual sentences for granular matching
    ans_sentences = [s.strip() for s in sentences(answer) if len(s.strip().split()) >= 4]
    if not ans_sentences:
        return {"score": 0.0, "matched_sentences": []}

    snippets_text = [r["snippet"] for r in valid]

    matched: list[dict] = []
    sentence_scores: list[float] = []

    for sent in ans_sentences:
        # FAISS batch: score this sentence against all web snippets at once
        faiss_scores = cosine_similarity_faiss_batch(sent, snippets_text)

        # TF-IDF: score this sentence against all web snippets
        vec = TfidfVectorizer(stop_words="english")
        try:
            tf = vec.fit_transform(
                [preprocess(sent)] + [preprocess(s) for s in snippets_text]
            )
            tfidf_scores = [
                float(cosine_similarity(tf[0], tf[i + 1])[0][0])
                for i in range(len(snippets_text))
            ]
        except ValueError:
            tfidf_scores = [0.0] * len(snippets_text)

        blended      = [(f + t) / 2 for f, t in zip(faiss_scores, tfidf_scores)]
        best_idx     = int(np.argmax(blended))
        best_score   = blended[best_idx]

        sentence_scores.append(best_score)

        # Only report sentences with meaningful similarity (≥ 25%)
        if best_score >= 0.25:
            matched.append({
                "sentence":       sent,
                "similarity":     round(best_score * 100, 2),
                "matched_source": snippets_text[best_idx],
                "source_title":   valid[best_idx]["title"],
                "source_link":    valid[best_idx]["link"],
            })

    # Overall score = mean of all per-sentence scores (not just the max),
    # so a single matching sentence doesn't inflate the whole answer.
    overall = round(float(np.mean(sentence_scores)) * 100, 2) if sentence_scores else 0.0

    # Sort matched sentences by similarity descending
    matched.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"[PLAGIARISM] overall={overall}  matched_sentences={len(matched)}/{len(ans_sentences)}")
    return {"score": overall, "matched_sentences": matched}

# ───────────────── AI DETECTION ─────────────────

def gpt2_prob(text: str) -> float:
    enc = gpt2_tok(text[:1000], return_tensors="pt", truncation=True)
    with torch.no_grad():
        loss = gpt2_model(**enc, labels=enc["input_ids"]).loss.item()
    return max(0.0, min(1.0, (4.5 - loss) / 3))

def classify(pipe, text: str) -> float:
    r     = pipe(text[:1200])[0]
    lbl   = r["label"].upper()
    score = float(r["score"])
    return score if lbl != "REAL" else 1 - score

def ai_probability(text: str) -> float:
    word_count = len(text.split())
    if word_count < MIN_WORDS_AI:
        print(f"[AI] Skipped — {word_count} words (threshold={MIN_WORDS_AI})")
        return 0.0
    s1 = classify(openai_pipe,  text)
    s2 = classify(chatgpt_pipe, text)
    s3 = gpt2_prob(text)
    print(f"[AI] openai={s1:.3f}  chatgpt={s2:.3f}  gpt2={s3:.3f}")
    return float(np.mean([s1, s2, s3]))

# ───────────────── VERDICT ─────────────────

def verdict(plag: float, ai: float) -> str:
    if plag > 50 and ai > 0.6:
        return "HIGH RISK: AI + Plagiarism"
    if plag > 50:
        return "HIGH RISK: Plagiarism"
    if ai > 0.6:
        return "HIGH RISK: AI Generated"
    if plag > 25 or ai > 0.4:
        return "MODERATE RISK"
    return "LOW RISK"

# ───────────────── ENDPOINT ─────────────────

@app.post("/check")
async def check(
    questions: str       = Form(...),
    pdf_file: UploadFile = File(...),
):
    qs = [Question(**q) for q in json.loads(questions)]

    raw        = await pdf_file.read()
    text, mode = extract_pdf(raw)
    answers    = segment_answers(text, qs)
    profiles   = load_profiles()
    student    = "default"
    results    = []

    for q in qs:
        ans      = answers.get(q.question_number, "")
        features = writing_features(ans)
        shift    = authorship_shift(profiles.get(student, {}), features)
        profiles[student] = features

        plag_result = await plagiarism(ans, q.question_text)
        plag        = plag_result["score"]
        ai          = ai_probability(ans)

        results.append({
            "question_number":   q.question_number,
            "answer_preview":    ans[:300],
            "plagiarism_score":  plag,
            "ai_probability":    round(ai, 3),
            "authorship_shift":  round(shift, 3),
            "verdict":           verdict(plag, ai),
            "matched_sentences": plag_result["matched_sentences"],
        })

        await asyncio.sleep(1)

    save_profiles(profiles)
    return {"pdf_mode": mode, "total_questions": len(qs), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("plagiarism:app", host="0.0.0.0", port=8000, reload=True)