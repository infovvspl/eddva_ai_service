"""
diagnose_models.py  —  run this to find exactly what is failing and why.

    python diagnose_models.py
"""
import sys

print("\n" + "="*65)
print("  PLAGIARISM DETECTOR — DIAGNOSTICS")
print("="*65)
print(f"\nPython: {sys.version}\n")

# ── Helper ────────────────────────────────────────────────────────
def ok(label, val=""):
    print(f"  ✅  {label}" + (f": {val}" if val else ""))

def fail(label, err, fix=""):
    print(f"  ❌  {label}: {err}")
    if fix:
        print(f"       Fix → {fix}")

# ── 1. numpy version (MOST COMMON CAUSE) ─────────────────────────
print("[1] numpy — must be <2.0 for mediapipe + torch compatibility")
try:
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        fail("numpy", f"{np.__version__} — TOO NEW",
             "pip install 'numpy<2.0'  then reinstall torch + mediapipe")
    else:
        ok("numpy", np.__version__)
except ImportError as e:
    fail("numpy", e, "pip install numpy")

# ── 2. torch ──────────────────────────────────────────────────────
print("\n[2] torch")
try:
    import torch
    ok("torch", torch.__version__)
    ok("GPU available", torch.cuda.is_available())
    # Quick tensor test
    x = torch.tensor([1.0, 2.0])
    ok("basic tensor op", x.sum().item())
except Exception as e:
    fail("torch", e, "pip install torch  (CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)")

# ── 3. transformers / tokenizers / huggingface-hub ────────────────
print("\n[3] HuggingFace stack — transformers + tokenizers + huggingface-hub")
try:
    import transformers
    ok("transformers", transformers.__version__)
except Exception as e:
    fail("transformers", e, "pip install transformers==4.44.2")

try:
    import tokenizers
    ok("tokenizers", tokenizers.__version__)
except Exception as e:
    fail("tokenizers", e, "pip install tokenizers==0.19.1")

try:
    import huggingface_hub
    ok("huggingface-hub", huggingface_hub.__version__)
except Exception as e:
    fail("huggingface-hub", e, "pip install huggingface-hub==0.24.7")

try:
    import safetensors
    ok("safetensors", safetensors.__version__)
except Exception as e:
    fail("safetensors", e, "pip install safetensors==0.4.4")

# ── 4. DistilBERT load test (similarity model) ────────────────────
print("\n[4] DistilBERT — similarity engine (primary)")
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    print("       ⏳ Loading distilbert-base-uncased (~250 MB on first run)…")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mdl = AutoModel.from_pretrained("distilbert-base-uncased")
    mdl.eval()
    enc = tok("Hello world", return_tensors="pt", truncation=True, max_length=16)
    with torch.no_grad():
        out = mdl(**enc)
    ok("DistilBERT", f"output shape {out.last_hidden_state.shape}")
except Exception as e:
    fail("DistilBERT", e,
         "pip install transformers==4.44.2 tokenizers==0.19.1  AND check numpy<2.0")

# ── 5. OpenAI RoBERTa AI detector ────────────────────────────────
print("\n[5] OpenAI RoBERTa — AI detector")
try:
    from transformers import pipeline
    print("       ⏳ Loading roberta-base-openai-detector (~500 MB on first run)…")
    pipe = pipeline("text-classification",
                    model="roberta-base-openai-detector", device=-1)
    r = pipe("This was written by a human.", truncation=True, max_length=64)
    ok("OpenAI RoBERTa", r)
except Exception as e:
    fail("OpenAI RoBERTa", e,
         "pip install transformers==4.44.2 safetensors==0.4.4")

# ── 6. ChatGPT RoBERTa AI detector ───────────────────────────────
print("\n[6] ChatGPT RoBERTa — AI detector")
try:
    from transformers import pipeline
    print("       ⏳ Loading Hello-SimpleAI/chatgpt-detector-roberta (~500 MB)…")
    pipe = pipeline("text-classification",
                    model="Hello-SimpleAI/chatgpt-detector-roberta", device=-1)
    r = pipe("This was written by a human.", truncation=True, max_length=64)
    ok("ChatGPT RoBERTa", r)
except Exception as e:
    fail("ChatGPT RoBERTa", e,
         "pip install transformers==4.44.2 safetensors==0.4.4")

# ── 7. GPT-2 perplexity ───────────────────────────────────────────
print("\n[7] GPT-2 — perplexity heuristic (Binoculars)")
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    print("       ⏳ Loading gpt2 (~500 MB on first run)…")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    mdl.eval()
    enc = tok("Hello world test", return_tensors="pt")
    with torch.no_grad():
        loss = mdl(**enc, labels=enc["input_ids"]).loss.item()
    ok("GPT-2", f"NLL loss on test text: {round(loss, 4)}")
except Exception as e:
    fail("GPT-2", e, "pip install transformers==4.44.2")

# ── 8. spaCy ──────────────────────────────────────────────────────
print("\n[8] spaCy")
try:
    import spacy
    ok("spacy", spacy.__version__)
    try:
        nlp = spacy.load("en_core_web_sm")
        ok("en_core_web_sm model", "loaded")
    except OSError:
        fail("en_core_web_sm", "model not downloaded",
             "python -m spacy download en_core_web_sm")
except Exception as e:
    fail("spacy", e, "pip install spacy")

# ── 9. NLTK ───────────────────────────────────────────────────────
print("\n[9] NLTK")
try:
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.tokenize import sent_tokenize
    ok("nltk + punkt", sent_tokenize("Hello. World."))
except Exception as e:
    fail("nltk", e, "pip install nltk  then: python -c \"import nltk; nltk.download('punkt_tab')\"")

# ── 10. faiss + datasketch ────────────────────────────────────────
print("\n[10] faiss + datasketch")
try:
    import faiss
    ok("faiss-cpu", faiss.__version__)
except Exception as e:
    fail("faiss-cpu", e, "pip install faiss-cpu==1.8.0")

try:
    from datasketch import MinHash
    m = MinHash(num_perm=16)
    ok("datasketch MinHash", "ok")
except Exception as e:
    fail("datasketch", e, "pip install datasketch==1.6.5")

# ── 11. PDF + OCR ─────────────────────────────────────────────────
print("\n[11] PDF processing")
try:
    import pdfplumber
    ok("pdfplumber", pdfplumber.__version__)
except Exception as e:
    fail("pdfplumber", e, "pip install pdfplumber==0.11.4")

try:
    import pytesseract
    ver = pytesseract.get_tesseract_version()
    ok("Tesseract OCR", ver)
except Exception as e:
    fail("Tesseract OCR", e,
         "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
         "         Linux: sudo apt install tesseract-ocr\n"
         "         macOS: brew install tesseract")

try:
    from pdf2image import convert_from_bytes
    ok("pdf2image (Poppler)", "available")
except Exception as e:
    fail("pdf2image / Poppler", e,
         "Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
         "         Linux: sudo apt install poppler-utils\n"
         "         macOS: brew install poppler")

try:
    from PIL import Image
    import PIL
    ok("Pillow", PIL.__version__)
except Exception as e:
    fail("Pillow", e, "pip install Pillow==10.4.0")

# ── 12. Google search ─────────────────────────────────────────────
print("\n[12] googlesearch-python")
try:
    from googlesearch import search
    ok("googlesearch-python", "imported ok")
    print("       ⏳ Quick search test (needs internet)…")
    results = list(search("python fastapi tutorial", num_results=1, advanced=True))
    if results:
        ok("search test", results[0].url)
    else:
        print("  ⚠️   Returned no results — may be rate-limited, try again in 30s")
except Exception as e:
    fail("googlesearch", e, "pip install googlesearch-python==1.3.0")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("  Done. Fix any ❌ items above, then run:")
print("  uvicorn plagiarism_detector:app --reload --port 8000")
print("  and call GET /health to verify all models show True.")
print("="*65 + "\n")