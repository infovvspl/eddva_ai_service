import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers import (
    AutoTokenizer,
    AutoModel,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    pipeline,
)

CACHE_DIR = "D:/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MARKER = os.path.join(CACHE_DIR, ".models_ready")
FIRST_RUN = not os.path.exists(MARKER)

print("\n" + "="*60)
print("  Downloading HuggingFace models (~1.5 GB total)")
print("  This only runs once — models are cached after download.")
print("="*60 + "\n")


# ── 1. DistilBERT ─────────────────────────────────────────────
print("─"*60)
print("[1/4] DistilBERT  (distilbert-base-uncased)  ~250 MB")
print("─"*60)
try:
    AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    AutoModel.from_pretrained(
        "distilbert-base-uncased",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    print("✅  DistilBERT downloaded successfully\n")
except Exception as e:
    print(f"❌  DistilBERT failed: {e}\n")


# ── 2. OpenAI RoBERTa ─────────────────────────────────────────
print("─"*60)
print("[2/4] OpenAI RoBERTa  (roberta-base-openai-detector)  ~500 MB")
print("─"*60)
try:
    AutoTokenizer.from_pretrained(
        "roberta-base-openai-detector",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    AutoModel.from_pretrained(
        "roberta-base-openai-detector",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    print("✅  OpenAI RoBERTa downloaded successfully\n")
except Exception as e:
    print(f"❌  OpenAI RoBERTa failed: {e}\n")


# ── 3. ChatGPT RoBERTa ────────────────────────────────────────
print("─"*60)
print("[3/4] ChatGPT RoBERTa  (Hello-SimpleAI/chatgpt-detector-roberta)  ~500 MB")
print("─"*60)
try:
    AutoTokenizer.from_pretrained(
        "Hello-SimpleAI/chatgpt-detector-roberta",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    AutoModel.from_pretrained(
        "Hello-SimpleAI/chatgpt-detector-roberta",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    print("✅  ChatGPT RoBERTa downloaded successfully\n")
except Exception as e:
    print(f"❌  ChatGPT RoBERTa failed: {e}\n")


# ── 4. GPT-2 ──────────────────────────────────────────────────
print("─"*60)
print("[4/4] GPT-2  (gpt2)  ~500 MB")
print("─"*60)
try:
    GPT2TokenizerFast.from_pretrained("gpt2",
    cache_dir=CACHE_DIR,
    force_download=FIRST_RUN
    )
    GPT2LMHeadModel.from_pretrained(
        "gpt2",
        cache_dir=CACHE_DIR,
        force_download=FIRST_RUN
    )
    print("✅  GPT-2 downloaded successfully\n")
except Exception as e:
    print(f"❌  GPT-2 failed: {e}\n")


# ── Quick smoke test ──────────────────────────────────────────
print("="*60)
print("  Running quick smoke test on all models...")
print("="*60)

import torch

all_ok = True

try:
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR)
    mdl = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR)
    mdl.eval()
    enc = tok("Hello world", return_tensors="pt", truncation=True, max_length=16)
    with torch.no_grad():
        out = mdl(**enc)
    print(f"✅  DistilBERT  — output shape: {out.last_hidden_state.shape}")
except Exception as e:
    print(f"❌  DistilBERT  — {e}")
    all_ok = False

try:
    pipe = pipeline("text-classification",
                    model="roberta-base-openai-detector",
                    tokenizer="roberta-base-openai-detector", device=-1,
                    model_kwargs={"cache_dir": CACHE_DIR},
                    tokenizer_kwargs={"cache_dir": CACHE_DIR})
    result = pipe("This was written by a human.", truncation=True, max_length=64)
    print(f"✅  OpenAI RoBERTa  — {result}")
except Exception as e:
    print(f"❌  OpenAI RoBERTa  — {e}")
    all_ok = False

try:
    pipe = pipeline("text-classification",
                    model="Hello-SimpleAI/chatgpt-detector-roberta",
                    tokenizer="Hello-SimpleAI/chatgpt-detector-roberta", device=-1,
                    model_kwargs={"cache_dir": CACHE_DIR},
                    tokenizer_kwargs={"cache_dir": CACHE_DIR})
    result = pipe("This was written by a human.", truncation=True, max_length=64)
    print(f"✅  ChatGPT RoBERTa  — {result}")
except Exception as e:
    print(f"❌  ChatGPT RoBERTa  — {e}")
    all_ok = False

try:
    tok = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=CACHE_DIR)
    mdl = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=CACHE_DIR)
    mdl.eval()
    enc = tok("Hello world", return_tensors="pt")
    with torch.no_grad():
        loss = mdl(**enc, labels=enc["input_ids"]).loss.item()
    print(f"✅  GPT-2  — NLL loss on test text: {round(loss, 4)}")
except Exception as e:
    print(f"❌  GPT-2  — {e}")
    all_ok = False

print("\n" + "="*60)
if all_ok and FIRST_RUN:
    open(MARKER, "w").close()
    print("  🎉 All models ready! You can now start the API:")
    print("     uvicorn plagiarism_detector:app --reload --port 8000")
else:
    print("  ⚠️  Some models failed. Check errors above.")
    print("  Make sure you have internet access and ~2 GB free disk space.")
    print("  Then delete the broken cache folder and re-run:")
    print(r"     rmdir /s /q C:\Users\karak\.cache\huggingface\hub")
    print("     python download_models.py")
print("="*60 + "\n")