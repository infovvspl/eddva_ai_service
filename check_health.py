import requests
import time


def check_ollama_health():
    """Check whether the RunPod Ollama endpoint is reachable and edvav2 is loaded."""
    import os
    ollama_url   = os.getenv("OLLAMA_URL",   "http://213.192.2.90:40077")
    ollama_model = os.getenv("OLLAMA_MODEL", "edvav2")
    try:
        import httpx
        r = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        models = [m["name"] for m in r.json().get("models", [])]
        model_loaded = any(ollama_model in m for m in models)
        return {
            "ollama": "ok" if model_loaded else "model_not_found",
            "endpoint": ollama_url,
            "model": ollama_model,
            "available_models": models,
        }
    except Exception as e:
        return {"ollama": "error", "endpoint": ollama_url, "detail": str(e)}


def check_health():
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        "/",
        "/feedback/health",
        "/notes/health",
        "/content/health",
        "/test/health",
        "/career/health",
        "/personalization/health",
    ]

    # ── Ollama / model check ────────────────────────────────────────────────
    print("\n[Ollama] Checking local LLM service...")
    ollama_status = check_ollama_health()
    status_label = ollama_status["ollama"]
    if status_label == "ok":
        print(f"[Ollama] OK — edvaqwen model is loaded")
    elif status_label == "model_not_found":
        available = ollama_status.get("available_models", [])
        print(f"[Ollama] WARNING — edvaqwen not found. Available: {available}")
        print("         Run: ollama pull edvaqwen")
    else:
        print(f"[Ollama] ERROR — {ollama_status.get('detail', 'unknown error')}")
        print("         Make sure Ollama is running:  ollama serve")

    # ── Django API health checks ────────────────────────────────────────────
    print(f"\n[Django] Waiting for server at {base_url}...")
    for _ in range(30):
        try:
            resp = requests.get(base_url, timeout=5)
            if resp.status_code == 200:
                print("[Django] Server is UP!\n")
                break
        except Exception:
            time.sleep(5)
    else:
        print("[Django] Server timed out.")
        return

    for ep in endpoints:
        try:
            resp = requests.get(base_url + ep, timeout=10)
            print(f"  [{ep}] {resp.status_code} — {resp.json()}")
        except Exception as e:
            print(f"  [{ep}] FAILED: {e}")


if __name__ == "__main__":
    check_health()
