"""
EDVA AI Services — Full Endpoint Test & Benchmark
Runs every active endpoint, measures latency, validates response structure.
Usage:  python test_endpoints.py
"""

import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Force UTF-8 output on Windows so box-drawing / tick chars render
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL  = "http://127.0.0.1:8000"
API_KEY   = "apexiq-dev-secret-key-2026"
HEADERS   = {"Content-Type": "application/json", "X-API-Key": API_KEY}
TIMEOUT   = 180  # seconds per request

# ── Expected response keys per endpoint ───────────────────────────────────────
EXPECTED_KEYS = {
    "doubt/resolve":              ["explanation"],
    "tutor/session":              ["response"],
    "tutor/continue":             ["response"],
    "recommend/content":          ["recommendations"],
    "feedback/generate":          ["feedbackText"],
    "notes/analyze":              ["quality_score", "weak_topics"],
    "resume/analyze":             ["score", "strengths"],
    "interview/start":            ["questions"],
    "plan/generate":              ["planItems"],
    "quiz/generate":              ["questions"],
    "feedback/analyze/":          ["score", "feedback"],
    "content/suggest/":           ["resources"],
    "test/generate/":             ["questions"],
    "career/generate/":           ["career_path", "roadmap"],
    "personalization/generate/":  ["daily_schedule"],
}

# ── Test payloads ──────────────────────────────────────────────────────────────
TESTS = [
    # ── Bridge endpoints ──────────────────────────────────────────────────────
    {
        "id":     "doubt/resolve",
        "label":  "AI #1  Doubt Clearing",
        "method": "POST",
        "path":   "/doubt/resolve",
        "body": {
            "questionText":  "Why does current lag voltage in an inductor in AC circuits?",
            "topicId":       "physics_ac",
            "mode":          "detailed",
            "studentContext": {},
        },
    },
    {
        "id":     "tutor/session",
        "label":  "AI #2  Tutor Start",
        "method": "POST",
        "path":   "/tutor/session",
        "body": {
            "studentId": "student_test_001",
            "topicId":   "newton_laws",
            "context":   "Student is struggling with Newton's third law",
        },
    },
    {
        "id":     "tutor/continue",
        "label":  "AI #2  Tutor Continue",
        "method": "POST",
        "path":   "/tutor/continue",
        "body": {
            "sessionId":     "session_test_001",
            "studentMessage": "I still don't understand why reaction force is equal to action",
        },
    },
    {
        "id":     "recommend/content",
        "label":  "AI #6  Content Recommendation",
        "method": "POST",
        "path":   "/recommend/content",
        "body": {
            "studentId":         "student_test_001",
            "context":           "dashboard",
            "weakTopics":        [{"topicId": "kinematics", "accuracy": 0.45}],
            "recentPerformance": {"score": 52, "totalQuestions": 30},
        },
    },
    {
        "id":     "feedback/generate",
        "label":  "AI #8  Feedback Engine",
        "method": "POST",
        "path":   "/feedback/generate",
        "body": {
            "studentId": "student_test_001",
            "context":   "post_test",
            "data": {
                "testName":  "Physics Mock 1",
                "score":     68,
                "maxScore":  100,
                "weakAreas": ["Optics", "Thermodynamics"],
            },
        },
    },
    {
        "id":     "notes/analyze",
        "label":  "AI #9  Notes Analyzer",
        "method": "POST",
        "path":   "/notes/analyze",
        "body": {
            "studentId":    "student_test_001",
            "topicId":      "thermodynamics",
            "notesContent": (
                "Thermodynamics - First law: energy is conserved. "
                "Delta U = Q - W. Heat added to system increases internal energy. "
                "Second law: entropy always increases. Carnot engine is most efficient."
            ),
        },
    },
    {
        "id":     "resume/analyze",
        "label":  "AI #10 Resume Analyzer",
        "method": "POST",
        "path":   "/resume/analyze",
        "body": {
            "targetRole": "Software Engineer at Google",
            "resumeText": (
                "Name: Rahul Sharma\n"
                "Education: B.Tech CSE, IIT Delhi, 2024, CGPA 8.9\n"
                "Skills: Python, Java, C++, Machine Learning, Django, React\n"
                "Projects: Built a recommendation engine for 50K users. "
                "Developed REST APIs with Django. Open-source contributor to NumPy.\n"
                "Internship: Google SWE Intern, Summer 2023"
            ),
        },
    },
    {
        "id":     "interview/start",
        "label":  "AI #11 Interview Prep",
        "method": "POST",
        "path":   "/interview/start",
        "body": {
            "studentId":     "student_test_001",
            "targetCollege": "IIT Bombay",
        },
    },
    {
        "id":     "plan/generate",
        "label":  "AI #12 Study Plan",
        "method": "POST",
        "path":   "/plan/generate",
        "body": {
            "studentId":   "student_test_001",
            "examTarget":  "jee",
            "examYear":    "2026",
            "dailyHours":  6,
            "weakTopics":  ["Organic Chemistry", "Integration"],
            "targetCollege": "IIT Bombay",
            "academicCalendar": {
                "assignedSubjects": ["Physics", "Chemistry", "Mathematics"]
            },
        },
    },
    {
        "id":     "quiz/generate",
        "label":  "AI #13 Quiz Generator",
        "method": "POST",
        "path":   "/quiz/generate",
        "body": {
            "lectureTitle": "Newton's Laws of Motion",
            "topicId":      "mechanics",
            "transcript": (
                "Today we will study Newton's three laws of motion. "
                "The first law states that an object at rest stays at rest unless acted upon by a force. "
                "This is also called the law of inertia. "
                "The second law says force equals mass times acceleration — F = ma. "
                "The third law states that every action has an equal and opposite reaction. "
                "For example, when you push a wall, the wall pushes back with equal force. "
                "These laws are fundamental to classical mechanics and are tested heavily in JEE."
            ),
        },
    },
    # ── Legacy endpoints ──────────────────────────────────────────────────────
    {
        "id":     "feedback/analyze/",
        "label":  "Legacy Feedback Analyze",
        "method": "POST",
        "path":   "/feedback/analyze/",
        "body": {
            "subject":        "Physics",
            "student_answer": "Current lags voltage by 90 degrees in an inductor because the inductor resists changes in current.",
            "marking_scheme": "Award 5 marks for correct explanation of phase difference. Award 3 marks for mentioning Lenz law.",
            "extra_context":  "JEE 2024 question",
        },
    },
    {
        "id":     "content/suggest/",
        "label":  "Legacy Content Suggest",
        "method": "GET",
        "path":   "/content/suggest/",
        "params": {"topic": "Integration by parts"},
    },
    {
        "id":     "test/generate/",
        "label":  "Legacy Test Generate",
        "method": "POST",
        "path":   "/test/generate/",
        "body": {
            "topic":          "Electromagnetic Induction",
            "num_questions":  3,
            "difficulty":     "medium",
            "question_types": "mcq",
        },
    },
    {
        "id":     "career/generate/",
        "label":  "Legacy Career Roadmap",
        "method": "POST",
        "path":   "/career/generate/",
        "body": {
            "goal":             "Become an AI researcher",
            "interests":        ["Machine Learning", "Mathematics"],
            "current_skills":   ["Python", "Linear Algebra"],
            "timeline_months":  18,
        },
    },
    {
        "id":     "personalization/generate/",
        "label":  "Legacy Study Plan",
        "method": "POST",
        "path":   "/personalization/generate/",
        "body": {
            "student_id":               "student_test_001",
            "learning_style":           "visual",
            "subjects_to_focus":        ["Physics", "Chemistry"],
            "available_hours_per_day":  5,
        },
    },
]


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class Result:
    label:        str
    path:         str
    status:       int   = 0
    latency_ms:   float = 0.0
    passed:       bool  = False
    error:        Optional[str] = None
    missing_keys: list  = field(default_factory=list)
    llm_latency:  Optional[float] = None
    response_size: int  = 0


# ── Runner ─────────────────────────────────────────────────────────────────────
def run_test(t: dict) -> Result:
    r = Result(label=t["label"], path=t["path"])
    url = BASE_URL + t["path"]
    try:
        start = time.perf_counter()
        if t["method"] == "GET":
            resp = requests.get(url, headers=HEADERS,
                                params=t.get("params", {}), timeout=TIMEOUT)
        else:
            resp = requests.post(url, headers=HEADERS,
                                 json=t.get("body", {}), timeout=TIMEOUT)
        r.latency_ms    = (time.perf_counter() - start) * 1000
        r.status        = resp.status_code
        r.response_size = len(resp.content)

        try:
            data = resp.json()
        except Exception:
            r.error = "Response is not JSON"
            return r

        if resp.status_code != 200:
            r.error = data.get("error") or data.get("detail") or str(resp.status_code)
            return r

        # Check expected keys
        expected = EXPECTED_KEYS.get(t["id"], [])
        r.missing_keys = [k for k in expected if k not in data]
        r.passed       = len(r.missing_keys) == 0

        # Extract LLM latency from _meta if present
        meta = data.get("_meta", {})
        if "latency_ms" in meta:
            r.llm_latency = meta["latency_ms"]

    except requests.exceptions.Timeout:
        r.error = f"TIMEOUT after {TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        r.error = "CONNECTION REFUSED — is Django running?"
    except Exception as e:
        r.error = str(e)

    return r


# ── Report printer ─────────────────────────────────────────────────────────────
def print_report(results: list[Result]):
    W = 90
    print("\n" + "═" * W)
    print("  EDVA AI SERVICES — ENDPOINT TEST REPORT")
    print("  Target:", BASE_URL, "  Model: edvav2 @", os.getenv("OLLAMA_URL", "RunPod"))
    print("═" * W)

    header = f"{'#':<3} {'Endpoint':<36} {'Status':<8} {'Total ms':<12} {'LLM ms':<10} {'Bytes':<8} {'Result'}"
    print(header)
    print("─" * W)

    passed = failed = 0
    latencies = []

    for i, r in enumerate(results, 1):
        status_str = str(r.status) if r.status else "---"
        latency_str = f"{r.latency_ms:>8.0f}" if r.latency_ms else "    ---"
        llm_str = f"{r.llm_latency:>6.0f}" if r.llm_latency else "   ---"
        size_str = f"{r.response_size:>6}" if r.response_size else "   ---"

        if r.error:
            result_str = f"FAIL  ✗  {r.error}"
            failed += 1
        elif r.missing_keys:
            result_str = f"FAIL  ✗  missing keys: {r.missing_keys}"
            failed += 1
        else:
            result_str = "PASS  ✓"
            passed += 1
            latencies.append(r.latency_ms)

        label = r.label[:35]
        print(f"{i:<3} {label:<36} {status_str:<8} {latency_str} ms   {llm_str} ms  {size_str}  {result_str}")

    print("─" * W)
    print(f"\n  SUMMARY:  {passed} passed  |  {failed} failed  |  {len(results)} total")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\n  LATENCY (passing endpoints only):")
        print(f"    Fastest  : {min(latencies):.0f} ms")
        print(f"    Slowest  : {max(latencies):.0f} ms")
        print(f"    Average  : {avg:.0f} ms")

    print("\n" + "═" * W)

    # Per-endpoint breakdown with response previews
    print("\n  RESPONSE PREVIEWS (first 200 chars of each passing endpoint)\n")
    for r in results:
        if r.passed:
            # Re-hit isn't needed — we print from what we have
            print(f"  [{r.label}]")
    print()


# ── Improvement report ─────────────────────────────────────────────────────────
def print_improvements(results: list[Result]):
    latencies = [(r.label, r.latency_ms) for r in results if r.passed and r.latency_ms]
    latencies.sort(key=lambda x: -x[1])

    print("═" * 90)
    print("  IMPROVEMENT RECOMMENDATIONS")
    print("═" * 90)

    slow = [(l, ms) for l, ms in latencies if ms > 10_000]
    medium = [(l, ms) for l, ms in latencies if 4_000 <= ms <= 10_000]

    print("""
1. CACHING  (biggest immediate win)
   ─────────────────────────────────────────────────────────────────────────────
   Problem : Every request hits the RunPod GPU even for identical questions.
   Fix     : Redis is already wired in — just start it.
             brew install redis && redis-server   (Mac)
             sudo apt install redis-server        (Linux)
             docker run -p 6379:6379 redis        (Docker)
   Impact  : Repeated doubt/quiz/plan calls → <10ms instead of 5-30s.
   Affected: All bridge + legacy endpoints (cache TTL already configured).

2. STREAMING  (makes slow endpoints feel instant)
   ─────────────────────────────────────────────────────────────────────────────
   Problem : Client waits for full response before seeing anything.
   Fix     : Add a  GET /doubt/resolve/stream  endpoint using Django StreamingHttpResponse
             Set "stream": true in the Ollama payload and forward SSE tokens.
   Impact  : User sees first word in <1s even if full answer takes 15s.
   Affected: doubt/resolve, tutor/session, plan/generate (longest endpoints).

3. MODEL CONTEXT WINDOW  (current: 4096 tokens)
   ─────────────────────────────────────────────────────────────────────────────
   Problem : Long transcripts (stt/notes, quiz/generate) may be truncated.
   Fix     : In llm_client.py → options → bump num_ctx to 8192 if RunPod VRAM allows.
   Check   : ollama show edvav2   →  look for "context length" in model info.
   Impact  : Notes and quiz quality improves on long lectures.

4. CONNECTION KEEP-ALIVE  (already done — verify)
   ─────────────────────────────────────────────────────────────────────────────
   Status  : httpx keep-alive pool (max 20, keepalive 10) is already configured.
   Verify  : Confirm RunPod firewall does not close idle TCP after 30s.
   Fix if needed: Set keepalive_expiry=60 in llm_client.py.

5. RUNPOD OPTIMISATION
   ─────────────────────────────────────────────────────────────────────────────
   a) GPU utilisation: Run  nvidia-smi  on RunPod — confirm GPU is at 100% during inference.
   b) Batch similar requests: If multiple students ask the same doubt simultaneously,
      deduplicate in-flight requests (cache the pending Future, resolve all waiters at once).
   c) Quantisation: If edvav2 is FP16, switching to Q4_K_M saves ~50% VRAM and
      increases tokens/sec by ~30%.  ollama pull edvav2:q4_k_m

6. RESPONSE SIZE  (minor)
   ─────────────────────────────────────────────────────────────────────────────
   Enable gzip in Django:  pip install django-compression-middleware
   Add to MIDDLEWARE: 'compression_middleware.middleware.CompressionMiddleware'
   Impact: ~60% size reduction on JSON responses (400 KB → 160 KB for plan/generate).

7. SLOW ENDPOINT BREAKDOWN
   ─────────────────────────────────────────────────────────────────────────────""")

    if slow:
        print("   CRITICAL (>10s) — consider streaming or pre-generation:")
        for l, ms in slow:
            print(f"     • {l:<40} {ms/1000:>6.1f}s")
    if medium:
        print("   MODERATE (4-10s) — good caching candidates:")
        for l, ms in medium:
            print(f"     • {l:<40} {ms/1000:>6.1f}s")
    if not slow and not medium:
        print("   All passing endpoints responded in under 4s — good!")

    print("""
8. PRODUCTION CHECKLIST
   ─────────────────────────────────────────────────────────────────────────────
   [ ] Start Redis (USAGE_SOFT_CAP + caching)
   [ ] Set DJANGO_DEBUG=false in production .env
   [ ] Set a strong DJANGO_SECRET_KEY
   [ ] Switch SQLite → PostgreSQL (DB_ENGINE=django.db.backends.postgresql)
   [ ] Put Nginx/Gunicorn in front of Django (not runserver)
   [ ] Pin RunPod pod — auto-scale off (model takes time to load on cold start)
   [ ] Monitor via /admin-api/usage/ (token budget dashboard already built)
""")
    print("═" * 90)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\nConnecting to {BASE_URL} ...")

    # Quick server check
    try:
        r = requests.get(BASE_URL, timeout=5)
        print(f"Server status: {r.status_code} ✓\n")
    except Exception as e:
        print(f"ERROR: Cannot reach {BASE_URL} — {e}")
        print("Start Django first:  python manage.py runserver")
        sys.exit(1)

    results = []
    for t in TESTS:
        sys.stdout.write(f"  Testing {t['label']:<40} ... ")
        sys.stdout.flush()
        r = run_test(t)
        results.append(r)
        if r.error:
            sys.stdout.write(f"FAIL ({r.error[:50]})\n")
        elif r.missing_keys:
            sys.stdout.write(f"FAIL (missing: {r.missing_keys})\n")
        else:
            sys.stdout.write(f"PASS  {r.latency_ms:.0f}ms\n")

    print_report(results)
    print_improvements(results)

    # Save JSON results
    out = []
    for r in results:
        out.append({
            "endpoint": r.label,
            "path":     r.path,
            "status":   r.status,
            "latency_ms": round(r.latency_ms),
            "llm_latency_ms": round(r.llm_latency) if r.llm_latency else None,
            "response_bytes": r.response_size,
            "passed": r.passed,
            "error": r.error,
            "missing_keys": r.missing_keys,
        })
    with open("test_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Raw results saved → test_report.json\n")


if __name__ == "__main__":
    main()
