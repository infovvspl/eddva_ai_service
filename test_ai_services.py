"""
Test script for ai_services — tests all endpoints via Django directly.

Usage:
    1. Start Django:  python manage.py runserver 8000
    2. Run tests:     python test_ai_services.py

Tests both:
  - Legacy Django endpoints (feedback, content, test, etc.)
  - NestJS ai-bridge endpoints (doubt, tutor, performance, etc.)
"""

import requests
import json
import sys
import time

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "ask_test_key_for_development_only_1234"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

# Also test NestJS-style auth (Bearer token)
HEADERS_BEARER = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

passed = 0
failed = 0
skipped = 0


def test(name, method, path, data=None, params=None, headers=None, expect_status=200):
    global passed, failed, skipped
    url = f"{BASE_URL}{path}"
    h = HEADERS if headers is None else headers

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {method} {path}")
    print(f"{'='*60}")

    try:
        if method == "GET":
            resp = requests.get(url, params=params, headers=h, timeout=60)
        else:
            resp = requests.post(url, json=data, headers=h, timeout=60)

        status_ok = resp.status_code == expect_status

        if status_ok:
            print(f"  STATUS: {resp.status_code} OK")
            passed += 1
        else:
            print(f"  STATUS: {resp.status_code} UNEXPECTED (expected {expect_status})")
            failed += 1

        # Print response (truncated)
        try:
            body = resp.json()
            preview = json.dumps(body, indent=2)
            if len(preview) > 500:
                preview = preview[:500] + "\n  ... (truncated)"
            print(f"  RESPONSE:\n{preview}")

            # Check for _meta (shows it went through the pipeline)
            if "_meta" in body:
                meta = body["_meta"]
                print(f"\n  _meta: source={meta.get('source')} model={meta.get('model')} "
                      f"tokens={meta.get('tokens')} latency={meta.get('latency_ms')}ms "
                      f"institute={meta.get('institute')}")
        except Exception:
            print(f"  RESPONSE: {resp.text[:300]}")

    except requests.ConnectionError:
        print(f"  CONNECTION ERROR - Is Django running on {BASE_URL}?")
        failed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1


def main():
    global passed, failed

    print("\n" + "#" * 60)
    print("#  AI SERVICES PRODUCTION TEST SUITE")
    print("#" * 60)

    # ── 0. Server check ──────────────────────────────────────────
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"\nServer is UP: {resp.status_code}")
    except requests.ConnectionError:
        print(f"\nERROR: Server not running at {BASE_URL}")
        print("Start it with: python manage.py runserver 8000")
        sys.exit(1)

    # ── 1. Health checks (no auth needed) ────────────────────────
    test("Health: Feedback", "GET", "/feedback/health/", headers={})
    test("Health: Content", "GET", "/content/health/", headers={})
    test("Health: Test", "GET", "/test/health/", headers={})

    # ── 2. Auth test — missing key should 401 ────────────────────
    test("Auth: Missing API key", "GET", "/content/suggest/",
         params={"topic": "Physics"}, headers={}, expect_status=401)

    # ── 3. Auth test — Bearer token (NestJS style) ───────────────
    test("Auth: Bearer token (NestJS style)", "GET", "/content/suggest/",
         params={"topic": "Thermodynamics"}, headers=HEADERS_BEARER)

    # ── 4. Legacy Django endpoints ───────────────────────────────
    test("Legacy: Content Suggest", "GET", "/content/suggest/",
         params={"topic": "Quantum Mechanics"})

    test("Legacy: Feedback Analyze", "POST", "/feedback/analyze/", data={
        "subject": "Physics",
        "student_answer": "Newton's third law states that for every action there is an equal and opposite reaction.",
        "marking_scheme": "Must mention action-reaction pairs and equal magnitude forces.",
    })

    test("Legacy: Generate Test", "POST", "/test/generate/", data={
        "topic": "Organic Chemistry - Alkanes",
        "num_questions": 3,
        "difficulty": "easy",
    })

    test("Legacy: Career Roadmap", "POST", "/career/generate/", data={
        "goal": "IIT Professor",
        "interests": ["Physics", "Research"],
        "timeline_months": 24,
    })

    test("Legacy: Study Plan", "POST", "/personalization/generate/", data={
        "student_id": "student_test_001",
        "learning_style": "visual",
        "subjects_to_focus": ["Physics", "Chemistry"],
        "available_hours_per_day": 4,
    })

    # ── 5. NestJS ai-bridge endpoints ────────────────────────────
    test("Bridge #1: Doubt Resolve", "POST", "/doubt/resolve", data={
        "questionText": "Why does a ball thrown upward come back down?",
        "topicId": "gravity-basics",
        "mode": "detailed",
    })

    test("Bridge #4: Grade Subjective", "POST", "/grade/subjective", data={
        "questionText": "Explain the first law of thermodynamics",
        "studentAnswer": "Energy cannot be created or destroyed, only transformed from one form to another.",
        "expectedAnswer": "The first law states that the total energy of an isolated system is constant. Energy can be transformed but not created or destroyed. dU = Q - W.",
        "maxMarks": 10,
    })

    test("Bridge #5: Engagement Detect", "POST", "/engage/detect", data={
        "studentId": "student_test_001",
        "context": "practice",
        "signals": {
            "rewindCount": 0,
            "pauseCount": 2,
            "answersPerMinute": 1.5,
            "accuracy": 0.65,
            "idleSeconds": 30,
        },
    })

    test("Bridge #6: Content Recommend", "POST", "/recommend/content", data={
        "studentId": "student_test_001",
        "context": "post_test",
        "weakTopics": ["Thermodynamics", "Organic Chemistry"],
    })

    test("Bridge #8: Feedback Generate", "POST", "/feedback/generate", data={
        "studentId": "student_test_001",
        "context": "post_test",
        "data": {"score": 72, "total": 100, "subject": "Physics"},
    })

    test("Bridge #12: Plan Generate", "POST", "/plan/generate", data={
        "studentId": "student_test_001",
        "examTarget": "jee",
        "examYear": "2027",
        "dailyHours": 5,
        "weakTopics": ["Calculus", "Optics"],
    })

    # ── 6. Admin API ─────────────────────────────────────────────
    test("Admin: Usage Dashboard", "GET", "/admin-api/usage/", params={"days": 1})
    test("Admin: Institute Info", "GET", "/admin-api/info/")

    # ── 7. Rate limit info (check _meta shows institute) ────────
    # Already covered in above tests — _meta.institute should show "test-institute"

    # ── Summary ──────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
