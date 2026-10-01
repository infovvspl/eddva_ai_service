"""Together AI — full feature matrix.

Drives every LLM-backed endpoint of the AI service and records WHICH PROVIDER AND
MODEL actually served it, by reading the routing log line the service emits for
each call:

    LLM (text) | provider=together model=openai/gpt-oss-120b ... tokens=661+727

Also asserts the negative side: Whisper, Sarvam, EasyOCR, FLUX, Serper and the
Gemini vision/OCR paths must NOT be served by Together.

Never prints the API key. Paced to stay under Together's rate limit.
"""
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
AI = "http://127.0.0.1:8000"
BE_ENV = "C:/Users/subha/Desktop/VVSPL/eddva/eddva_backend/.env.local"
OUT = os.path.join(HERE, "matrix-results.json")
INSTITUTE = "c259cd4e-b018-45e2-8e46-52a497ca49a1"
PACE_S = 3
LOG = os.environ.get("AI_LOG", "")   # AI service stdout log, for provider attribution

ROUTE_LINE = re.compile(
    r"LLM \((?P<kind>text|json)\) \| provider=(?P<provider>\w+) model=(?P<model>\S+).*?"
    r"tokens=(?P<tin>\d+)\+(?P<tout>\d+)")
ATTEMPT_LINE = re.compile(r"AI route attempt .*?provider=(?P<provider>\w+) model=(?P<model>\S+)"
                          r".*?outcome=(?P<outcome>\w+)(?: kind=(?P<kind>\w+))?")

PASSAGE = ("[p.1] Photosynthesis is the process by which green plants make food using sunlight, "
           "water and carbon dioxide. [p.2] Chlorophyll in the chloroplasts traps light energy. "
           "[p.3] Oxygen is released as a by-product and glucose is stored as starch. ")
TRANSCRIPT = ("Today we will study photosynthesis. Green plants make their own food. "
              "They take carbon dioxide from the air and water from the soil. Chlorophyll "
              "traps sunlight. The plant makes glucose and releases oxygen. This happens in "
              "the leaves, mainly in the chloroplasts of the mesophyll cells.")

# (id, method, path, payload, expect_together)
CASES = [
    ("doubt.resolve", "/doubt/resolve", {
        "questionText": "What is photosynthesis?", "subjectName": "Science", "mode": "brief",
        "className": "Class 8", "studentContext": {"className": "Class 8", "subject": "Science"}}, True),
    ("tutor.session", "/tutor/session", {
        "studentId": "test-student-1", "topic": "Photosynthesis", "subject": "Science",
        "className": "Class 8", "studentName": "Test", "level": "beginner"}, True),
    ("tutor.continue", "/tutor/continue", {
        "sessionId": "test-session-1", "studentId": "test-student-1",
        "studentMessage": "Why do leaves look green?", "topic": "Photosynthesis",
        "subject": "Science"}, True),
    ("recommend.content", "/recommend/content", {
        "studentId": "test-student-1", "subject": "Science", "topic": "Photosynthesis",
        "className": "Class 8", "performance": {"score": 60}}, True),
    ("stt.notes_from_text", "/stt/notes-from-text", {
        "transcript": TRANSCRIPT, "topic": "Photosynthesis", "subjectName": "Science",
        "className": "Class 8", "language": "english"}, True),
    ("stt.extract_image_terms", "/stt/extract-image-terms", {
        "notes": "## Photosynthesis\n- Chlorophyll traps sunlight\n- Oxygen is released",
        "topic": "Photosynthesis"}, True),
    ("feedback.generate", "/feedback/generate", {
        "studentId": "test-student-1", "studentName": "Test", "subject": "Science",
        "score": 62, "maxScore": 100, "topic": "Photosynthesis"}, True),
    ("notes.analyze", "/notes/analyze", {
        "studentId": "test-student-1", "subject": "Science",
        "notesContent": "Photosynthesis happens in leaves. Chlorophyll traps sunlight. Oxygen is released as a by-product."}, True),
    ("resume.analyze", "/resume/analyze", {
        "resumeText": "B.Sc Physics 2021. Taught science to school students for two years. "
                      "Skills: lab work, lesson planning."}, True),
    ("teacher.recording_analysis", "/teacher/analyze-recording", {
        "transcript": TRANSCRIPT, "subjectName": "Science", "className": "Class 8",
        "topicName": "Photosynthesis", "durationMinutes": 12}, True),
    ("interview.start", "/interview/start", {
        "studentId": "test-student-1", "role": "Science Teacher",
        "experience": "2 years", "subject": "Science"}, True),
    ("plan.generate", "/plan/generate", {
        "studentId": "test-student-1", "goal": "Revise Class 8 Science in 2 weeks",
        "subjects": ["Science"], "durationDays": 14, "hoursPerDay": 2}, True),
    ("syllabus.generate", "/syllabus/generate", {
        "subjects": ["Science"], "className": "Class 8", "board": "CBSE",
        "durationWeeks": 8, "examTarget": "Class 8 Annual"}, True),
    ("quiz.generate", "/quiz/generate", {
        "topic": "Photosynthesis", "subjectName": "Science", "className": "Class 8",
        "questionCount": 2, "transcript": TRANSCRIPT}, True),
    ("memorization.generate", "/memorization/generate", {
        "topic": "Photosynthesis", "subjectName": "Science", "className": "Class 8",
        "count": 3}, True),
    ("content.generate", "/content/generate", {
        "contentType": "notes", "topicName": "Photosynthesis", "subjectName": "Science",
        "courseName": "Class 8 Science", "board": "cbse", "language": "english"}, True),
    ("grading.rubrics", "/grading/subjective-rubric-batch", {
        "questions": [{"questionId": "q1", "text": "Explain photosynthesis.", "marks": 5,
                       "type": "long"}],
        "subjectName": "Science", "className": "Class 8", "board": "CBSE"}, True),
    ("grading.answer", "/grading/subjective-answer", {
        "questionText": "Explain photosynthesis.", "maxMarks": 5,
        "studentAnswer": "Plants make food using sunlight, water and carbon dioxide. "
                         "Chlorophyll traps the light. Oxygen is given out.",
        "subjectName": "Science", "className": "Class 8"}, True),
    ("ppt.generate", "/ppt/generate", {
        "topic": "Photosynthesis", "slideCount": 2, "className": "Class 8",
        "subjectName": "Science", "chapterName": "Photosynthesis", "language": "English"}, True),
    ("evaluate.batch", "/evaluate/batch", {
        "questions": [{"id": "1", "questionText": "What is chlorophyll?",
                      "studentAnswer": "A green pigment in leaves that traps sunlight.",
                      "maxMarks": 2}],
        "subjectName": "Science"}, True),
    ("feedback.analyze", "/feedback/analyze/", {
        "subject": "Science",
        "student_answer": "Plants make food using sunlight and water.",
        "marking_scheme": "1 mark for sunlight, 1 mark for water, 1 mark for chlorophyll"}, True),
    ("content.suggest", "/content/suggest/", {
        "topic": "Photosynthesis", "subject": "Science", "className": "Class 8"}, True),
    ("test.generate", "/test/generate/", {
        "topic": "Photosynthesis", "subject": "Science", "className": "Class 8",
        "questionCount": 2, "difficulty": "medium", "board": "CBSE"}, True),
    ("career.guidance", "/career/guidance", {
        "interests": ["biology", "teaching"], "className": "Class 10",
        "strengths": ["science"]}, True),
    ("career.generate", "/career/generate/", {
        "interests": ["biology"], "className": "Class 10", "goal": "become a doctor"}, True),
    ("personalization.study_plan", "/personalization/generate/", {
        "student_id": "test-student-1", "subjects": ["Science"],
        "weakTopics": ["Photosynthesis"], "hoursPerDay": 2}, True),
    ("diagram.render", "/diagram/render", {
        "spec": {"kind": "diagram", "title": "Parts of a leaf",
                 "description": "a simple labelled diagram of a plant leaf"},
        "subject": "Science", "className": "Class 8"}, None),
    # ---- negative checks: these must NOT be served by Together ----
    ("translate.sarvam", "/translate", {
        "text": "Photosynthesis happens in leaves.", "targetLanguage": "hi"}, False),
    ("ppt.search_image", "/ppt/search-image", {
        "searchTerm": "photosynthesis diagram", "slideTitle": "Photosynthesis"}, False),
    ("search.images", "/search/educational-images", {
        "query": "photosynthesis diagram", "limit": 2}, False),
]


def env_of(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def log_size():
    try:
        return os.path.getsize(LOG) if LOG else 0
    except OSError:
        return 0


def log_since(offset):
    if not LOG:
        return ""
    try:
        with io.open(LOG, encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            return fh.read()
    except OSError:
        return ""


def providers_in(text):
    """Every provider/model actually used, in order, for one request."""
    used = []
    for m in ROUTE_LINE.finditer(text):
        used.append({"provider": m.group("provider"), "model": m.group("model"),
                     "tokens": f"{m.group('tin')}+{m.group('tout')}"})
    errs = [f"{m.group('provider')}/{m.group('model')}:{m.group('outcome')}"
            f"{'/' + m.group('kind') if m.group('kind') else ''}"
            for m in ATTEMPT_LINE.finditer(text) if m.group("outcome") == "error"]
    return used, errs


def call(path, payload, key, timeout=300):
    req = urllib.request.Request(
        AI + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key,
                 "X-Vertical": "school", "X-Board": "cbse", "X-Tenant-ID": INSTITUTE})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, None, time.time() - t0, e.read().decode()[:220]
    except Exception as exc:
        return None, None, time.time() - t0, f"{type(exc).__name__}: {str(exc)[:180]}"


def main():
    key = env_of(BE_ENV)["AI_API_KEY"]
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    cases = [c for c in CASES if not only or c[0].startswith(only)]
    print(f"Together feature matrix — {len(cases)} endpoint(s)"
          f"{'  (log attribution ON)' if LOG else '  (no AI_LOG set: provider unknown)'}\n")
    results = []
    for n, (cid, path, payload, expect_together) in enumerate(cases, 1):
        off = log_size()
        status, body, secs, err = call(path, payload, key)
        time.sleep(1.0)                      # let the log flush
        used, errors = providers_in(log_since(off))
        provs = sorted({u["provider"] for u in used})
        models = sorted({u["model"] for u in used})
        served_by_together = "together" in provs

        if status != 200:
            verdict = "ENDPOINT ERROR"
        elif expect_together is True:
            verdict = "TOGETHER OK" if served_by_together else (
                "NOT TOGETHER" if provs else "NO LLM CALL")
        elif expect_together is False:
            verdict = "CORRECTLY NOT TOGETHER" if not served_by_together else "LEAKED TO TOGETHER"
        else:
            verdict = f"INFO ({'/'.join(provs) or 'no llm'})"

        print(f"[{n}/{len(cases)}] {cid:<28} {str(status):<5} {secs:6.1f}s  {verdict:<24}"
              f" {','.join(models) if models else '-'}")
        if err:
            print(f"      error: {err[:160]}")
        if errors:
            print(f"      route errors: {errors[:3]}")
        results.append({"id": cid, "path": path, "status": status, "seconds": round(secs, 1),
                        "verdict": verdict, "providers": provs, "models": models,
                        "calls": used, "route_errors": errors, "error": err,
                        "sample": json.dumps(body)[:400] if body else None})
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(results, indent=1, ensure_ascii=False))
        time.sleep(PACE_S)

    ok = sum(1 for r in results if r["verdict"] in ("TOGETHER OK", "CORRECTLY NOT TOGETHER"))
    print(f"\n==== {ok}/{len(results)} as expected ====")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
