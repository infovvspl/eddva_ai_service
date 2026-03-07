import os
import json
import re
import glob
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import requests as http_requests

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "grade_results")
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), "performance_reports")

app = FastAPI(title="Student Performance Analysis API")


# ── Helpers ──────────────────────────────────────────────────────────────────


def save_analysis_report(student_id: str, report: dict) -> str:
    """Save the performance analysis report to a JSON file and return the file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{timestamp}.json"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[PERF] Report saved → {filepath}")
    return filepath

def ollama_generate(prompt: str, temperature: float = 0.3) -> str:
    """Call Ollama's local API and return the generated text."""
    resp = http_requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",          # force JSON output mode
            "options": {
                "temperature":   temperature,
                "num_predict":   3000,
            },
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from potentially messy LLM output."""
    # Strip markdown fences
    text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    # Fall back to parsing from start
    return json.loads(text[start:])


def load_student_results(student_id: str) -> list[dict]:
    """Load all JSON result files for a given student, sorted newest-first."""
    pattern = os.path.join(RESULTS_DIR, f"{student_id}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    results = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = os.path.basename(fpath)
                results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[PERF] Skipping {fpath}: {e}")
    return results


def build_analysis_context(all_results: list[dict]) -> str:
    """Flatten all exam results into a concise text summary for Ollama."""
    lines = []
    for exam in all_results:
        lines.append(f"\n=== Exam: {exam.get('_source_file', '?')} ===")
        lines.append(f"Overall: {exam.get('overall_percentage', 0)}%  "
                      f"Grade: {exam.get('letter_grade', '?')}  "
                      f"Marks: {exam.get('total_marks_awarded', 0)}/{exam.get('total_max_marks', 0)}")

        for q in exam.get("results", []):
            qn = q.get("question_number", "?")
            qt = q.get("question_text", "")
            ss = q.get("sub_scores", {})
            gr = q.get("grade", {})
            lines.append(f"\n  Q{qn}: {qt}")
            lines.append(f"    Correctness={ss.get('correctness',0):.1f}  "
                          f"Keywords={ss.get('keyword_coverage',0):.1f}  "
                          f"Completeness={ss.get('completeness',0):.1f}  "
                          f"Writing={ss.get('writing_quality',{}).get('score',0):.1f}")
            lines.append(f"    Marks={gr.get('marks_awarded',0)}/{gr.get('max_marks',0)}  "
                          f"Penalised={gr.get('penalised_score',0):.1f}")
            lines.append(f"    Keywords: {', '.join(q.get('keywords_used', []))}")

        # Include AI feedback if available
        for fb in exam.get("ai_feedback", []):
            qn = fb.get("question_number", "?")
            lines.append(f"\n  [AI Feedback Q{qn}]")
            lines.append(f"    Marking Reason: {fb.get('marking_reason', '')}")
            lines.append(f"    Strong: {fb.get('strong_parts', [])}")
            lines.append(f"    Weak:   {fb.get('weak_parts', [])}")
            mp = fb.get("mistake_patterns", {})
            lines.append(f"    Mistakes: conceptual={mp.get('conceptual_mistakes',0)} "
                          f"calculation={mp.get('calculation_mistakes',0)} "
                          f"time={mp.get('time_management_issues',0)} "
                          f"guessing={mp.get('guessing_behavior',0)}")
            lines.append(f"    Mistake Details: {mp.get('details', '')}")
            da = fb.get("difficulty_analysis", {})
            lines.append(f"    Difficulty: {da.get('ai_assigned_difficulty','?')}  "
                          f"Accuracy@Level: {da.get('student_accuracy_at_level',0)}")
            sk = fb.get("skill_scores", {})
            lines.append(f"    Skills: PS={sk.get('problem_solving',0)} "
                          f"CC={sk.get('concept_clarity',0)} "
                          f"Acc={sk.get('accuracy',0)} "
                          f"LI={sk.get('overall_learning_index',0)}")

    return "\n".join(lines)


def generate_performance_analysis(context: str) -> dict:
    """Call Ollama to produce a structured performance analysis."""

    prompt = f"""You are an expert educational AI analyst. Below is the complete exam performance data
for a student across multiple exam sessions. Analyse ALL the data and produce a comprehensive
performance report.

{context}

Return ONLY valid JSON (no markdown fences, no extra text) with exactly this structure:
{{
  "topic_wise_performance": [
    {{
      "topic": "topic or concept name identified from the questions",
      "performance": "Strong 💪" or "Average ⚠️" or "Weak ❗",
      "accuracy_percent": number 0-100,
      "questions_attempted": number,
      "remarks": "brief note on performance in this topic"
    }}
  ],
  "strengths": ["list of specific things the student is good at"],
  "weaknesses": ["list of specific areas where the student struggles"],
  "mistake_patterns_by_topic": [
    {{
      "topic": "topic or subject name",
      "conceptual_mistakes": "description of conceptual mistakes made in this topic, or 'None'",
      "calculation_mistakes": "description of calculation mistakes made in this topic, or 'None'",
      "time_management_issues": "description of time issues in this topic, or 'None'",
      "guessing_behavior": "description of guessing patterns in this topic, or 'None'"
    }}
  ],
  "difficulty_analysis": {{
    "easy": {{
      "total_questions": number,
      "accuracy_percent": number 0-100
    }},
    "medium": {{
      "total_questions": number,
      "accuracy_percent": number 0-100
    }},
    "hard": {{
      "total_questions": number,
      "accuracy_percent": number 0-100
    }}
  }},
  "skill_scores": {{
    "problem_solving": number 1-10,
    "concept_clarity": number 1-10,
    "accuracy": number 1-10,
    "overall_learning_index": float average of the three scores above
  }},
  "overall_summary": "2-3 sentence summary of the student's overall performance, key strengths, and most critical area to improve"
}}

IMPORTANT:
- Identify topics/concepts from the QUESTION TEXT, not from pre-defined categories.
- For mistake_patterns_by_topic, group mistakes by the TOPIC or SUBJECT they belong to (e.g. "Photosynthesis", "Newton's Laws"), NOT by score.
- Aggregate skill scores across all exams to give an overall score.
- Be constructive, specific, and educational.
- Use the FULL 1-10 range for skill scores — do not default to giving all high scores.
"""

    MAX_RETRIES = 3
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[PERF] Attempt {attempt}/{MAX_RETRIES} ...")
            raw = ollama_generate(prompt, temperature=0.3)
            data = _extract_json(raw)

            # Validate skill scores
            sk = data.get("skill_scores", {})
            for key in ("problem_solving", "concept_clarity", "accuracy"):
                sk[key] = max(1, min(10, int(sk.get(key, 5))))
            sk["overall_learning_index"] = round(
                sum(sk[k] for k in ("problem_solving", "concept_clarity", "accuracy")) / 3, 1
            )
            data["skill_scores"] = sk

            print("[PERF] Performance analysis generated successfully.")
            return data

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[PERF] Attempt {attempt} JSON parse error: {e}")
        except http_requests.ConnectionError:
            print("[PERF] ERROR: Cannot connect to Ollama. Is it running?")
            return {"error": "Cannot connect to Ollama. Make sure it is running (ollama serve)."}
        except Exception as e:
            last_error = e
            print(f"[PERF] Attempt {attempt} ERROR: {e}")

    return {"error": f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"}


# ── API Endpoint ─────────────────────────────────────────────────────────────

@app.get("/analyze")
async def analyze_performance(
    student_id: str = Query(..., description="Student ID to analyse, e.g. student_001"),
    latest_n: Optional[int] = Query(None, description="Only analyse the N most recent exams"),
):
    """
    GET /analyze?student_id=student_001
    Reads all saved results for the student and generates a comprehensive
    performance analysis using Ollama (local LLM).
    """
    all_results = load_student_results(student_id)
    if not all_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for student '{student_id}' in {RESULTS_DIR}"
        )

    if latest_n and latest_n > 0:
        all_results = all_results[:latest_n]

    print(f"[PERF] Analysing {len(all_results)} exam(s) for {student_id}")

    context  = build_analysis_context(all_results)
    analysis = generate_performance_analysis(context)

    # Wrap with metadata
    output = {
        "student_id":     student_id,
        "exams_analysed": len(all_results),
        "exam_files":     [r["_source_file"] for r in all_results],
        "analysis":       analysis,
    }

    # Save report to disk for later use
    saved_path = save_analysis_report(student_id, output)
    output["saved_report_path"] = saved_path

    return JSONResponse(content=output)


@app.get("/students")
async def list_students():
    """List all student IDs that have saved results."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    ids = set()
    for f in files:
        base = os.path.basename(f)
        # Format: student_001_20260306_130852.json  →  extract student_001
        parts = base.split("_")
        if len(parts) >= 2:
            # Rejoin everything before the date portion (8 digits)
            id_parts = []
            for p in parts:
                if len(p) == 8 and p.isdigit():
                    break
                id_parts.append(p)
            if id_parts:
                ids.add("_".join(id_parts))
    return {"students": sorted(ids)}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "performance_analysis", "llm": f"ollama/{OLLAMA_MODEL}"}


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"[PERF] Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    uvicorn.run("performance_analysis:app", host="0.0.0.0", port=8001, reload=True)
