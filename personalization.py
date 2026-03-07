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
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), "performance_reports")
PLANS_DIR       = os.path.join(os.path.dirname(__file__), "study_plans")

app = FastAPI(title="Personalised Study Plan API")


# ── Helpers ──────────────────────────────────────────────────────────────────


def ollama_generate(prompt: str, temperature: float = 0.4) -> str:
    """Call Ollama's local API and return the generated text."""
    resp = http_requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",
            "options": {
                "temperature":   temperature,
                "num_predict":   16000,
            },
        },
        timeout=360,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from potentially messy LLM output."""
    text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
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
    return json.loads(text[start:])


def load_latest_report(student_id: str) -> dict | None:
    """Load the most recent performance analysis report for a student."""
    pattern = os.path.join(REPORTS_DIR, f"{student_id}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                data["_report_file"] = os.path.basename(fpath)
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[PLAN] Skipping {fpath}: {e}")
    return None


def save_study_plan(student_id: str, plan: dict) -> str:
    """Save the generated study plan to a JSON file and return the file path."""
    os.makedirs(PLANS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{timestamp}.json"
    filepath = os.path.join(PLANS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"[PLAN] Study plan saved → {filepath}")
    return filepath


# ── Core Logic ───────────────────────────────────────────────────────────────


def _build_student_context(report: dict) -> str:
    """Build the shared student performance context used by both prompts."""
    analysis = report.get("analysis", {})

    # ── Topic performance summary ────────────────────────────────────────
    topic_lines = []
    for t in analysis.get("topic_wise_performance", []):
        topic_lines.append(
            f"  • {t['topic']}: {t['performance']} — "
            f"accuracy {t.get('accuracy_percent', '?')}%, "
            f"{t.get('questions_attempted', '?')} questions, "
            f"{t.get('remarks', '')}"
        )
    topics_text = "\n".join(topic_lines) or "  (no topic data)"

    # ── Strengths & weaknesses ───────────────────────────────────────────
    strengths  = "\n".join(f"  • {s}" for s in analysis.get("strengths", []))  or "  (none listed)"
    weaknesses = "\n".join(f"  • {w}" for w in analysis.get("weaknesses", [])) or "  (none listed)"

    # ── Mistake patterns ─────────────────────────────────────────────────
    mistake_lines = []
    for m in analysis.get("mistake_patterns_by_topic", []):
        parts = []
        if m.get("conceptual_mistakes", "None") != "None":
            parts.append(f"conceptual: {m['conceptual_mistakes']}")
        if m.get("calculation_mistakes", "None") != "None":
            parts.append(f"calculation: {m['calculation_mistakes']}")
        if m.get("time_management_issues", "None") != "None":
            parts.append(f"time mgmt: {m['time_management_issues']}")
        if m.get("guessing_behavior", "None") != "None":
            parts.append(f"guessing: {m['guessing_behavior']}")
        if parts:
            mistake_lines.append(f"  • {m['topic']}: {'; '.join(parts)}")
    mistakes_text = "\n".join(mistake_lines) or "  (no notable mistakes)"

    # ── Difficulty analysis ──────────────────────────────────────────────
    diff = analysis.get("difficulty_analysis", {})
    diff_text = (
        f"  Easy  → {diff.get('easy',{}).get('total_questions',0)} questions, "
        f"{diff.get('easy',{}).get('accuracy_percent',0)}% accuracy\n"
        f"  Medium → {diff.get('medium',{}).get('total_questions',0)} questions, "
        f"{diff.get('medium',{}).get('accuracy_percent',0)}% accuracy\n"
        f"  Hard  → {diff.get('hard',{}).get('total_questions',0)} questions, "
        f"{diff.get('hard',{}).get('accuracy_percent',0)}% accuracy"
    )

    # ── Skill scores ─────────────────────────────────────────────────────
    sk = analysis.get("skill_scores", {})
    skill_text = (
        f"  Problem Solving  : {sk.get('problem_solving', '?')}/10\n"
        f"  Concept Clarity  : {sk.get('concept_clarity', '?')}/10\n"
        f"  Accuracy         : {sk.get('accuracy', '?')}/10\n"
        f"  Learning Index   : {sk.get('overall_learning_index', '?')}/10"
    )

    overall = analysis.get("overall_summary", "(no summary)")

    return f"""Student: "{report.get('student_id', '?')}" — {report.get('exams_analysed', '?')} exams analysed.

─── TOPIC-WISE PERFORMANCE ───
{topics_text}

─── STRENGTHS ───
{strengths}

─── WEAKNESSES ───
{weaknesses}

─── MISTAKE PATTERNS ───
{mistakes_text}

─── DIFFICULTY ANALYSIS ───
{diff_text}

─── SKILL SCORES ───
{skill_text}

─── OVERALL SUMMARY ───
{overall}"""


# ── Prompt builders (split into two focused calls) ───────────────────────


def _build_timetable_prompt(context: str) -> str:
    """Prompt for ONLY the 4-week timetable (Call 1)."""
    return f"""You are an expert educational AI tutor and study planner.

{context}

Create a PERSONALISED 4-WEEK (1-MONTH) STUDY TIMETABLE for this student.
The plan should progressively shift focus across the 4 weeks:
  • Week 1 — Foundation & Weak Areas: heavy focus on weakest topics.
  • Week 2 — Practice & Reinforcement: targeted practice and concept deepening.
  • Week 3 — New Material & Revision: introduce next topics while revising.
  • Week 4 — Consolidation & Mock Tests: mock tests, timed practice, final review.

Return ONLY valid JSON with this structure:
{{
  "monthly_timetable": [
    {{
      "week": 1,
      "theme": "Foundation & Weak Areas",
      "days": [
        {{
          "day": "Monday",
          "slots": [
            {{
              "time": "e.g. 6:00 AM – 7:30 AM",
              "topic": "topic name",
              "activity": "Revision" or "Practice Problems" or "New Material" or "Mock Test" or "Light Review",
              "priority": "High" or "Medium" or "Low",
              "notes": "what to do"
            }}
          ]
        }}
      ]
    }}
  ]
}}

RULES:
- Include all 4 weeks, each with 7 days (Monday–Sunday), 2-3 slots per day.
- Week 1: MOST time on WEAK topics. Week 4: mock tests & consolidation.
- Strong topics appear with lower frequency (light review).
- Include varied activities to avoid burnout.
"""


def _build_insights_prompt(context: str) -> str:
    """Prompt for focus areas, recommendations, tips, goals (Call 2)."""
    return f"""You are an expert educational AI tutor and study planner.

{context}

Based on this data, generate personalised study INSIGHTS and RECOMMENDATIONS.

Return ONLY valid JSON with this structure:
{{
  "focus_areas": [
    {{
      "topic": "weak topic name",
      "urgency": "Critical" or "High" or "Medium",
      "current_accuracy": number,
      "target_accuracy": number,
      "recommended_strategy": "specific actionable advice"
    }}
  ],
  "what_to_study_next": [
    {{
      "topic": "next topic or chapter to study",
      "reason": "why this should be studied next",
      "prerequisites": ["prerequisite topics"],
      "estimated_hours": number
    }}
  ],
  "study_tips": [
    {{
      "tip": "specific personalised tip",
      "applies_to": "topic or skill this targets",
      "based_on": "which mistake pattern or weakness this addresses"
    }}
  ],
  "milestone_goals": {{
    "end_of_week_1": ["measurable goal for end of week 1"],
    "end_of_week_2": ["measurable goal for end of week 2"],
    "end_of_week_3": ["measurable goal for end of week 3"],
    "end_of_month": ["measurable goal for end of the month"]
  }}
}}

RULES:
- Focus areas must rank weak topics by urgency with specific strategies.
- Study tips must be SPECIFIC to this student's mistake patterns, not generic.
- what_to_study_next should recommend topics that logically follow studied ones.
- Milestone goals must be concrete and measurable.
- Be constructive, motivating, and educational.
"""


# ── Generation with retries ──────────────────────────────────────────────


def _call_ollama_with_retries(prompt: str, required_keys: list[str], label: str) -> dict:
    """Call Ollama, extract JSON, validate required keys, retry up to 3 times."""
    MAX_RETRIES = 3
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[PLAN] {label} — attempt {attempt}/{MAX_RETRIES} ...")
            raw = ollama_generate(prompt, temperature=0.4)
            data = _extract_json(raw)

            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"[PLAN] {label} attempt {attempt}: missing keys {missing}, retrying...")
                last_error = f"Missing keys: {missing}"
                continue

            print(f"[PLAN] {label} — success.")
            return data

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[PLAN] {label} attempt {attempt} JSON parse error: {e}")
        except http_requests.ConnectionError:
            print("[PLAN] ERROR: Cannot connect to Ollama. Is it running?")
            return {"error": "Cannot connect to Ollama. Make sure it is running (ollama serve)."}
        except Exception as e:
            last_error = e
            print(f"[PLAN] {label} attempt {attempt} ERROR: {e}")

    return {"error": f"{label} failed after {MAX_RETRIES} attempts. Last error: {last_error}"}


# ── API Endpoints ────────────────────────────────────────────────────────────


@app.get("/plan")
async def create_study_plan(
    student_id: str = Query(..., description="Student ID, e.g. student_001"),
):
    """
    GET /plan?student_id=student_001
    Reads the latest performance report for the student and generates
    a personalised study plan using Ollama (local LLM).
    Uses two separate LLM calls: one for the timetable, one for insights.
    """
    report = load_latest_report(student_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No performance report found for '{student_id}' in {REPORTS_DIR}. "
                f"Run the performance analysis first (GET /analyze on port 8001)."
            ),
        )

    print(f"[PLAN] Generating study plan for {student_id} "
          f"(report: {report.get('_report_file', '?')})")

    context = _build_student_context(report)

    # ── Call 1: 4-week timetable ─────────────────────────────────────────
    timetable_prompt = _build_timetable_prompt(context)
    timetable_data = _call_ollama_with_retries(
        timetable_prompt, ["monthly_timetable"], "Timetable"
    )

    # ── Call 2: focus areas, tips, recommendations, goals ────────────────
    insights_prompt = _build_insights_prompt(context)
    insights_data = _call_ollama_with_retries(
        insights_prompt,
        ["focus_areas", "what_to_study_next", "study_tips", "milestone_goals"],
        "Insights",
    )

    # ── Merge both results ───────────────────────────────────────────────
    plan = {**timetable_data, **insights_data}

    # Wrap with metadata
    output = {
        "student_id":       student_id,
        "source_report":    report.get("_report_file", ""),
        "exams_analysed":   report.get("exams_analysed", 0),
        "generated_at":     datetime.now().isoformat(),
        "study_plan":       plan,
    }

    # Save to disk
    saved_path = save_study_plan(student_id, output)
    output["saved_plan_path"] = saved_path

    return JSONResponse(content=output)


@app.get("/plans")
async def list_saved_plans(
    student_id: str = Query(..., description="Student ID, e.g. student_001"),
):
    """List all previously saved study plans for a student."""
    pattern = os.path.join(PLANS_DIR, f"{student_id}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    plans = []
    for fpath in files:
        plans.append({
            "file":      os.path.basename(fpath),
            "size_kb":   round(os.path.getsize(fpath) / 1024, 1),
            "modified":  datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
        })
    return {"student_id": student_id, "total_plans": len(plans), "plans": plans}


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "personalization",
        "llm":     f"ollama/{OLLAMA_MODEL}",
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"[PLAN] Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    uvicorn.run("personalization:app", host="0.0.0.0", port=8002, reload=True)
