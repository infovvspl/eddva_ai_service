import os
import json
from groq import Groq
import re
import glob
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import requests as http_requests


# ── Configuration ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PLANS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "study_plans"
)
CONTENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "content_resources"
)

router = APIRouter(prefix="/content", tags=["AI Content"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def groq_generate(prompt: str, system_prompt: str = "") -> str:
    """
    Calls Groq to generate suggestions.
    """
    if not GROQ_API_KEY:
        print("[GROO-ERROR] GROQ_API_KEY missing.")
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(3):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[GROQ-RETRY] Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                print("[GROQ-FATAL] All retries failed.")
                return ""
    return ""


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
                return json.loads(text[start : i + 1])
    return json.loads(text[start:])


def _call_groq_with_retries(prompt: str, required_keys: list[str], label: str) -> dict:
    """Call Groq, extract JSON, validate required keys, retry up to 3×."""
    MAX_RETRIES = 3
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[CONTENT] {label} — attempt {attempt}/{MAX_RETRIES} ...")
            raw = groq_generate(prompt)
            data = _extract_json(raw)

            missing = [k for k in required_keys if k not in data]
            if missing:
                print(
                    f"[CONTENT] {label} attempt {attempt}: missing keys {missing}, retrying..."
                )
                last_error = f"Missing keys: {missing}"
                continue

            print(f"[CONTENT] {label} — success.")
            return data

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[CONTENT] {label} attempt {attempt} JSON parse error: {e}")
        except Exception as e:
            last_error = e
            print(f"[CONTENT] {label} attempt {attempt} ERROR: {e}")

    return {
        "error": f"{label} failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    }


def load_latest_plan(student_id: str) -> dict | None:
    """Load the most recent study plan for a student."""
    pattern = os.path.join(PLANS_DIR, f"{student_id}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                data["_plan_file"] = os.path.basename(fpath)
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[CONTENT] Skipping {fpath}: {e}")
    return None


def save_content_resources(student_id: str, resources: dict) -> str:
    """Save the generated resource recommendations to a JSON file."""
    os.makedirs(CONTENT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{timestamp}.json"
    filepath = os.path.join(CONTENT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(resources, f, indent=2, ensure_ascii=False)
    print(f"[CONTENT] Resources saved → {filepath}")
    return filepath


# ── Topic Extraction ─────────────────────────────────────────────────────────


def extract_topics_from_plan(plan: dict) -> list[str]:
    """Extract all unique topics from the study plan for resource generation."""
    topics = set()
    study_plan = plan.get("study_plan", {})

    # From monthly timetable — every day/slot
    for week in study_plan.get("monthly_timetable", []):
        for day in week.get("days", []):
            for slot in day.get("slots", []):
                topic = slot.get("topic", "").strip()
                if topic and topic.lower() != "mock test":
                    topics.add(topic)

    # From focus areas
    for fa in study_plan.get("focus_areas", []):
        topic = fa.get("topic", "").strip()
        if topic:
            topics.add(topic)

    # From what to study next
    for ns in study_plan.get("what_to_study_next", []):
        topic = ns.get("topic", "").strip()
        if topic:
            topics.add(topic)

    # From study tips
    for tip in study_plan.get("study_tips", []):
        applies = tip.get("applies_to", "").strip()
        if applies:
            topics.add(applies)

    return sorted(topics)


def build_daily_schedule_context(plan: dict) -> str:
    """Build a text summary of the full day-by-day timetable for the LLM."""
    study_plan = plan.get("study_plan", {})
    lines = []

    for week in study_plan.get("monthly_timetable", []):
        lines.append(f"\n── Week {week.get('week', '?')}: {week.get('theme', '')} ──")
        for day in week.get("days", []):
            day_name = day.get("day", "?")
            for slot in day.get("slots", []):
                topic = slot.get("topic", "?")
                activity = slot.get("activity", "?")
                notes = slot.get("notes", "")
                priority = slot.get("priority", "")
                lines.append(
                    f"  {day_name} | {topic} | {activity} | "
                    f"Priority: {priority} | {notes}"
                )
    return "\n".join(lines)


def build_focus_context(plan: dict) -> str:
    """Build text summary of focus areas, study-next, and tips."""
    study_plan = plan.get("study_plan", {})
    lines = []

    # Focus areas
    lines.append("\n── FOCUS AREAS (Weak Topics) ──")
    for fa in study_plan.get("focus_areas", []):
        lines.append(
            f"  • {fa['topic']}  Urgency: {fa.get('urgency','?')}  "
            f"Accuracy: {fa.get('current_accuracy','?')}% → {fa.get('target_accuracy','?')}%  "
            f"Strategy: {fa.get('recommended_strategy', '')}"
        )

    # What to study next
    lines.append("\n── WHAT TO STUDY NEXT ──")
    for ns in study_plan.get("what_to_study_next", []):
        lines.append(
            f"  • {ns['topic']}  Reason: {ns.get('reason','')}  "
            f"Est. hours: {ns.get('estimated_hours','?')}"
        )

    # Study tips
    lines.append("\n── STUDY TIPS ──")
    for tip in study_plan.get("study_tips", []):
        lines.append(f"  • [{tip.get('applies_to','')}] {tip.get('tip','')}")

    return "\n".join(lines)


# ── Prompt Builders ──────────────────────────────────────────────────────────


def _build_topic_resources_prompt(
    topic: str, activity_context: str, focus_context: str
) -> str:
    """Prompt to generate learning resources for one topic."""
    return f"""You are an expert educational content curator. A student needs learning resources
for the topic "{topic}".

Here is the student's study timetable context showing when and how this topic appears:
{activity_context}

Here is additional context about the student's strengths/weaknesses and study strategy:
{focus_context}

Generate a comprehensive set of LEARNING RESOURCES for "{topic}" that the student can use
when studying this topic on each scheduled day.

Return ONLY valid JSON with this structure:
{{
  "topic": "{topic}",
  "resources": {{
    "youtube_videos": [
      {{
        "title": "descriptive video title",
        "url": "https://www.youtube.com/results?search_query=<URL-encoded+search+query>",
        "why": "why this video helps for this topic",
        "difficulty": "Beginner" or "Intermediate" or "Advanced"
      }}
    ],
    "free_courses": [
      {{
        "title": "course or lesson name",
        "platform": "Khan Academy" or "Coursera" or "MIT OCW" or "edX" or "Udemy",
        "url": "direct URL to the course or search URL on the platform",
        "why": "why this course is recommended"
      }}
    ],
    "articles_and_notes": [
      {{
        "title": "article or notes title",
        "url": "URL to the article, Wikipedia page, or educational site",
        "source": "Wikipedia" or "GeeksforGeeks" or "BBC Bitesize" or "CK-12" or other,
        "why": "why this article is useful"
      }}
    ],
    "practice_and_quizzes": [
      {{
        "title": "quiz/practice site name",
        "url": "URL to practice problems or quizzes for this topic",
        "type": "MCQ Quiz" or "Flashcards" or "Worksheet" or "Interactive Exercises",
        "why": "why this practice resource is helpful"
      }}
    ]
  }}
}}

RULES:
- Provide 3-4 items per category (youtube_videos, free_courses, articles_and_notes, practice_and_quizzes).
- YouTube URLs should use search query format: https://www.youtube.com/results?search_query=<topic+keywords>
- For free courses, use REAL platform URLs (e.g. https://www.khanacademy.org/search?referer=%2F&page_search_query=<topic>).
- For articles, prefer well-known educational sites (Wikipedia, BBC Bitesize, CK-12, GeeksforGeeks).
- For practice, include sites like Quizlet, Kahoot, CK-12, or subject-specific quiz sites.
- Tailor difficulty and focus to the student's weakness level in this topic.
- URLs must be properly formatted and realistic.
"""


def _build_daily_resources_prompt(plan: dict) -> str:
    """Prompt to map resources to each day in the weekly plan."""
    study_plan = plan.get("study_plan", {})

    # Build a compact day → topics mapping
    day_topics = []
    for week in study_plan.get("monthly_timetable", []):
        week_num = week.get("week", "?")
        for day in week.get("days", []):
            day_name = day.get("day", "?")
            slots_info = []
            for slot in day.get("slots", []):
                topic = slot.get("topic", "?")
                activity = slot.get("activity", "?")
                notes = slot.get("notes", "")
                slots_info.append(f"{topic} ({activity}: {notes})")
            day_topics.append(f"  Week {week_num} {day_name}: {' | '.join(slots_info)}")

    schedule_text = "\n".join(day_topics)

    return f"""You are an expert educational content curator. Below is a student's complete
4-week study timetable. For EACH day, recommend 2-3 specific resources that directly match
the topic and activity planned for that day.

SCHEDULE:
{schedule_text}

Return ONLY valid JSON with this structure:
{{
  "daily_resources": [
    {{
      "week": 1,
      "day": "Monday",
      "resources": [
        {{
          "topic": "the topic being studied",
          "activity": "the activity type",
          "resource_title": "specific resource name",
          "resource_url": "URL to the resource",
          "resource_type": "Video" or "Article" or "Quiz" or "Course" or "Interactive",
          "notes": "how to use this resource for the day's activity"
        }}
      ]
    }}
  ]
}}

RULES:
- Cover ALL days across ALL 4 weeks.
- Each day should have 2-3 resources matched to its topic and activity.
- For "Revision" days: recommend summary videos or cheat sheets.
- For "Practice Problems" days: recommend quiz sites and problem sets.
- For "New Material" days: recommend introductory videos and articles.
- For "Mock Test" days: recommend full practice tests and timed quiz platforms.
- For "Light Review" days: recommend quick flashcards or short summaries.
- Use realistic URLs (YouTube search URLs, Khan Academy, Wikipedia, Quizlet, etc.).
- Keep resource titles descriptive and specific.
"""


# ── Generation Logic ─────────────────────────────────────────────────────────


def generate_topic_resources(
    topics: list[str], activity_ctx: str, focus_ctx: str
) -> list[dict]:
    """Generate resources for each unique topic using Groq."""
    all_topic_resources = []

    for topic in topics:
        print(f"[CONTENT] Generating resources for topic: {topic}")
        prompt = _build_topic_resources_prompt(topic, activity_ctx, focus_ctx)
        result = _call_groq_with_retries(
            prompt, ["topic", "resources"], f"Resources({topic})"
        )
        if "error" not in result:
            all_topic_resources.append(result)
        else:
            all_topic_resources.append(
                {
                    "topic": topic,
                    "resources": {},
                    "error": result["error"],
                }
            )

    return all_topic_resources


def generate_daily_resources(plan: dict) -> dict:
    """Generate day-by-day resource mapping using Groq."""
    print("[CONTENT] Generating daily resource mapping ...")
    prompt = _build_daily_resources_prompt(plan)
    result = _call_groq_with_retries(prompt, ["daily_resources"], "DailyResources")
    return result


# ── API Endpoints ────────────────────────────────────────────────────────────


@router.get("/resources")
async def create_resource_recommendations(
    student_id: str = Query(..., description="Student ID, e.g. student_001"),
):
    """
    GET /resources?student_id=student_001
    Reads the latest study plan for the student and generates learning
    resource recommendations for every topic and every scheduled day.
    """
    plan = load_latest_plan(student_id)
    if not plan:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No study plan found for '{student_id}' in {PLANS_DIR}. "
                f"Run the personalization service first (GET /plan on port 8002)."
            ),
        )

    print(
        f"[CONTENT] Generating resources for {student_id} "
        f"(plan: {plan.get('_plan_file', '?')})"
    )

    # ── Extract unique topics ────────────────────────────────────────────
    topics = extract_topics_from_plan(plan)
    print(f"[CONTENT] Found {len(topics)} unique topics: {topics}")

    # ── Build context strings ────────────────────────────────────────────
    activity_ctx = build_daily_schedule_context(plan)
    focus_ctx = build_focus_context(plan)

    # ── Call 1: per-topic resources (detailed) ───────────────────────────
    topic_resources = generate_topic_resources(topics, activity_ctx, focus_ctx)

    # ── Call 2: daily resource mapping (day-by-day recommendations) ──────
    daily_resources = generate_daily_resources(plan)

    # ── Merge into final output ──────────────────────────────────────────
    output = {
        "student_id": student_id,
        "source_plan": plan.get("_plan_file", ""),
        "generated_at": datetime.now().isoformat(),
        "topics_covered": topics,
        "topic_resources": topic_resources,
        "daily_resources": daily_resources.get("daily_resources", []),
    }

    # Save to disk
    saved_path = save_content_resources(student_id, output)
    output["saved_resources_path"] = saved_path

    return JSONResponse(content=output)


@router.get("/resources/list")
async def list_saved_resources(
    student_id: str = Query(..., description="Student ID, e.g. student_001"),
):
    """List all previously saved resource recommendation files for a student."""
    pattern = os.path.join(CONTENT_DIR, f"{student_id}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    resources = []
    for fpath in files:
        resources.append(
            {
                "file": os.path.basename(fpath),
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "modified": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
            }
        )
    return {
        "student_id": student_id,
        "total_files": len(resources),
        "resources": resources,
    }


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ai_content",
        "llm": f"groq/{GROQ_MODEL}",
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"[CONTENT] Using Groq model: {GROQ_MODEL}")
    app = FastAPI(title="AI Content API Standalone")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)
