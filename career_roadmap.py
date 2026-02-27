from __future__ import annotations

import json
import re
from typing import Any

import httpx
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Career Roadmap Builder",
    description="LLM-powered career roadmap builder using Llama 3.2 via Ollama.",
    version="1.0.0",
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2"

# ─────────────────────────────────────────────────────────────────────────────
# Request Schema
# ─────────────────────────────────────────────────────────────────────────────
class RoadmapRequest(BaseModel):
    interest: str = Field(
        ...,
        description="What career are you interested in? (e.g. 'I want to become a Data Scientist')",
    )
    current_skills: list[str] = Field(
        default=[],
        description="Skills you already have (used to compute readiness %)",
    )
    related_roles_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="How many related roles to discover and build roadmaps for",
    )
    timeline_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Target learning timeline in months",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "interest": "I want to become a Data Scientist",
                "current_skills": ["Python", "SQL"],
                "related_roles_count": 10,
                "timeline_months": 12,
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Helper
# ─────────────────────────────────────────────────────────────────────────────
async def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.15,
) -> str:

    messages: list[dict] = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=600) as client:
        try:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            r.raise_for_status()

            data = r.json()
            content = (
                data.get("message", {})
                .get("content", "")
                .strip()
            )

            # ✅ HARD SAFETY CHECK
            if not content:
                raise HTTPException(
                    status_code=502,
                    detail="LLM returned empty response (Ollama generation failed)."
                )

            return content

        except httpx.ConnectError:
            raise HTTPException(
                503,
                "Ollama is not running. Start it with: ollama serve"
            )

        except Exception as exc:
            raise HTTPException(
                500,
                f"LLM call failed: {exc}"
            )


async def extract_json(text: str, retry_fn=None):
    """
    Ultra-robust JSON extractor for LLM outputs.

    Handles:
    - empty responses
    - truncated JSON
    - extra text
    - partial objects
    - auto-repair retry
    """

    import json, re

    if not text or not text.strip():
        raise ValueError("LLM returned empty response")

    # remove markdown
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # --------------------------------------------------
    # 1️⃣ direct parse
    # --------------------------------------------------
    try:
        return json.loads(text)
    except Exception:
        pass

    # --------------------------------------------------
    # 2️⃣ balanced JSON search
    # --------------------------------------------------
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

    # --------------------------------------------------
    # 3️⃣ partial object recovery
    # --------------------------------------------------
    objs = []
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1

        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i+1]
                try:
                    objs.append(json.loads(candidate))
                except Exception:
                    pass

    if objs:
        return objs

    # --------------------------------------------------
    # 4️⃣ AUTO JSON REPAIR (CRITICAL ADDITION)
    # --------------------------------------------------
    if retry_fn:
        repair_prompt = f"""
Fix this output and return ONLY valid JSON.
Do not explain.

Output:
{text}
"""
        repaired = await retry_fn(repair_prompt)

        try:
            return json.loads(repaired)
        except Exception:
            pass

    # --------------------------------------------------
    raise ValueError(
        f"No valid JSON found in LLM output:\n{text[:500]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────
async def extract_role(user_input: str) -> str:
    system = (
        "You are a career counsellor. Identify the single most relevant job title "
        "the user is describing. Reply ONLY with a JSON object:\n"
        '{"role": "<Canonical Job Title>"}'
    )
    prompt = (
        f'User said: "{user_input}"\n'
        "What is the canonical job title they want? Be specific "
        "(e.g. 'Data Scientist', 'ML Engineer', 'Full Stack Developer', "
        "'Cybersecurity Analyst', 'Cloud Architect', 'Product Manager')."
    )
    raw  = await call_llm(prompt, system, max_tokens=80)
    data = await extract_json(raw, retry_fn=call_llm)
    role = str(data.get("role", "")).strip()
    if not role:
        raise HTTPException(400, f"Could not identify a role from: '{user_input}'")
    return role


async def discover_related_roles(primary_role: str, top_k: int) -> list[dict]:
    system = (
        "You are a career advisor. "
        "Return ONLY a valid JSON array — no prose, no markdown fences."
    )
    prompt = f"""
You are a JSON generator.

Output ONLY valid JSON.
No markdown.
No explanation.

IMPORTANT:
Your response MUST start with '[' and end with ']'.

Schema:
[
  {{
    "role": "Role Name",
    "similarity_score": 0.85,
    "reason": "One sentence explaining the relationship."
  }}
]

Generate exactly {top_k} roles related to "{primary_role}".
"""
    raw  = await call_llm(prompt, system, max_tokens=800)
    data = await extract_json(raw, retry_fn=call_llm)

    # ---- normalize possible LLM formats ----
    if isinstance(data, dict):
        # common LLM mistake: wraps array inside object
        for key in ["related_roles", "roles", "data", "items"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise HTTPException(
            500,
            f"LLM returned invalid related roles format.\nRaw output:\n{raw[:500]}"
        )

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role   = str(item.get("role", "")).strip()
        score  = float(item.get("similarity_score", 0.5))
        reason = str(item.get("reason", "")).strip()
        if role and role.lower() != primary_role.lower():
            results.append({
                "role":             role,
                "similarity_score": round(min(max(score, 0.0), 1.0), 3),
                "reason":           reason,
            })

    return results[:top_k]


async def generate_role_skills(primary_role: str, related_roles: list[str]) -> dict[str, list[str]]:
    all_roles = [primary_role] + related_roles
    system = (
        "You are a senior technical recruiter and curriculum designer. "
        "Return ONLY a valid JSON object — no prose, no markdown fences."
    )
    prompt = f"""
For each career role below, list ALL skills a professional in that role needs.
Include: programming languages, frameworks, tools, concepts, methodologies, and soft skills.

Roles: {json.dumps(all_roles)}

Return format (JSON object, keys = role names, values = arrays of skill strings):
{{
  "Role Name": ["skill1", "skill2", ...],
  ...
}}
"""
    raw  = await call_llm(prompt, system, max_tokens=3000)
    data = await extract_json(raw, retry_fn=call_llm)

    if not isinstance(data, dict):
        raise HTTPException(500, "LLM returned unexpected format for role skills.")

    result: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            result[k.strip()] = [s.strip() for s in v if isinstance(s, str) and s.strip()]

    # Case-insensitive fallback
    if primary_role not in result:
        for k in list(result.keys()):
            if k.lower() == primary_role.lower():
                result[primary_role] = result.pop(k)
                break
        else:
            raise HTTPException(500, f"LLM did not return skills for '{primary_role}'.")

    return result


async def generate_skill_graph(all_skills: list[str]) -> dict[str, list[str]]:
    system = (
        "You are a curriculum designer. "
        "Return ONLY a valid JSON object — no prose, no markdown fences."
    )
    prompt = f"""
Given this exact set of skills, produce a prerequisite dependency graph.
For each skill, list ONLY skills from THIS SAME SET that must be learned BEFORE it.
If a skill has no prerequisites, use an empty array [].
Do NOT invent skills not in the list.

Skills: {json.dumps(sorted(all_skills))}

Return format (JSON object only):
{{
  "skill_name": ["prereq1", "prereq2"],
  ...
}}
"""
    raw  = await call_llm(prompt, system, max_tokens=4096)
    data = await extract_json(raw, retry_fn=call_llm)

    if not isinstance(data, dict):
        raise HTTPException(500, "LLM returned unexpected format for skill graph.")

    skill_set = set(all_skills)
    return {
        k.strip(): [
            p.strip() for p in v
            if isinstance(p, str) and p.strip() in skill_set
        ]
        for k, v in data.items()
        if isinstance(v, list) and k.strip() in skill_set
    }


# ─────────────────────────────────────────────────────────────────────────────
# Roadmap + Plan Builders
# ─────────────────────────────────────────────────────────────────────────────
def build_nx_graph(skill_graph: dict[str, list[str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for skill, prereqs in skill_graph.items():
        G.add_node(skill)
        for prereq in prereqs:
            if prereq in skill_graph:
                G.add_edge(prereq, skill)
    return G


def compute_learning_path(required_skills: list[str], owned: set[str], skill_graph: dict[str, list[str]]) -> list[str]:
    G      = build_nx_graph(skill_graph)
    needed: set[str] = set()
    for skill in required_skills:
        if skill not in owned:
            needed.add(skill)
            if skill in G:
                needed |= nx.ancestors(G, skill) - owned

    sub = G.subgraph(needed)
    try:
        order = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        order = list(needed)

    return [s for s in order if s not in owned]


def make_monthly_plan(path: list[str], months: int) -> list[dict]:
    if not path:
        return []
    per = max(1, months // len(path))
    plan = []
    for i, skill in enumerate(path):
        start = i * per + 1
        end   = min(start + per - 1, months)
        plan.append({"month_start": start, "month_end": end, "skill": skill})
    return plan


def build_single_roadmap(
    role: str,
    required_skills: list[str],
    owned_skills: list[str],
    skill_graph: dict[str, list[str]],
    timeline_months: int,
    similarity_score: float | None = None,
    similarity_reason: str | None = None,
) -> dict:
    owned_set    = set(owned_skills)
    required_set = set(required_skills)
    gap          = [s for s in required_skills if s not in owned_set]
    path         = compute_learning_path(required_skills, owned_set, skill_graph)
    plan         = make_monthly_plan(path, timeline_months)
    readiness    = round(len(owned_set & required_set) / max(len(required_set), 1) * 100, 1)

    rm = {
        "role":               role,
        "readiness_percent":  readiness,
        "skills_you_have":    sorted(owned_set & required_set),
        "skill_gap":          gap,
        "learning_path":      path,
        "monthly_plan":       plan,
        "total_skills":       len(required_skills),
        "skills_to_learn":    len(path),
        "timeline_months":    timeline_months,
    }
    if similarity_score is not None:
        rm["similarity_to_primary"] = similarity_score
    if similarity_reason:
        rm["similarity_reason"] = similarity_reason
    return rm


def compute_skill_comparison(
    primary_role: str,
    related_roles: list[str],
    role_skills: dict[str, list[str]],
) -> dict:
    primary_set  = set(role_skills.get(primary_role, []))
    related_sets = {r: set(role_skills.get(r, [])) for r in related_roles if r in role_skills}

    all_sets   = [primary_set] + list(related_sets.values())
    common_all = set.intersection(*all_sets) if len(all_sets) > 1 else primary_set.copy()
    union_of_related = set.union(*list(related_sets.values())) if related_sets else set()

    pairwise: dict[str, dict] = {}
    for role, r_set in related_sets.items():
        jaccard = len(primary_set & r_set) / max(len(primary_set | r_set), 1)
        pairwise[role] = {
            "jaccard_similarity":             round(jaccard, 3),
            "skills_shared_with_primary":     sorted(primary_set & r_set),
            "extra_skills_for_this_role":     sorted(r_set - primary_set),
            "unique_to_primary_not_here":     sorted(primary_set - r_set),
        }

    return {
        "common_foundation_skills": sorted(common_all),
        "note": "Learn 'common_foundation_skills' first — they apply across all paths.",
        "unique_to_primary_role":   sorted(primary_set - union_of_related),
        "pairwise_comparison":      pairwise,
    }


async def generate_advice(
    user_input: str,
    primary_role: str,
    related_roles: list[str],
    skill_comparison: dict,
    primary_roadmap: dict,
    current_skills: list[str],
) -> str:
    system = (
        "You are an expert career counsellor. "
        "Give structured, motivating, and actionable advice in plain text with numbered sections."
    )
    prompt = f"""
User's interest: "{user_input}"
Primary role matched: {primary_role}
Related roles: {', '.join(related_roles)}
Current skills: {', '.join(current_skills) or 'None listed'}
Readiness for {primary_role}: {primary_roadmap['readiness_percent']}%
Skills still needed: {', '.join(primary_roadmap['skill_gap'][:15]) or 'None — already qualified!'}
Common foundation across all roles: {', '.join(skill_comparison['common_foundation_skills'][:12])}
Skills unique to {primary_role}: {', '.join(skill_comparison['unique_to_primary_role'][:10])}

Write a personalised career report with these 5 sections:
1. ROLE OVERVIEW — What does a {primary_role} actually do day-to-day?
2. RELATED ROLES — Why each related role was suggested and how it differs
3. FOUNDATION SKILLS — Which common skills to learn first and why
4. YOUR STARTING POINT — Based on current skills, where to begin specifically
5. ACTION PLAN — What to do this week, this month, and this quarter
"""
    return await call_llm(prompt, system, max_tokens=2000, temperature=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Main Route
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/career-roadmap",
    summary="Generate a complete career roadmap",
    tags=["Roadmap"],
)
async def career_roadmap(req: RoadmapRequest) -> dict:
    """
    Send your career interest in plain English and get back:
    - Your matched primary role
    - Related roles with similarity scores
    - Skill gap analysis and learning path for each role
    - Monthly plan based on your timeline
    - Common vs unique skill comparison across all roles
    - Personalised career advice

    ⏳ Takes 60–180 seconds (multiple LLM calls). Do not cancel the request.
    """
    # Step 1: Extract canonical role
    primary_role = await extract_role(req.interest)

    # Step 2: Discover related roles
    related_role_data = await discover_related_roles(primary_role, req.related_roles_count)
    related_roles     = [r["role"] for r in related_role_data]

    # Step 3: Generate skill lists per role
    role_skills = await generate_role_skills(primary_role, related_roles)

    all_skills = sorted({
        skill
        for skills_list in role_skills.values()
        for skill in skills_list
    })

    # Step 4: Build skill prerequisite graph
    skill_graph = await generate_skill_graph(all_skills)

    # Step 5: Build roadmaps
    primary_roadmap = build_single_roadmap(
        role=primary_role,
        required_skills=role_skills.get(primary_role, []),
        owned_skills=req.current_skills,
        skill_graph=skill_graph,
        timeline_months=req.timeline_months,
    )

    sim_lookup = {r["role"]: r for r in related_role_data}
    related_roadmaps: list[dict] = []
    for role in related_roles:
        role_required = role_skills.get(role, [])
        if not role_required:
            continue
        meta = sim_lookup.get(role, {})
        rm   = build_single_roadmap(
            role=role,
            required_skills=role_required,
            owned_skills=req.current_skills,
            skill_graph=skill_graph,
            timeline_months=req.timeline_months,
            similarity_score=meta.get("similarity_score"),
            similarity_reason=meta.get("reason"),
        )
        related_roadmaps.append(rm)

    # Step 6: Skill comparison
    roles_present    = [r for r in related_roles if r in role_skills]
    skill_comparison = compute_skill_comparison(primary_role, roles_present, role_skills)

    # Step 7: LLM career advice
    advice = await generate_advice(
        user_input=req.interest,
        primary_role=primary_role,
        related_roles=related_roles,
        skill_comparison=skill_comparison,
        primary_roadmap=primary_roadmap,
        current_skills=req.current_skills,
    )

    return {
        "matched_role":      primary_role,
        "related_roles":     related_role_data,
        "primary_roadmap":   primary_roadmap,
        "related_roadmaps":  related_roadmaps,
        "skill_comparison":  skill_comparison,
        "career_advice":     advice,
    }


@app.get("/health", tags=["Utility"], summary="Health check")
async def health() -> dict:
    ollama_ok     = False
    ollama_models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                ollama_ok     = True
                ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    return {
        "status": "ok",
        "ollama": {
            "reachable":        ollama_ok,
            "base_url":         OLLAMA_BASE_URL,
            "model_in_use":     OLLAMA_MODEL,
            "available_models": ollama_models,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("career_roadmap:app", host="0.0.0.0", port=8000, reload=True)