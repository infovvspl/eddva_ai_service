# Phase 0 — Reconciliation Report (Coaching ↔ School)

**Goal:** Merge the two forked AI-service codebases into a single, vertical-aware codebase.
**Canonical base:** `AI_Study` (coaching). **Tenancy:** one service, per-request `vertical` discriminator.
**Status:** Analysis only — no behavior changed. Work branch: `feat/unified-verticals`.

Compared:
- **Coaching:** `c:/Users/HP/Desktop/VVSPL/AI_Study`
- **School:** `C:/Users/HP/Desktop/eddva-school/eddva_ai_service_school`

---

## TL;DR — the forks barely diverged

The raw `diff` looked terrifying (e.g. `bridge.py` showed 8468 changed lines on a 4226-line file). **That was an illusion.** Coaching files use **CRLF** line endings, school files use **LF**, so naive `diff` flagged *every line* as changed.

After normalizing line endings + whitespace, the **entire real divergence** across the production (Django) codebase is:

| File | Raw diff lines | **True diff lines** | Nature of difference |
|---|---:|---:|---|
| `core/prompt_templates.py` | 1183 | **602** | Essentially **one prompt** — `DOUBT_SYSTEM`. The other 17 prompts are byte-identical. |
| `core/llm_client.py` | 1000 | **78** | School added a **DeepSeek provider** + minor JSON-mode prompt tweaks. |
| `views/bridge.py` | 8468 | **36** | Model-name choice, a JSON-escaping prompt fix, and scientific-solver routing in the doubt path. |
| `core/model_tier.py` | 128 | **0** | Identical (pure CRLF noise). |
| everything else in `ai_services/` | 100s | **0** | Identical. |

**Conclusion:** This is not a hard merge. It is a small, surgical one. The "heavy fork drift" concern is resolved — there are only **four** real differences to reconcile, and most become *configuration*, not code forks.

---

## The 18 AI features and their prompt divergence

Both repos register the **same 18 features** with the **same prompt-constant names**. Content comparison (whitespace-normalized):

| Feature prompt | Coaching | School | Differs? |
|---|---:|---:|:--:|
| `DOUBT_SYSTEM` | 22,333 ch | 38,331 ch | **YES** ← the only one |
| CAREER_ROADMAP, CONTENT_SUGGEST, FEEDBACK_ANALYZE, FEEDBACK_GENERATE, INTERVIEW, NOTES_ANALYZE, NOTES_GENERATE, PERSONALIZATION, PLAN_GENERATE, QUIZ_GENERATE, RECOMMEND, RESUME, STT_NOTES, SYLLABUS_GENERATE, TEST_GENERATE, TUTOR_CONTINUE, TUTOR (17 prompts) | — | — | identical |

**Implication:** School's "verticalization" is **incomplete today** — only the doubt-solving prompt was customized; the other 17 features behave identically to coaching. The framework we build will let the team fill in school-specific prompts feature-by-feature over time, without touching engine code.

> Note: School's longer `DOUBT_SYSTEM` is **not** simplified for class 1–10. It still references JEE/NEET/NCERT (12× NEET, 9× JEE). It reads as an *improved/expanded* version of the same exam-oriented prompt, not a genuinely school-flavored one. **Decision needed (see Q1).**

---

## The four real differences (full inventory)

### 1. `DOUBT_SYSTEM` prompt — the one true content fork
- **What:** School's doubt prompt is ~16k chars longer.
- **Target design:** `TEMPLATES["doubt_resolve"]` becomes `{ "base": <coaching>, "school": <school> }`. `get_template("doubt_resolve", vertical)` returns the right one, falling back to `base`.
- **Open question:** is school's version actually *better for everyone* (then make it `base`) or genuinely *school-specific* (then keep as `school` override)? — **Q1**.

### 2. LLM provider/model choice — becomes configuration, not a fork
- **Coaching:** Groq-hosted `openai/gpt-oss-120b` (reasoning) + `qwen/qwen3-32b` (math).
- **School:** `deepseek-v4-pro` (reasoning + math), via an OpenAI-compatible DeepSeek client (`DEEPSEEK_KEY`).
- **School added real code:** a `deepseek` branch in `llm_client.complete()` (new provider integration). This is a **genuine capability** coaching lacks.
- **Target design:**
  - Merge the DeepSeek provider branch into the **shared** `llm_client` (gated by `DEEPSEEK_KEY` env — inert if unset).
  - Move model selection into `get_model_for_task(feature, vertical)` + per-vertical `model_overrides`, so coaching→Groq and school→DeepSeek is **config**, not duplicated code.

### 3. Prompt micro-fixes school added — promote to `base` (help both verticals)
- `"ALWAYS double escape backslashes in JSON (e.g. \\frac, \\sqrt) so they parse correctly"` (bridge.py, 3 spots).
- `"START YOUR RESPONSE DIRECTLY WITH '{'"` relocated/added in JSON-mode suffixes (llm_client.py).
- These are objective JSON-reliability fixes → fold into base; benefit coaching too.

### 4. Scientific-solver routing in the doubt path — school improvement
- School's `bridge.resolve_doubt` routes physics/chemistry/math doubts to `app.scientific_solver` first, with a clean LLM fallback + defensive `.get()` access. Coaching's does not.
- **Target design:** adopt school's routing into `base` (it degrades gracefully). The scientific solver itself stays a shared library.

---

## The two-runtime problem (the actual source of maintenance pain)

Each repo ships **two parallel implementations** of the same 12 services:

| Runtime | Entry point | What runs it | Used in prod? |
|---|---|---|:--:|
| **Django** | `ai_study_project.wsgi` → `ai_services/views/bridge.py` | PM2 / `deploy/ecosystem.config.js` (gunicorn) | **YES** |
| **FastAPI** | `main.py` → `app/*.py` | `Dockerfile` (uvicorn) | No (parallel/legacy) |

**`app/` dependency reality (verified):**
- `bridge.py` (Django, prod) imports **only** `app.scientific_solver` (school) — nothing else from `app/`.
- `app/scientific_solver.py` → needs `app/formula_retriever.py` + `core/llm_client`.
- **Keep set from `app/`:** `scientific_solver.py` + `formula_retriever.py` (cohesive "scientific solver" library; promote under `ai_services/` or a `solver/` package).
- **Retire set (Phase 4, archived not deleted):** `main.py` + `app/`{`ai_content, ai_notes, ai_test, career_roadmap, cheating_main, doubt_resolve, feedback, performance_analysis, personalization, translate`} — all reimplemented by `bridge.py`. These are the FastAPI-only files carrying the scary-looking inline-prompt drift, and **none of them serve production traffic.**

Collapsing to the single Django runtime removes ~half the surface area and is what makes "one codebase" actually maintainable.

---

## School-only orphan: `views/evaluate.py`
- A batch question fact-checker (`evaluate_batch`). **Not wired into `urls.py`** in school → currently dead/unreachable.
- Uses an inline CBSE/JEE/NEET prompt (coaching-flavored, not school-specific).
- **Decision:** port as a real, registry-backed, profile-gated feature **or** drop it. Low priority. — **Q2**.

---

## Proposed reconciliation decisions (for the unified `base`)

| # | Item | Decision |
|---|---|---|
| D1 | 17 identical prompts | Keep as single `base` (zero duplication). |
| D2 | `DOUBT_SYSTEM` | `base` (coaching) + `school` override. Final placement pending **Q1**. |
| D3 | DeepSeek provider | Merge into shared `llm_client`, env-gated (`DEEPSEEK_KEY`). |
| D4 | Model selection | `get_model_for_task(feature, vertical)` + per-vertical `model_overrides`. Coaching→Groq, School→DeepSeek as config. |
| D5 | JSON-escaping / "start with {" fixes | Promote to `base`. |
| D6 | Scientific-solver doubt routing | Adopt into `base`. |
| D7 | `scientific_solver` + `formula_retriever` | Promote to a shared package; keep. |
| D8 | FastAPI `main.py` + rest of `app/` | Archive in Phase 4 after parity check. |
| D9 | `evaluate.py` | Pending **Q2**. |

---

## Open questions for you

- **Q1 — `DOUBT_SYSTEM`:** Is school's longer doubt prompt (a) strictly better for *all* verticals → make it the shared `base`; or (b) intended only for school → keep coaching as `base`, school as override? (It still reads exam-oriented, so I lean (a) unless you tell me school needs class 1–10 simplification.)
- **Q2 — `evaluate.py`:** Port it as a proper feature, or drop it?
- **Q3 — verticals to seed now:** Confirm initial set = `coaching`, `school`. Any third on the horizon to design for (e.g. `foundation`, `college`)?

---

## What Phase 1 will do (next, on approval)
1. Add `vertical` to `Institute` model (+ migration), default `coaching` (existing tenants unaffected).
2. Add `core/verticals.py` (`VerticalProfile` + `PROFILES` registry).
3. Extend `TenantAuthMiddleware` → resolve `request.vertical` (precedence: request → institute → `DEFAULT_VERTICAL` env).
4. Make `get_template(feature, vertical)` + `get_model_for_task(feature, vertical)` vertical-aware **with `base` fallback** → byte-identical behavior when no vertical is sent.
5. Add `vertical` to cache key + `UsageLog`.

All additive and backward-compatible: with no `vertical` supplied, the unified service behaves exactly as coaching does today.
