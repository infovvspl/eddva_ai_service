# Vertical Integration Guide

How the unified AI service serves multiple product verticals (coaching, school,
…) from one codebase, and what callers/ops need to know.

## Concepts

- **Vertical** — a product line with its own academic framing. Seeded:
  `coaching` (JEE/NEET/competitive) and `school` (Class 1–10, CBSE/ICSE/State).
  Defined in `ai_services/core/verticals.py` (`PROFILES`).
- **Canonical base** — coaching. Shared prompts/models live in the base
  registries; a vertical only overrides what genuinely differs.

## How the vertical is chosen (precedence)

Resolved once per request in `TenantAuthMiddleware` and attached as
`request.vertical` / `request.profile`:

1. **`X-Vertical` header** (or `?vertical=` query param) — explicit per-request override
2. **`Institute.vertical`** — the tenant's default (set in Django admin)
3. **`DEFAULT_VERTICAL` env** — deployment default
4. **`coaching`** — hard fallback

Unknown/empty values fall back to the default; the service never errors on a
bad vertical.

> **You do NOT need per-request headers to go live.** Set each tenant's
> `Institute.vertical` to `coaching` or `school` and everything routes correctly.
> `X-Vertical` is only for overriding on a specific call.

## NestJS `ai-bridge` forwarding (implemented)

`AiBridgeService.headers()`/`post()` accept an optional `vertical` that is sent
as the `X-Vertical` header (alongside `Authorization: Bearer` + `X-Tenant-ID`).

- The **school module** (`school/material/school-material.service.ts`) passes
  `'school'` to `generateTopicContent(...)`, so school content generation is
  served with school framing.
- Coaching controllers pass no vertical → no `X-Vertical` → the AI service falls
  back to `Institute.vertical` (coaching by default). Backward compatible.

To verticalize another school AI call, pass `'school'` as the trailing
`vertical` argument to the relevant `AiBridgeService` method (add the param to
that method the same way `generateTopicContent` has it).

## Response metadata

Every AI response includes the resolved vertical:

```json
{ "...": "...", "_meta": { "vertical": "school", "model": "...", "source": "llm" } }
```

The doubt endpoint also reports `"source": "solver"` when answered by the
scientific solver instead of the LLM.

## What currently differs by vertical

| Area | coaching (base) | school |
|---|---|---|
| **16 registry prompts** — tutor, feedback, test, quiz, plan, syllabus, notes, content, career, evaluate, … | JEE/NEET / competitive framing | derived school variant: Classes 1-10, CBSE/ICSE/State board, simple language, board-exam style |
| `doubt` (`bridge._build_solver_system_prompt`) | "CBSE/NEET Subject Matter Expert" | "CBSE/ICSE School Teacher (Classes 1-10)", zero JEE/NEET |
| `content/generate` | exam target defaults to JEE | defaults to Class 10 board rules |
| **Scientific-solver formula KB** | grounded with the JEE/NEET formula sheets in `data/knowledge_base/` | **skipped** — IIT-JEE formulae would push a Class-6 answer above grade level |
| `resume_analyze`, `interview_prep` | available | **403 (gated off)** — a 10-year-old has no résumé and is not prepping for college interviews |
| everything else | shared base prompt | shared base prompt |
| model provider | Groq | Groq (same) |

### How the school prompts are built (important)

School prompts are **derived from the base at import time** by
`ai_services/core/prompts/school.py::schoolify()` — competitive framing is
neutralised and a Classes 1-10 audience block is prepended. They are deliberately
NOT hand-written second copies, because:

* the base prompts embed the exact JSON schema the views parse — a hand-written
  variant that drifted from it would silently break response parsing (a test
  asserts every schema key survives `schoolify()`);
* an edit to a base prompt is inherited by the school variant automatically —
  there is no second copy to forget to update.

To give a feature a school variant: add its name to `SCHOOL_OVERRIDE_FEATURES`.
To gate one off for a vertical: add it to that profile's `disabled_features`.

### The guard that keeps this honest

`tests_school_prompts.py` asserts that **any prompt WITHOUT a school override is
free of competitive framing**. So if someone later adds "for JEE aspirants" to a
shared prompt, CI fails instead of quietly leaking it to school students.

## Adding a new vertical

1. Add a `VerticalProfile` to `PROFILES` in `core/verticals.py`.
2. Add its choice to `Institute.VERTICAL_CHOICES` (+ migration).
3. Add prompt overrides in `VERTICAL_OVERRIDES` / doubt framing
   (`bridge._DOUBT_VERTICAL_FRAMING`) and model overrides in
   `VERTICAL_MODEL_OVERRIDES` — only where it differs from base.

No changes to the engine, cache, middleware, or pipeline are required.

## Ops

- Set a tenant's vertical: Django admin → Institute → **Vertical** field
  (list is filterable by vertical).
- Per-vertical usage/billing: `GET /admin-api/usage/` returns
  `vertical_breakdown`; `UsageLog` is filterable by `vertical` in admin.
- Caches and usage logs are vertical-scoped, so a coaching answer is never
  served to a school request.

