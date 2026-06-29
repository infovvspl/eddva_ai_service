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

| Feature | coaching (base) | school |
|---|---|---|
| `doubt` (`_build_solver_system_prompt`) | "JEE/NEET Subject Matter Expert", JEE/NEET framing | "CBSE/ICSE School Teacher (Classes 1–10)", no JEE/NEET |
| `evaluate_batch` | CBSE/JEE/NEET evaluator | CBSE/ICSE school (Classes 1–10) evaluator |
| `content/generate` | exam target defaults to JEE | defaults to Class 10 board rules (no competitive framing) when none supplied |
| all other features | shared base prompt | shared base prompt (no override yet) |
| model provider | Groq | Groq (same) |

Everything not listed is shared — adding a school variant later is just a new
entry in `VERTICAL_OVERRIDES` (prompts) or `VERTICAL_MODEL_OVERRIDES` (models).

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

