# CBSE Class 8 Science — AI feature test suite

Phase 1 artifact: **test cases only. No tests have been executed.**

- Class: 8
- Board: CBSE
- Subject: Science
- Chapters covered: 18 (every Class 8 Science chapter configured for the EDDVA school)
- Total test cases: 180
- Branch: `test/class-08-science`

## Layout

```
test/class-08-science/
├── README.md            this file — scope, method, deviations
├── DATA-AUDIT.md        grounding-data integrity findings (read before judging failures)
├── cases/
│   ├── ch01-05.jsonl    chapters 1-5,  50 cases
│   ├── ch06-11.jsonl    chapters 6-11, 60 cases
│   └── ch12-18.jsonl    chapters 12-18, 70 cases
└── results/             Phase 2 output goes here (kept separate; cases are never edited)
```

Cases are stored as JSONL so the suite is re-runnable and diffable. `render.py`
prints any case in the plain Test-Case-ID format for reading or pasting.

## Case ID scheme

`SCI-08-<chapter number 01-18>-<sequence 001-010>` — unique and sequential.
Chapter numbers follow CBSE syllabus order, which is **not** the alphabetical
order the chapters appear in the database.

## Coverage

Every chapter has exactly one case of each of the ten required types:

| Type | Per chapter | Total |
|---|---|---|
| Basic Knowledge | 1 | 18 |
| Conceptual Understanding | 1 | 18 |
| Application-Based | 1 | 18 |
| Reasoning | 1 | 18 |
| Numerical | 1 | 18 |
| Short Answer | 1 | 18 |
| Long Answer | 1 | 18 |
| Common Misconception | 1 | 18 |
| Adversarial / Boundary | 1 | 18 |
| Slide Generation | 1 | 18 |

**Numerical: 10 of 18 chapters are marked `Not Applicable`** — the biology and
descriptive chapters (Microorganisms, Synthetic Materials, Metals and Non-metals,
Conservation of Biodiversity, The Cell, Reproduction, Reaching the Age of
Adolescence, Friction, Chemical Effects of Current, Air and Water Pollution)
prescribe no calculations at Class 8 level. These are recorded rather than
omitted, so coverage is provable. The 8 chapters with genuine numericals are
Crop Production (yield), Fuels (calorific value), Force and Pressure (pressure),
Sound (frequency), Some Natural Phenomena (Richter scale), Light (angle from the
normal), The Universe (light year) and Nature of Matter (fixed-ratio mass).

## Which feature each case exercises

- `feature: doubt_resolver` — 161 cases. Sent as a student question; tests the
  model's own CBSE knowledge. **No source passages are supplied**, so these are
  unaffected by the textbook-indexing problems in DATA-AUDIT.md.
- `feature: ppt_generate` — 18 cases, one per chapter, `slideCount: 2`.
  Real `textbook_chunks` passages are supplied exactly as NestJS does, so these
  run the grounded path.
- `feature: none` — 1 case per N/A numerical; recorded, never executed.

## Expected answers

Written from established CBSE/NCERT Class 8 subject knowledge, independently of
anything the model produces. Per the workflow, an expected answer is **never**
revised to match a model output. Where a case is genuinely uncertain it is to be
marked MANUAL REVIEW in Phase 2 rather than forced to PASS or FAIL.

## Deviations from the brief — both need your awareness

1. **Slide count floor.** The endpoint clamped `slideCount` with `max(3, …)`, so
   a request for 2 slides silently became 3 and "exactly 2 slides" was untestable.
   On your explicit authorisation the floor was lowered to 2 (`_MIN_SLIDES = 2`),
   and the coverage planner was changed to `min(slide_count, len(sub_areas) + 2)`
   so it can only shorten a deck, never re-inflate a 2-slide request to 3.
   Covered by `ai_services/tests_ppt_slide_count.py` (8 tests). This is the only
   source change made while authoring test cases.
2. **Grounding data is partly wrong.** Three Class 8 Science chapters are
   unusable as sources (two hold Class 10 Light content, one has no content at
   all) and three more are indexed with question banks instead of chapter text.
   See DATA-AUDIT.md. Slide cases for those chapters will fail for data reasons;
   that must not be recorded as a model defect.

## Phase 2 (not started)

On `TEST` / `START TESTING`, each case is executed against the running AI
service, the actual response captured verbatim, and compared against the expected
answer using independent subject knowledge — checking factual correctness,
reasoning, Class 8 appropriateness, hallucination, formatting, and for slide
cases that **exactly 2 slides** are returned. Results are written to `results/`
with PASS / FAIL / MANUAL REVIEW, a failure reason and a severity, leaving
`cases/` untouched so the suite can be re-run after fixes.
