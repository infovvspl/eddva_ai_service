# Photosynthesis — the V2 reference case

Two slides, CBSE Class 8 Science. This is the case the PPT V2 infographic
system is built against: a concept slide with a labelled diagram, and a
process slide with ordered steps.

## What is here

| File | What it is |
|---|---|
| `planned_deck.json` | Captured **planner output** — what each slide teaches and what the learner should see. No layout, colour, image source or geometry. |

The fixture stops at the storyboard stage on purpose. Everything after it —
layout choice, theme, visual strategy, image prompt, geometry — is computed by
`ai_services/core/ppt_v2/compose.py`, which is exactly what the tests need to
exercise. Freezing the planner output also means the tests never call an LLM,
so they run in milliseconds and give the same answer every time.

## Running it

```bash
python manage.py test ai_services.tests_ppt_v2_compose
```

The reference assertions live in
`ai_services/tests_ppt_v2_compose.py::PhotosynthesisReferenceTests`. They cover
the deck composing cleanly, the two slides getting different layouts, the four
diagram annotations surviving, biology theming, the microscopic content
(`chlorophyll`, `stomata`, `chloroplast`) going to a generated figure rather
than a stock photo, and the deck still composing when AI image generation —
and then everything else — is unavailable.

## What these tests do NOT verify

**Nothing here renders a pixel.** There is no server-side renderer in this
stage: the deck is drawn by `pptExport.js` in the browser. So the tests can
show that the specification is structurally sound — on-slide geometry, no
colliding text regions, readable contrast, a visual strategy with a working
fallback — and they cannot show that the result *looks* like the reference
image, that a generated figure is scientifically correct, or that an image
suits the slide it landed on.

Those are human judgements on a rendered file. Treat a green test run as
"nothing is structurally broken", not as visual sign-off.
