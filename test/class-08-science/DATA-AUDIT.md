# Grounding data audit — Class 8 Science (EDDVA school)

Run before any testing, because a mis-indexed chapter produces failures that look
like model defects but are not. This happened already: a Class 10 PPT for
"Carbon and its Compounds" came back full of Life Processes content, and the
model was behaving correctly — the chapter was indexed with the wrong textbook.

Institute `c259cd4e…`, class `ae2e49a2…` (Class 8), subject Science, 18 chapters,
264 indexed chunks in `textbook_chunks`.

## Findings that will affect Phase 2

| Severity | Chapter | Finding |
|---|---|---|
| **Critical** | Air and Water Pollution | Indexed content is **Class 10 "Light – Reflection and Refraction"**. First chunk: `"9 CHAPTER Light – Reflection and Refraction We see a vari…"`. Nothing about pollution is in the source. |
| **Critical** | Light | **Byte-identical content to Air and Water Pollution** (same MD5, 27 chunks each) — i.e. the same Class 10 Light chapter is indexed twice, under two different Class 8 chapter names. |
| **Critical** | The Universe | **Zero chunks indexed.** Grounded generation is impossible; the feature can only fall back to general knowledge. |
| High | Conservation of Biodiversity | First chunk reads `"CHAPTER 13 BIODIVERSITY AND CONSERVATION 13.1 Biodiversit…"` — section numbering and register suggest a **senior-secondary** book, not the Class 8 chapter "Conservation of Plants and Animals". Needs a human check. |
| High | Sound | First chunk reads `"Chapter Sound Waves: Characteristics and 10 Applications"` — title does not match the NCERT Class 8 chapter. Needs a human check. |
| Medium | Friction | First chunk is `"68 EXEMPLAR PROBLEMS…"` — an **NCERT Exemplar question bank**, not the textbook chapter (only 5 chunks). |
| Medium | Synthetic Materials | First chunk is `"3 Synthetic Fibres and Plastics MULTIPLE CHOICE QUESTIONS"` — question bank rather than chapter text (5 chunks). |
| Medium | Reaching the Age of Adolescence | First chunk is `"10 Reaching the Age of Adolescence MULTIPLE CHOICE QUESTI…"` — question bank rather than chapter text (8 chunks). |
| Low | Crop Production and Management, Force and Pressure, Reproduction, Fuels Combustion and Flame, Chemical Effects of Current | **OCR letter-doubling on headings**, e.g. `"CC PP MM RROOPP RROODDUUCCTTIIOONN AANNDD AANNAAGGEEMMEENN"`. Body text appears intact; headings are corrupted. May degrade retrieval and can leak into generated slide titles. |

Chapters with no flag: Metals and Non-metals, Microorganisms, The Cell,
Some Natural Phenomena, Nature of matter, Heredity-style content not present.

## What this means for interpreting results

1. For the three **Critical** chapters, a grounded answer or deck cannot be
   correct about the stated chapter. Any failure there must be recorded as a
   **data defect**, not a model defect, and the model should arguably be
   *credited* for staying faithful to the source it was given.
2. For the **question-bank** chapters (Friction, Synthetic Materials,
   Reaching the Age of Adolescence), the source has no explanatory prose, so
   grounded answers may be thin or oddly framed. Judge content accuracy, not richness.
3. The `feature: doubt_resolver` cases do **not** supply source passages, so
   they test the model's own CBSE knowledge and are unaffected by this audit.
   Only `feature: ppt_generate` cases, which pass real passages, are affected.

## Recommended fix before re-running Phase 2

Re-upload the correct NCERT Class 8 PDFs for: Air and Water Pollution, Light,
The Universe, and (after checking) Conservation of Biodiversity and Sound.
Ingestion deletes and re-inserts a chapter's chunks, so a re-upload replaces the
bad content cleanly.

A guard at ingestion time — comparing the chapter name against the `CHAPTER <name>`
heading in the PDF's own first page, and warning on a clear mismatch — would have
caught all three critical cases at upload instead of at teaching time.
