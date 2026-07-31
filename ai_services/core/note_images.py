import json
import logging
import os
import re
import base64
from typing import Any

from ai_services.core.image_generation import (
    can_generate_note_images,
    generate_note_image,
    image_generation_config,
)


logger = logging.getLogger("ai_services.images")

_NOTE_IMAGE_OVERLAY_MARKER = "<<NOTE_IMAGE_OVERLAY:"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _strip_md_marks(text: str) -> str:
    return re.sub(r"^[#*\-\s>`]+|[*_`]+", "", str(text or "")).strip()


def _extract_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _encode_overlay_marker(labels: list[dict[str, Any]]) -> str:
    if not labels:
        return ""
    payload = json.dumps({"labels": labels}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f" {_NOTE_IMAGE_OVERLAY_MARKER}{encoded}>>"


def _notes_context_for_candidate(notes: str, candidate: dict[str, Any], max_chars: int = 2200) -> str:
    evidence = str(candidate.get("evidence_quote") or "")
    evidence_norm = _norm(evidence)
    if not evidence_norm:
        return str(notes or "")[:max_chars]

    lines = str(notes or "").splitlines()
    for idx, line in enumerate(lines):
        if evidence_norm in _norm(line):
            start = max(0, idx - 8)
            end = min(len(lines), idx + 9)
            return "\n".join(lines[start:end]).strip()[:max_chars]

    notes_text = str(notes or "")
    pos = _norm(notes_text).find(evidence_norm)
    if pos < 0:
        return notes_text[:max_chars]
    start = max(0, pos - max_chars // 2)
    return notes_text[start : start + max_chars]


def _sanitize_overlay_labels(raw_labels: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(raw_labels, list):
        return []
    grounded_text = _norm(context)
    labels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or item.get("label") or "")).strip()
        key = _norm(text)
        if len(key) < 2 or key in seen or (len(key) >= 3 and key not in grounded_text):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if not (0.03 <= x <= 0.97 and 0.03 <= y <= 0.97):
            continue
        seen.add(key)
        labels.append({"text": text[:60], "x": round(x, 4), "y": round(y, 4)})
        if len(labels) >= 8:
            break
    return labels


def _candidate_label_terms(candidate: dict[str, Any], context: str) -> list[str]:
    raw_terms = candidate.get("label_terms") or candidate.get("labels") or candidate.get("required_labels") or []
    if isinstance(raw_terms, str):
        raw_terms = re.split(r"[,;\n]+", raw_terms)
    if not isinstance(raw_terms, list):
        raw_terms = []

    grounded_text = _norm(context)
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw_terms:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        key = _norm(text)
        if len(key) < 2 or key in seen:
            continue
        if len(key) >= 3 and grounded_text and key not in grounded_text:
            continue
        seen.add(key)
        terms.append(text[:60])
        if len(terms) >= 8:
            break
    return terms


def _image_path_to_data_uri(path: str, mime_type: str = "image/png") -> str:
    data = open(path, "rb").read()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type or 'image/png'};base64,{encoded}"


def _overlay_label_prompt(candidate: dict[str, Any], context: str) -> str:
    label_terms = _candidate_label_terms(candidate, context)
    required_label_text = "\n".join(f"- {term}" for term in label_terms) or "- Use the clearest visible note-supported parts."
    return (
        "You are labeling an already generated educational diagram. Identify only the visible regions/parts in the image "
        "that match explicit terms in the lecture notes context. Return normalized coordinates for where a small label "
        "badge should be placed on the image. Do not invent labels. Do not label decorative or unclear regions.\n\n"
        "Return JSON only in this exact shape:\n"
        "{\n"
        '  "labels": [\n'
        '    {"text": "exact label from notes", "x": 0.50, "y": 0.50}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- x and y must be normalized from 0 to 1 relative to image width and height.\n"
        "- Prefer the required label targets below. Place every required target that is clearly visible in the image.\n"
        "- Use 1 to 8 labels only when the image clearly shows the part.\n"
        "- Every label text must appear verbatim in the notes context, ignoring case.\n"
        "- Put the coordinate on the actual visible structure, not in empty margin or on a leader line.\n"
        "- If the generated image is anatomically or conceptually wrong, label only the parts that are still clearly correct.\n"
        "- If accurate placement is not possible, return an empty labels array.\n\n"
        f"REQUIRED LABEL TARGETS:\n{required_label_text}\n\n"
        f"Figure caption: {candidate.get('caption', '')}\n"
        f"Intended diagram: {candidate.get('visual_description', '')}\n"
        f"Grounding quote: {candidate.get('evidence_quote', '')}\n\n"
        f"NOTES CONTEXT:\n{context}"
    )


def _label_generated_note_image_with_gemini(
    image_path: str,
    mime_type: str,
    prompt: str,
    context: str,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from google import genai
        from google.genai import types
        from ai_services.core.gemini_keys import (
            gemini_key_count,
            get_rotated_gemini_keys,
            is_gemini_permanent_key_error,
            is_gemini_retryable_error,
            mark_gemini_key_disabled,
        )
    except Exception as exc:
        logger.warning("Gemini vision overlay labeling unavailable: %s", exc)
        return [], "overlay_label_gemini_unavailable"

    try:
        image_bytes = open(image_path, "rb").read()
    except Exception as exc:
        logger.warning("Could not read generated image for Gemini overlay labels: %s", exc)
        return [], "overlay_label_image_encode_failed"

    model = os.getenv("NOTES_IMAGE_OVERLAY_GEMINI_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
    last_error = ""
    for key_index, api_key in get_rotated_gemini_keys():
        try:
            client = genai.Client(api_key=api_key)
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/png"),
            ]
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=600,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except TypeError:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=600,
                        response_mime_type="application/json",
                    ),
                )
            content = _extract_json_object(getattr(response, "text", "") or "{}")
            labels = _sanitize_overlay_labels(content.get("labels"), context)
            logger.info(
                "Gemini vision overlay labels | key=%d/%d | model=%s | labels=%d",
                key_index,
                gemini_key_count(),
                model,
                len(labels),
            )
            return labels, None
        except Exception as exc:
            last_error = str(exc)
            if is_gemini_permanent_key_error(last_error):
                mark_gemini_key_disabled(api_key)
                logger.warning("Gemini overlay label key %d/%d disabled: %s", key_index, gemini_key_count(), last_error[:160])
                continue
            if is_gemini_retryable_error(last_error):
                logger.warning("Gemini overlay label key %d/%d retryable error; rotating: %s", key_index, gemini_key_count(), last_error[:180])
                continue
            logger.warning("Gemini vision overlay labeling failed: %s", last_error[:180])
            break
    return [], f"overlay_label_gemini_failed: {last_error[:100]}" if last_error else "overlay_label_gemini_failed"


def label_generated_note_image(
    image_result: dict[str, Any],
    candidate: dict[str, Any],
    notes: str,
    language: str = "",
) -> tuple[list[dict[str, Any]], str | None]:
    if os.getenv("NOTES_IMAGE_OVERLAY_LABELS_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return [], None
    image_path = str(image_result.get("path") or "")
    if not image_path:
        return [], "overlay_label_image_path_missing"

    context = _notes_context_for_candidate(notes, candidate)
    prompt = _overlay_label_prompt(candidate, context)
    mime_type = str(image_result.get("mime_type") or "image/png")

    if _is_odia_language(language):
        return _label_generated_note_image_with_gemini(image_path, mime_type, prompt, context)

    try:
        from groq import Groq
        from ai_services.core.groq_keys import get_rotated_groq_keys
    except Exception as exc:
        logger.warning("Groq vision overlay labeling unavailable: %s", exc)
        return [], "overlay_label_groq_unavailable"

    keys = get_rotated_groq_keys()
    if not keys:
        return [], "overlay_label_no_groq_key"

    data_uri = ""
    try:
        data_uri = _image_path_to_data_uri(image_path, mime_type)
    except Exception as exc:
        logger.warning("Could not encode generated image for Groq vision overlay labels: %s", exc)
        return [], "overlay_label_image_encode_failed"

    models = [
        os.getenv("NOTES_IMAGE_OVERLAY_GROQ_MODEL", "").strip(),
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-11b-vision-preview",
        "qwen/qwen3.6-27b",  # last resort — reasoning suppressed below
    ]
    models = [model for idx, model in enumerate(models) if model and model not in models[:idx]]
    last_error = ""
    for api_key in keys:
        for model in models:
            try:
                client = Groq(api_key=api_key, timeout=45.0)
                call_kw = dict(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=600,
                )
                if "qwen" in model.lower():
                    call_kw["reasoning_effort"] = "none"
                response = client.chat.completions.create(**call_kw)
                content = _extract_json_object(response.choices[0].message.content or "{}")
                labels = _sanitize_overlay_labels(content.get("labels"), context)
                logger.info("Groq vision overlay labels | model=%s | labels=%d", model, len(labels))
                return labels, None
            except Exception as exc:
                last_error = str(exc)
                lower = last_error.lower()
                if "not found" in lower or "not supported" in lower or "invalid model" in lower or "404" in lower:
                    continue
                if "rate" in lower or "429" in lower:
                    break
                logger.warning("Groq vision overlay labeling failed | model=%s | %s", model, last_error[:180])
                continue
    return [], f"overlay_label_failed: {last_error[:100]}" if last_error else "overlay_label_failed"


def _is_odia_language(language: str) -> bool:
    return str(language or "").strip().lower() in {"od", "odia", "or", "od-in", "or-in"}


def _gemini_image_plan(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
        from ai_services.core.gemini_keys import (
            gemini_key_count,
            get_rotated_gemini_keys,
            is_gemini_permanent_key_error,
            is_gemini_retryable_error,
            mark_gemini_key_disabled,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini image planner unavailable: {exc}") from exc

    model = os.getenv("NOTES_IMAGE_PLANNER_GEMINI_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
    last_exc: Exception | None = None
    for key_index, api_key in get_rotated_gemini_keys():
        try:
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except TypeError:
                response = client.models.generate_content(
                    model=model,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                )
            logger.info("Gemini note image planning OK | key=%d/%d | model=%s", key_index, gemini_key_count(), model)
            return _extract_json_object(getattr(response, "text", "") or "")
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if is_gemini_permanent_key_error(msg):
                mark_gemini_key_disabled(api_key)
                logger.warning("Gemini image planner key %d/%d disabled: %s", key_index, gemini_key_count(), msg[:180])
                continue
            if is_gemini_retryable_error(msg):
                logger.warning("Gemini image planner key %d/%d retryable error; rotating: %s", key_index, gemini_key_count(), msg[:220])
                continue
            raise RuntimeError(f"Gemini image planner failed: {exc}") from exc
    raise RuntimeError(f"Gemini image planner failed across all keys: {last_exc}") from last_exc


def _section_heading_exists(notes: str, heading: str) -> bool:
    target = _norm(_strip_md_marks(heading))
    if not target:
        return False
    for line in notes.splitlines():
        if re.match(r"^\s*#{1,4}\s+", line):
            if _norm(_strip_md_marks(line)) == target:
                return True
    return False


def _evidence_is_grounded(notes: str, evidence: str) -> bool:
    evidence_norm = _norm(evidence)
    if len(evidence_norm) < 24:
        return False
    return evidence_norm in _norm(notes)


def _diagram_key(section: str, caption: str, visual: str, diagram_group: str = "") -> str:
    group = _norm(diagram_group)
    if group:
        return group[:120]
    caption_text = _norm(caption)
    text = _norm(f"{section} {caption} {visual}")
    tokens = re.findall(r"[a-z0-9\u0b00-\u0b7f]+", text)
    stop = {
        "a", "an", "and", "are", "as", "for", "in", "of", "on", "or", "the", "to", "with",
        "diagram", "figure", "image", "label", "labels", "labeled", "show", "showing", "simple",
        "educational", "structure", "structures", "function", "functions", "part", "parts",
    }
    caption_tokens = [tok for tok in re.findall(r"[a-z0-9\u0b00-\u0b7f]+", caption_text) if tok not in stop and len(tok) > 2]
    if len(caption_tokens) >= 2:
        return " ".join(sorted(set(caption_tokens))[:8])
    useful = [tok for tok in tokens if tok not in stop and len(tok) > 2]
    return " ".join(sorted(set(useful))[:10]) or _norm(caption)


def _merge_text(existing: str, addition: str, max_chars: int) -> str:
    existing = str(existing or "").strip()
    addition = str(addition or "").strip()
    if not addition or _norm(addition) in _norm(existing):
        return existing[:max_chars]
    if not existing:
        return addition[:max_chars]
    return f"{existing}; {addition}"[:max_chars]


def _clean_candidates(notes: str, raw_candidates: list[Any], limit: int | None = None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    for item in raw_candidates:
        if not isinstance(item, dict):
            continue

        section = str(item.get("section_heading") or "").strip()
        diagram_group = str(item.get("diagram_group") or item.get("diagram_target") or "").strip()
        caption = str(item.get("caption") or "").strip()
        visual = str(item.get("visual_description") or item.get("prompt") or "").strip()
        evidence = str(item.get("evidence_quote") or "").strip()
        label_terms = _candidate_label_terms(item, f"{notes}\n{evidence}\n{visual}")

        if not caption or not visual or not evidence:
            continue
        if not _evidence_is_grounded(notes, evidence):
            logger.info("Skipping note image candidate without exact evidence grounding: %s", caption[:80])
            continue

        key = _diagram_key(section, caption, visual, diagram_group)
        existing = by_key.get(key)
        if existing:
            existing["caption"] = _merge_text(existing["caption"], caption, 160)
            existing["visual_description"] = _merge_text(existing["visual_description"], visual, 700)
            existing["evidence_quote"] = _merge_text(existing["evidence_quote"], evidence, 900)
            existing_terms = existing.get("label_terms") or []
            for term in label_terms:
                if _norm(term) not in {_norm(t) for t in existing_terms} and len(existing_terms) < 8:
                    existing_terms.append(term)
            existing["label_terms"] = existing_terms
            continue

        by_key[key] = {
            "section_heading": section,
            "diagram_group": diagram_group[:160],
            "caption": caption[:160],
            "visual_description": visual[:700],
            "evidence_quote": evidence[:900],
            "label_terms": label_terms,
            "section_exists": str(_section_heading_exists(notes, section)).lower(),
        }
        cleaned = list(by_key.values())
        if limit is not None and len(cleaned) >= limit:
            continue

    return list(by_key.values())[:limit] if limit is not None else list(by_key.values())


def _merge_cleaned_candidates(candidates: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        section = candidate.get("section_heading", "")
        diagram_group = candidate.get("diagram_group", "")
        caption = candidate.get("caption", "")
        visual = candidate.get("visual_description", "")
        key = _diagram_key(section, caption, visual, diagram_group)
        existing = by_key.get(key)
        if existing:
            existing["caption"] = _merge_text(existing["caption"], caption, 160)
            existing["visual_description"] = _merge_text(existing["visual_description"], visual, 700)
            existing["evidence_quote"] = _merge_text(existing["evidence_quote"], candidate.get("evidence_quote", ""), 900)
            existing_terms = existing.get("label_terms") or []
            for term in candidate.get("label_terms") or []:
                if _norm(term) not in {_norm(t) for t in existing_terms} and len(existing_terms) < 8:
                    existing_terms.append(term)
            existing["label_terms"] = existing_terms
            continue
        by_key[key] = dict(candidate)
    values = list(by_key.values())
    return values[:limit] if limit is not None else values


def _split_notes_for_image_planning(notes: str, max_chars: int) -> list[str]:
    text = str(notes or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sections = re.split(r"(?m)(?=^\s*#{1,3}\s+)", text)
    batches: list[str] = []
    current = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) > max_chars:
            if current:
                batches.append(current.strip())
                current = ""
            start = 0
            while start < len(section):
                end = min(start + max_chars, len(section))
                boundary = section.rfind("\n", start, end)
                if boundary <= start + int(max_chars * 0.5):
                    boundary = section.rfind(". ", start, end)
                if boundary > start + int(max_chars * 0.5):
                    end = boundary + 1
                batches.append(section[start:end].strip())
                start = end
            continue
        if current and len(current) + len(section) + 2 > max_chars:
            batches.append(current.strip())
            current = section
        else:
            current = f"{current}\n\n{section}".strip() if current else section
    if current:
        batches.append(current.strip())
    return batches


def _planning_limit_text(max_images: int, already_planned: int) -> tuple[int | None, str]:
    hard_limit = max(1, max_images)
    remaining = hard_limit - already_planned
    return remaining, (
        f"- Safety limit for this request: return at most {remaining} more images.\n"
        if remaining > 0
        else "- Safety limit reached; return an empty images array.\n"
    )


def plan_note_images(notes: str, topic_id: str, language: str, institute_id: str, llm) -> tuple[list[dict[str, Any]], list[str]]:
    cfg = image_generation_config()
    if not can_generate_note_images():
        return [], []
    if len(str(notes or "").strip()) < 500:
        return [], ["notes_too_short_for_images"]

    try:
        planner_max_chars = int(os.getenv("NOTES_IMAGE_PLANNER_MAX_CHARS", "3500") or "3500")
    except (TypeError, ValueError):
        planner_max_chars = 3500

    configured_max = int(cfg.get("max_images") or 0)
    hard_max = int(cfg.get("hard_max_images") or 12)
    effective_limit = configured_max if configured_max > 0 else hard_max
    note_batches = _split_notes_for_image_planning(notes, max(1200, planner_max_chars))
    if not note_batches:
        return [], []

    all_candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    model = os.getenv("NOTES_IMAGE_PLANNER_MODEL", "llama-3.1-8b-instant")
    use_gemini_planner = (
        _is_odia_language(language)
        or os.getenv("NOTES_IMAGE_PLANNER_PROVIDER", "").strip().lower() == "gemini"
    )

    for batch_index, notes_excerpt in enumerate(note_batches, start=1):
        remaining, limit_rule = _planning_limit_text(effective_limit, len(all_candidates))
        if remaining is not None and remaining <= 0:
            break
        system_prompt = (
            "You are a strict educational diagram planner. Identify visual aids only when they are "
            "directly supported by exact text in the supplied lecture notes excerpt. Do not invent topics, "
            "examples, labels, processes, or diagrams absent from the excerpt. If the excerpt does not "
            "contain diagram-worthy content, return an empty images array."
        )
        user_prompt = (
            f"Lecture topic: {topic_id or 'General'}\n"
            f"Source language: {language}\n"
            f"Notes excerpt: {batch_index} of {len(note_batches)}\n\n"
            "Return JSON only in this exact shape:\n"
            "{\n"
            '  "images": [\n'
            "    {\n"
            '      "section_heading": "exact Markdown section heading from the notes if possible",\n'
            '      "diagram_group": "short stable name for the one object/process/graph this image should depict",\n'
            '      "caption": "student-facing figure caption",\n'
            '      "visual_description": "specific educational diagram to generate",\n'
            '      "label_terms": ["exact visible part name from the notes", "another visible part name"],\n'
            '      "evidence_quote": "exact contiguous quote from the notes proving this visual belongs here"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Return every genuinely useful educational visual required by this excerpt, not an arbitrary fixed count.\n"
            f"{limit_rule}"
            "- Every evidence_quote must be copied exactly from the excerpt.\n"
            "- Prefer diagrams, processes, apparatus, graphs, structures, mechanisms, cycles, or spatial relationships.\n"
            "- Avoid duplicate visuals. If multiple labels or closely related parts belong to the same object/process, "
            "combine them into one composite diagram instead of returning separate images. "
            "Use the same diagram_group for candidates that should become one image.\n"
            "- label_terms is required for diagrams with visible parts: list the exact structures/process items from "
            "the notes that must be visible and later overlaid as labels. Use only words present in the excerpt.\n"
            "- Do not request decorative, motivational, generic classroom, portrait, logo, or unrelated images.\n"
            "- Do not create images for content that is only implied but not written in the notes.\n\n"
            f"NOTES EXCERPT:\n{notes_excerpt}"
        )
        try:
            if use_gemini_planner:
                content = _gemini_image_plan(system_prompt, user_prompt, max_tokens=800)
            else:
                result = llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=0.1,
                    max_tokens=800,
                    json_mode=True,
                    institute_id=institute_id,
                )
                content = _extract_json_object(result.get("content"))
        except Exception as exc:
            logger.warning("Note image planning failed for batch %d/%d: %s", batch_index, len(note_batches), exc)
            errors.append(f"planner_batch_{batch_index}_failed: {str(exc)[:120]}")
            continue

        raw_images = content.get("images", [])
        if not isinstance(raw_images, list):
            errors.append(f"planner_batch_{batch_index}_returned_invalid_images")
            continue

        batch_candidates = _clean_candidates(notes, raw_images, remaining)
        all_candidates.extend(batch_candidates)

    deduped = _merge_cleaned_candidates(all_candidates, effective_limit)
    return deduped, errors


def _build_image_prompt(candidate: dict[str, Any], topic_id: str) -> str:
    return (
        "You are generating educational note images for students.\n\n"
        "TASK:\n"
        "Create only the image that is genuinely required to explain the given notes and subtopic. "
        "Do not create decorative, random, symbolic, or unrelated images.\n\n"
        "INPUT:\n"
        f"Notes:\n{candidate.get('evidence_quote', '')}\n\n"
        f"Subtopic:\n{candidate.get('section_heading') or topic_id or candidate.get('caption', 'General')}\n\n"
        "IMAGE SELECTION RULES:\n"
        "1. Analyze the notes first.\n"
        "2. Generate an image only if a visual diagram will improve understanding.\n"
        "3. If the subtopic is theoretical and does not need a diagram, do not generate an image for it.\n"
        "4. The image must be directly connected to one specific subtopic from the notes.\n"
        "5. Do not invent concepts, objects, organs, processes, or examples that are not present in the notes.\n"
        "6. Do not generate generic classroom, student, teacher, book, AI, or abstract education images.\n"
        "7. Prefer scientifically/academically correct diagrams, process illustrations, anatomy diagrams, apparatus diagrams, "
        "flow diagrams, cycle diagrams, or concept visuals depending on the topic.\n\n"
        "IMAGE STYLE:\n"
        "- Clean educational textbook-style illustration\n"
        "- Accurate and directly related to the selected subtopic\n"
        "- White or light background\n"
        "- No text\n"
        "- No labels\n"
        "- No arrows\n"
        "- No pointer lines\n"
        "- No leader lines\n"
        "- No numbers\n"
        "- No letters\n"
        "- No captions\n"
        "- No diagrams with annotation marks\n"
        "- No decorative or unrelated objects\n\n"
        "UNLABELLED IMAGE REQUIREMENT:\n"
        "The image must be a clean visual only.\n\n"
        "Strictly forbidden:\n"
        "- written labels\n"
        "- blank label spaces\n"
        "- pointer lines\n"
        "- leader lines\n"
        "- arrows\n"
        "- callout boxes\n"
        "- numbers\n"
        "- letters\n"
        "- captions\n"
        "- titles\n"
        "- annotation marks\n\n"
        "Show only the actual object, process, structure, apparatus, scene, or concept needed for the subtopic.\n\n"
        "ACCURACY RULES:\n"
        "- The visual must match the exact concept described in the notes.\n"
        "- Do not mix different chapters or unrelated concepts.\n"
        "- Do not use metaphorical or fantasy visuals.\n"
        "- For biology, show correct organs/parts/processes.\n"
        "- For chemistry, show correct apparatus, particles, reaction setup, or molecular-level concept where applicable.\n"
        "- For physics, show correct experimental setup, forces, rays, circuits, motion, or field diagrams.\n"
        "- For mathematics, show clean geometric or graph-based visuals only when required.\n\n"
        "FINAL IMAGE PROMPT:\n"
        f"{candidate.get('visual_description', '')}\n"
    )


def _image_markdown(image: dict[str, Any]) -> str:
    caption = str(image.get("caption") or "Generated educational visual").strip()
    url = str(image.get("url") or "").strip()
    safe_caption = re.sub(r"[\[\]\n\r|]+", " ", caption).strip()
    overlay_marker = _encode_overlay_marker(image.get("overlay_labels") or [])
    lines = [
        f"![{safe_caption}{overlay_marker}]({url})",
        "",
        f"*Figure: {safe_caption}*",
    ]
    return "\n\n" + "\n".join(lines) + "\n"


def _insert_after_section(notes: str, section_heading: str, markdown: str) -> tuple[str, bool]:
    target = _norm(_strip_md_marks(section_heading))
    if not target:
        return notes, False

    lines = notes.splitlines()
    for idx, line in enumerate(lines):
        if not re.match(r"^\s*#{1,4}\s+", line):
            continue
        if _norm(_strip_md_marks(line)) != target:
            continue
        insert_at = idx + 1
        lines[insert_at:insert_at] = markdown.strip("\n").splitlines()
        return "\n".join(lines).strip(), True
    return notes, False


def enrich_notes_with_images(notes: str, topic_id: str, language: str, institute_id: str, llm) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    cfg = image_generation_config()
    meta = {
        "enabled": bool(cfg["enabled"]),
        "provider": cfg["provider"],
        "model": cfg["model"],
        "count": 0,
        "errors": [],
    }

    if not can_generate_note_images():
        if not cfg["enabled"]:
            meta["errors"].append("image_generation_disabled")
        elif not cfg["has_key"]:
            meta["errors"].append("image_generation_not_configured")
        return notes, [], meta

    candidates, plan_errors = plan_note_images(notes, topic_id, language, institute_id, llm)
    meta["errors"].extend(plan_errors)
    if not candidates:
        return notes, [], meta

    generated: list[dict[str, Any]] = []
    for candidate in candidates:
        image_prompt = _build_image_prompt(candidate, topic_id)
        image_result = generate_note_image(image_prompt)
        if not image_result.get("ok"):
            meta["errors"].append(str(image_result.get("error") or "image_generation_failed"))
            if image_result.get("fatal"):
                logger.warning("Stopping note image generation after fatal provider error: %s", image_result.get("error"))
                break
            continue
        overlay_labels, overlay_error = label_generated_note_image(image_result, candidate, notes, language)
        if overlay_error:
            meta["errors"].append(overlay_error)

        generated.append(
            {
                "url": image_result["url"],
                "caption": candidate["caption"],
                "visual_description": candidate["visual_description"],
                "overlay_labels": overlay_labels,
                "section_heading": candidate["section_heading"],
                "evidence_quote": candidate["evidence_quote"],
                "prompt": image_prompt,
                "provider": image_result["provider"],
                "model": image_result["model"],
                "image_size": image_result["image_size"],
                "aspect_ratio": image_result.get("aspect_ratio"),
                "embedded_text_removed": image_result.get("embedded_text_removed", 0),
                "embedded_text_strip_error": image_result.get("embedded_text_strip_error"),
            }
        )

    enriched = notes
    fallback_blocks: list[str] = []
    for image in generated:
        block = _image_markdown(image)
        enriched, inserted = _insert_after_section(enriched, image.get("section_heading", ""), block)
        if not inserted:
            fallback_blocks.append(block)

    if fallback_blocks:
        enriched = enriched.rstrip() + "\n\n## Visual Summary" + "".join(fallback_blocks)

    meta["count"] = len(generated)
    return enriched, generated, meta
