import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


logger = logging.getLogger("ai_services.images")

_ocr_reader = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        logger.warning("Invalid integer env var %s=%r; using %d", name, os.getenv(name), default)
        return default


def _generated_images_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / "data" / "generated_note_images"
    return Path(os.getenv("NOTES_IMAGE_OUTPUT_DIR", str(default_dir))).resolve()


def _public_base_url() -> str:
    return os.getenv("NOTES_IMAGE_PUBLIC_BASE_URL", os.getenv("AI_PUBLIC_BASE_URL", "http://localhost:8000")).rstrip("/")


def _huggingface_token() -> str:
    return (
        os.getenv("HF_IMAGE_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_IMAGE_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )


def image_generation_config() -> dict[str, Any]:
    enabled = _env_bool("NOTES_IMAGE_GENERATION_ENABLED", "false")
    width = max(128, _env_int("HF_IMAGE_WIDTH", _env_int("NOTES_IMAGE_WIDTH", 768)))
    height = max(128, _env_int("HF_IMAGE_HEIGHT", _env_int("NOTES_IMAGE_HEIGHT", 512)))
    model = (
        os.getenv("HF_IMAGE_MODEL", "").strip()
        or os.getenv("HUGGINGFACE_IMAGE_MODEL", "").strip()
        or os.getenv("NOTES_IMAGE_MODEL", "").strip()
        or "black-forest-labs/FLUX.1-schnell"
    )
    router_base_url = os.getenv("HF_IMAGE_ROUTER_BASE_URL", "https://router.huggingface.co/hf-inference/models").rstrip("/")
    return {
        "enabled": enabled,
        "provider": "huggingface",
        "model": model,
        "url": f"{router_base_url}/{quote(model, safe='/')}",
        "width": width,
        "height": height,
        "timeout": max(30, _env_int("HF_IMAGE_TIMEOUT_SECONDS", 180)),
        # 0 means dynamic: let the note-image planner decide how many grounded
        # visuals are needed. NOTES_IMAGE_HARD_MAX_PER_LECTURE is only a safety
        # guard against accidental runaway provider calls.
        "max_images": max(0, _env_int("NOTES_IMAGE_MAX_PER_LECTURE", 0)),
        "hard_max_images": max(1, _env_int("NOTES_IMAGE_HARD_MAX_PER_LECTURE", 12)),
        "has_key": bool(_huggingface_token()),
        "output_dir": str(_generated_images_dir()),
        "public_base_url": _public_base_url(),
    }


def can_generate_note_images() -> bool:
    cfg = image_generation_config()
    return bool(cfg["enabled"] and cfg["model"] and cfg["has_key"])


def _is_fatal_huggingface_error(status_code: int | None, message: str) -> bool:
    text = str(message or "").lower()
    if status_code in {401, 403, 404}:
        return True
    return any(
        token in text
        for token in (
            "invalid token",
            "unauthorized",
            "forbidden",
            "not found",
            "does not exist",
            "gated",
            "license",
            "payment required",
        )
    )


def _save_generated_image(image_bytes: bytes, mime_type: str) -> dict[str, str]:
    output_dir = _generated_images_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(mime_type) or ".png"
    if ext == ".jpe":
        ext = ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = output_dir / filename
    path.write_bytes(image_bytes)
    return {
        "path": str(path),
        "url": f"{_public_base_url()}/generated-note-images/{filename}",
        "mime_type": mime_type,
    }


def _text_strip_languages() -> list[str]:
    raw = os.getenv("NOTES_IMAGE_TEXT_STRIP_LANGS", "en").strip()
    langs = [part.strip() for part in raw.split(",") if part.strip()]
    return langs or ["en"]


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(_text_strip_languages(), gpu=False)
    return _ocr_reader


def _strip_embedded_text_from_image(path: str) -> dict[str, Any]:
    if not _env_bool("NOTES_IMAGE_STRIP_EMBEDDED_TEXT", "true"):
        return {"removed": 0, "enabled": False}

    try:
        import cv2
    except Exception as exc:
        logger.warning("Generated image text stripping unavailable: %s", exc)
        return {"removed": 0, "error": "text_strip_dependencies_unavailable"}

    try:
        image = cv2.imread(path)
        if image is None:
            return {"removed": 0, "error": "text_strip_image_read_failed"}

        min_conf = float(os.getenv("NOTES_IMAGE_TEXT_STRIP_MIN_CONFIDENCE", "0.35") or "0.35")
        pad = max(0, _env_int("NOTES_IMAGE_TEXT_STRIP_PADDING", 8))
        reader = _get_ocr_reader()
        detections = reader.readtext(path)
        removed = 0

        height, width = image.shape[:2]
        for detection in detections:
            if not isinstance(detection, (list, tuple)) or len(detection) < 3:
                continue
            box, text, confidence = detection[0], str(detection[1] or "").strip(), float(detection[2] or 0)
            if not text or confidence < min_conf:
                continue

            points = []
            for point in box or []:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((int(point[0]), int(point[1])))
            if not points:
                continue

            x1 = max(0, min(x for x, _ in points) - pad)
            y1 = max(0, min(y for _, y in points) - pad)
            x2 = min(width, max(x for x, _ in points) + pad)
            y2 = min(height, max(y for _, y in points) + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            image[y1:y2, x1:x2] = (255, 255, 255)
            removed += 1

        if removed:
            cv2.imwrite(path, image)
            logger.info("Stripped %d embedded text region(s) from generated note image", removed)
        return {"removed": removed}
    except Exception as exc:
        logger.warning("Generated image text stripping failed: %s", exc)
        return {"removed": 0, "error": "text_strip_failed"}


def generate_note_image(prompt: str) -> dict[str, Any]:
    """
    Generate one strictly grounded educational image through Hugging Face Router.

    The upstream planner only sends prompts backed by exact note evidence. This
    function mirrors the provided HF call shape, retries once when the model is
    still loading, stores returned image bytes, and returns the public URL.
    """
    cfg = image_generation_config()
    if not cfg["enabled"]:
        return {"ok": False, "error": "image_generation_disabled"}
    if not can_generate_note_images():
        return {"ok": False, "error": "huggingface_image_generation_not_configured", "fatal": True}

    token = _huggingface_token()
    image_prompt = (
        f"{str(prompt or '').strip()}\n\n"
        "Generate one clean educational note illustration only. Use only the facts explicitly present in the prompt. "
        "Do not add unsupported facts, extra labels, logos, watermarks, decorative backgrounds, or unrelated objects."
    )
    body = {
        "inputs": image_prompt,
        "parameters": {
            "width": cfg["width"],
            "height": cfg["height"],
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/png",
    }

    last_error = ""
    last_status: int | None = None
    for attempt in range(2):
        try:
            logger.info(
                "HF image generation request | model=%s | size=%dx%d | attempt=%d/2",
                cfg["model"],
                cfg["width"],
                cfg["height"],
                attempt + 1,
            )
            response = requests.post(cfg["url"], headers=headers, json=body, timeout=cfg["timeout"])
            last_status = response.status_code
            content_type = response.headers.get("content-type", "")

            if response.ok and content_type.startswith("image/"):
                image_bytes = response.content
                mime_type = content_type.split(";", 1)[0] or "image/png"
                saved = _save_generated_image(image_bytes, mime_type)
                text_strip = _strip_embedded_text_from_image(saved["path"])
                logger.info(
                    "HF image generation OK | model=%s | size=%dx%d",
                    cfg["model"],
                    cfg["width"],
                    cfg["height"],
                )
                return {
                    "ok": True,
                    "url": saved["url"],
                    "path": saved["path"],
                    "mime_type": saved["mime_type"],
                    "provider": cfg["provider"],
                    "model": cfg["model"],
                    "image_size": f"{cfg['width']}x{cfg['height']}",
                    "aspect_ratio": f"{cfg['width']}:{cfg['height']}",
                    "embedded_text_removed": text_strip.get("removed", 0),
                    "embedded_text_strip_error": text_strip.get("error"),
                }

            if response.status_code == 503 and attempt == 0:
                wait_ms = 8000
                try:
                    payload = response.json()
                    estimated_time = payload.get("estimated_time")
                    if estimated_time:
                        wait_ms = min(20000, int(float(estimated_time) * 1000 + 999))
                except Exception:
                    pass
                logger.info("HF model %s loading; retrying in %dms", cfg["model"], wait_ms)
                time.sleep(wait_ms / 1000)
                continue

            err_text = response.text[:500]
            last_error = f"HF image API error {response.status_code}: {err_text}"
            logger.warning("HF image gen failed (%s): %s", response.status_code, err_text[:200])
            break
        except Exception as exc:
            last_error = f"HF image gen error: {exc}"
            logger.warning("HF image gen error: %s", exc)
            break

    return {
        "ok": False,
        "error": last_error[:240] or "huggingface_image_generation_failed",
        "fatal": _is_fatal_huggingface_error(last_status, last_error),
    }
