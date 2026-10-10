import base64
import logging
import mimetypes
import os
import threading
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


def _together_token() -> str:
    """The key the text router already uses.

    Image generation deliberately shares it rather than introducing a second
    Together credential for the same account.
    """
    return os.getenv("TOGETHER_API_KEY", "").strip()


PROVIDER_HUGGINGFACE = "huggingface"
PROVIDER_TOGETHER = "together"
SUPPORTED_IMAGE_PROVIDERS = (PROVIDER_HUGGINGFACE, PROVIDER_TOGETHER)

# Verified live against GET https://api.together.ai/v1/models: present, with
# "type": "image". Note that FLUX.1-schnell - the Hugging Face default - is
# NOT served by Together, so the model id cannot simply be carried across.
DEFAULT_TOGETHER_IMAGE_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_TOGETHER_IMAGE_BASE_URL = "https://api.together.ai/v1"
# The docs recommend raising steps to 30-40 for production-quality output.
# 0 (or less) omits the field and lets the provider apply its own default.
DEFAULT_TOGETHER_IMAGE_STEPS = 30


def _image_provider() -> str:
    """Which provider answers ``generate_note_image``.

    Defaults to Hugging Face, so a deployment that never heard of
    NOTES_IMAGE_PROVIDER behaves exactly as it did before. An unrecognised
    value is NOT coerced to a working provider: it reports "not configured",
    so a typo disables generation loudly instead of quietly billing a
    provider nobody chose.
    """
    return (os.getenv("NOTES_IMAGE_PROVIDER", "") or "").strip().lower() or PROVIDER_HUGGINGFACE


def _scrub(text: Any, secret: str) -> str:
    """Never let a key reach a log line or an error string."""
    out = str(text or "")
    if secret:
        out = out.replace(secret, "<redacted>")
    return out


def image_generation_config() -> dict[str, Any]:
    enabled = _env_bool("NOTES_IMAGE_GENERATION_ENABLED", "false")
    width = max(128, _env_int("HF_IMAGE_WIDTH", _env_int("NOTES_IMAGE_WIDTH", 768)))
    height = max(128, _env_int("HF_IMAGE_HEIGHT", _env_int("NOTES_IMAGE_HEIGHT", 512)))
    provider = _image_provider()

    if provider == PROVIDER_TOGETHER:
        model = os.getenv("TOGETHER_IMAGE_MODEL", "").strip() or DEFAULT_TOGETHER_IMAGE_MODEL
        base_url = (os.getenv("TOGETHER_IMAGE_BASE_URL", "").strip()
                    or DEFAULT_TOGETHER_IMAGE_BASE_URL).rstrip("/")
        # "Both values should be multiples of 8" - rounding down here beats
        # a 400 from the provider on every slide of the deck.
        width -= width % 8
        height -= height % 8
        url = f"{base_url}/images/generations"
        timeout = max(30, _env_int("TOGETHER_IMAGE_TIMEOUT_SECONDS", 120))
        steps = _env_int("TOGETHER_IMAGE_STEPS", DEFAULT_TOGETHER_IMAGE_STEPS)
        has_key = bool(_together_token())
    elif provider == PROVIDER_HUGGINGFACE:
        model = (
            os.getenv("HF_IMAGE_MODEL", "").strip()
            or os.getenv("HUGGINGFACE_IMAGE_MODEL", "").strip()
            or os.getenv("NOTES_IMAGE_MODEL", "").strip()
            or "black-forest-labs/FLUX.1-schnell"
        )
        router_base_url = os.getenv("HF_IMAGE_ROUTER_BASE_URL", "https://router.huggingface.co/hf-inference/models").rstrip("/")
        url = f"{router_base_url}/{quote(model, safe='/')}"
        timeout = max(30, _env_int("HF_IMAGE_TIMEOUT_SECONDS", 180))
        steps = 0
        has_key = bool(_huggingface_token())
    else:
        # Unknown provider: no model and no key, so can_generate_note_images()
        # is False and nothing is ever called.
        logger.warning("Unknown NOTES_IMAGE_PROVIDER %r; image generation disabled "
                       "(supported: %s)", provider, ", ".join(SUPPORTED_IMAGE_PROVIDERS))
        model, url, steps = "", "", 0
        timeout = max(30, _env_int("HF_IMAGE_TIMEOUT_SECONDS", 180))
        has_key = False

    return {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "url": url,
        "width": width,
        "height": height,
        "timeout": timeout,
        "steps": steps,
        # 0 means dynamic: let the note-image planner decide how many grounded
        # visuals are needed. NOTES_IMAGE_HARD_MAX_PER_LECTURE is only a safety
        # guard against accidental runaway provider calls.
        "max_images": max(0, _env_int("NOTES_IMAGE_MAX_PER_LECTURE", 0)),
        "hard_max_images": max(1, _env_int("NOTES_IMAGE_HARD_MAX_PER_LECTURE", 12)),
        "has_key": has_key,
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
    """Erase text the generator drew into the picture.

    Reports `detected` and `removed` separately, and samples what was read.
    They are different questions and the first live run needed both: "did
    FLUX draw labels at all" is a prompt problem, "how much did we paint out"
    is an image-damage problem, and a single count cannot tell them apart.
    `detected: None` means nobody looked, not that the image was clean.
    """
    if not _env_bool("NOTES_IMAGE_STRIP_EMBEDDED_TEXT", "true"):
        return {"removed": 0, "detected": None, "texts": [], "enabled": False}

    try:
        import cv2
    except Exception as exc:
        logger.warning("Generated image text stripping unavailable: %s", exc)
        return {"removed": 0, "detected": None, "texts": [],
                "error": "text_strip_dependencies_unavailable"}

    try:
        image = cv2.imread(path)
        if image is None:
            return {"removed": 0, "detected": None, "texts": [],
                    "error": "text_strip_image_read_failed"}

        min_conf = float(os.getenv("NOTES_IMAGE_TEXT_STRIP_MIN_CONFIDENCE", "0.35") or "0.35")
        pad = max(0, _env_int("NOTES_IMAGE_TEXT_STRIP_PADDING", 8))
        reader = _get_ocr_reader()
        detections = reader.readtext(path)
        removed = 0
        detected = 0
        texts: list[str] = []

        height, width = image.shape[:2]
        for detection in detections:
            if not isinstance(detection, (list, tuple)) or len(detection) < 3:
                continue
            box, text, confidence = detection[0], str(detection[1] or "").strip(), float(detection[2] or 0)
            if not text or confidence < min_conf:
                continue
            detected += 1
            if len(texts) < 24:
                texts.append(text[:40])

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
        logger.info("Embedded text in generated image: detected=%d removed=%d%s",
                    detected, removed,
                    (" | " + "; ".join(texts[:8])) if texts else "")
        return {"removed": removed, "detected": detected, "texts": texts}
    except Exception as exc:
        logger.warning("Generated image text stripping failed: %s", exc)
        return {"removed": 0, "detected": None, "texts": [], "error": "text_strip_failed"}


def _is_fatal_together_error(status_code: int | None) -> bool:
    """Will this call fail the same way for every remaining slide?

    Keyed on the status alone. The Hugging Face classifier also reads error
    wording, which is that provider's phrasing and must not be carried
    across. 401/403 is a credential problem and 404 is a model that is not
    served; both repeat identically on slide two, so the deck should stop
    asking. 429 and 5xx are transient by definition and stay non-fatal.
    """
    return status_code in {401, 403, 404}


def _generate_huggingface(cfg: dict[str, Any], image_prompt: str) -> dict[str, Any]:
    """One Hugging Face Router call. Returns raw bytes, not a saved file.

    Behaviour is unchanged from before the provider split: the router answers
    with the image bytes themselves, and a cold model answers 503 once before
    it is warm.
    """
    token = _huggingface_token()
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
                logger.info(
                    "HF image generation OK | model=%s | size=%dx%d",
                    cfg["model"],
                    cfg["width"],
                    cfg["height"],
                )
                return {
                    "ok": True,
                    "bytes": response.content,
                    "mime": content_type.split(";", 1)[0] or "image/png",
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
        "error": last_error or "huggingface_image_generation_failed",
        "fatal": _is_fatal_huggingface_error(last_status, last_error),
    }


# --- Together rate limits ----------------------------------------------------
#
# A 10-slide deck asked for three images at once; Together answered two of
# them with 429 ("too many requests in a short window") and, with no retry,
# both slides fell back to a drawn diagram and an icon row. Two changes:
#
#   * image calls to Together queue behind a small semaphore (one at a time by
#     default), so a deck never bursts the account's window by itself;
#   * a 429 or 5xx is retried after the delay the provider asks for
#     (Retry-After), or a short backoff when it does not say, a bounded number
#     of times. Anything else - and above all 401/403/404, which repeat
#     identically - is answered at once, as before.
_TOGETHER_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_TOGETHER_MAX_RETRY_WAIT_S = 20.0
_together_slots: dict[int, threading.BoundedSemaphore] = {}
_together_slots_lock = threading.Lock()
# Indirection so tests never really sleep.
_sleep = time.sleep


def _together_slot() -> threading.BoundedSemaphore:
    """The semaphore every Together image call in this process shares."""
    limit = max(1, _env_int("TOGETHER_IMAGE_MAX_CONCURRENCY", 1))
    with _together_slots_lock:
        slot = _together_slots.get(limit)
        if slot is None:
            slot = _together_slots[limit] = threading.BoundedSemaphore(limit)
        return slot


def _together_retry_delay(response: Any, attempt: int) -> float:
    """Seconds to wait before retry ``attempt`` (0-based)."""
    try:
        asked = float(str((response.headers or {}).get("Retry-After") or "").strip())
    except (TypeError, ValueError, AttributeError):
        asked = 0.0
    delay = asked if asked > 0 else 2.0 * (2 ** attempt)
    return min(delay, _TOGETHER_MAX_RETRY_WAIT_S)


def _generate_together(cfg: dict[str, Any], image_prompt: str,
                       slot: threading.BoundedSemaphore | None = None,
                       retries: int | None = None, tenant: Any = "") -> dict[str, Any]:
    """One Together ``/v1/images/generations`` call. Returns raw bytes.

    Together answers with JSON rather than image bytes, and the base64 the
    request asks for comes back under a differently spelled key: the request
    field is ``response_format="base64"``, the response key is ``b64_json``.

    Base64 rather than a URL is deliberate. The pipeline needs the bytes on
    disk anyway - the OCR text strip rewrites the file in place - and the
    lifetime of a returned URL is not documented, so holding one would be a
    bet on somebody else's retention policy.
    """
    token = _together_token()
    body: dict[str, Any] = {
        "model": cfg["model"],
        "prompt": image_prompt,
        "width": cfg["width"],
        "height": cfg["height"],
        "n": 1,
        "response_format": "base64",
        # The provider default is jpeg; everything downstream assumes png.
        "output_format": "png",
    }
    steps = int(cfg.get("steps") or 0)
    if steps > 0:
        body["steps"] = steps

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    retries = max(0, _env_int("TOGETHER_IMAGE_RETRIES", 2) if retries is None else retries)
    # This process's own limit, then the service-wide one: every worker on
    # every server shares the Together account's image allowance, so the
    # slot that counts is the shared one (see provider_gate).
    from ai_services.core import provider_gate
    gate = provider_gate.together_images()
    with (slot or _together_slot()):
        try:
            # Long enough to ride out a busy hour: a slide that cannot get a
            # slot fails, and is only retried later in the deck.
            with gate.slot(tenant, max_wait_s=float(_env_int("TOGETHER_IMAGE_MAX_WAIT_S", 300))) as ticket:
                response = _together_attempts(cfg, headers, body, steps, retries, token,
                                              gate, ticket)
        except provider_gate.GateBusy as exc:
            logger.warning("Together image gen: no slot free (%s)", exc)
            return {"ok": False, "error": "image service busy; try again shortly", "fatal": False}
    if isinstance(response, dict):
        return response
    return _together_result(response, cfg, token)


def _together_attempts(cfg: dict[str, Any], headers: dict[str, str], body: dict[str, Any],
                       steps: int, retries: int, token: str, gate: Any, ticket: Any) -> Any:
    """The call and its retries, inside a held slot. Returns the response, or an error dict."""
    response = None
    for attempt in range(retries + 1):
        ticket.renew()
        try:
            logger.info(
                "Together image generation request | model=%s | size=%dx%d | steps=%s%s",
                cfg["model"], cfg["width"], cfg["height"], steps or "provider default",
                " | retry %d/%d" % (attempt, retries) if attempt else "",
            )
            response = requests.post(cfg["url"], headers=headers, json=body,
                                     timeout=cfg["timeout"])
        except Exception as exc:
            message = _scrub(exc, token)
            logger.warning("Together image gen error: %s", message)
            return {"ok": False, "error": f"Together image gen error: {message}",
                    "fatal": False}
        if (response.ok or response.status_code not in _TOGETHER_RETRYABLE
                or attempt == retries):
            break
        delay = _together_retry_delay(response, attempt)
        logger.info("Together image gen got %s; retrying in %.1fs",
                    response.status_code, delay)
        if response.status_code == 429:
            # The account's limit, not this call's: every process waits.
            gate.cool_down(delay)
        _sleep(delay)
    return response


def _together_result(response: Any, cfg: dict[str, Any], token: str) -> dict[str, Any]:
    """The image bytes from a finished response, or why there are none."""
    if not response.ok:
        err_text = _scrub(response.text[:500], token)
        logger.warning("Together image gen failed (%s): %s", response.status_code, err_text[:200])
        return {
            "ok": False,
            "error": f"Together image API error {response.status_code}: {err_text}",
            "fatal": _is_fatal_together_error(response.status_code),
        }

    try:
        payload = response.json()
    except Exception:
        return {"ok": False, "error": "Together image API returned a non-JSON body", "fatal": False}

    items = payload.get("data") if isinstance(payload, dict) else None
    first = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
    encoded = first.get("b64_json")
    if not encoded:
        if first.get("url"):
            # Asking for base64 and being handed a link is not something to
            # paper over by quietly downloading it: that would reintroduce
            # the URL lifetime this request exists to avoid.
            return {"ok": False,
                    "error": "Together returned a url; base64 was requested",
                    "fatal": False}
        return {"ok": False, "error": "Together response had no b64_json image", "fatal": False}

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return {"ok": False, "error": "Together returned undecodable base64 image data", "fatal": False}
    if not image_bytes:
        return {"ok": False, "error": "Together returned an empty image", "fatal": False}

    logger.info(
        "Together image generation OK | model=%s | size=%dx%d",
        cfg["model"], cfg["width"], cfg["height"],
    )
    return {"ok": True, "bytes": image_bytes, "mime": "image/png"}


_slide_slots: dict[int, threading.BoundedSemaphore] = {}


def _slide_slot() -> threading.BoundedSemaphore:
    """Whole-slide images queue separately from note/figure images.

    A deck of image slides needs several in flight to finish inside the
    request budget (one at a time, ten slides took longer than the API waits),
    and must not starve - or be starved by - the one-at-a-time figure queue.
    """
    limit = max(1, _env_int("SLIDE_IMAGE_CONCURRENCY", 3))
    with _together_slots_lock:
        slot = _slide_slots.get(limit)
        if slot is None:
            slot = _slide_slots[limit] = threading.BoundedSemaphore(limit)
        return slot


def generate_slide_image(prompt: str, *, model: str, width: int = 1536,
                         height: int = 864, tenant: Any = "") -> dict[str, Any]:
    """Paint one whole presentation slide. Together only.

    Deliberately NOT generate_note_image: that appends a grounding clause
    written for figures and then OCR-strips every piece of text from the
    result - which on a slide whose content IS its text would erase the
    slide. Here the text is the point, so the bytes are saved untouched.
    """
    if not _together_token():
        return {"ok": False, "error": "together_image_generation_not_configured"}
    cfg = image_generation_config()
    cfg = dict(cfg, model=model, width=width - width % 8, height=height - height % 8,
               # Slide models take their own default step count.
               steps=0,
               url=f"{(os.getenv('TOGETHER_IMAGE_BASE_URL', '').strip() or DEFAULT_TOGETHER_IMAGE_BASE_URL).rstrip('/')}/images/generations",
               timeout=max(30, _env_int("SLIDE_IMAGE_TIMEOUT_SECONDS", 90)))
    # More patience than a figure gets: a deck of slides is a burst, and a
    # live six-slide deck lost one slide to 429 after two retries at four
    # requests in flight. Waits are capped, so this stays inside the budget.
    # ``tenant`` (the school) is what the shared image gate shares fairly.
    result = _generate_together(cfg, prompt, slot=_slide_slot(),
                                retries=max(0, _env_int("SLIDE_IMAGE_RETRIES", 4)),
                                tenant=tenant)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error") or "generation failed",
                "fatal": bool(result.get("fatal"))}
    saved = _save_generated_image(result["bytes"], result.get("mime") or "image/png")
    return {"ok": True, "url": saved["url"], "path": saved["path"],
            "provider": PROVIDER_TOGETHER, "model": model}


def generate_note_image(prompt: str) -> dict[str, Any]:
    """
    Generate one strictly grounded educational image.

    Which provider answers is chosen by NOTES_IMAGE_PROVIDER and defaults to
    Hugging Face, so an unset deployment behaves exactly as it did before.
    Everything a provider does NOT decide stays shared: the grounding clause
    appended to the prompt, saving the bytes, stripping embedded text, and
    the shape of the dict that comes back - three callers depend on it.

    The upstream planner only sends prompts backed by exact note evidence.
    """
    cfg = image_generation_config()
    if not cfg["enabled"]:
        return {"ok": False, "error": "image_generation_disabled"}
    if not can_generate_note_images():
        return {"ok": False,
                "error": "%s_image_generation_not_configured" % cfg["provider"],
                "fatal": True}

    image_prompt = (
        f"{str(prompt or '').strip()}\n\n"
        "Generate one clean educational note illustration only. Use only the facts explicitly present in the prompt. "
        "Do not add unsupported facts, extra labels, logos, watermarks, decorative backgrounds, or unrelated objects."
    )

    if cfg["provider"] == PROVIDER_TOGETHER:
        produced = _generate_together(cfg, image_prompt)
    elif cfg["provider"] == PROVIDER_HUGGINGFACE:
        produced = _generate_huggingface(cfg, image_prompt)
    else:  # pragma: no cover - can_generate_note_images() already blocked it
        produced = {"ok": False,
                    "error": "unsupported image provider %r" % cfg["provider"],
                    "fatal": True}

    if not produced.get("ok"):
        return {
            "ok": False,
            "error": str(produced.get("error") or "")[:240] or "image_generation_failed",
            "fatal": bool(produced.get("fatal")),
        }

    saved = _save_generated_image(produced["bytes"], produced.get("mime") or "image/png")
    text_strip = _strip_embedded_text_from_image(saved["path"])
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
        # What the generator drew, separately from what we painted out. None
        # means the strip did not run, which is not the same as a clean image.
        "embedded_text_detected": text_strip.get("detected"),
        "embedded_text_samples": list(text_strip.get("texts") or []),
        "embedded_text_strip_error": text_strip.get("error"),
    }
