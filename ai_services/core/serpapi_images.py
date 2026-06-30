"""Small Serper.dev Google Images adapter used by lecture-note enrichment."""

import os

import logging
import requests

logger = logging.getLogger(__name__)

SERPER_SEARCH_URL = "https://google.serper.dev/images"


def search_google_images(query: str, limit: int = 5, language: str = "") -> list[dict]:
    logger.info("search_google_images: query=%r, limit=%r, language=%r", query, limit, language)
    api_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY or SERPER_KEY is not configured")

    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        logger.info("search_google_images: empty query, returning []")
        return []

    normalized_language = str(language or "").strip().lower()
    google_language = "or" if normalized_language in {"od", "odia", "od-in", "or-in"} else normalized_language

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": cleaned_query,
        "num": max(1, min(int(limit or 5), 20)),
    }

    if google_language:
        payload["hl"] = google_language
    if google_language == "or":
        payload["gl"] = "in"

    response = requests.post(
        SERPER_SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("images") or []:
        image_url = item.get("imageUrl") or item.get("thumbnailUrl")
        if not image_url:
            continue
        results.append(
            {
                "imageUrl": image_url,
                "thumbnailUrl": item.get("thumbnailUrl"),
                "title": item.get("title"),
                "source": item.get("source"),
                "sourcePage": item.get("link"),
            }
        )
        if len(results) >= max(1, min(int(limit or 5), 10)):
            break
    logger.info("search_google_images: found %d results", len(results))
    return results
