"""Small SerpApi Google Images adapter used by lecture-note enrichment."""

import os

import requests


SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


def search_google_images(query: str, limit: int = 5, language: str = "") -> list[dict]:
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPAPI_KEY is not configured")

    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return []

    normalized_language = str(language or "").strip().lower()
    google_language = "or" if normalized_language in {"od", "odia", "od-in", "or-in"} else normalized_language
    params = {
        "engine": "google_images",
        "q": cleaned_query,
        "api_key": api_key,
        "safe": "active",
        "ijn": 0,
    }
    if google_language:
        params["hl"] = google_language
    if google_language == "or":
        params["gl"] = "in"

    response = requests.get(
        SERPAPI_SEARCH_URL,
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))

    results = []
    for item in payload.get("images_results") or []:
        image_url = item.get("original") or item.get("thumbnail")
        if not image_url or item.get("unsafe") is True:
            continue
        results.append(
            {
                "imageUrl": image_url,
                "thumbnailUrl": item.get("thumbnail"),
                "title": item.get("title"),
                "source": item.get("source"),
                "sourcePage": item.get("link"),
            }
        )
        if len(results) >= max(1, min(int(limit or 5), 10)):
            break
    return results
