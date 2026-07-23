"""Small Serper.dev Google Images adapter used by lecture-note enrichment.

Falls back to Wikipedia REST API when no Serper key is configured — Wikipedia
images are free, always available, and educationally appropriate.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

SERPER_SEARCH_URL = "https://google.serper.dev/images"


def _search_wikipedia_images(query: str, limit: int = 3) -> list[dict]:
    """Return images from Wikipedia for `query` — no API key required."""
    results: list[dict] = []

    def _try_summary(title: str) -> dict | None:
        try:
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                headers={"User-Agent": "eddva-education-app/1.0"},
                timeout=10,
            )
            if not resp.ok:
                return None
            data = resp.json()
            img = data.get("thumbnail") or data.get("originalimage")
            if img and img.get("source"):
                return {
                    "imageUrl": img["source"],
                    "thumbnailUrl": img.get("source"),
                    "title": data.get("title", title),
                    "source": "Wikipedia",
                    "sourcePage": (data.get("content_urls") or {}).get("desktop", {}).get("page"),
                }
        except Exception:
            pass
        return None

    # 1. Direct summary lookup with the raw query
    hit = _try_summary(query)
    if hit:
        results.append(hit)

    # 2. If nothing yet, do a Wikipedia search and try the top result
    if not results:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "srlimit": 3, "format": "json"},
                headers={"User-Agent": "eddva-education-app/1.0"},
                timeout=10,
            )
            if resp.ok:
                for item in (resp.json().get("query") or {}).get("search", []):
                    hit = _try_summary(item["title"])
                    if hit and hit not in results:
                        results.append(hit)
                    if len(results) >= limit:
                        break
        except Exception:
            pass

    logger.info("Wikipedia fallback: found %d image(s) for query %r", len(results), query)
    return results[:limit]


def search_google_images(query: str, limit: int = 5, language: str = "") -> list[dict]:
    logger.info("search_google_images: query=%r, limit=%r, language=%r", query, limit, language)
    api_key = (
        os.getenv("SERPER_API_KEY") or
        os.getenv("SERPER_KEY") or
        os.getenv("SERPAPI_KEY") or
        ""
    ).strip()

    if not api_key:
        logger.warning("No Serper API key configured — falling back to Wikipedia image search")
        return _search_wikipedia_images(query, limit)

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

    try:
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
        logger.info("search_google_images: found %d results via Serper", len(results))

        # If Serper returned nothing, try Wikipedia as a fallback
        if not results:
            logger.info("Serper returned 0 results — trying Wikipedia fallback")
            return _search_wikipedia_images(cleaned_query, limit)

        return results

    except Exception as exc:
        logger.warning("Serper API error (%s) — falling back to Wikipedia: %s", type(exc).__name__, exc)
        return _search_wikipedia_images(cleaned_query, limit)
