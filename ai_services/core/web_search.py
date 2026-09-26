"""Student-safe Google search (web, images, videos) for the AI Tutor, via Serper.

Every request asks Google for SafeSearch ("safe": "active"), and results are
filtered again here because the answers are shown to Class 1-12 students:
blocked domains and explicit words are dropped, social media is dropped, and
education sites are ranked first. Only YouTube results are returned as videos,
so every video can be embedded and played inside the app.

Without a Serper key, web search falls back to Wikipedia's search API (the same
pattern serpapi_images.py uses for images); images and videos return [].

Only titles, snippets and thumbnails are used — pages are never fetched — so
no third-party page content is copied into an answer.
"""
import html
import logging
import os
import re
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger("ai_services.web_search")

SERPER_BASE_URL = "https://google.serper.dev"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Ranked first when present. Override with TUTOR_WEB_PREFERRED_DOMAINS (comma-separated).
_DEFAULT_PREFERRED_DOMAINS = (
    "ncert.nic.in", "cbseacademic.nic.in", "diksha.gov.in", "gov.in", "nic.in",
    "wikipedia.org", "britannica.com", "khanacademy.org", "ck12.org",
    "bbc.co.uk", "nationalgeographic.com", "nasa.gov", "physicsclassroom.com",
    "mathsisfun.com", "byjus.com", "vedantu.com", "toppr.com", "geeksforgeeks.org",
    "learncbse.in", "teachoo.com", "shaalaa.com", "libretexts.org",
)

# Never shown. Adult/gambling sites, plus social media whose results are user
# posts rather than explanations. Extend with TUTOR_WEB_BLOCKED_DOMAINS.
_DEFAULT_BLOCKED_DOMAINS = (
    "pornhub.com", "xvideos.com", "xnxx.com", "xhamster.com", "redtube.com", "youporn.com",
    "onlyfans.com", "chaturbate.com", "stripchat.com", "bet365.com", "1xbet.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "reddit.com", "pinterest.com", "snapchat.com", "tumblr.com",
)

# Whole-word matches on title/snippet/URL. Deliberately excludes clinical words
# like "sex" — sexual reproduction is part of the CBSE Class 10 syllabus.
_BLOCKED_WORDS = re.compile(
    r"\b(porn\w*|xxx|nudes?|naked|nsfw|onlyfans|hentai|erotic\w*|escorts?|"
    r"casinos?|betting|gambling|sexy)\b",
    re.IGNORECASE,
)


def _domains_from_env(name: str, default: tuple) -> tuple:
    raw = os.getenv(name, "").strip()
    extra = tuple(d.strip().lower() for d in raw.split(",") if d.strip())
    if name == "TUTOR_WEB_BLOCKED_DOMAINS":
        return default + extra
    return extra or default


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _matches(host: str, domains: tuple) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_safe_result(url: str, *texts: str) -> bool:
    """False for blocked domains, unparseable URLs or explicit words."""
    host = _host(url)
    if not host or _matches(host, _domains_from_env("TUTOR_WEB_BLOCKED_DOMAINS", _DEFAULT_BLOCKED_DOMAINS)):
        return False
    return not _BLOCKED_WORDS.search(" ".join([url or "", *[t or "" for t in texts]]))


def is_preferred(url: str) -> bool:
    return _matches(_host(url), _domains_from_env("TUTOR_WEB_PREFERRED_DOMAINS", _DEFAULT_PREFERRED_DOMAINS))


def _serper_key() -> str:
    return (
        os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY") or os.getenv("SERPAPI_KEY") or ""
    ).strip()


def _serper(endpoint: str, query: str, num: int) -> dict:
    response = requests.post(
        f"{SERPER_BASE_URL}/{endpoint}",
        headers={"X-API-KEY": _serper_key(), "Content-Type": "application/json"},
        json={"q": query, "num": num, "gl": "in", "hl": "en", "safe": "active"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "")).strip()[:300]


# ── Web ──────────────────────────────────────────────────────────────────────

def _search_wikipedia(query: str, fetch: int) -> list:
    response = requests.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": fetch, "format": "json", "utf8": 1,
        },
        headers={"User-Agent": "EDDVA-AI-Tutor/1.0"},
        timeout=10,
    )
    response.raise_for_status()
    results = []
    for item in (response.json().get("query") or {}).get("search") or []:
        title = item.get("title") or ""
        results.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "snippet": html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet") or "")),
        })
    return results


def _quick_facts(data: dict) -> list:
    """Google's answer box and knowledge panel as short fact strings.

    These are Google's own direct answers ("Pressure is force per unit area"),
    usually more precise than any single result snippet.
    """
    facts = []
    box = data.get("answerBox") or {}
    box_text = box.get("answer") or box.get("snippet") or ""
    if box_text and is_safe_result(box.get("link") or "https://google.com", box.get("title"), box_text):
        facts.append(f"{box.get('title') or 'Answer'}: {box_text}".strip())
    graph = data.get("knowledgeGraph") or {}
    if graph.get("description") and is_safe_result(graph.get("descriptionLink") or "https://google.com",
                                                    graph.get("title"), graph.get("description")):
        attrs = "; ".join(f"{k}: {v}" for k, v in list((graph.get("attributes") or {}).items())[:5])
        facts.append(" ".join(s for s in (
            f"{graph.get('title') or ''} ({graph.get('type') or 'topic'}): {graph['description']}", attrs,
        ) if s).strip())
    return [f[:600] for f in facts]


def search_google(query: str, limit: int = 6) -> dict:
    """{results: [...], facts: [...]} — safe Google results (education sites first)
    plus Google's quick facts. Never raises; any failure returns empty lists."""
    empty = {"results": [], "facts": []}
    cleaned = _clean_query(query)
    if not cleaned:
        return empty
    facts = []
    try:
        if _serper_key():
            data = _serper("search", cleaned, 10)
            facts = _quick_facts(data)
            raw = [
                {"title": o.get("title") or "", "url": o.get("link") or "", "snippet": o.get("snippet") or ""}
                for o in (data.get("organic") or [])
            ]
        else:
            raw = _search_wikipedia(cleaned, 10)
    except Exception as exc:
        logger.warning("Tutor web search failed for %r: %s", cleaned, exc)
        return empty

    results, seen = [], set()
    for item in raw:
        url = item["url"]
        if url in seen or not item["snippet"] or not is_safe_result(url, item["title"], item["snippet"]):
            continue
        # YouTube results are shown as videos instead.
        if _matches(_host(url), ("youtube.com", "youtu.be")):
            continue
        seen.add(url)
        results.append({**item, "site": _host(url).removeprefix("www.")})
    results.sort(key=lambda r: 0 if is_preferred(r["url"]) else 1)  # stable: keeps Google's order within a tier
    return {"results": results[:limit], "facts": facts}


def search_web(query: str, limit: int = 4) -> list:
    """Up to `limit` safe Google results as {title, url, snippet, site}, education sites first."""
    return search_google(query, limit)["results"]


# ── Images ───────────────────────────────────────────────────────────────────

def search_images(query: str, limit: int = 10) -> list:
    """Safe Google Images results as {title, imageUrl, thumbnailUrl, source, pageUrl}.

    Unranked beyond safety filtering — the caller orders them for relevance.
    """
    cleaned = _clean_query(query)
    if not cleaned or not _serper_key():
        return []
    try:
        items = _serper("images", cleaned, max(limit, 10)).get("images") or []
    except Exception as exc:
        logger.warning("Tutor image search failed for %r: %s", cleaned, exc)
        return []
    results = []
    for item in items:
        image_url = item.get("imageUrl") or ""
        page_url = item.get("link") or ""
        if not image_url.startswith("http"):
            continue
        if not is_safe_result(page_url or image_url, item.get("title"), item.get("source")):
            continue
        results.append({
            "title": item.get("title") or "",
            "imageUrl": image_url,
            "thumbnailUrl": item.get("thumbnailUrl") or image_url,
            "source": item.get("source") or _host(page_url).removeprefix("www."),
            "pageUrl": page_url,
        })
    return results[:limit]


# ── Videos ───────────────────────────────────────────────────────────────────

_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_video_id(url: str) -> str:
    """The 11-character id of a YouTube watch/short/youtu.be URL, or ''."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            candidate = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
    return candidate if _YOUTUBE_ID.match(candidate) else ""


def search_videos(query: str, limit: int = 3) -> list:
    """Safe YouTube results as {title, url, videoId, thumbnailUrl, channel, duration},
    in Google's order — the caller ranks them for relevance and length."""
    cleaned = _clean_query(query)
    if not cleaned or not _serper_key():
        return []
    try:
        items = _serper("videos", cleaned, 10).get("videos") or []
    except Exception as exc:
        logger.warning("Tutor video search failed for %r: %s", cleaned, exc)
        return []
    results, seen = [], set()
    for item in items:
        url = item.get("link") or ""
        video_id = youtube_video_id(url)
        if not video_id or video_id in seen:
            continue
        if not is_safe_result(url, item.get("title"), item.get("snippet"), item.get("channel")):
            continue
        seen.add(video_id)
        results.append({
            "title": item.get("title") or "YouTube video",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "videoId": video_id,
            "thumbnailUrl": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            "channel": item.get("channel") or "",
            "duration": item.get("duration") or "",
        })
        if len(results) >= limit:
            break
    return results
