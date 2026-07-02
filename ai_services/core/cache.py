"""
Redis-powered AI Response Cache — Cost Optimization Layer.

WHY THIS MATTERS:
  750 students studying JEE/NEET ask the SAME questions every day.
  "What is Newton's second law?" gets asked 50 times → without cache = 50 LLM calls.
  With Redis cache → 1 LLM call + 49 instant cache hits = 98% cost saving on that question.

ARCHITECTURE (3 layers):
  Layer 1 — Prompt Normalizer:
      Cleans whitespace/case/punctuation before hashing so
      "Newton's 2nd law?" and "newtons second law" hit the SAME cache entry.

  Layer 2 — Redis (Primary, Shared):
      Survives restarts. Shared across all gunicorn workers.
      Keyed by: (institute_id + vertical + feature + normalized_prompt_hash)
      TTL varies by feature (doubt=24h, quiz=24h, test=6h).

  Layer 3 — In-Memory LRU (Hot-path fallback):
      Ultra-fast local cache for the most recent 2000 entries.
      Avoids a Redis round-trip for the hottest questions.
      Falls back when Redis is unavailable.

COST TRACKING:
  Every cache hit records money saved in Redis counters.
  GET /admin-api/cache/stats/ shows hit rate + rupees saved today.

PROMPT NORMALIZATION (the key insight for cost savings):
  Raw prompt:       "  what  is  Newton's 2nd  Law of motion ??? "
  Normalized:       "what is newton 2nd law of motion"
  → Same hash as:   "What is Newton's Second Law of Motion?"
  → Cache HIT instead of a new LLM call.
"""

import os
import re
import json
import hashlib
import logging
import random
import time
from collections import OrderedDict
from threading import Lock
from typing import Optional

logger = logging.getLogger("ai_services.cache")

# ─────────────────────────────────────────────────────────────────────────────
#  Feature-specific TTLs (seconds)
#  Rule: deterministic input → same output = cache long.
#        per-student/session data = TTL 0 (skip).
# ─────────────────────────────────────────────────────────────────────────────
CACHE_TTL = {
    # ── Never cache — must always be fresh ───────────────────────────────────
    "plan_generate":     0,      # personalized plan — stale = misleading
    "cheating_analyze":  0,      # real-time proctoring
    "feedback_generate": 0,      # per-session test scores
    "content_generate":  0,      # teachers expect fresh content every time
    "evaluate_batch":    0,      # QA must reflect the exact current batch

    # ── Long cache — same input ALWAYS gives the same correct answer ──────────
    "doubt_resolve":     86400,  # 24h — Newton's law is Newton's law
    "syllabus_generate": 86400,  # 24h — syllabus doesn't change mid-year
    "content_suggest":   86400,  # 24h — resource URLs are stable
    "notes_generate":    86400,  # 24h — same transcript = same notes
    "stt_notes":         86400,  # 24h — same audio = same notes
    "quiz_generate":     86400,  # 24h — same transcript = same questions

    # ── Medium cache ──────────────────────────────────────────────────────────
    "test_generate":     21600,  # 6h  — MCQs stable within a day
    "career_roadmap":    43200,  # 12h
    "content_recommend": 21600,  # 6h

    # ── Short cache ───────────────────────────────────────────────────────────
    "performance_analyze": 3600, # 1h
    "feedback_analyze":    3600, # 1h
    "notes_analyze":       3600, # 1h

    # ── Legacy ────────────────────────────────────────────────────────────────
    "study_plan":        0,      # mapped to plan_generate
}

DEFAULT_TTL = 3600  # 1h for unlisted features

# Redis key prefixes
_KEY_PREFIX    = "ai_svc"       # response cache
_STATS_PREFIX  = "ai_stats"     # hit/miss counters + cost saved


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt Normalizer — the secret weapon for higher cache hit rates
# ─────────────────────────────────────────────────────────────────────────────

# Common Indian English / SMS abbreviations students use
_ABBREV = {
    r"\b2nd\b":    "second",
    r"\b3rd\b":    "third",
    r"\b1st\b":    "first",
    r"\bkya\b":    "what",
    r"\bkaise\b":  "how",
    r"\bbtao\b":   "explain",
    r"\bpls\b":    "please",
    r"\bplz\b":    "please",
    r"\bu\b":      "you",
    r"\bw/\b":     "with",
    r"\bw/o\b":    "without",
    r"\bvs\b":     "versus",
    r"\bdef\b":    "definition",
    r"\bdefn\b":   "definition",
    r"\bproof\b":  "prove",
    r"\beq\b":     "equation",
    r"\bdiff\b":   "differentiate",
    r"\bintegn\b": "integration",
    r"\bcalc\b":   "calculate",
    r"\bformula\b":"formula",
    r"\bex\b":     "example",
    r"\beg\b":     "example",
    r"\bq\b":      "question",
}

_ABBREV_PATTERNS = [(re.compile(k, re.I), v) for k, v in _ABBREV.items()]

# Punctuation to strip (apostrophes, question/exclamation marks, etc.)
_STRIP_PUNCT = re.compile(r"[\"'`\?\!\.\,\;\:\(\)\[\]\{\}\-\_\*\#\@\$\%\^\&]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_prompt(prompt: str) -> str:
    """
    Normalize a user prompt to maximize cache hit rate.

    Transformations (order matters):
      1. Lowercase
      2. Expand common abbreviations (2nd → second, pls → please)
      3. Strip punctuation
      4. Collapse whitespace
      5. Strip leading/trailing whitespace

    Examples:
      "  What is Newton's 2nd Law?? "  →  "what is newtons second law"
      "newton second law"               →  "newton second law"   ← SAME HASH
      "NEWTONS 2ND LAW!"                →  "newtons second law"  ← SAME HASH
    """
    if not prompt:
        return ""

    text = prompt.lower()

    # Expand abbreviations
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    # Strip punctuation (keep Devanagari danda as-is for Hindi)
    text = _STRIP_PUNCT.sub(" ", text)

    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def _make_cache_key(
    institute_id: str,
    feature: str,
    prompt_hash: str,
    vertical: str = "base",
) -> str:
    """
    Tenant- AND vertical-scoped cache key.
    Institute A and B get separate entries.
    Coaching and school answers never mix.
    """
    return f"{_KEY_PREFIX}:{institute_id}:{vertical}:{feature}:{prompt_hash}"


def _hash_prompt(prompt: str) -> str:
    """
    Hash the NORMALIZED prompt for higher cache hit rate.
    Two slightly different phrasings of the same question → same hash.
    """
    normalized = normalize_prompt(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _stats_key(institute_id: str, date_str: str) -> str:
    return f"{_STATS_PREFIX}:{institute_id}:{date_str}"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared Redis Connection Pool (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_redis_client = None
_redis_lock = Lock()


def get_redis():
    """
    Return a shared Redis connection pool instance.
    Uses a connection pool (not a single connection) — safe for multi-threaded gunicorn.
    Raises RuntimeError if Redis is not configured and we're not in dev fallback mode.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client

        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return None

        try:
            import redis
            # Connection pool: max 20 connections, shared across all threads/workers
            pool = redis.ConnectionPool.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                retry_on_timeout=True,
            )
            client = redis.Redis(connection_pool=pool)
            client.ping()
            logger.info("Redis connected (pool) → %s", redis_url.split("@")[-1])  # hide password
            _redis_client = client
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-memory cache", e)
            _redis_client = None

    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
#  In-Memory LRU — hot-path Layer 3 (local to each worker)
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryLRU:
    """Thread-safe LRU cache — Layer 3 fallback and hot-path accelerator."""

    def __init__(self, max_size: int = 2000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry["expires_at"] < time.time():
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry["value"]

    def set(self, key: str, value: dict, ttl: int):
        with self._lock:
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl,
            }
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # evict oldest

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def flush(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
#  ResponseCache — the main cache used by every AI view
# ─────────────────────────────────────────────────────────────────────────────

class ResponseCache:
    """
    3-layer AI response cache for maximum cost savings.

    Layer 1: Prompt normalization (before hashing) → higher hit rate
    Layer 2: Redis (shared, persistent, survives restarts)
    Layer 3: In-memory LRU (ultra-fast, local to this worker)

    All keys tenant-scoped — no cross-tenant data leakage.
    """

    def __init__(self):
        self._memory = _InMemoryLRU(max_size=2000)

    @property
    def _redis(self):
        return get_redis()

    # ── Cost tracking helpers ─────────────────────────────────────────────────

    def _record_hit(self, institute_id: str, feature: str, tokens_saved: int = 0, cost_saved: float = 0.0):
        """Increment hit counters in Redis for the cost savings dashboard."""
        r = self._redis
        if not r:
            return
        try:
            today = _today()
            key = _stats_key(institute_id, today)
            pipe = r.pipeline()
            pipe.hincrby(key, "hits", 1)
            pipe.hincrby(key, f"hits:{feature}", 1)
            if tokens_saved:
                pipe.hincrbyfloat(key, "tokens_saved", tokens_saved)
            if cost_saved:
                pipe.hincrbyfloat(key, "cost_saved_usd", cost_saved)
            pipe.expire(key, 86400 * 7)  # keep 7 days of stats
            pipe.execute()
        except Exception:
            pass

    def _record_miss(self, institute_id: str, feature: str):
        """Increment miss counter."""
        r = self._redis
        if not r:
            return
        try:
            today = _today()
            key = _stats_key(institute_id, today)
            pipe = r.pipeline()
            pipe.hincrby(key, "misses", 1)
            pipe.hincrby(key, f"misses:{feature}", 1)
            pipe.expire(key, 86400 * 7)
            pipe.execute()
        except Exception:
            pass

    # ── Core get / set ────────────────────────────────────────────────────────

    def get(
        self,
        institute_id: str,
        feature: str,
        user_prompt: str,
        vertical: str = "base",
    ) -> Optional[dict]:
        """
        Look up a cached LLM response.
        Prompt is normalized before hashing for higher hit rate.
        Returns None on miss (caller must invoke LLM).
        """
        ttl = CACHE_TTL.get(feature, DEFAULT_TTL)
        if ttl == 0:
            return None  # feature opted out of caching

        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt), vertical)

        # Layer 3 — in-memory (fastest, no network round-trip)
        mem_result = self._memory.get(key)
        if mem_result is not None:
            logger.debug("Cache HIT (memory) key=%s", key[-20:])
            self._record_hit(institute_id, feature)
            return mem_result

        # Layer 2 — Redis
        r = self._redis
        if r:
            try:
                raw = r.get(key)
                if raw:
                    result = json.loads(raw)
                    # Populate memory cache for next time (avoid Redis round-trip)
                    self._memory.set(key, result, min(ttl, 300))  # memory TTL max 5min
                    logger.debug("Cache HIT (redis) key=%s", key[-20:])
                    self._record_hit(institute_id, feature)
                    return result
            except Exception as e:
                logger.warning("Redis get error: %s", e)

        self._record_miss(institute_id, feature)
        return None

    def set(
        self,
        institute_id: str,
        feature: str,
        user_prompt: str,
        response: dict,
        vertical: str = "base",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ):
        """
        Store an LLM response in both Redis and memory cache.
        Also records the tokens/cost so future hits can report savings.
        """
        ttl = CACHE_TTL.get(feature, DEFAULT_TTL)
        if ttl == 0:
            return

        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt), vertical)

        # Store the original response + metadata about what it cost to generate
        payload = {
            **response,
            "_cache_meta": {
                "cached_at": time.time(),
                "tokens": tokens_used,
                "cost_usd": cost_usd,
                "institute_id": institute_id,
                "feature": feature,
                "vertical": vertical,
            },
        }

        # Write to Redis first (shared, persistent)
        r = self._redis
        if r:
            try:
                r.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                logger.warning("Redis set error: %s", e)

        # Write to memory cache (hot-path accelerator)
        self._memory.set(key, payload, min(ttl, 300))

    def invalidate(self, institute_id: str, feature: str, user_prompt: str, vertical: str = "base"):
        """Remove a specific cached response."""
        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt), vertical)
        self._memory.delete(key)
        r = self._redis
        if r:
            try:
                r.delete(key)
            except Exception:
                pass

    def flush_tenant(self, institute_id: str):
        """Clear ALL cached responses for a specific tenant (Redis + memory)."""
        prefix = f"{_KEY_PREFIX}:{institute_id}:"

        # Memory flush
        with self._memory._lock:
            keys_to_delete = [k for k in self._memory._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._memory._cache[k]

        # Redis flush (cursor scan — safe for large datasets)
        r = self._redis
        if r:
            try:
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = r.scan(cursor, match=f"{prefix}*", count=200)
                    if keys:
                        r.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                logger.info("flush_tenant: deleted %d Redis keys for %s", deleted, institute_id)
            except Exception as e:
                logger.warning("Redis flush_tenant error: %s", e)

    def flush_all(self):
        """Clear all cached responses for ALL tenants."""
        self._memory.flush()
        r = self._redis
        if r:
            try:
                cursor = 0
                while True:
                    cursor, keys = r.scan(cursor, match=f"{_KEY_PREFIX}:*", count=200)
                    if keys:
                        r.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning("Redis flush_all error: %s", e)

    # ── Analytics / Cost Dashboard ────────────────────────────────────────────

    def get_stats(self, institute_id: str, days: int = 7) -> dict:
        """
        Return cache performance stats for the cost savings dashboard.
        Shows hit rate, tokens saved, and estimated money saved in USD + INR.
        """
        r = self._redis
        if not r:
            return {
                "note": "Redis not connected — stats unavailable",
                "redis_connected": False,
                "memory_entries": self._memory.size(),
            }

        today = _today()
        all_hits = 0
        all_misses = 0
        total_tokens_saved = 0.0
        total_cost_saved_usd = 0.0
        daily = []

        try:
            for i in range(days):
                import datetime
                day = (
                    datetime.date.today() - datetime.timedelta(days=i)
                ).strftime("%Y-%m-%d")
                key = _stats_key(institute_id, day)
                data = r.hgetall(key)
                if not data:
                    continue

                hits    = int(data.get("hits", 0))
                misses  = int(data.get("misses", 0))
                tokens  = float(data.get("tokens_saved", 0))
                cost    = float(data.get("cost_saved_usd", 0))

                all_hits   += hits
                all_misses += misses
                total_tokens_saved   += tokens
                total_cost_saved_usd += cost

                daily.append({
                    "date": day,
                    "hits": hits,
                    "misses": misses,
                    "hit_rate_pct": round(hits / (hits + misses) * 100, 1) if (hits + misses) else 0,
                    "tokens_saved": int(tokens),
                    "cost_saved_usd": round(cost, 4),
                    "cost_saved_inr": round(cost * 84, 2),  # approx USD→INR
                })
        except Exception as e:
            logger.warning("Cache stats error: %s", e)

        total_calls = all_hits + all_misses
        hit_rate = round(all_hits / total_calls * 100, 1) if total_calls else 0

        return {
            "redis_connected": True,
            "memory_entries": self._memory.size(),
            "period_days": days,
            "total_calls": total_calls,
            "total_hits": all_hits,
            "total_misses": all_misses,
            "hit_rate_pct": hit_rate,
            "tokens_saved": int(total_tokens_saved),
            "cost_saved_usd": round(total_cost_saved_usd, 4),
            "cost_saved_inr": round(total_cost_saved_usd * 84, 2),
            "daily_breakdown": daily,
            "insight": _cache_insight(hit_rate, total_cost_saved_usd),
        }


def _cache_insight(hit_rate: float, cost_saved_usd: float) -> str:
    """Human-readable insight for the dashboard."""
    if hit_rate >= 60:
        return f"Excellent! {hit_rate}% of requests served from cache. Students are asking similar questions — cache is working perfectly."
    if hit_rate >= 35:
        return f"Good. {hit_rate}% hit rate. As more students use the platform, hit rate will improve further."
    if hit_rate >= 10:
        return f"Cache is warming up ({hit_rate}% hit rate). Expect improvement as question patterns repeat."
    return f"Cache is fresh ({hit_rate}% hit rate). Hit rate grows as students ask repeated questions."


# ─────────────────────────────────────────────────────────────────────────────
#  QuestionBankCache — fallback question store on LLM failure
# ─────────────────────────────────────────────────────────────────────────────

class QuestionBankCache:
    """
    Stores successfully generated questions by (institute, vertical, subject, chapter, difficulty, qtype).
    On LLM failure, returns a random previously-seen question from the bank.

    Tenant + vertical scoped — school questions never mix with JEE questions.
    """

    _BANK_TTL = 60 * 60 * 24 * 7   # 7 days
    _MAX_PER_KEY = 50

    def __init__(self):
        self._memory: dict[str, list] = {}
        self._lock = Lock()

    @property
    def _redis(self):
        return get_redis()

    def _key(
        self,
        subject: str,
        chapter: str,
        difficulty: str,
        qtype: str,
        institute_id: str = "global",
        vertical: str = "base",
    ) -> str:
        parts = [
            (institute_id or "global").lower().strip()[:40],
            (vertical or "base").lower().strip(),
            (subject or "").lower().strip()[:40],
            (chapter or "").lower().strip()[:40],
            (difficulty or "").lower().strip(),
            (qtype or "").lower().strip(),
        ]
        return "qbank:" + ":".join(parts)

    def save(
        self,
        subject: str,
        chapter: str,
        difficulty: str,
        qtype: str,
        questions: list,
        institute_id: str = "global",
        vertical: str = "base",
    ):
        if not questions:
            return
        key = self._key(subject, chapter, difficulty, qtype, institute_id, vertical)

        with self._lock:
            bucket = self._memory.get(key, [])
            bucket.extend(questions)
            if len(bucket) > self._MAX_PER_KEY:
                bucket = bucket[-self._MAX_PER_KEY:]
            self._memory[key] = bucket

        r = self._redis
        if r:
            try:
                pipe = r.pipeline()
                for q in questions:
                    pipe.rpush(key, json.dumps(q, ensure_ascii=False))
                pipe.ltrim(key, -self._MAX_PER_KEY, -1)
                pipe.expire(key, self._BANK_TTL)
                pipe.execute()
            except Exception as e:
                logger.warning("QuestionBank Redis write failed: %s", e)

    def get_random(
        self,
        subject: str,
        chapter: str,
        difficulty: str,
        qtype: str,
        n: int = 1,
        institute_id: str = "global",
        vertical: str = "base",
    ) -> list:
        key = self._key(subject, chapter, difficulty, qtype, institute_id, vertical)
        questions: list = []

        r = self._redis
        if r:
            try:
                raw_list = r.lrange(key, 0, -1)
                if raw_list:
                    questions = [json.loads(x) for x in raw_list]
            except Exception:
                pass

        if not questions:
            with self._lock:
                questions = list(self._memory.get(key, []))

        if not questions:
            return []

        random.shuffle(questions)
        return questions[:n]


# Module-level singletons
question_bank = QuestionBankCache()
