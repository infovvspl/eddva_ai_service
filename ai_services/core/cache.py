"""
Tenant-scoped response caching — Redis-backed with in-memory fallback.

Same JEE/NEET questions get asked millions of times. Caching identical
or near-identical prompts saves 15-20% on LLM costs.

IMPORTANT: Cache keys are scoped by (institute_id + feature + prompt_hash)
so Institute A never sees Institute B's cached responses.

Strategy:
  - Hash the (institute + feature + user_prompt) to create a cache key
  - TTL varies by feature (content=24h, feedback=1h, tests=6h)
  - Falls back to in-memory LRU cache if Redis is unavailable
"""

import os
import json
import hashlib
import logging
import random
import time
from collections import OrderedDict
from threading import Lock
from typing import Optional

logger = logging.getLogger("ai_services.cache")

# Feature-specific TTLs in seconds.
# Rule of thumb:
#   - Same inputs ALWAYS produce the same useful output → cache long (24h)
#   - Output is per-session / per-student-state / must be fresh → TTL=0 (skip cache)
#   - Not listed here → DEFAULT_TTL (1 hour)
CACHE_TTL = {
    # ── Never cache — must always be fresh ────────────────────────────────────
    "plan_generate":     0,      # Regenerate must produce a new plan every time
    "cheating_analyze":  0,      # Real-time proctoring data
    "feedback_generate": 0,      # Per-session test scores — stale feedback is misleading

    # ── Long cache — deterministic, stable content ─────────────────────────────
    "doubt_resolve":     86400,  # 24h — same physics/chemistry question = same answer
    "syllabus_generate": 86400,  # 24h — exam syllabus doesn't change mid-year
    "content_suggest":   86400,  # 24h — resource URLs don't change often
    "notes_generate":    86400,  # 24h — same transcript = same notes
    "stt_notes":         86400,  # 24h — same audio = same notes

    # ── Long cache — deterministic, stable content (bridge endpoints) ─────────
    "quiz_generate":     86400,  # 24h — same transcript always produces same questions
    "content_generate":  0,      # never cache — teachers expect fresh content on every generate

    # ── Medium cache ──────────────────────────────────────────────────────────
    "test_generate":     21600,  # 6h  — topic MCQs are stable within a day
    "career_roadmap":    43200,  # 12h
    "study_plan":        0,      # legacy key — mapped to plan_generate now; disable
    "content_recommend": 21600,  # 6h  — recommendations change as weak topics evolve

    # ── Short cache ───────────────────────────────────────────────────────────
    "performance_analyze": 3600, # 1h — scores change after new tests
    "feedback_analyze":    3600, # 1h
    "notes_analyze":       3600, # 1h
}

DEFAULT_TTL = 3600  # 1 hour — for features not explicitly listed above


def _make_cache_key(institute_id: str, feature: str, prompt_hash: str) -> str:
    """Tenant-scoped cache key. Institute A and B get separate cache entries."""
    return f"ai_svc:{institute_id}:{feature}:{prompt_hash}"


def _hash_prompt(prompt: str) -> str:
    """Deterministic hash of the user prompt (not system prompt — that's constant)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class _InMemoryLRU:
    """Thread-safe LRU cache as Redis fallback. Max 5000 entries."""

    def __init__(self, max_size: int = 5000):
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
                self._cache.popitem(last=False)

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def flush(self):
        with self._lock:
            self._cache.clear()


class ResponseCache:
    """
    Dual-layer cache: Redis (primary) → In-memory LRU (fallback).
    All keys are tenant-scoped — no cross-tenant data leakage.
    """

    def __init__(self):
        self._redis = None
        self._memory = _InMemoryLRU()
        self._init_redis()

    def _init_redis(self):
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            logger.info("REDIS_URL not set — using in-memory LRU cache")
            return
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True, socket_timeout=1)
            self._redis.ping()
            logger.info("Redis cache connected: %s", redis_url)
        except Exception as e:
            logger.warning("Redis unavailable (%s), using in-memory LRU fallback", e)
            self._redis = None

    def get(self, institute_id: str, feature: str, user_prompt: str) -> Optional[dict]:
        """Look up cached LLM response scoped to this tenant. Returns None on miss."""
        ttl = CACHE_TTL.get(feature, DEFAULT_TTL)
        if ttl == 0:
            return None  # feature opted out of caching

        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt))

        # Try Redis first
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    logger.debug("Cache HIT (redis) %s", key)
                    return json.loads(raw)
            except Exception:
                pass

        # Fallback to memory
        result = self._memory.get(key)
        if result:
            logger.debug("Cache HIT (memory) %s", key)
        return result

    def set(self, institute_id: str, feature: str, user_prompt: str, response: dict):
        """Store LLM response in tenant-scoped cache."""
        ttl = CACHE_TTL.get(feature, DEFAULT_TTL)
        if ttl == 0:
            return

        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt))

        # Write to both layers
        self._memory.set(key, response, ttl)

        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(response))
            except Exception as e:
                logger.warning("Redis write failed: %s", e)

    def invalidate(self, institute_id: str, feature: str, user_prompt: str):
        """Remove a specific cached response for a tenant."""
        key = _make_cache_key(institute_id, feature, _hash_prompt(user_prompt))
        self._memory.delete(key)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def flush_tenant(self, institute_id: str):
        """Clear ALL cached responses for a specific tenant."""
        # Memory cache — scan and delete matching keys
        prefix = f"ai_svc:{institute_id}:"
        with self._memory._lock:
            keys_to_delete = [k for k in self._memory._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._memory._cache[k]

        # Redis — scan and delete
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass

    def flush_all(self):
        """Clear all cached responses for ALL tenants."""
        self._memory.flush()
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match="ai_svc:*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass


class QuestionBankCache:
    """
    Stores successfully generated questions by (subject, chapter, difficulty, qtype).
    On LLM failure, returns a random previously-seen question from the bank.

    Storage:
      - Redis LIST (RPUSH/LTRIM) when REDIS_URL is set — survives restarts, shared across workers
      - Module-level dict fallback when Redis is unavailable — in-memory only, lost on restart
        (still useful for demo: bank fills up during the session)
    """

    _BANK_TTL = 60 * 60 * 24 * 7   # 7 days
    _MAX_PER_KEY = 50               # max questions stored per bucket

    def __init__(self):
        self._redis = None
        self._memory: dict[str, list] = {}
        self._lock = Lock()
        self._init_redis()

    def _init_redis(self):
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True, socket_timeout=1)
            self._redis.ping()
            logger.info("QuestionBank: Redis connected")
        except Exception as e:
            logger.warning("QuestionBank: Redis unavailable (%s), using in-memory fallback", e)
            self._redis = None

    def _key(self, subject: str, chapter: str, difficulty: str, qtype: str) -> str:
        parts = [
            (subject or "").lower().strip()[:40],
            (chapter or "").lower().strip()[:40],
            (difficulty or "").lower().strip(),
            (qtype or "").lower().strip(),
        ]
        return "qbank:" + ":".join(parts)

    def save(self, subject: str, chapter: str, difficulty: str, qtype: str, questions: list):
        """Append successfully generated questions into the bank."""
        if not questions:
            return
        key = self._key(subject, chapter, difficulty, qtype)

        # In-memory store (always written)
        with self._lock:
            bucket = self._memory.get(key, [])
            bucket.extend(questions)
            if len(bucket) > self._MAX_PER_KEY:
                bucket = bucket[-self._MAX_PER_KEY:]
            self._memory[key] = bucket

        # Redis store (bounded list)
        if self._redis:
            try:
                pipe = self._redis.pipeline()
                for q in questions:
                    pipe.rpush(key, json.dumps(q, ensure_ascii=False))
                pipe.ltrim(key, -self._MAX_PER_KEY, -1)
                pipe.expire(key, self._BANK_TTL)
                pipe.execute()
            except Exception as e:
                logger.warning("QuestionBank Redis write failed: %s", e)

    def get_random(self, subject: str, chapter: str, difficulty: str, qtype: str, n: int = 1) -> list:
        """Return up to n random cached questions. Returns [] if the bank is empty."""
        key = self._key(subject, chapter, difficulty, qtype)
        questions: list = []

        # Try Redis first
        if self._redis:
            try:
                raw_list = self._redis.lrange(key, 0, -1)
                if raw_list:
                    questions = [json.loads(r) for r in raw_list]
            except Exception:
                pass

        # Fallback to in-memory
        if not questions:
            with self._lock:
                questions = list(self._memory.get(key, []))

        if not questions:
            return []

        random.shuffle(questions)
        return questions[:n]


question_bank = QuestionBankCache()
