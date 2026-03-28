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
import time
from collections import OrderedDict
from threading import Lock
from typing import Optional

logger = logging.getLogger("ai_services.cache")

# Feature-specific TTLs in seconds
CACHE_TTL = {
    "content_suggest": 86400,       # 24 hours — resources don't change often
    "test_generate": 21600,         # 6 hours — tests can be reused
    "career_roadmap": 43200,        # 12 hours
    "performance_analyze": 3600,    # 1 hour — scores change
    "feedback_analyze": 3600,       # 1 hour — answers vary
    "study_plan": 7200,             # 2 hours
    "cheating_analyze": 0,          # never cache — always real-time
    "notes_generate": 86400,        # 24 hours — same video = same notes
}

DEFAULT_TTL = 3600  # 1 hour


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
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
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
