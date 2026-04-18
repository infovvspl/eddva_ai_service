"""
Per-institute usage caps with concurrent request throttling.

Two layers of protection:
  1. Daily token budget — soft cap (warn) + hard cap (block)
     Caps are read from Institute model in DB, not env vars.
  2. Concurrent request limit — prevents one tenant from exhausting
     Ollama's concurrency limits for everyone else (noisy neighbor).

Falls back to in-memory tracking if Redis is unavailable.
"""

import os
import time
import logging
from threading import Lock, Semaphore
from typing import Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("ai_services.rate_limiter")

# Fallback defaults if Institute model is not available
FALLBACK_SOFT_CAP = int(os.getenv("USAGE_SOFT_CAP_TOKENS", "500000"))
FALLBACK_HARD_CAP = int(os.getenv("USAGE_HARD_CAP_TOKENS", "1000000"))
FALLBACK_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))


def _day_key() -> str:
    """Current date as string for daily bucketing."""
    return time.strftime("%Y-%m-%d")


class UsageLimiter:
    """
    Track and enforce per-institute daily token budgets + concurrency limits.
    Redis-backed with in-memory fallback.
    """

    def __init__(self):
        self._redis = None
        self._memory: dict = defaultdict(lambda: defaultdict(int))
        self._lock = Lock()
        # Per-tenant semaphores for concurrent request limiting
        self._semaphores: dict = {}
        self._sem_lock = Lock()
        self._init_redis()

    def _init_redis(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
            self._redis.ping()
        except Exception:
            self._redis = None

    def _redis_key(self, institute_id: str) -> str:
        return f"ai_svc:usage:{institute_id}:{_day_key()}"

    def _get_semaphore(self, institute_id: str, max_concurrent: int) -> Semaphore:
        """Get or create a per-tenant semaphore for concurrency control."""
        with self._sem_lock:
            if institute_id not in self._semaphores:
                self._semaphores[institute_id] = Semaphore(max_concurrent)
            return self._semaphores[institute_id]

    def check_budget(
        self,
        institute_id: str,
        soft_cap: Optional[int] = None,
        hard_cap: Optional[int] = None,
    ) -> Tuple[bool, bool, int]:
        """
        Check current usage against tenant-specific caps.

        Args:
            institute_id: Tenant slug
            soft_cap: Override from Institute model (or fallback)
            hard_cap: Override from Institute model (or fallback)

        Returns:
            (is_allowed, is_warning, current_usage)
        """
        soft = soft_cap or FALLBACK_SOFT_CAP
        hard = hard_cap or FALLBACK_HARD_CAP
        current = self._get_usage(institute_id)

        is_warning = current >= soft
        is_allowed = current < hard

        if not is_allowed:
            logger.warning(
                "HARD CAP hit | institute=%s usage=%d cap=%d",
                institute_id, current, hard,
            )
        elif is_warning:
            logger.warning(
                "SOFT CAP hit | institute=%s usage=%d soft_cap=%d",
                institute_id, current, soft,
            )

        return is_allowed, is_warning, current

    def acquire_concurrency_slot(
        self, institute_id: str, max_concurrent: int = FALLBACK_MAX_CONCURRENT, timeout: float = 30.0
    ) -> bool:
        """
        Try to acquire a concurrency slot for this tenant.
        Returns False if the tenant has too many in-flight requests.
        """
        sem = self._get_semaphore(institute_id, max_concurrent)
        acquired = sem.acquire(timeout=timeout)
        if not acquired:
            logger.warning(
                "Concurrency limit reached | institute=%s max=%d",
                institute_id, max_concurrent,
            )
        return acquired

    def release_concurrency_slot(self, institute_id: str):
        """Release a concurrency slot after an LLM call completes."""
        with self._sem_lock:
            sem = self._semaphores.get(institute_id)
            if sem:
                try:
                    sem.release()
                except ValueError:
                    pass  # already released

    def record_usage(self, institute_id: str, tokens_used: int):
        """Record token consumption after an LLM call."""
        if self._redis:
            try:
                key = self._redis_key(institute_id)
                pipe = self._redis.pipeline()
                pipe.incrby(key, tokens_used)
                pipe.expire(key, 86400)  # auto-expire after 24h
                pipe.execute()
                return
            except Exception:
                pass

        # Fallback to memory
        with self._lock:
            day = _day_key()
            bucket_key = f"{institute_id}:{day}"
            self._memory[bucket_key]["tokens"] += tokens_used

    def _get_usage(self, institute_id: str) -> int:
        if self._redis:
            try:
                val = self._redis.get(self._redis_key(institute_id))
                return int(val) if val else 0
            except Exception:
                pass

        with self._lock:
            bucket_key = f"{institute_id}:{_day_key()}"
            return self._memory[bucket_key]["tokens"]

    def get_usage_summary(self, institute_id: str, soft_cap: int = None, hard_cap: int = None) -> dict:
        """Return current usage stats for dashboard / admin."""
        soft = soft_cap or FALLBACK_SOFT_CAP
        hard = hard_cap or FALLBACK_HARD_CAP
        current = self._get_usage(institute_id)
        return {
            "institute_id": institute_id,
            "date": _day_key(),
            "tokens_used": current,
            "soft_cap": soft,
            "hard_cap": hard,
            "soft_cap_pct": round(current / soft * 100, 1) if soft else 0,
            "hard_cap_pct": round(current / hard * 100, 1) if hard else 0,
        }
