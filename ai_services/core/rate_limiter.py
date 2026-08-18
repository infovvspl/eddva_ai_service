"""
Per-institute usage caps with concurrent request throttling.

Two layers of protection:
  1. Daily token budget — soft cap (warn) + hard cap (block)
     Caps are read from Institute model in DB, not env vars.
  2. Concurrent request limit — prevents one tenant from exhausting the LLM
     provider's concurrency for everyone else (noisy neighbour).

Both layers are Redis-backed so they are enforced across ALL gunicorn workers.
The concurrency gate uses an atomic Lua script over a per-tenant ZSET of
in-flight slot tokens; each slot carries a lease (CONCURRENCY_LEASE_SECONDS) so
a worker killed mid-request cannot leak a slot forever.

Falls back to per-process, in-memory tracking if Redis is unavailable — in that
mode limits are NOT enforced across workers (a warning is logged at startup).
"""

import os
import time
import uuid
import logging
import threading
from threading import Lock, Semaphore
from typing import Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("ai_services.rate_limiter")

# Fallback defaults if Institute model is not available
FALLBACK_SOFT_CAP = int(os.getenv("USAGE_SOFT_CAP_TOKENS", "500000"))
FALLBACK_HARD_CAP = int(os.getenv("USAGE_HARD_CAP_TOKENS", "1000000"))
FALLBACK_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# How long a concurrency slot may be held before it is considered leaked and
# reclaimed. Guards against a worker that is SIGKILLed mid-request never
# releasing its slot. Keep it above the gunicorn --timeout (120s).
CONCURRENCY_LEASE_SECONDS = int(os.getenv("CONCURRENCY_LEASE_SECONDS", "180"))

# Marker pushed when a slot was taken from the in-process semaphore instead of
# Redis (so release() knows which mechanism to give it back to).
_MEMORY_SLOT = "__memory__"

# Atomic acquire: prune leaked slots, count in-flight, take one if under the cap.
# Must be atomic — otherwise two workers can both read "cur < max" and overshoot.
_ACQUIRE_LUA = """
local key   = KEYS[1]
local now   = tonumber(ARGV[1])
local max   = tonumber(ARGV[2])
local lease = tonumber(ARGV[3])
local token = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - lease)
local cur = redis.call('ZCARD', key)
if cur < max then
  redis.call('ZADD', key, now, token)
  redis.call('EXPIRE', key, lease + 60)
  return 1
end
return 0
"""

# Thread-local stack of slot tokens held by the current request/thread. Acquire
# and release always happen on the same thread (sync gunicorn worker), so this
# lets release_concurrency_slot() stay a single-argument API for its callers.
_slot_tokens = threading.local()


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
            self._acquire_script = self._redis.register_script(_ACQUIRE_LUA)
            logger.info("Concurrency gate: Redis-backed (enforced across all workers)")
        except Exception:
            self._redis = None
            self._acquire_script = None
            logger.warning(
                "Concurrency gate: Redis unavailable — falling back to per-worker "
                "in-process semaphores. Limits are NOT enforced across workers."
            )

    def _redis_key(self, institute_id: str) -> str:
        return f"ai_svc:usage:{institute_id}:{_day_key()}"

    def _conc_key(self, institute_id: str) -> str:
        """Key holding the set of in-flight slot tokens for a tenant."""
        return f"ai_svc:conc:{institute_id}"

    # ── thread-local slot bookkeeping ────────────────────────────────────────
    @staticmethod
    def _push_token(institute_id: str, token: str):
        stacks = getattr(_slot_tokens, "stacks", None)
        if stacks is None:
            stacks = {}
            _slot_tokens.stacks = stacks
        stacks.setdefault(institute_id, []).append(token)

    @staticmethod
    def _pop_token(institute_id: str) -> Optional[str]:
        stacks = getattr(_slot_tokens, "stacks", None)
        if not stacks:
            return None
        stack = stacks.get(institute_id)
        return stack.pop() if stack else None

    def _get_semaphore(self, institute_id: str, max_concurrent: int) -> Semaphore:
        """
        Get or create a per-tenant semaphore for concurrency control.

        FIX BUG-7 (partial): If max_concurrent_requests changes in the DB
        (e.g. via admin), the old cached semaphore had the wrong count.
        We now store the configured max alongside the semaphore and rebuild
        it when the count changes.

        NOTE: In-process semaphores are NOT shared across gunicorn workers, so
        this is only the FALLBACK used when Redis is unavailable. The primary
        gate is the Redis ZSET in acquire_concurrency_slot(), which enforces the
        cap across every worker (audit finding H1).
        """
        with self._sem_lock:
            entry = self._semaphores.get(institute_id)
            if entry is None or entry["max"] != max_concurrent:
                # Build a new semaphore if one doesn't exist or the limit changed
                self._semaphores[institute_id] = {
                    "sem": Semaphore(max_concurrent),
                    "max": max_concurrent,
                }
            return self._semaphores[institute_id]["sem"]

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

        Redis-backed so the cap is enforced across ALL gunicorn workers (a
        per-process semaphore would let N workers each admit `max_concurrent`).
        Slots carry a lease so a worker killed mid-request cannot leak one
        forever. Degrades to the in-process semaphore if Redis is unavailable.
        """
        if self._redis and self._acquire_script:
            key = self._conc_key(institute_id)
            token = uuid.uuid4().hex
            deadline = time.time() + timeout
            backoff = 0.05
            while True:
                try:
                    got = self._acquire_script(
                        keys=[key],
                        args=[time.time(), max_concurrent, CONCURRENCY_LEASE_SECONDS, token],
                    )
                except Exception as e:
                    logger.warning(
                        "Redis concurrency gate failed (%s) — using in-process semaphore", e
                    )
                    return self._acquire_in_memory(institute_id, max_concurrent, timeout)

                if got == 1:
                    self._push_token(institute_id, token)
                    return True

                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.warning(
                        "Concurrency limit reached | institute=%s max=%d",
                        institute_id, max_concurrent,
                    )
                    return False
                time.sleep(min(backoff, remaining))
                backoff = min(backoff * 1.5, 0.5)

        return self._acquire_in_memory(institute_id, max_concurrent, timeout)

    def _acquire_in_memory(self, institute_id: str, max_concurrent: int, timeout: float) -> bool:
        """Per-worker fallback. NOT enforced across workers — Redis is preferred."""
        sem = self._get_semaphore(institute_id, max_concurrent)
        acquired = sem.acquire(timeout=timeout)
        if acquired:
            self._push_token(institute_id, _MEMORY_SLOT)
        else:
            logger.warning(
                "Concurrency limit reached | institute=%s max=%d (in-process)",
                institute_id, max_concurrent,
            )
        return acquired

    def release_concurrency_slot(self, institute_id: str):
        """Release a concurrency slot after an LLM call completes."""
        token = self._pop_token(institute_id)

        # Slot came from Redis → drop its token from the in-flight set.
        if token and token != _MEMORY_SLOT:
            if self._redis:
                try:
                    self._redis.zrem(self._conc_key(institute_id), token)
                    return
                except Exception as e:
                    # Not fatal: the lease will reclaim the slot within
                    # CONCURRENCY_LEASE_SECONDS.
                    logger.warning("Failed to release Redis concurrency slot (%s)", e)
            return

        # Slot came from the in-process semaphore (or acquire was never called).
        with self._sem_lock:
            entry = self._semaphores.get(institute_id)
            if entry:
                try:
                    entry["sem"].release()
                except ValueError:
                    pass  # already released (defensive)


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


# ── Shared instance ──────────────────────────────────────────────────────────
# Constructing a UsageLimiter opens a Redis connection, so building one per call
# would add a connect/handshake to the hot path and leak sockets under load.
# Callers that only need the shared counters should use this rather than
# instantiating their own.
_SHARED_LIMITER: Optional["UsageLimiter"] = None
_SHARED_LOCK = Lock()


def get_shared_limiter() -> "UsageLimiter":
    global _SHARED_LIMITER
    if _SHARED_LIMITER is None:
        with _SHARED_LOCK:
            if _SHARED_LIMITER is None:
                _SHARED_LIMITER = UsageLimiter()
    return _SHARED_LIMITER
