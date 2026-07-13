"""
Tests for the Redis-backed concurrency gate (audit finding H1).

The gate must enforce a tenant's max_concurrent_requests across ALL gunicorn
workers, not per-process. We simulate Redis with a fake that implements the same
ZSET semantics the Lua script relies on, so the Python-side orchestration
(atomic take, retry-until-timeout, token bookkeeping, lease reclaim, release) is
exercised without needing a live Redis.
"""

import time
from django.test import SimpleTestCase

from ai_services.core import rate_limiter
from ai_services.core.rate_limiter import UsageLimiter, _MEMORY_SLOT


class _FakeZSetRedis:
    """Minimal stand-in implementing the ZSET ops the acquire script uses."""

    def __init__(self):
        self.zsets = {}  # key -> {token: score}

    # The limiter calls the registered script like a callable.
    def register_script(self, _lua):
        def _script(keys, args):
            key = keys[0]
            now, max_c, lease, token = float(args[0]), int(args[1]), float(args[2]), args[3]
            z = self.zsets.setdefault(key, {})
            # ZREMRANGEBYSCORE key 0 (now - lease)  → drop leaked slots
            for t, score in list(z.items()):
                if score <= now - lease:
                    del z[t]
            if len(z) < max_c:
                z[token] = now
                return 1
            return 0
        return _script

    def ping(self):
        return True

    def zrem(self, key, token):
        self.zsets.get(key, {}).pop(token, None)

    def zcard(self, key):
        return len(self.zsets.get(key, {}))


def _limiter_with_fake_redis(fake):
    lim = UsageLimiter.__new__(UsageLimiter)  # bypass __init__ (no real Redis)
    from threading import Lock
    from collections import defaultdict
    lim._redis = fake
    lim._acquire_script = fake.register_script("")
    lim._memory = defaultdict(lambda: defaultdict(int))
    lim._lock = Lock()
    lim._semaphores = {}
    lim._sem_lock = Lock()
    # Clear thread-local slot stacks between tests.
    rate_limiter._slot_tokens.stacks = {}
    return lim


class RedisConcurrencyGateTests(SimpleTestCase):
    def setUp(self):
        self.fake = _FakeZSetRedis()
        self.lim = _limiter_with_fake_redis(self.fake)

    def test_cap_is_enforced(self):
        # 2 slots available → third caller is rejected.
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=2, timeout=0.1))
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=2, timeout=0.1))
        self.assertFalse(self.lim.acquire_concurrency_slot("t1", max_concurrent=2, timeout=0.1))
        self.assertEqual(self.fake.zcard("ai_svc:conc:t1"), 2)

    def test_release_frees_a_slot(self):
        self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1)
        self.assertFalse(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        self.lim.release_concurrency_slot("t1")
        self.assertEqual(self.fake.zcard("ai_svc:conc:t1"), 0)
        # Slot is available again.
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))

    def test_tenants_are_isolated(self):
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        # A different tenant is unaffected by t1 being full.
        self.assertTrue(self.lim.acquire_concurrency_slot("t2", max_concurrent=1, timeout=0.1))

    def test_leaked_slot_is_reclaimed_by_lease(self):
        key = "ai_svc:conc:t1"
        # Simulate a worker killed mid-request: a token older than the lease.
        self.fake.zsets[key] = {"stale": time.time() - (rate_limiter.CONCURRENCY_LEASE_SECONDS + 10)}
        # The cap is 1, but the stale slot must be pruned, so this succeeds.
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        self.assertNotIn("stale", self.fake.zsets[key])

    def test_falls_back_to_semaphore_when_redis_errors(self):
        def _boom(keys, args):
            raise RuntimeError("redis down")
        self.lim._acquire_script = _boom
        # Must not raise — degrades to the in-process semaphore.
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        # It recorded a memory slot, and the cap still applies within the worker.
        self.assertFalse(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        self.lim.release_concurrency_slot("t1")
        self.assertTrue(self.lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))


class InMemoryFallbackTests(SimpleTestCase):
    def test_no_redis_still_caps_within_worker(self):
        lim = UsageLimiter.__new__(UsageLimiter)
        from threading import Lock
        from collections import defaultdict
        lim._redis = None
        lim._acquire_script = None
        lim._memory = defaultdict(lambda: defaultdict(int))
        lim._lock = Lock()
        lim._semaphores = {}
        lim._sem_lock = Lock()
        rate_limiter._slot_tokens.stacks = {}

        self.assertTrue(lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        self.assertFalse(lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
        lim.release_concurrency_slot("t1")
        self.assertTrue(lim.acquire_concurrency_slot("t1", max_concurrent=1, timeout=0.1))
