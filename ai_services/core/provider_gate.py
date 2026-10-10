"""How many calls to one provider are in flight, across every worker and server.

Together answered "429 too many requests" to a single deck sending three slide
paintings at once, and each process only limited its own calls - with many
teachers generating at the same time, every process sent its share and most of
it was refused. A gate is one Redis-backed counter of slots for a provider,
shared by every gunicorn worker and every server:

  * at most ``capacity`` calls are in flight, whoever makes them;
  * the next free slot goes to the school with the fewest slots in use (oldest
    request first among equals), so one school generating fifty decks cannot
    keep the others waiting - and no school may hold more than ``per_tenant``;
  * a request that cannot have a slot waits for one rather than failing, and
    learns its place in the line;
  * when the provider says "429", every process pauses (``cool_down``) instead
    of each retrying into the same limit;
  * slots are leases: a worker killed mid-call frees its slot when the lease
    runs out. A long call renews its lease.

Without Redis the gate falls back to a semaphore in this process - the old
behaviour - and says so in the log.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger("ai_services.provider_gate")

# Atomic: expire leases, forget waiters that stopped asking, then hand a free
# slot to this request if it is next in fair order. Returns {granted, place}.
_ACQUIRE_LUA = """
local p = KEYS[1]
local held, owner = p .. ':held', p .. ':owner'
local wait, seen, wtenant = p .. ':wait', p .. ':seen', p .. ':wtenant'
local cool = p .. ':cool'
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local per_tenant = tonumber(ARGV[3])
local lease = tonumber(ARGV[4])
local waiter_ttl = tonumber(ARGV[5])
local token, tenant = ARGV[6], ARGV[7]

for _, t in ipairs(redis.call('ZRANGEBYSCORE', held, 0, now)) do redis.call('HDEL', owner, t) end
redis.call('ZREMRANGEBYSCORE', held, 0, now)
for _, t in ipairs(redis.call('ZRANGEBYSCORE', seen, 0, now - waiter_ttl)) do
  redis.call('ZREM', wait, t)
  redis.call('HDEL', wtenant, t)
end
redis.call('ZREMRANGEBYSCORE', seen, 0, now - waiter_ttl)

if not redis.call('ZSCORE', wait, token) then
  redis.call('ZADD', wait, now, token)
  redis.call('HSET', wtenant, token, tenant)
end
redis.call('ZADD', seen, now, token)
for _, k in ipairs({held, owner, wait, seen, wtenant}) do
  redis.call('PEXPIRE', k, lease + 600000)
end

local place = redis.call('ZRANK', wait, token)
if redis.call('EXISTS', cool) == 1 then return {0, place} end

local counts = {}
for _, t in ipairs(redis.call('HVALS', owner)) do counts[t] = (counts[t] or 0) + 1 end
local free = capacity - redis.call('ZCARD', held)
local waiters = redis.call('ZRANGE', wait, 0, -1)
local tenant_of = {}
for _, t in ipairs(waiters) do tenant_of[t] = redis.call('HGET', wtenant, t) or '' end
local picked = {}
for _ = 1, free do
  local best, best_count = nil, nil
  for _, t in ipairs(waiters) do
    if not picked[t] then
      local c = counts[tenant_of[t]] or 0
      if c < per_tenant and (best_count == nil or c < best_count) then
        best, best_count = t, c
      end
    end
  end
  if best == nil then break end
  if best == token then
    redis.call('ZREM', wait, token)
    redis.call('ZREM', seen, token)
    redis.call('HDEL', wtenant, token)
    redis.call('ZADD', held, now + lease, token)
    redis.call('HSET', owner, token, tenant)
    return {1, 0}
  end
  picked[best] = true
  counts[tenant_of[best]] = best_count + 1
end
return {0, place}
"""

_RELEASE_LUA = """
local p = KEYS[1]
redis.call('ZREM', p .. ':held', ARGV[1])
redis.call('HDEL', p .. ':owner', ARGV[1])
redis.call('ZREM', p .. ':wait', ARGV[1])
redis.call('ZREM', p .. ':seen', ARGV[1])
redis.call('HDEL', p .. ':wtenant', ARGV[1])
return 1
"""

_RENEW_LUA = """
local held = KEYS[1] .. ':held'
if redis.call('ZSCORE', held, ARGV[1]) then
  redis.call('ZADD', held, tonumber(ARGV[2]), ARGV[1])
  return 1
end
return 0
"""

# Only ever lengthens a pause: a short "retry in 2s" must not cut a long one short.
_COOL_LUA = """
local cool = KEYS[1] .. ':cool'
local ms = tonumber(ARGV[1])
if redis.call('PTTL', cool) < ms then
  redis.call('SET', cool, '1', 'PX', ms)
end
return 1
"""


class GateBusy(Exception):
    """No slot came free within the time a caller was willing to wait."""


class _NoRedis(Exception):
    """Redis is unavailable; the caller falls back to this process's semaphore."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _default_client():
    from ai_services.core.cache import get_redis
    return get_redis()


class Gate:
    """A shared limit on calls in flight to one provider. See the module docstring."""

    POLL_S = 0.3              # how often a waiting request asks again
    WAITER_TTL_S = 5.0        # a waiter that has not asked for this long has gone
    RECHECK_REDIS_S = 30.0    # how long to stay on the local fallback before trying Redis again

    def __init__(self, name: str, *, capacity: int, per_tenant: Optional[int] = None,
                 lease_s: float = 180.0, client_factory: Callable[[], Any] = _default_client,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.name = name
        self.capacity = max(1, int(capacity))
        self.per_tenant = max(1, int(per_tenant or self.capacity))
        self.lease_s = float(lease_s)
        self._client_factory = client_factory
        self._clock = clock
        self._sleep = sleep
        self._key = "gate:{%s}" % name
        self._local = threading.BoundedSemaphore(self.capacity)
        self._scripts: Optional[Dict[str, Any]] = None
        self._client = None
        self._redis_down_until = 0.0
        self._warned = False
        self._lock = threading.Lock()

    # ── Redis plumbing ──────────────────────────────────────────────────────
    def _redis(self):
        """The client and its scripts, or None while Redis is unavailable."""
        now = self._clock()
        with self._lock:
            if self._client is not None:
                return self._client, self._scripts
            if now < self._redis_down_until:
                return None
            try:
                client = self._client_factory()
            except Exception:
                client = None
            if client is None:
                self._redis_down_until = now + self.RECHECK_REDIS_S
                if not self._warned:
                    logger.warning("provider gate %s: no Redis; limiting this process only "
                                   "(capacity %d)", self.name, self.capacity)
                    self._warned = True
                return None
            self._client = client
            self._scripts = {name: client.register_script(src) for name, src in (
                ("acquire", _ACQUIRE_LUA), ("release", _RELEASE_LUA),
                ("renew", _RENEW_LUA), ("cool", _COOL_LUA))}
            return self._client, self._scripts

    def _redis_failed(self, exc: Exception) -> None:
        logger.warning("provider gate %s: Redis error (%s); limiting this process only",
                       self.name, exc)
        with self._lock:
            self._client, self._scripts = None, None
            self._redis_down_until = self._clock() + self.RECHECK_REDIS_S

    # ── public API ──────────────────────────────────────────────────────────
    @contextmanager
    def slot(self, tenant: Any = "", *, max_wait_s: float = 240.0,
             on_wait: Optional[Callable[[int], None]] = None) -> Iterator["Ticket"]:
        """Hold one slot for the body of the ``with``. Raises GateBusy after ``max_wait_s``."""
        ticket = self._acquire(str(tenant or ""), max_wait_s, on_wait)
        try:
            yield ticket
        finally:
            ticket.release()

    def cool_down(self, seconds: float) -> None:
        """Pause every process's next call to this provider for ``seconds``."""
        if seconds <= 0:
            return
        conn = self._redis()
        if conn is None:
            return            # one process: its own retry delay already pauses it
        try:
            conn[1]["cool"](keys=[self._key], args=[int(seconds * 1000)])
        except Exception as exc:
            self._redis_failed(exc)

    def in_flight(self) -> Optional[int]:
        """Slots held right now across all processes (None without Redis). For monitoring."""
        conn = self._redis()
        if conn is None:
            return None
        try:
            now_ms = int(self._clock() * 1000)
            return int(conn[0].zcount(self._key + ":held", now_ms + 1, "+inf"))
        except Exception as exc:
            self._redis_failed(exc)
            return None

    def _acquire(self, tenant: str, max_wait_s: float,
                 on_wait: Optional[Callable[[int], None]]) -> "Ticket":
        token = uuid.uuid4().hex
        deadline = self._clock() + max_wait_s
        last_place = None
        while True:
            try:
                granted, place = self._attempt(token, tenant)
            except _NoRedis:
                return self._acquire_local(deadline)
            if granted:
                return Ticket(self, token, shared=True)
            if self._clock() >= deadline:
                self._forget(token)
                raise GateBusy("%s: no slot within %.0fs" % (self.name, max_wait_s))
            if on_wait and place != last_place:
                try:
                    on_wait(place)
                except Exception:
                    logger.exception("provider gate %s: on_wait failed", self.name)
                last_place = place
            # Jitter so waiters on many workers do not all ask at the same instant.
            self._sleep(self.POLL_S * (0.75 + 0.5 * random.random()))

    def _attempt(self, token: str, tenant: str):
        """One try for a slot: (granted, place in line). Raises _NoRedis without Redis."""
        conn = self._redis()
        if conn is None:
            raise _NoRedis()
        try:
            granted, place = conn[1]["acquire"](keys=[self._key], args=[
                int(self._clock() * 1000), self.capacity, self.per_tenant,
                int(self.lease_s * 1000), int(self.WAITER_TTL_S * 1000), token, tenant])
        except Exception as exc:
            self._redis_failed(exc)
            raise _NoRedis() from exc
        return int(granted) == 1, (int(place) if place is not None else 0)

    def _acquire_local(self, deadline: float) -> "Ticket":
        if not self._local.acquire(timeout=max(0.0, deadline - self._clock())):
            raise GateBusy("%s: no local slot" % self.name)
        return Ticket(self, None, shared=False)

    def _forget(self, token: str) -> None:
        conn = self._redis()
        if conn is None:
            return
        try:
            conn[1]["release"](keys=[self._key], args=[token])
        except Exception as exc:
            self._redis_failed(exc)


class Ticket:
    """One held slot. ``renew`` before a long call outlives the lease; ``release`` once."""

    def __init__(self, gate: Gate, token: Optional[str], *, shared: bool):
        self._gate, self._token, self.shared = gate, token, shared
        self._released = False

    def renew(self) -> None:
        if not self.shared or self._released:
            return
        conn = self._gate._redis()
        if conn is None:
            return
        try:
            expiry = int((self._gate._clock() + self._gate.lease_s) * 1000)
            conn[1]["renew"](keys=[self._gate._key], args=[self._token, expiry])
        except Exception as exc:
            self._gate._redis_failed(exc)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.shared:
            self._gate._forget(self._token)
        else:
            self._gate._local.release()


# ── the gates the service uses ──────────────────────────────────────────────
_gates: Dict[str, Gate] = {}
_gates_lock = threading.Lock()


def together_images() -> Gate:
    """Every call to Together's image endpoint - slide paintings and figures alike.

    TOGETHER_IMAGE_GLOBAL_SLOTS is the number of image calls in flight across
    the whole service: set it to what the Together account (or a dedicated
    endpoint) actually sustains. TOGETHER_IMAGE_SLOTS_PER_SCHOOL caps one school.
    """
    capacity = max(1, _env_int("TOGETHER_IMAGE_GLOBAL_SLOTS", 3))
    per_tenant = max(1, _env_int("TOGETHER_IMAGE_SLOTS_PER_SCHOOL", capacity))
    key = "together-images:%d:%d" % (capacity, per_tenant)
    with _gates_lock:
        gate = _gates.get(key)
        if gate is None:
            gate = _gates[key] = Gate("together-images", capacity=capacity,
                                      per_tenant=per_tenant,
                                      lease_s=_env_int("TOGETHER_IMAGE_SLOT_LEASE_S", 180))
        return gate
