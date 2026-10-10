"""A Redis queue of deck jobs, run by worker processes apart from the web server.

Decks used to run as threads inside the gunicorn worker that took the request:
a deploy, a restart, or gunicorn recycling the worker killed every deck in it
half-way, and two minutes of generation competed with the web requests on the
same process. Now ``/ppt/generate/start`` puts the job on this queue and
returns; ``manage.py run_ppt_worker`` (any number of them, on any server) takes
jobs off it.

Nothing is lost on the way:

  * a job moves atomically from ``pending`` to ``processing`` (BLMOVE) and only
    leaves ``processing`` when it has finished;
  * a worker beats a heartbeat for each job it runs. A job whose heartbeat has
    stopped (its worker died) goes back on the queue, at most
    ``max_attempts`` times in all, then is marked failed;
  * on SIGTERM a worker takes no new job and finishes the ones it has.

When no worker is alive, or Redis is unavailable, ``should_queue`` says no and
the caller runs the deck in-process as before - so deploying this before any
worker exists changes nothing. ``PPT_JOB_RUNNER=thread`` forces that.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PENDING = "pptq:pending"
PROCESSING = "pptq:processing"
WORKERS = "pptq:workers"          # zset: worker id -> last heartbeat (unix seconds)
_SPEC = "pptq:spec:%s"
_BEAT = "pptq:beat:%s"
_ORPHAN = "pptq:orphan:%s"

SPEC_TTL_S = 2 * 60 * 60          # a job's payload outlives any wait or retry
BEAT_TTL_S = 60                   # a job with no beat for this long has lost its worker
BEAT_EVERY_S = 15
WORKER_ALIVE_S = 30               # a worker that has not beaten for this long is gone
ORPHAN_GRACE_S = 20               # a just-taken job may not have beaten yet


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def max_attempts() -> int:
    return max(1, _env_int("PPT_JOB_MAX_ATTEMPTS", 2))


def _client():
    try:
        from ai_services.core.cache import get_redis
        return get_redis()
    except Exception:
        return None


def workers_alive(client=None) -> int:
    """Workers that have beaten in the last WORKER_ALIVE_S seconds."""
    client = client if client is not None else _client()
    if client is None:
        return 0
    try:
        return int(client.zcount(WORKERS, time.time() - WORKER_ALIVE_S, "+inf"))
    except Exception as exc:
        logger.warning("ppt queue: cannot read workers (%s)", exc)
        return 0


def should_queue(client=None) -> bool:
    """Send this deck to a worker (True) or run it in this process (False)."""
    if (os.getenv("PPT_JOB_RUNNER", "auto") or "auto").strip().lower() == "thread":
        return False
    return workers_alive(client) > 0


def enqueue(job_id: str, spec: Dict[str, Any], client=None) -> bool:
    """Put a job on the queue. False (nothing queued) when Redis fails."""
    client = client if client is not None else _client()
    if client is None:
        return False
    try:
        payload = json.dumps(dict(spec, attempts=0), default=str)
        pipe = client.pipeline()
        pipe.setex(_SPEC % job_id, SPEC_TTL_S, payload)
        pipe.lpush(PENDING, job_id)
        pipe.execute()
        return True
    except Exception as exc:
        logger.warning("ppt queue: enqueue failed (%s)", exc)
        return False


def queue_position(job_id: str, client=None) -> Optional[int]:
    """How many jobs will be taken before this one (0 = next); None if not waiting."""
    client = client if client is not None else _client()
    if client is None:
        return None
    try:
        waiting = client.lrange(PENDING, 0, -1)       # oldest at the right
    except Exception:
        return None
    if job_id not in waiting:
        return None
    return len(waiting) - 1 - waiting.index(job_id)


def reap(client, *, on_give_up: Callable[[str], None], now: Optional[float] = None) -> Dict[str, list]:
    """Requeue jobs whose worker died; give up on those out of attempts.

    A job in ``processing`` without a beat is first only marked: the worker that
    has just taken it may not have beaten yet. Seen again ORPHAN_GRACE_S later
    and still silent, it is requeued (or given up on).
    """
    now = time.time() if now is None else now
    done: Dict[str, list] = {"requeued": [], "failed": [], "dropped": []}
    for job_id in client.lrange(PROCESSING, 0, -1):
        if client.exists(_BEAT % job_id):
            continue
        marked = client.get(_ORPHAN % job_id)
        if marked is None:
            client.setex(_ORPHAN % job_id, 600, str(now))
            continue
        if now - float(marked) < ORPHAN_GRACE_S:
            continue
        client.delete(_ORPHAN % job_id)
        raw = client.get(_SPEC % job_id)
        if raw is None:                                # payload expired: nothing to run
            client.lrem(PROCESSING, 0, job_id)
            done["dropped"].append(job_id)
            continue
        spec = json.loads(raw)
        if int(spec.get("attempts") or 0) >= max_attempts():
            client.lrem(PROCESSING, 0, job_id)
            client.delete(_SPEC % job_id)
            done["failed"].append(job_id)
            try:
                on_give_up(job_id)
            except Exception:
                logger.exception("ppt queue: give-up callback failed for %s", job_id)
            continue
        pipe = client.pipeline()
        pipe.lrem(PROCESSING, 0, job_id)
        pipe.rpush(PENDING, job_id)                   # the right end: taken next
        pipe.execute()
        done["requeued"].append(job_id)
        logger.warning("ppt queue: job %s lost its worker; requeued", job_id)
    return done


class Worker:
    """Takes jobs off the queue and runs ``handler(job_id, spec)``, ``concurrency`` at a time."""

    def __init__(self, client, handler: Callable[[str, Dict[str, Any]], None], *,
                 concurrency: int = 3, on_give_up: Callable[[str], None] = lambda job_id: None,
                 take_timeout_s: float = 2.0):
        self.client = client
        self.handler = handler
        self.concurrency = max(1, int(concurrency))
        self.on_give_up = on_give_up
        # A blocking take must end before the connection's read timeout, or every
        # quiet wait is an error: the service's shared pool times reads out at 1s.
        try:
            read_timeout = client.connection_pool.connection_kwargs.get("socket_timeout")
        except Exception:
            read_timeout = None
        if read_timeout and take_timeout_s >= read_timeout:
            take_timeout_s = max(0.1, read_timeout / 2)
        self.take_timeout_s = take_timeout_s
        self.id = "%s:%d:%s" % (socket.gethostname(), os.getpid(), uuid.uuid4().hex[:6])
        self.stop = threading.Event()
        self._running = 0
        self._lock = threading.Lock()

    # One job, start to finish. Never raises: a broken job must not stop the worker.
    def run_one(self, job_id: str) -> None:
        raw = self.client.get(_SPEC % job_id)
        if raw is None:
            logger.warning("ppt queue: job %s has no payload; dropped", job_id)
            self.client.lrem(PROCESSING, 0, job_id)
            return
        spec = json.loads(raw)
        spec["attempts"] = int(spec.get("attempts") or 0) + 1
        self.client.setex(_SPEC % job_id, SPEC_TTL_S, json.dumps(spec, default=str))
        self.client.setex(_BEAT % job_id, BEAT_TTL_S, self.id)
        beating = threading.Event()

        def beat():
            while not beating.wait(BEAT_EVERY_S):
                try:
                    self.client.setex(_BEAT % job_id, BEAT_TTL_S, self.id)
                except Exception as exc:
                    logger.warning("ppt queue: heartbeat for %s failed (%s)", job_id, exc)
        threading.Thread(target=beat, name="pptq-beat-%s" % job_id[:8], daemon=True).start()
        try:
            self.handler(job_id, spec)
        except Exception:
            logger.exception("ppt queue: job %s raised", job_id)
        finally:
            beating.set()
            try:
                pipe = self.client.pipeline()
                pipe.lrem(PROCESSING, 0, job_id)
                pipe.delete(_SPEC % job_id, _BEAT % job_id, _ORPHAN % job_id)
                pipe.execute()
            except Exception as exc:
                logger.warning("ppt queue: cleanup for %s failed (%s)", job_id, exc)

    def take(self) -> Optional[str]:
        """The oldest waiting job, moved to ``processing``; None after the timeout."""
        return self.client.blmove(PENDING, PROCESSING, self.take_timeout_s, "RIGHT", "LEFT")

    def beat_worker(self) -> None:
        self.client.zadd(WORKERS, {self.id: time.time()})
        self.client.zremrangebyscore(WORKERS, 0, time.time() - 10 * WORKER_ALIVE_S)

    def run_forever(self) -> None:
        """Until ``stop`` is set; then finish the jobs in hand and return."""
        logger.info("ppt worker %s: started (%d at a time)", self.id, self.concurrency)
        last_beat = last_reap = 0.0
        threads = []
        while not self.stop.is_set():
            now = time.time()
            try:
                if now - last_beat >= 5:
                    self.beat_worker()
                    last_beat = now
                if now - last_reap >= 10:
                    reap(self.client, on_give_up=self.on_give_up)
                    last_reap = now
            except Exception as exc:
                logger.warning("ppt worker %s: housekeeping failed (%s)", self.id, exc)
            threads = [t for t in threads if t.is_alive()]
            if len(threads) >= self.concurrency:
                self.stop.wait(0.2)
                continue
            try:
                job_id = self.take()
            except Exception as exc:
                logger.warning("ppt worker %s: cannot take a job (%s)", self.id, exc)
                self.stop.wait(2)
                continue
            if job_id is None:
                self.stop.wait(0.05)
                continue
            t = threading.Thread(target=self.run_one, args=(job_id,),
                                 name="pptq-job-%s" % job_id[:8], daemon=True)
            t.start()
            threads.append(t)
        logger.info("ppt worker %s: stopping; finishing %d job(s)", self.id,
                    sum(1 for t in threads if t.is_alive()))
        for t in threads:
            t.join()
        try:
            self.client.zrem(WORKERS, self.id)
        except Exception:
            pass
        logger.info("ppt worker %s: stopped", self.id)
