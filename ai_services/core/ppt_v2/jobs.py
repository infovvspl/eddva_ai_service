"""Background deck generation: a job id now, the deck as it is built.

A deck takes one to three minutes. Held open as one request it ran into every
timeout on the way (the production web worker kills a request at 120s, the
backend stops waiting at 240s), and the teacher watched a spinner with no
idea whether anything was happening. Instead the request starts a job and
returns at once; the studio polls for its status, sees real progress, and gets
the slides as soon as their text exists - pictures arrive while they read.

State lives in Redis when it is configured (required in production, where any
of several workers may answer a poll) and in process memory otherwise.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# How long a job (and a finished deck) can still be fetched: a teacher who
# sends a deck to the background may open it hours later (PPT_JOB_TTL_S).
JOB_TTL_S = 24 * 60 * 60


def job_ttl_s() -> int:
    import os
    try:
        return max(600, int(os.getenv("PPT_JOB_TTL_S", "") or JOB_TTL_S))
    except ValueError:
        return JOB_TTL_S
# A running job that has not reported for this long was interrupted (a worker
# restart kills its thread); report that rather than poll forever.
STALE_AFTER_S = 300
# Decks one institute may have in flight (running or queued) at once. It was
# 3: with decks sent to the background, three teachers of one school starting a
# deck each was enough for the next to be refused. Decks now wait in the worker
# queue and share the provider gate fairly, so this is only a runaway guard
# (PPT_MAX_RUNNING_PER_INSTITUTE).
MAX_RUNNING_PER_INSTITUTE = 20


def max_running_per_institute() -> int:
    import os
    try:
        return max(1, int(os.getenv("PPT_MAX_RUNNING_PER_INSTITUTE", "") or MAX_RUNNING_PER_INSTITUTE))
    except ValueError:
        return MAX_RUNNING_PER_INSTITUTE
# A job still waiting for a worker after this long is reported as failed: the
# workers are down or hopelessly behind, and the teacher should hear so.
QUEUED_MAX_WAIT_S = 600

_KEY = "ppt_job:%s"
_RUNNING_KEY = "ppt_jobs_running:%s"

_memory: Dict[str, str] = {}
_memory_running: Dict[str, int] = {}
_lock = threading.Lock()


def _redis():
    try:
        from ai_services.core.cache import get_redis
        return get_redis()
    except Exception:
        return None


def _write(job_id: str, job: Dict[str, Any]) -> None:
    payload = json.dumps(job, default=str)
    client = _redis()
    if client is not None:
        try:
            client.setex(_KEY % job_id, job_ttl_s(), payload)
            return
        except Exception as exc:
            logger.warning("ppt job store: redis write failed (%s); using memory", exc)
    with _lock:
        _memory[job_id] = payload


def get(job_id: str) -> Optional[Dict[str, Any]]:
    raw = None
    client = _redis()
    if client is not None:
        try:
            raw = client.get(_KEY % job_id)
        except Exception as exc:
            logger.warning("ppt job store: redis read failed (%s)", exc)
    if raw is None:
        with _lock:
            raw = _memory.get(job_id)
    if raw is None:
        return None
    job = json.loads(raw)
    if job.get("status") == "running" and time.time() - job.get("updatedAt", 0) > STALE_AFTER_S:
        job.update(status="failed", error="generation was interrupted; please try again")
    # Waiting in the worker queue is not being stuck: only a long wait fails.
    if job.get("status") == "queued" and time.time() - job.get("createdAt", 0) > QUEUED_MAX_WAIT_S:
        job.update(status="failed", error="the presentation service is busy; please try again")
    return job


def update(job_id: str, **fields: Any) -> None:
    job = get(job_id) or {}
    job.update(fields)
    job["updatedAt"] = time.time()
    _write(job_id, job)


def _running_delta(institute_id: str, delta: int) -> int:
    key = _RUNNING_KEY % (institute_id or "default")
    client = _redis()
    if client is not None:
        try:
            value = client.incrby(key, delta)
            client.expire(key, STALE_AFTER_S * 2)
            return int(value)
        except Exception as exc:
            logger.warning("ppt job store: redis counter failed (%s)", exc)
    with _lock:
        _memory_running[key] = max(0, _memory_running.get(key, 0) + delta)
        return _memory_running[key]


class Progress:
    """What a running deck reports. Callable from any thread.

    ``progress(stage, done=, total=, partial=)``: ``partial`` is the deck as
    far as it exists - the same shape the finished response carries - so the
    studio can show slides before the pictures are in.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._lock = threading.Lock()

    # The latest this many steps are kept; a deck makes a few dozen.
    MAX_ACTIVITY = 40

    def event(self, key: str, text: str, state: str = "active", **extra: Any) -> None:
        """One line of the activity feed, updated in place by ``key``.

        ``state`` is "active" (in progress), "done" or "warn". The studio
        shows these as a live timeline: what is being done, right now, and
        what has been done.
        """
        try:
            with self._lock:
                job = get(self.job_id) or {}
                feed = job.get("activity") or []
                entry = next((e for e in feed if e.get("key") == key), None)
                if entry is None:
                    entry = {"key": key, "startedAt": time.time()}
                    feed.append(entry)
                entry.update(text=text, state=state, at=time.time(), **extra)
                update(self.job_id, activity=feed[-self.MAX_ACTIVITY:])
        except Exception:
            logger.exception("ppt job %s: activity update failed", self.job_id)

    def __call__(self, stage: str, done: Optional[int] = None,
                 total: Optional[int] = None, partial: Optional[Dict[str, Any]] = None) -> None:
        fields: Dict[str, Any] = {"stage": stage}
        if done is not None:
            fields["done"] = done
        if total is not None:
            fields["total"] = total
        if partial is not None:
            fields["partial"] = partial
        try:
            with self._lock:
                update(self.job_id, **fields)
        except Exception:  # progress is a courtesy; never fail the deck over it
            logger.exception("ppt job %s: progress update failed", self.job_id)


def start(run: Callable[[Progress], Any], *, institute_id: str,
          meta: Optional[Dict[str, Any]] = None,
          scope: Optional[Dict[str, Any]] = None,
          spec: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Start a deck in the background. None when the institute is at its cap.

    ``run(progress)`` returns a DRF Response; its status and body become the
    result. With ``spec`` - the request as plain data - and a live worker, the
    job goes to the worker queue (see job_queue) and a worker runs it from
    ``spec``; otherwise ``run`` runs on a thread in this process, as before.
    ``scope`` is what the deck is being written for, exactly as the generator
    reads it; every status carries it, so the studio can show the teacher -
    from the first poll - which topic and chapter are actually being made.
    """
    from ai_services.core.ppt_v2 import job_queue

    if _running_delta(institute_id, +1) > max_running_per_institute():
        _running_delta(institute_id, -1)
        return None
    job_id = uuid.uuid4().hex
    record = {"jobId": job_id, "status": "running", "stage": "queued",
              "instituteId": institute_id or "", "meta": meta or {},
              "scope": scope or {},
              "createdAt": time.time(), "updatedAt": time.time()}
    if spec is not None and job_queue.should_queue():
        _write(job_id, dict(record, status="queued"))
        if job_queue.enqueue(job_id, dict(spec, instituteId=institute_id or "")):
            return job_id
        logger.warning("ppt job %s: queue unavailable; running in this process", job_id)
    _write(job_id, record)
    threading.Thread(target=execute, args=(job_id, run, institute_id),
                     name="ppt-job-%s" % job_id[:8], daemon=True).start()
    return job_id


def execute(job_id: str, run: Callable[[Progress], Any], institute_id: str) -> None:
    """Run one job to its end and record the result. Used by both runners."""
    update(job_id, status="running", startedAt=time.time())
    progress = Progress(job_id)
    try:
        response = run(progress)
        code = getattr(response, "status_code", 500)
        body = getattr(response, "data", None)
        update(job_id, status="done" if code < 400 else "failed",
               httpStatus=code, result=body, stage="done",
               error=None if code < 400 else (body or {}).get("error"))
    except Exception as exc:
        logger.exception("ppt job %s failed", job_id)
        update(job_id, status="failed", stage="failed", httpStatus=500,
               error="generation failed: %s" % str(exc)[:200])
    finally:
        _running_delta(institute_id, -1)
        # Usage logging opened database connections on this thread; a
        # job thread that leaves them open leaks one per deck.
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass


def give_up(job_id: str) -> None:
    """A queued job whose workers kept dying: report it failed and free its place."""
    job = get(job_id) or {}
    update(job_id, status="failed", stage="failed", httpStatus=500,
           error="generation was interrupted; please try again")
    _running_delta(job.get("instituteId") or "", -1)
