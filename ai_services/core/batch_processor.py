"""
Async batch processor for MCQ/content generation jobs.

For bulk operations (e.g., "generate 50 MCQs for Physics Chapter 3"),
this queues jobs and processes them in controlled batches instead of
hammering the LLM API with 50 concurrent requests.

FIX BUG-5: Jobs are now persisted to the BatchJob DB model (not a global
in-memory dict). The in-memory dict was an unbounded memory leak — every
job lived forever until the process restarted. DB-backed storage survives
restarts and is garbage-collected via the `purge_expired_batch_jobs`
management command (runs as a cron job / scheduled task).

Benefits:
  - Controlled concurrency prevents the LLM API from being overwhelmed
  - Rate-limit friendly
  - Retries failed items without re-running the whole batch
  - Results stored in DB for later retrieval (survives restarts)
  - Jobs expire after a configurable TTL (default 24h)
"""

import json
import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from threading import Lock

logger = logging.getLogger("ai_services.batch")

# Max parallel LLM calls per batch — keeps rate-limit pressure manageable
MAX_CONCURRENCY = 5
MAX_RETRIES = 2
# How long job results are kept in DB before expiry (configurable by plan tier)
JOB_TTL_HOURS = 24


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchItem:
    item_id: str
    user_prompt: str
    status: JobStatus = JobStatus.QUEUED
    result: Optional[dict] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class BatchJob:
    job_id: str
    feature: str
    institute_id: str
    items: List[BatchItem] = field(default_factory=list)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def progress(self) -> dict:
        total = len(self.items)
        done = sum(1 for i in self.items if i.status == JobStatus.COMPLETED)
        failed = sum(1 for i in self.items if i.status == JobStatus.FAILED)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": total,
            "completed": done,
            "failed": failed,
            "pct": round(done / total * 100, 1) if total else 0,
        }


# ---------------------------------------------------------------------------
# In-memory job store — used ONLY as a fast local cache during the lifetime
# of a request. The authoritative store is the BatchJob DB model.
# FIX BUG-5: This dict is bounded to MAX_MEMORY_JOBS entries. Once the limit
# is reached, the oldest entry is evicted (jobs remain in DB, just not in RAM).
# ---------------------------------------------------------------------------
_MAX_MEMORY_JOBS = 200  # sensible upper bound for a single process
_jobs: Dict[str, BatchJob] = {}
_jobs_lock = Lock()


def _evict_memory_jobs_if_needed():
    """Evict oldest in-memory entries when the dict grows beyond MAX_MEMORY_JOBS."""
    with _jobs_lock:
        if len(_jobs) > _MAX_MEMORY_JOBS:
            # Sort by created_at and remove oldest 20%
            sorted_ids = sorted(_jobs.keys(), key=lambda jid: _jobs[jid].created_at)
            to_evict = sorted_ids[:max(1, _MAX_MEMORY_JOBS // 5)]
            for jid in to_evict:
                del _jobs[jid]
            logger.info(
                "BatchProcessor: evicted %d old in-memory job entries (DB records preserved)",
                len(to_evict),
            )


def _persist_job_to_db(job: BatchJob, institute=None):
    """Write/update a BatchJob row in the database."""
    try:
        from ai_services.models import BatchJob as DBBatchJob
        expires = datetime.now(timezone.utc) + timedelta(hours=JOB_TTL_HOURS)
        completed_dt = (
            datetime.fromtimestamp(job.completed_at, tz=timezone.utc)
            if job.completed_at else None
        )
        results = {
            item.item_id: {
                "status": item.status,
                "result": item.result,
                "error": item.error,
            }
            for item in job.items
        }
        DBBatchJob.objects.update_or_create(
            job_id=job.job_id,
            defaults=dict(
                institute=institute,
                institute_id_str=job.institute_id,
                feature=job.feature,
                status=str(job.status),
                total_items=len(job.items),
                completed_items=sum(1 for i in job.items if i.status == JobStatus.COMPLETED),
                failed_items=sum(1 for i in job.items if i.status == JobStatus.FAILED),
                results_json=results,
                completed_at=completed_dt,
                expires_at=expires,
            ),
        )
    except Exception as e:
        logger.warning("BatchProcessor: DB persist failed (non-fatal): %s", e)


class BatchProcessor:
    """
    Process LLM calls in controlled batches.

    Usage:
        processor = BatchProcessor()
        job = processor.create_job("test_generate", "inst_001", prompts)
        processor.run(job)  # synchronous
        # or
        processor.run_async(job)  # background thread
    """

    MAX_CONCURRENCY = MAX_CONCURRENCY
    MAX_RETRIES = MAX_RETRIES

    def __init__(self, llm_client=None):
        from .llm_client import LLMClient
        self._llm = llm_client or LLMClient()

    def create_job(
        self,
        feature: str,
        institute_id: str,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        institute=None,
    ) -> BatchJob:
        """Create a batch job from a list of user prompts and persist it to DB."""
        job = BatchJob(
            job_id=str(uuid.uuid4()),
            feature=feature,
            institute_id=institute_id,
            items=[
                BatchItem(item_id=str(uuid.uuid4()), user_prompt=p)
                for p in prompts
            ],
        )
        with _jobs_lock:
            _jobs[job.job_id] = job
        _evict_memory_jobs_if_needed()
        # Persist to DB immediately so the job survives process restarts
        _persist_job_to_db(job, institute=institute)
        logger.info("Batch job created: %s (%d items)", job.job_id, len(job.items))
        return job

    def run(self, job: BatchJob, system_prompt: Optional[str] = None, institute=None):
        """Process all items in the batch (blocking). Persists progress to DB."""
        from .model_tier import get_model_for_task
        from .prompt_templates import get_template

        job.status = JobStatus.PROCESSING
        model = get_model_for_task(job.feature)

        if system_prompt is None:
            template = get_template(job.feature)
            system_prompt = template.system

        pending = [i for i in job.items if i.status != JobStatus.COMPLETED]

        # FIX SCALE-2: cap max_workers to prevent thread explosion
        workers = min(len(pending), self.MAX_CONCURRENCY)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for item in pending:
                item.status = JobStatus.PROCESSING
                future = executor.submit(
                    self._process_item, item, system_prompt, model, job.institute_id
                )
                future_map[future] = item

            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    result = future.result()
                    item.result = result["content"]
                    item.status = JobStatus.COMPLETED
                except Exception as e:
                    item.error = str(e)
                    item.attempts += 1
                    item.status = (
                        JobStatus.FAILED if item.attempts >= self.MAX_RETRIES
                        else JobStatus.QUEUED
                    )

        # Retry failed items once more
        retryable = [i for i in job.items if i.status == JobStatus.QUEUED]
        if retryable:
            logger.info("Retrying %d failed items in batch %s", len(retryable), job.job_id)
            workers = min(len(retryable), self.MAX_CONCURRENCY)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {}
                for item in retryable:
                    item.status = JobStatus.PROCESSING
                    future = executor.submit(
                        self._process_item, item, system_prompt, model, job.institute_id
                    )
                    future_map[future] = item

                for future in as_completed(future_map):
                    item = future_map[future]
                    try:
                        result = future.result()
                        item.result = result["content"]
                        item.status = JobStatus.COMPLETED
                    except Exception as e:
                        item.error = str(e)
                        item.status = JobStatus.FAILED

        # Final status
        failed_count = sum(1 for i in job.items if i.status == JobStatus.FAILED)
        if failed_count == 0:
            job.status = JobStatus.COMPLETED
        elif failed_count == len(job.items):
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.PARTIAL

        job.completed_at = time.time()
        logger.info("Batch job %s finished: %s", job.job_id, job.progress)

        # Persist final result to DB
        _persist_job_to_db(job, institute=institute)

    def _process_item(
        self, item: BatchItem, system_prompt: str, model: str, institute_id: str
    ) -> dict:
        return self._llm.complete(
            system_prompt=system_prompt,
            user_prompt=item.user_prompt,
            model=model,
            institute_id=institute_id,
        )

    @staticmethod
    def get_job(job_id: str) -> Optional[BatchJob]:
        """Look up a job — in-memory first, then DB fallback."""
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job:
            return job

        # DB fallback for jobs that were evicted from memory or on a different worker
        try:
            from ai_services.models import BatchJob as DBBatchJob
            db_job = DBBatchJob.objects.filter(job_id=job_id).first()
            if db_job:
                # Reconstruct a lightweight BatchJob from DB for status reporting
                job = BatchJob(
                    job_id=str(db_job.job_id),
                    feature=db_job.feature,
                    institute_id=db_job.institute_id_str,
                    status=JobStatus(db_job.status),
                    created_at=db_job.created_at.timestamp(),
                    completed_at=db_job.completed_at.timestamp() if db_job.completed_at else None,
                )
                # Reconstruct items from results_json for the progress report
                for item_id, data in (db_job.results_json or {}).items():
                    item = BatchItem(
                        item_id=item_id,
                        user_prompt="",
                        status=JobStatus(data.get("status", "completed")),
                        result=data.get("result"),
                        error=data.get("error"),
                    )
                    job.items.append(item)
                return job
        except Exception as e:
            logger.warning("BatchProcessor: DB job lookup failed: %s", e)

        return None

    @staticmethod
    def list_jobs(institute_id: Optional[str] = None) -> List[dict]:
        """List jobs from memory cache (fast path). For full history, query DB directly."""
        with _jobs_lock:
            jobs = list(_jobs.values())
        if institute_id:
            jobs = [j for j in jobs if j.institute_id == institute_id]
        return [j.progress for j in jobs]
