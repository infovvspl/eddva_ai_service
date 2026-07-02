"""
Management command: purge_expired_batch_jobs

Deletes BatchJob rows whose expires_at timestamp has passed.
Run this via cron or a scheduled task:

    # Every hour — keep only recent jobs
    0 * * * * python manage.py purge_expired_batch_jobs

    # Or with Django management:
    python manage.py purge_expired_batch_jobs --dry-run

FIX BUG-5: This command is the TTL eviction half of the batch job memory-leak fix.
The other half (bounded in-memory dict + DB persistence) is in batch_processor.py.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "Delete expired BatchJob rows (TTL eviction). Safe to run frequently."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report how many rows would be deleted without actually deleting them.",
        )
        parser.add_argument(
            "--older-than-hours",
            type=int,
            default=0,
            help=(
                "Also delete completed/failed jobs older than N hours, even if expires_at is not set. "
                "Useful for cleaning up legacy rows created before the expires_at column was added."
            ),
        )

    def handle(self, *args, **options):
        from ai_services.models import BatchJob

        dry_run = options["dry_run"]
        older_than_hours = options["older_than_hours"]

        now = timezone.now()

        # Primary eviction: rows past their explicit expires_at
        qs_expired = BatchJob.objects.filter(expires_at__lte=now)

        # Optional: legacy cleanup for rows without expires_at
        qs_legacy = BatchJob.objects.none()
        if older_than_hours > 0:
            cutoff = now - timezone.timedelta(hours=older_than_hours)
            qs_legacy = BatchJob.objects.filter(
                expires_at__isnull=True,
                created_at__lte=cutoff,
                status__in=("completed", "failed", "partial"),
            )

        expired_count = qs_expired.count()
        legacy_count = qs_legacy.count()
        total = expired_count + legacy_count

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"[DRY RUN] Would delete {expired_count} expired rows "
                    f"+ {legacy_count} legacy rows = {total} total."
                )
            )
            return

        deleted_expired, _ = qs_expired.delete()
        deleted_legacy, _ = qs_legacy.delete()
        total_deleted = deleted_expired + deleted_legacy

        self.stdout.write(
            self.style.SUCCESS(
                f"Purged {total_deleted} batch job rows "
                f"({deleted_expired} expired + {deleted_legacy} legacy)."
            )
        )
