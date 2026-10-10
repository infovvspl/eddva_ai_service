"""
Run deck jobs from the Redis queue, apart from the web server.

    python manage.py run_ppt_worker [--concurrency 3]

Run one or more of these (pm2, systemd, a container) beside gunicorn. While at
least one is alive, /ppt/generate/start queues decks for them; with none, decks
run inside the web process as before (see ai_services.core.ppt_v2.job_queue).

On SIGTERM or Ctrl+C the worker takes no new job and finishes the decks it
has, so give the process manager a stop timeout longer than a deck takes
(pm2: --kill-timeout 300000).
"""
import logging
import signal

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger("ai_services.ppt_worker")


class Command(BaseCommand):
    help = "Run PPT deck jobs from the Redis queue"

    def add_arguments(self, parser):
        parser.add_argument("--concurrency", type=int, default=None,
                            help="decks this worker runs at once (default PPT_WORKER_CONCURRENCY or 3)")

    def handle(self, *args, **opts):
        import os

        import redis

        from ai_services.core.ppt_v2 import job_queue, jobs
        from ai_services.views.ppt import run_queued_job

        # Its own connection: the service's shared pool times reads out at 1s,
        # shorter than a worker's blocking wait for the next job.
        url = (os.getenv("REDIS_URL") or "").strip()
        try:
            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=15,
                                          socket_connect_timeout=5, health_check_interval=30)
            client.ping()
        except Exception as exc:
            raise CommandError("REDIS_URL is not set or Redis is unreachable (%s); "
                               "the deck queue lives in Redis." % type(exc).__name__)
        concurrency = opts["concurrency"] or int(os.getenv("PPT_WORKER_CONCURRENCY", "") or 3)
        worker = job_queue.Worker(client, run_queued_job, concurrency=concurrency,
                                  on_give_up=jobs.give_up)

        def stop(signum, frame):
            logger.info("ppt worker: signal %s; finishing the decks in hand", signum)
            worker.stop.set()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, stop)
            except (ValueError, OSError):        # not every platform has every signal
                pass
        self.stdout.write("PPT worker %s running %d deck(s) at a time" % (worker.id, concurrency))
        worker.run_forever()
