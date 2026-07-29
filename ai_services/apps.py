import os
import sys
import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger("ai_services")


class AiServicesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_services"

    def ready(self):
        from ai_services.core.cache import get_redis
        from ai_services.core.llm_client import _get_groq_client, check_groq_keys
        from ai_services.core.rate_limiter import UsageLimiter

        # ── SQLite tuning ─────────────────────────────────────────────────────
        # Every request writes a UsageLog row. SQLite locks the whole file on
        # write, so with multiple gunicorn workers that serialises and can raise
        # "database is locked". WAL lets readers proceed during a write, and the
        # busy_timeout makes a blocked writer wait rather than fail instantly.
        # (No-op on Postgres.)
        from django.db.backends.signals import connection_created
        from django.dispatch import receiver

        @receiver(connection_created)
        def _tune_sqlite(sender, connection, **kwargs):
            if connection.vendor != "sqlite":
                return
            try:
                with connection.cursor() as cursor:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    cursor.execute("PRAGMA busy_timeout=20000;")
            except Exception as exc:  # never block startup on a pragma
                logger.warning("Could not apply SQLite pragmas: %s", exc)

        self._tune_sqlite = _tune_sqlite  # keep a strong ref (receivers are weak)

        # ── LLM Client Init ───────────────────────────────────────────────────
        try:
            _get_groq_client()
            logger.info("LLM client (Groq) initialized")
        except Exception as e:
            logger.error("Failed to init LLM client: %s", e)

        # ── Groq Key Health Check (background) ───────────────────────────────
        def _run_health_check():
            try:
                check_groq_keys()
            except Exception as exc:
                logger.error("Groq health check crashed: %s", exc)

        t = threading.Thread(target=_run_health_check, name="groq-health-check", daemon=True)
        t.start()

        # ── Redis Validation ──────────────────────────────────────────────────
        # Attempt Redis connection. Log a prominent warning if it's missing,
        # because in production Redis is critical for cost savings and shared state.
        redis_url = os.getenv("REDIS_URL", "")
        redis_client = get_redis()

        if redis_client:
            logger.info("Redis connected — response caching active (cost optimization ON)")
        else:
            is_debug = os.getenv("DJANGO_DEBUG", "false").lower() in ("true", "1", "yes")
            if is_debug:
                logger.warning(
                    "Redis NOT connected — falling back to per-worker in-memory cache. "
                    "This is OK for local dev. Set REDIS_URL in .env for production."
                )
            else:
                # Production without Redis = shared state broken, cache not shared across workers
                logger.error(
                    "═══════════════════════════════════════════════════════════\n"
                    "  PRODUCTION WARNING: Redis is NOT connected!\n"
                    "  REDIS_URL=%s\n"
                    "  Without Redis:\n"
                    "    ✗ LLM responses are NOT shared between gunicorn workers\n"
                    "    ✗ Rate limits are NOT enforced correctly (per-worker only)\n"
                    "    ✗ No AI cost savings from caching\n"
                    "  Fix: Set REDIS_URL=redis://your-redis-host:6379 in .env\n"
                    "═══════════════════════════════════════════════════════════",
                    redis_url or "(not set)",
                )

        # ── Rate Limiter Init ─────────────────────────────────────────────────
        UsageLimiter()

        # ── Local dev self-heal ──────────────────────────────────────────────
        # SQLite dev DBs (DB_ENGINE unset, see settings.py) start empty. Without
        # migrations + a service-account Institute row matching NESTJS_SERVICE_
        # API_KEY, every call from the NestJS ai-bridge 500s ("no such table")
        # or 401s. Auto-repair on `runserver` so a fresh clone / wiped db.sqlite3
        # just works — both operations are idempotent, safe to run every start.
        # Scoped to `runserver` only so `migrate`/`makemigrations`/`test`/`shell`
        # aren't affected, and skipped entirely once DB_ENGINE points at a real
        # (production) database.
        if not os.getenv("DB_ENGINE") and "runserver" in sys.argv:
            try:
                from django.core.management import call_command
                call_command("migrate", verbosity=0, interactive=False)
                service_key = os.getenv("NESTJS_SERVICE_API_KEY", "").strip()
                if service_key:
                    call_command("ensure_service_account", api_key=service_key, verbosity=0)
                else:
                    logger.warning(
                        "NESTJS_SERVICE_API_KEY not set — skipping service-account "
                        "self-heal. Calls from the NestJS backend will 401 until "
                        "`python manage.py ensure_service_account --api-key <AI_API_KEY>` "
                        "is run manually."
                    )
            except Exception as exc:
                logger.error("Local dev self-heal (migrate/ensure_service_account) failed: %s", exc)

        logger.info("AI Services ready — all components initialized")
