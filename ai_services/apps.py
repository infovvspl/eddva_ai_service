import os
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
        logger.info("AI Services ready — all components initialized")
