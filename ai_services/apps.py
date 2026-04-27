import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger("ai_services")


class AiServicesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_services"

    def ready(self):
        from ai_services.core.llm_client import _get_groq_client, check_groq_keys
        from ai_services.core.cache import ResponseCache
        from ai_services.core.rate_limiter import UsageLimiter

        # Quick key-count check (sync, instant — no network call)
        try:
            _get_groq_client()
            logger.info("LLM client (Groq) initialized")
        except Exception as e:
            logger.error("Failed to init LLM client: %s", e)

        # Full per-key health check in background — makes one tiny API call per key.
        # Runs after Django is ready so it never delays request handling.
        def _run_health_check():
            try:
                check_groq_keys()
            except Exception as exc:
                logger.error("Groq health check crashed: %s", exc)

        t = threading.Thread(target=_run_health_check, name="groq-health-check", daemon=True)
        t.start()

        # These will attempt Redis connection and fallback gracefully
        ResponseCache()
        UsageLimiter()
        logger.info("AI Services ready — cache and rate limiter initialized")
