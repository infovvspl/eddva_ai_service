import logging
from django.apps import AppConfig

logger = logging.getLogger("ai_services")


class AiServicesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_services"

    def ready(self):
        # Pre-warm singletons on startup (not per-request)
        from ai_services.core.llm_client import _get_groq_client
        from ai_services.core.cache import ResponseCache
        from ai_services.core.rate_limiter import UsageLimiter

        try:
            _get_groq_client()
            logger.info("LLM client (Groq) singleton initialized")
        except Exception as e:
            logger.error("Failed to init LLM client: %s", e)

        # These will attempt Redis connection and fallback gracefully
        ResponseCache()
        UsageLimiter()
        logger.info("AI Services ready — cache and rate limiter initialized")
