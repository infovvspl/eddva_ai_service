"""
Django settings for ai_study_project project.
Production-ready configuration with environment variable overrides.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Security ─────────────────────────────────────────────────────────────────
# FIX BUG-3: No hardcoded fallback. If DJANGO_SECRET_KEY is missing in production,
# the app must CRASH at startup rather than silently use an insecure known key.
# In dev, you can set it in .env. In production, inject it as an environment secret.
_secret = os.getenv("DJANGO_SECRET_KEY", "")
if not _secret:
    if os.getenv("DJANGO_DEBUG", "false").lower() in ("true", "1", "yes"):
        # Dev mode: provide a safe local default so runserver works out of the box
        import warnings
        _secret = "django-dev-only-insecure-key-change-before-deploying"
        warnings.warn(
            "DJANGO_SECRET_KEY is not set. Using an insecure dev-only key. "
            "Set DJANGO_SECRET_KEY in your .env before deploying to production.",
            stacklevel=1,
        )
    else:
        raise RuntimeError(
            "DJANGO_SECRET_KEY environment variable is required in production. "
            "Generate one with: python -c \"from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())\""
        )
SECRET_KEY = _secret
DEBUG = os.getenv("DJANGO_DEBUG", "false").lower() in ("true", "1", "yes")
# ALLOWED_HOSTS — fail loud rather than silently rejecting everything.
#
# The deploy writes .env from CI secrets; an unset secret lands here as an EMPTY
# string, and `"".split(",")` == [""], which is not a valid host. Django then
# answers EVERY request with 400 (DisallowedHost) — the service looks "up" (the
# deploy is green, Redis connects) while serving nothing. That actually happened.
# So: default only when the value is genuinely absent, and refuse to boot in
# production when it is empty.
_allowed_hosts = os.getenv("ALLOWED_HOSTS", "").strip()
if not _allowed_hosts:
    if DEBUG:
        _allowed_hosts = "localhost,127.0.0.1,0.0.0.0"
    else:
        raise RuntimeError(
            "ALLOWED_HOSTS is empty in production. Django would reject EVERY request "
            "with 400 (DisallowedHost). Set it in .env / the deploy workflow, e.g. "
            "ALLOWED_HOSTS=ai.eddva.in,localhost,127.0.0.1"
        )
ALLOWED_HOSTS = [h.strip() for h in _allowed_hosts.split(",") if h.strip()]

# ── Apps ─────────────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "ai_services.apps.AiServicesConfig",
]

# ── Middleware ────────────────────────────────────────────────────────────────
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Tenant auth — validates API key and resolves institute on every request
    "ai_services.middleware.TenantAuthMiddleware",
]

# ── CORS ─────────────────────────────────────────────────────────────────────
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _cors_origins:
    CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    CORS_ALLOW_ALL_ORIGINS = False
else:
    # Dev fallback — allow all in debug mode only
    CORS_ALLOW_ALL_ORIGINS = DEBUG

ROOT_URLCONF = "ai_study_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ai_study_project.wsgi.application"

# ── Database ─────────────────────────────────────────────────────────────────
_db_engine = os.getenv("DB_ENGINE", "")

if _db_engine:
    # Production: PostgreSQL (or whatever DB_ENGINE is set to)
    DATABASES = {
        "default": {
            "ENGINE": _db_engine,
            "NAME": os.getenv("DB_NAME", "ai_services_db"),
            "USER": os.getenv("DB_USER", "postgres"),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
            "HOST": os.getenv("DB_HOST", "localhost"),
            "PORT": os.getenv("DB_PORT", "5432"),
            "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "600")),
            "OPTIONS": {
                "connect_timeout": 5,
            },
        }
    }
else:
    # SQLite. Fine for dev, and workable in production at low traffic — but note
    # SQLite locks the whole file on write and we write a UsageLog row on every
    # request, so concurrent workers can hit "database is locked". The busy
    # timeout below makes writers WAIT for the lock instead of failing instantly;
    # WAL mode (set in ai_services/apps.py) lets readers run during a write.
    if not DEBUG:
        import warnings
        warnings.warn(
            "DB_ENGINE is not set — running PRODUCTION on SQLite (db.sqlite3). "
            "Acceptable at low traffic, but writes serialise on a single file lock "
            "and the data lives only on this box's disk (lost if the instance is "
            "replaced). Set DB_ENGINE/DB_NAME/DB_USER/DB_PASSWORD/DB_HOST to move "
            "to Postgres.",
            stacklevel=1,
        )
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            "OPTIONS": {
                # Wait for a busy write lock instead of erroring immediately.
                "timeout": int(os.getenv("SQLITE_TIMEOUT", "20")),
            },
        }
    }

# ── Auth ─────────────────────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── i18n ─────────────────────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

# ── Static ───────────────────────────────────────────────────────────────────
STATIC_URL = "static/"

# ── DRF ──────────────────────────────────────────────────────────────────────
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.getenv("DRF_ANON_THROTTLE", "100/min"),
    },
}

# ── Redis ─────────────────────────────────────────────────────────────────────
# Redis is used for:
#   1. AI response caching (primary cost optimization — avoids repeated LLM calls)
#   2. Question bank persistence (survives restarts, shared across workers)
#   3. Cache hit/miss statistics (cost savings dashboard)
#
# Required in production. Falls back to in-memory (per-worker) if not set.
#
# For local dev:   redis://localhost:6379
# For EC2:         redis://127.0.0.1:6379  (Redis on same server)
# For Redis Cloud: redis://:password@host:6379
#
# IMPORTANT Redis memory policy: Set `maxmemory-policy allkeys-lru` in redis.conf
# or via: redis-cli CONFIG SET maxmemory-policy allkeys-lru
# This means Redis will evict the least-recently-used keys when memory is full,
# which is correct behavior for a cache (never crash, always make space for new entries).
#
# Recommended maxmemory for 750 students: 256mb–512mb
# Set via: redis-cli CONFIG SET maxmemory 256mb

REDIS_URL = os.getenv("REDIS_URL", "")

# Redis is REQUIRED in production. Without it, response caching, per-tenant token
# budget caps, and cross-worker concurrency limits all silently degrade to
# per-worker in-memory state — caps become unenforced (billing/abuse risk) and
# every worker re-pays the LLM. Fail loud instead of running mis-configured.
#
# Escape hatch: REDIS_OPTIONAL=true explicitly accepts the in-memory fallback
# (e.g. a one-off single-worker box). Do NOT use it for real multi-worker traffic.
if not REDIS_URL and not DEBUG:
    if os.getenv("REDIS_OPTIONAL", "false").lower() in ("true", "1", "yes"):
        import warnings
        warnings.warn(
            "REDIS_URL is not set and REDIS_OPTIONAL=true. Running on per-worker "
            "in-memory fallback: token caps and concurrency limits are NOT enforced "
            "globally and caching is disabled. Not safe for multi-worker production.",
            stacklevel=1,
        )
    else:
        raise RuntimeError(
            "REDIS_URL environment variable is required in production. Redis backs "
            "response caching, per-tenant token budgets, and cross-worker concurrency "
            "limits. Set REDIS_URL=redis://localhost:6379 (or your endpoint). To run "
            "without it on a single-worker box, set REDIS_OPTIONAL=true."
        )

# ── Logging ──────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "ai_services": {
            "handlers": ["console"],
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}

