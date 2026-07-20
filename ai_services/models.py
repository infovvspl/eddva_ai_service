import uuid
from django.db import models


class Institute(models.Model):
    """
    Tenant registry — each institute is a tenant with its own config.

    Every API request is scoped to an institute via API key.
    Caps, features, and billing are per-institute.
    """

    PLAN_CHOICES = [
        ("free", "Free"),
        ("basic", "Basic"),
        ("premium", "Premium"),
        ("enterprise", "Enterprise"),
    ]

    VERTICAL_CHOICES = [
        ("coaching", "Coaching (JEE/NEET/Competitive)"),
        ("school", "School (Class 1-10)"),
    ]

    # Keep in sync with ai_services.core.boards.BOARDS AND with the board options the
    # admin UI offers (eddva_frontend school/admin/Institutes.jsx): CBSE, ICSE,
    # State Board, IB.
    BOARD_CHOICES = [
        ("cbse", "CBSE (NCERT)"),
        ("icse", "ICSE / CISCE"),
        ("state", "State Board"),
        ("ib", "IB (International Baccalaureate)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, help_text="URL-safe identifier, e.g. 'allen-kota'")
    vertical = models.CharField(
        max_length=32,
        choices=VERTICAL_CHOICES,
        default="coaching",
        db_index=True,
        help_text=(
            "Product vertical this tenant belongs to. Selects prompt/model variants. "
            "Can be overridden per-request via the X-Vertical header."
        ),
    )
    board = models.CharField(
        max_length=32,
        choices=BOARD_CHOICES,
        default="cbse",
        db_index=True,
        help_text=(
            "Education board for a school tenant (CBSE / ICSE / State). Selects syllabus, "
            "textbook and paper-pattern framing. Usually supplied per-request via the "
            "X-Board header (the backend reads it from institutes.board); this is the "
            "fallback when the header is absent."
        ),
    )
    api_key = models.CharField(
        max_length=64, unique=True, db_index=True,
        help_text="Sent via X-API-Key or Authorization: Bearer header",
    )
    external_tenant_id = models.CharField(
        max_length=64, unique=True, null=True, blank=True, db_index=True,
        help_text="UUID from NestJS Tenant table — links this Institute to the apexiq-backend tenant",
    )
    plan = models.CharField(max_length=20, choices=PLAN_CHOICES, default="free")
    is_active = models.BooleanField(default=True)

    # FIX BUG-9: Service-account flag.
    # Only service accounts (e.g. the NestJS ai-bridge) are allowed to use the
    # X-Tenant-ID header to operate on behalf of a different tenant.
    # Regular institute API keys MUST NEVER be able to switch tenant context.
    is_service_account = models.BooleanField(
        default=False,
        db_index=True,
        help_text=(
            "If True, this API key belongs to a trusted internal service (e.g. NestJS ai-bridge). "
            "Only service accounts may use X-Tenant-ID to operate on behalf of another tenant. "
            "Regular institute keys are NEVER allowed to impersonate another tenant."
        ),
    )

    # Per-tenant token caps (override global defaults)
    daily_soft_cap = models.IntegerField(
        default=500_000,
        help_text="Warn after this many tokens/day",
    )
    daily_hard_cap = models.IntegerField(
        default=1_000_000,
        help_text="Block after this many tokens/day",
    )
    max_concurrent_requests = models.IntegerField(
        default=10,
        help_text="Max parallel LLM calls for this tenant",
    )

    # Feature toggles — disable features per tenant
    features_enabled = models.JSONField(
        default=dict,
        blank=True,
        help_text='e.g. {"feedback": true, "cheating": true, "batch": false}',
    )

    # Billing
    contact_email = models.EmailField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.plan})"

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled for this tenant."""
        if not self.features_enabled:
            return True  # all features on by default
        return self.features_enabled.get(feature, True)

    @classmethod
    def get_by_api_key(cls, api_key: str):
        """Lookup tenant by API key. Returns None if not found or inactive."""
        try:
            return cls.objects.get(api_key=api_key, is_active=True)
        except cls.DoesNotExist:
            return None


class UsageLog(models.Model):
    """Track every LLM call per institute for billing and monitoring."""

    institute = models.ForeignKey(
        Institute, on_delete=models.CASCADE, related_name="usage_logs",
        null=True, blank=True,
    )
    # institute_id_str is a denormalized copy of institute.slug kept for:
    #   1. Fast filtering in admin_api.py without a JOIN
    #   2. Logging anonymous / unauthenticated calls where FK is null
    # Rule: always set institute_id_str == institute.slug when FK is present.
    institute_id_str = models.CharField(
        max_length=128, db_index=True,
        help_text="Denormalized slug — must match institute.slug. Used for fast filter queries.",
    )
    feature = models.CharField(max_length=64, db_index=True)
    vertical = models.CharField(
        max_length=32, default="coaching", db_index=True,
        help_text="Vertical this call was served under (for per-vertical usage/billing breakdown)",
    )
    board = models.CharField(
        max_length=32, default="", blank=True, db_index=True,
        help_text="Education board this call was served under (school tenants), for per-board usage breakdown",
    )
    model_used = models.CharField(max_length=64)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0)
    cache_hit = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["institute_id_str", "created_at"]),
            models.Index(fields=["feature", "created_at"]),
        ]

    def __str__(self):
        return f"{self.institute_id_str} | {self.feature} | {self.total_tokens} tokens"


class BatchJob(models.Model):
    """
    Persistent record of batch processing jobs.

    FIX BUG-5: Jobs are persisted in DB (not in-memory dict) and carry an
    expires_at timestamp. A management command / cron job deletes expired rows.
    """

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("partial", "Partial"),
    ]

    job_id = models.UUIDField(unique=True, db_index=True)
    institute = models.ForeignKey(
        Institute, on_delete=models.CASCADE, related_name="batch_jobs",
        null=True, blank=True,
    )
    institute_id_str = models.CharField(max_length=128, db_index=True)
    feature = models.CharField(max_length=64)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    total_items = models.IntegerField(default=0)
    completed_items = models.IntegerField(default=0)
    failed_items = models.IntegerField(default=0)
    results_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    # TTL for garbage collection — management command deletes rows past this date.
    # Default: 24 hours after creation. Adjust per plan tier if needed.
    expires_at = models.DateTimeField(
        null=True, blank=True, db_index=True,
        help_text="Row is safe to delete after this time. Set on job creation.",
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Batch {self.job_id} | {self.feature} | {self.status}"
