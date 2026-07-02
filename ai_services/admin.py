import uuid
from django.contrib import admin
from django.db.models import Sum, Count
from django.utils import timezone
from .models import Institute, UsageLog, BatchJob


@admin.register(Institute)
class InstituteAdmin(admin.ModelAdmin):
    list_display = (
        "name", "slug", "vertical", "plan", "is_active", "is_service_account",
        "daily_soft_cap", "daily_hard_cap", "max_concurrent_requests", "created_at",
    )
    list_filter = ("vertical", "plan", "is_active", "is_service_account")
    search_fields = ("name", "slug", "contact_email")
    readonly_fields = ("id", "api_key", "created_at", "updated_at")
    fieldsets = (
        ("Identity", {
            "fields": (
                "id", "name", "slug", "vertical", "api_key",
                "external_tenant_id", "contact_email", "is_active", "is_service_account",
            ),
        }),
        ("Plan & Limits", {
            "fields": ("plan", "daily_soft_cap", "daily_hard_cap", "max_concurrent_requests"),
        }),
        ("Feature Toggles", {
            "fields": ("features_enabled",),
            "description": 'JSON: {"feedback": true, "batch": false}',
        }),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def save_model(self, request, obj, form, change):
        if not change and not obj.api_key:
            obj.api_key = f"ask_{uuid.uuid4().hex[:32]}"
        super().save_model(request, obj, form, change)
        # FIX BUG-4: Invalidate BOTH caches (api_key cache + tenant_id cache)
        # so NestJS-routed requests immediately see the updated config.
        from .middleware import invalidate_institute_cache
        invalidate_institute_cache(
            api_key=obj.api_key,
            external_tenant_id=obj.external_tenant_id if obj.external_tenant_id else None,
        )


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = (
        "institute_id_str", "vertical", "feature", "model_used",
        "total_tokens", "cache_hit", "latency_ms", "created_at",
    )
    list_filter = ("vertical", "institute_id_str", "feature", "cache_hit", "model_used")
    date_hierarchy = "created_at"
    readonly_fields = (
        "institute", "institute_id_str", "vertical", "feature", "model_used",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_ms", "cache_hit", "created_at",
    )


@admin.register(BatchJob)
class BatchJobAdmin(admin.ModelAdmin):
    list_display = (
        "job_id", "institute_id_str", "feature", "status",
        "total_items", "completed_items", "failed_items",
        "created_at", "expires_at",
    )
    list_filter = ("status", "feature", "institute_id_str")
    date_hierarchy = "created_at"
    readonly_fields = (
        "job_id", "institute", "institute_id_str", "feature", "status",
        "total_items", "completed_items", "failed_items",
        "results_json", "created_at", "completed_at", "expires_at",
    )
