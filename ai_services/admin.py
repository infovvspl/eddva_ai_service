import uuid
from django.contrib import admin
from django.db.models import Sum, Count
from django.utils import timezone
from .models import Institute, UsageLog, BatchJob


@admin.register(Institute)
class InstituteAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "plan", "is_active", "daily_soft_cap", "daily_hard_cap", "max_concurrent_requests", "created_at")
    list_filter = ("plan", "is_active")
    search_fields = ("name", "slug", "contact_email")
    readonly_fields = ("id", "api_key", "created_at", "updated_at")
    fieldsets = (
        ("Identity", {"fields": ("id", "name", "slug", "api_key", "external_tenant_id", "contact_email", "is_active")}),
        ("Plan & Limits", {"fields": ("plan", "daily_soft_cap", "daily_hard_cap", "max_concurrent_requests")}),
        ("Feature Toggles", {"fields": ("features_enabled",), "description": "JSON: {\"feedback\": true, \"batch\": false}"}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def save_model(self, request, obj, form, change):
        if not change and not obj.api_key:
            obj.api_key = f"ask_{uuid.uuid4().hex[:32]}"
        super().save_model(request, obj, form, change)
        # Invalidate middleware cache when institute config changes
        from .middleware import invalidate_institute_cache
        invalidate_institute_cache(obj.api_key)


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = ("institute_id_str", "feature", "model_used", "total_tokens", "cache_hit", "latency_ms", "created_at")
    list_filter = ("institute_id_str", "feature", "cache_hit", "model_used")
    date_hierarchy = "created_at"
    readonly_fields = ("institute", "institute_id_str", "feature", "model_used", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms", "cache_hit", "created_at")


@admin.register(BatchJob)
class BatchJobAdmin(admin.ModelAdmin):
    list_display = ("job_id", "institute_id_str", "feature", "status", "total_items", "completed_items", "failed_items", "created_at")
    list_filter = ("status", "feature", "institute_id_str")
    date_hierarchy = "created_at"
