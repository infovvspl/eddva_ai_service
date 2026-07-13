"""
Ensure the NestJS ai-bridge service-account Institute exists.

Every request to this service is authenticated against the Institute table. If
no Institute row matches the key the backend sends, EVERY call 401s — which is
exactly what happened on EC2 (the deploy only ever *updated* an existing row and
never created one, so the tenant table stayed empty).

Idempotent: safe to run on every deploy.

    python manage.py ensure_service_account --api-key "$AI_API_KEY"
"""

from django.core.management.base import BaseCommand, CommandError

from ai_services.models import Institute


class Command(BaseCommand):
    help = "Create/repair the service-account Institute used by the NestJS ai-bridge."

    def add_arguments(self, parser):
        parser.add_argument("--api-key", required=True, help="Key the backend sends as X-API-Key / Bearer")
        parser.add_argument("--slug", default="apexiq-nestjs-service")
        parser.add_argument("--name", default="ApexIQ NestJS Service")
        parser.add_argument("--vertical", default="coaching", help="Default vertical for this tenant")

    def handle(self, *args, **opts):
        api_key = (opts["api_key"] or "").strip()
        if not api_key:
            raise CommandError(
                "--api-key is empty. The backend's AI_API_KEY must be set, otherwise "
                "no request can authenticate."
            )

        institute, created = Institute.objects.get_or_create(
            api_key=api_key,
            defaults={
                "name": opts["name"],
                "slug": opts["slug"],
                "vertical": opts["vertical"],
                "is_active": True,
                "is_service_account": True,
            },
        )

        # Repair an existing row that is inactive or not marked as a service
        # account (a service account is what may switch tenant via X-Tenant-ID).
        changed = []
        if not institute.is_active:
            institute.is_active = True
            changed.append("is_active")
        if not institute.is_service_account:
            institute.is_service_account = True
            changed.append("is_service_account")
        if not institute.slug:
            institute.slug = opts["slug"]
            changed.append("slug")
        if changed:
            institute.save(update_fields=changed)

        # The middleware caches Institute lookups for 5 minutes — drop it so the
        # new/updated row is picked up immediately after a deploy.
        try:
            from ai_services.middleware import invalidate_institute_cache
            invalidate_institute_cache()
        except Exception:
            pass

        if created:
            self.stdout.write(self.style.SUCCESS(
                f"Created service-account Institute '{institute.slug}' (vertical={institute.vertical})"
            ))
        elif changed:
            self.stdout.write(self.style.SUCCESS(
                f"Repaired service-account Institute '{institute.slug}' ({', '.join(changed)})"
            ))
        else:
            self.stdout.write(f"Service-account Institute '{institute.slug}' already OK")

        self.stdout.write(f"Active institutes: {Institute.objects.filter(is_active=True).count()}")
