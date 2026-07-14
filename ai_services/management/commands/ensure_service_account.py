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

        # Look the tenant up by SLUG, not by api_key.
        #
        # There is exactly ONE service-account tenant; the API key is a property of
        # it, not its identity. Keying on api_key was wrong twice over:
        #   * rotating the key tried to CREATE a second row, which blew up on the
        #     unique slug ("UNIQUE constraint failed: ai_services_institute.slug");
        #   * even if it had succeeded, the OLD key would have stayed active — a
        #     backdoor that survives every rotation.
        # Keying on slug makes rotation an update: the new key works, the old one
        # stops working immediately.
        institute, created = Institute.objects.get_or_create(
            slug=opts["slug"],
            defaults={
                "name": opts["name"],
                "api_key": api_key,
                "vertical": opts["vertical"],
                "is_active": True,
                "is_service_account": True,
            },
        )

        # Rotate the key onto the existing row, and repair a tenant that is inactive
        # or not flagged as a service account (only a service account may switch
        # tenant via X-Tenant-ID).
        changed = []
        if institute.api_key != api_key:
            institute.api_key = api_key      # rotation: new key works, old one dies
            changed.append("api_key")
        if not institute.is_active:
            institute.is_active = True
            changed.append("is_active")
        if not institute.is_service_account:
            institute.is_service_account = True
            changed.append("is_service_account")
        if changed:
            institute.save(update_fields=changed)

        # Belt and braces: make sure no OTHER active tenant is still carrying the
        # weak default key. Otherwise `apexiq-dev-secret-key-2026` — which is in the
        # git history and the docs — stays a valid credential on an internet-facing
        # service.
        WEAK_DEFAULT = "apexiq-dev-secret-key-2026"
        if api_key != WEAK_DEFAULT:
            stale = (
                Institute.objects
                .filter(api_key=WEAK_DEFAULT, is_active=True)
                .exclude(pk=institute.pk)
                .update(is_active=False)
            )
            if stale:
                self.stdout.write(self.style.WARNING(
                    f"Deactivated {stale} tenant(s) still using the weak default API key"
                ))

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
