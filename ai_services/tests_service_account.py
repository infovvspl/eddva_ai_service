"""
Regression tests for `ensure_service_account`.

This command is what stands between a deploy and a service that 401s every
request, so its edge cases matter:

  * Rotating the API key must UPDATE the single service-account tenant, not try to
    create a second one. Creating blew up with
    "UNIQUE constraint failed: ai_services_institute.slug" and left production
    with zero active tenants.
  * After rotation the OLD key must stop working — otherwise every rotation leaves
    a live backdoor behind.
"""

from django.core.management import call_command
from django.test import TestCase

from ai_services.models import Institute

WEAK_DEFAULT = "apexiq-dev-secret-key-2026"


class EnsureServiceAccountTests(TestCase):
    def _run(self, key):
        call_command("ensure_service_account", "--api-key", key, verbosity=0)

    def test_creates_the_tenant_when_the_table_is_empty(self):
        self.assertEqual(Institute.objects.count(), 0)
        self._run("key-one")
        i = Institute.objects.get(api_key="key-one")
        self.assertTrue(i.is_active)
        self.assertTrue(i.is_service_account)

    def test_is_idempotent(self):
        self._run("key-one")
        self._run("key-one")
        self.assertEqual(Institute.objects.count(), 1)

    def test_rotating_the_key_updates_in_place_and_does_not_crash(self):
        """The exact production failure: a second key hit the unique slug."""
        self._run("old-key")
        self._run("new-key")          # used to raise IntegrityError
        self.assertEqual(Institute.objects.count(), 1, "rotation must not create a second tenant")
        i = Institute.objects.get()
        self.assertEqual(i.api_key, "new-key")
        self.assertTrue(i.is_active)

    def test_the_old_key_stops_working_after_rotation(self):
        self._run("old-key")
        self._run("new-key")
        self.assertIsNone(Institute.get_by_api_key("old-key"), "old key must not authenticate")
        self.assertIsNotNone(Institute.get_by_api_key("new-key"))

    def test_reactivates_a_deactivated_tenant(self):
        """Recovers the state prod was left in: tenant present but is_active=False."""
        self._run("key-one")
        Institute.objects.update(is_active=False)
        self.assertIsNone(Institute.get_by_api_key("key-one"))
        self._run("key-one")
        self.assertIsNotNone(Institute.get_by_api_key("key-one"))

    def test_weak_default_key_is_deactivated_when_a_real_key_is_set(self):
        Institute.objects.create(
            name="legacy", slug="legacy", api_key=WEAK_DEFAULT, is_active=True,
        )
        self._run("a-real-strong-key")
        self.assertIsNone(
            Institute.get_by_api_key(WEAK_DEFAULT),
            "the weak default must not remain a valid credential",
        )
        self.assertIsNotNone(Institute.get_by_api_key("a-real-strong-key"))

    def test_empty_key_is_rejected(self):
        from django.core.management.base import CommandError
        with self.assertRaises(CommandError):
            call_command("ensure_service_account", "--api-key", "", verbosity=0)
