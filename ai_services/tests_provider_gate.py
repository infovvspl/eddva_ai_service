"""One limit on calls in flight to a provider, shared by every process.

A single deck sending three slide paintings at once got "429 too many requests"
from Together, and each worker only limited itself. These tests drive the gate
on a fake Redis (with Lua) and a fake clock: two Gate objects on one server
stand for two processes.
"""
import threading
import time
from unittest import mock, skipUnless

from django.test import SimpleTestCase

from ai_services.core import provider_gate
from ai_services.core.provider_gate import Gate, GateBusy

try:
    import fakeredis
    HAVE_FAKEREDIS = True
except ImportError:          # a CI image without it skips the Redis tests
    HAVE_FAKEREDIS = False


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def gates(n=2, **kw):
    """n gates (processes) on one fake Redis, with one shared fake clock."""
    server = fakeredis.FakeServer()
    clock = Clock()
    made = [Gate("test", client_factory=lambda: fakeredis.FakeRedis(server=server,
                                                                     decode_responses=True),
                 clock=clock, sleep=clock.sleep, **kw) for _ in range(n)]
    return made, clock


@skipUnless(HAVE_FAKEREDIS, "fakeredis[lua] is not installed")
class SharedLimitTests(SimpleTestCase):
    def test_the_limit_holds_across_processes(self):
        (p1, p2), clock = gates(capacity=2)
        self.assertEqual(p1._attempt("a", "school-a")[0], True)
        self.assertEqual(p2._attempt("b", "school-b")[0], True)
        granted, place = p1._attempt("c", "school-c")
        self.assertFalse(granted)
        self.assertEqual(place, 0)                          # first in line
        p2._forget("a")                                     # the first call finished
        self.assertTrue(p1._attempt("c", "school-c")[0])

    def test_the_next_slot_goes_to_the_school_with_fewest_in_use(self):
        (p1, p2), clock = gates(capacity=2)
        p1._attempt("a1", "A")
        p1._attempt("a2", "A")                              # school A has both slots
        clock.t += 1
        self.assertFalse(p1._attempt("a3", "A")[0])        # A asks again first...
        clock.t += 1
        self.assertFalse(p2._attempt("b1", "B")[0])        # ...then B
        p1._forget("a1")
        self.assertFalse(p1._attempt("a3", "A")[0])        # A still holds one; B holds none
        self.assertTrue(p2._attempt("b1", "B")[0])
        p1._forget("a2")
        self.assertTrue(p1._attempt("a3", "A")[0])

    def test_no_school_holds_more_than_its_share(self):
        (p1, p2), clock = gates(capacity=3, per_tenant=1)
        self.assertTrue(p1._attempt("a1", "A")[0])
        self.assertFalse(p1._attempt("a2", "A")[0])        # a free slot, but A is at its cap
        self.assertTrue(p2._attempt("b1", "B")[0])

    def test_a_dead_workers_slot_is_freed_when_its_lease_ends(self):
        (p1, p2), clock = gates(capacity=1, lease_s=60)
        self.assertTrue(p1._attempt("dead", "A")[0])       # this worker is then killed
        self.assertFalse(p2._attempt("next", "B")[0])
        clock.t += 61
        self.assertTrue(p2._attempt("next", "B")[0])

    def test_a_long_call_renews_its_lease(self):
        (p1, p2), clock = gates(capacity=1, lease_s=60)
        with p1.slot("A") as ticket:
            clock.t += 50
            ticket.renew()
            clock.t += 50                                   # 100s in: past the first lease
            self.assertFalse(p2._attempt("next", "B")[0])
        self.assertTrue(p2._attempt("next", "B")[0])

    def test_a_request_that_stopped_waiting_does_not_hold_the_line(self):
        (p1, p2), clock = gates(capacity=1)
        p1._attempt("holder", "A")
        p1._attempt("gone", "B")                            # waits, then its worker dies
        clock.t += 1
        p2._attempt("w2", "C")
        p1._forget("holder")
        self.assertFalse(p2._attempt("w2", "C")[0])        # "gone" is still in line
        clock.t += Gate.WAITER_TTL_S + 1
        self.assertTrue(p2._attempt("w2", "C")[0])

    def test_a_429_pauses_every_process(self):
        (p1, p2), clock = gates(capacity=3)
        p1.cool_down(0.3)
        self.assertFalse(p2._attempt("x", "A")[0])
        time.sleep(0.4)                                     # the pause is a real Redis TTL
        self.assertTrue(p2._attempt("x", "A")[0])

    def test_a_short_pause_never_cuts_a_longer_one(self):
        (p1, p2), clock = gates(capacity=3)
        p1.cool_down(5)
        p2.cool_down(0.05)
        time.sleep(0.1)
        self.assertFalse(p2._attempt("x", "A")[0])


@skipUnless(HAVE_FAKEREDIS, "fakeredis[lua] is not installed")
class WaitingTests(SimpleTestCase):
    def test_a_request_waits_its_turn_and_hears_its_place(self):
        (p1, p2), clock = gates(capacity=1)
        p1._attempt("holder", "A")
        places = []
        orig_sleep = clock.sleep

        def sleep(s):                                       # the holder finishes while we wait
            orig_sleep(s)
            if len(places) and clock.t > 1_000_001:
                p1._forget("holder")
        p2._sleep = sleep
        with p2.slot("B", max_wait_s=30, on_wait=places.append):
            pass
        self.assertEqual(places, [0])

    def test_gives_up_after_the_longest_wait_and_leaves_the_line(self):
        (p1, p2), clock = gates(capacity=1)
        p1._attempt("holder", "A")
        with self.assertRaises(GateBusy):
            with p2.slot("B", max_wait_s=5):
                pass
        p1._forget("holder")
        self.assertTrue(p1._attempt("someone", "C")[0])    # the quitter is not ahead of them

    def test_a_slot_is_given_back_when_the_call_raises(self):
        (p1, p2), clock = gates(capacity=1)
        with self.assertRaises(ValueError):
            with p1.slot("A"):
                raise ValueError("provider exploded")
        self.assertTrue(p2._attempt("next", "B")[0])


class WithoutRedisTests(SimpleTestCase):
    def test_without_redis_the_gate_limits_this_process(self):
        clock = Clock()
        gate = Gate("test", capacity=1, client_factory=lambda: None, clock=time.monotonic)
        held = threading.Event()
        done = threading.Event()

        def hold():
            with gate.slot("A"):
                held.set()
                done.wait(2)
        t = threading.Thread(target=hold)
        t.start()
        held.wait(2)
        with self.assertRaises(GateBusy):
            with gate.slot("B", max_wait_s=0.1):
                pass
        done.set()
        t.join(2)
        with gate.slot("B", max_wait_s=1):
            pass

    def test_redis_is_not_asked_again_on_every_call_while_it_is_down(self):
        factory = mock.Mock(return_value=None)
        gate = Gate("test", capacity=2, client_factory=factory, clock=time.monotonic)
        for _ in range(5):
            with gate.slot("A"):
                pass
        self.assertEqual(factory.call_count, 1)

    @skipUnless(HAVE_FAKEREDIS, "fakeredis[lua] is not installed")
    def test_a_redis_error_mid_way_falls_back_instead_of_failing_the_call(self):
        broken = mock.Mock()
        broken.register_script.return_value = mock.Mock(side_effect=ConnectionError("gone"))
        gate = Gate("test", capacity=1, client_factory=lambda: broken, clock=time.monotonic)
        with gate.slot("A"):
            pass


class ConfigTests(SimpleTestCase):
    def test_the_image_gate_is_sized_from_the_environment(self):
        with mock.patch.dict("os.environ", {"TOGETHER_IMAGE_GLOBAL_SLOTS": "6",
                                            "TOGETHER_IMAGE_SLOTS_PER_SCHOOL": "2"}):
            gate = provider_gate.together_images()
        self.assertEqual((gate.capacity, gate.per_tenant), (6, 2))
        with mock.patch.dict("os.environ", {"TOGETHER_IMAGE_GLOBAL_SLOTS": "6",
                                            "TOGETHER_IMAGE_SLOTS_PER_SCHOOL": "2"}):
            self.assertIs(provider_gate.together_images(), gate)


class TogetherIntegrationTests(SimpleTestCase):
    """The Together image call takes its slot from the shared gate."""

    def call(self, statuses, gate):
        from ai_services.core import image_generation as ig

        responses = [mock.Mock(ok=(s == 200), status_code=s, headers={}, text="x",
                               json=mock.Mock(return_value={"data": [{"b64_json": "aGk="}]}))
                     for s in statuses]
        cfg = {"model": "m", "width": 64, "height": 64, "url": "u", "timeout": 5, "steps": 0}
        with mock.patch.object(ig, "_together_token", return_value="key"), \
             mock.patch.object(ig.requests, "post", side_effect=responses), \
             mock.patch.object(ig, "_sleep"), \
             mock.patch.object(provider_gate, "together_images", return_value=gate):
            return ig._generate_together(cfg, "a slide", retries=3, tenant="school-a")

    def test_a_429_pauses_every_process_for_the_time_together_asks(self):
        gate = mock.MagicMock()
        result = self.call([429, 200], gate)
        self.assertTrue(result["ok"])
        gate.slot.assert_called_once()
        self.assertEqual(gate.slot.call_args[0][0], "school-a")
        gate.cool_down.assert_called_once_with(2.0)

    def test_a_5xx_is_retried_without_pausing_everyone(self):
        gate = mock.MagicMock()
        self.assertTrue(self.call([503, 200], gate)["ok"])
        gate.cool_down.assert_not_called()

    def test_no_free_slot_is_a_clean_failure_not_a_hang(self):
        gate = mock.MagicMock()
        gate.slot.return_value.__enter__.side_effect = GateBusy("full")
        result = self.call([200], gate)
        self.assertFalse(result["ok"])
        self.assertFalse(result["fatal"])
        self.assertIn("busy", result["error"])

    def test_each_attempt_renews_the_lease(self):
        gate = mock.MagicMock()
        ticket = gate.slot.return_value.__enter__.return_value
        self.call([429, 429, 200], gate)
        self.assertEqual(ticket.renew.call_count, 3)
