"""Finished decks are kept and served at once for an identical request.

Schools on one board teach the same chapters from the same books; the second
teacher asking for exactly the same deck should not wait two minutes - or
spend the provider's limits - for work already done. But a deck is only ever
reused for the same request made from the same text.
"""
import json
import os
import tempfile
from unittest import mock, skipUnless

from django.test import SimpleTestCase
from rest_framework.response import Response

from ai_services.core.ppt_v2 import deck_cache

try:
    import fakeredis
    HAVE_FAKEREDIS = True
except ImportError:
    HAVE_FAKEREDIS = False

BOOK = [{"page_no": 100, "content": "Respiration releases energy from food."},
        {"page_no": 101, "content": "Glucose is broken down into pyruvate."}]
REQUEST = {"topic": "Respiration", "className": "Class 10", "subjectName": "Science",
           "chapterName": "Life Processes", "topicName": "Respiration", "slideCount": 8,
           "language": "English", "pptVersion": "image", "deckTheme": "ocean-blue",
           "sourcePassages": BOOK}


def deck(**gen):
    return {"title": "Respiration",
            "slides": [{"title": "S%d" % i, "slideImage": {"url": "http://ai/generated-note-images/s%d.png" % i}}
                       for i in range(1, 4)],
            "generation": dict({"version": "image", "failedSlides": []}, **gen)}


class KeyTests(SimpleTestCase):
    def key(self, **changes):
        return deck_cache.key_for(dict(REQUEST, **changes), board="CBSE", vertical="school")

    def test_the_same_request_gets_the_same_key_however_it_is_spelt(self):
        self.assertEqual(self.key(), self.key(chapterName="  life   PROCESSES ", slideCount="8"))

    def test_any_difference_that_changes_the_deck_changes_the_key(self):
        base = self.key()
        for change in ({"topicName": "Nutrition"}, {"language": "Hindi"}, {"pptVersion": "v2"},
                       {"className": "Class 9"}, {"topic": "Respiration in plants"}):
            self.assertNotEqual(self.key(**change), base, change)
        self.assertNotEqual(deck_cache.key_for(REQUEST, board="ICSE", vertical="school"), base)

    def test_by_default_one_deck_serves_a_topic_whatever_count_or_theme(self):
        # The school demos (2026-10-09): a deck already made for the topic is shown
        # at once, rather than made again because the count or theme differs.
        base = self.key()
        for change in ({"slideCount": 10}, {"slideCount": "auto"}, {"deckTheme": "clean-white"}):
            self.assertEqual(self.key(**change), base, change)

    def test_exact_matching_can_be_turned_back_on(self):
        with mock.patch.dict("os.environ", {"PPT_DECK_CACHE_MATCH": "exact"}):
            base = self.key()
            for change in ({"slideCount": 10}, {"slideCount": "auto"}, {"deckTheme": "clean-white"}):
                self.assertNotEqual(self.key(**change), base, change)
            self.assertEqual(self.key(slideCount="8"), base)
        self.assertNotEqual(self.key(), base)          # the two modes never share a key

    def test_a_different_book_never_shares_a_deck(self):
        other = [dict(BOOK[0], content="Respiration releases energy."), BOOK[1]]
        self.assertNotEqual(self.key(sourcePassages=other), self.key())
        self.assertNotEqual(self.key(sourcePassages=[]), self.key())
        # The same pages in another order are the same book.
        self.assertEqual(self.key(sourcePassages=list(reversed(BOOK))), self.key())

    def test_decks_can_be_kept_within_one_school_only(self):
        with mock.patch.dict("os.environ", {"PPT_DECK_CACHE_SCOPE": "institute"}):
            a = deck_cache.key_for(REQUEST, board="CBSE", institute_id="school-a")
            b = deck_cache.key_for(REQUEST, board="CBSE", institute_id="school-b")
        self.assertNotEqual(a, b)
        self.assertEqual(deck_cache.key_for(REQUEST, board="CBSE", institute_id="school-a"),
                         deck_cache.key_for(REQUEST, board="CBSE", institute_id="school-b"))


class CompleteTests(SimpleTestCase):
    def test_only_whole_decks_in_the_asked_for_style_are_kept(self):
        self.assertTrue(deck_cache.complete(deck(), "image"))
        self.assertFalse(deck_cache.complete(deck(failedSlides=[2]), "image"))
        self.assertFalse(deck_cache.complete(deck(fallback=True), "image"))
        self.assertFalse(deck_cache.complete(deck(version="v1"), "image"))   # a stand-in deck
        self.assertFalse(deck_cache.complete(deck(cached=True), "image"))    # never re-keep a copy
        unpainted = deck()
        del unpainted["slides"][1]["slideImage"]
        self.assertFalse(deck_cache.complete(unpainted, "image"))
        self.assertFalse(deck_cache.complete({"slides": []}, ""))
        self.assertFalse(deck_cache.complete(None, ""))


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class StoreTests(SimpleTestCase):
    def setUp(self):
        self.redis = fakeredis.FakeRedis(decode_responses=True)
        self.images = tempfile.mkdtemp()
        for i in range(1, 4):
            open(os.path.join(self.images, "s%d.png" % i), "wb").close()
        p1 = mock.patch.object(deck_cache, "_client", return_value=self.redis)
        p2 = mock.patch.object(deck_cache, "_images_dir",
                               return_value=__import__("pathlib").Path(self.images))
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_a_kept_deck_comes_back_marked_as_reused(self):
        self.assertTrue(deck_cache.put("k", deck(), "image"))
        got = deck_cache.get("k")
        self.assertEqual(got["title"], "Respiration")
        self.assertTrue(got["generation"]["cached"])
        self.assertIsNotNone(got["generation"]["cachedAt"])
        self.assertGreater(self.redis.ttl("k"), 29 * 86400)

    def test_a_kept_deck_longer_than_the_cap_is_dropped_not_served(self):
        # Decks went from at most 15 slides to at most 10 (2026-10-09); a deck
        # matched by topic must never bring back a longer one.
        long_deck = deck()
        long_deck["slides"] = long_deck["slides"] * 5          # 15 slides
        self.assertTrue(deck_cache.put("k", long_deck, ""))
        self.assertIsNone(deck_cache.get("k", max_slides=10))
        self.assertFalse(self.redis.exists("k"))

    def test_an_incomplete_deck_is_not_kept(self):
        self.assertFalse(deck_cache.put("k", deck(failedSlides=[1]), "image"))
        self.assertIsNone(deck_cache.get("k"))

    def test_a_deck_whose_images_are_gone_is_dropped_not_served_broken(self):
        deck_cache.put("k", deck(), "image")
        os.remove(os.path.join(self.images, "s2.png"))
        self.assertIsNone(deck_cache.get("k"))
        self.assertFalse(self.redis.exists("k"))

    def test_an_unreadable_entry_is_dropped(self):
        self.redis.set("k", "{not json")
        self.assertIsNone(deck_cache.get("k"))
        self.assertFalse(self.redis.exists("k"))

    def test_the_cache_can_be_switched_off(self):
        with mock.patch.dict("os.environ", {"PPT_DECK_CACHE": "false"}):
            self.assertFalse(deck_cache.put("k", deck(), "image"))
            self.assertIsNone(deck_cache.get("k"))

    def test_without_redis_nothing_is_kept_and_nothing_fails(self):
        with mock.patch.object(deck_cache, "_client", return_value=None):
            self.assertFalse(deck_cache.put("k", deck(), "image"))
            self.assertIsNone(deck_cache.get("k"))


@skipUnless(HAVE_FAKEREDIS, "fakeredis is not installed")
class JobPathTests(SimpleTestCase):
    """The job endpoint answers from the cache, and keeps what it makes."""

    def setUp(self):
        self.redis = fakeredis.FakeRedis(decode_responses=True)
        p1 = mock.patch.object(deck_cache, "_client", return_value=self.redis)
        p2 = mock.patch.object(deck_cache, "_images_present", return_value=True)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def snapshot(self, **changes):
        snap = mock.Mock()
        snap.data = dict(REQUEST, **changes)
        snap.headers = {"X-Board": "CBSE", "X-Vertical": "school"}
        snap.board, snap.vertical, snap.institute_id, snap.ppt_version = None, None, "school-a", "image"
        return snap

    def run_job(self, snap, generated=None):
        from ai_services.views import ppt

        progress = mock.Mock(fallback_reason=None)
        made = Response({"success": True, "data": generated or deck()})
        with mock.patch.object(ppt, "_generate_presentation", return_value=made) as gen:
            response = ppt._cached_or_generated(snap, progress)
        return response, gen, progress

    def test_the_second_identical_request_is_answered_without_generating(self):
        first, gen1, _ = self.run_job(self.snapshot())
        self.assertEqual(gen1.call_count, 1)
        second, gen2, progress = self.run_job(self.snapshot())
        self.assertEqual(gen2.call_count, 0)
        self.assertTrue(second.data["data"]["generation"]["cached"])
        keys = [c.args[0] for c in progress.event.call_args_list]
        self.assertIn("cache", keys)

    def test_make_a_fresh_one_generates_again_and_replaces_the_kept_deck(self):
        self.run_job(self.snapshot())
        fresh = deck()
        fresh["title"] = "Respiration, again"
        response, gen, _ = self.run_job(self.snapshot(fresh=True), generated=fresh)
        self.assertEqual(gen.call_count, 1)
        again, gen3, _ = self.run_job(self.snapshot())
        self.assertEqual(gen3.call_count, 0)
        self.assertEqual(again.data["data"]["title"], "Respiration, again")

    def test_a_failed_or_fallen_back_deck_is_never_kept(self):
        self.run_job(self.snapshot(), generated=deck(fallback=True))
        _, gen, _ = self.run_job(self.snapshot())
        self.assertEqual(gen.call_count, 1)

    def test_a_different_request_is_generated(self):
        self.run_job(self.snapshot())
        _, gen, _ = self.run_job(self.snapshot(topicName="Nutrition"))
        self.assertEqual(gen.call_count, 1)


class RedisOutageTests(SimpleTestCase):
    def test_a_redis_outage_costs_one_connection_attempt_not_one_per_deck(self):
        deck_cache._down_until = 0.0
        with mock.patch("ai_services.core.cache.get_redis", return_value=None) as get_redis:
            for _ in range(5):
                self.assertIsNone(deck_cache.get("k"))
                deck_cache.put("k", deck(), "image")
        self.assertEqual(get_redis.call_count, 1)
        deck_cache._down_until = 0.0
