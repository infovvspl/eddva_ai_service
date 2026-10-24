"""/learning/topic-video: the one best video for a topic, class and board. Search mocked."""

from unittest.mock import patch
from django.test import TestCase

from ai_services.models import Institute
from ai_services.middleware import invalidate_institute_cache
from ai_services.views.learning import best_video


def v(video_id, title, channel="Some Channel", duration="8:00"):
    return {"videoId": video_id, "title": title, "channel": channel, "duration": duration,
            "url": f"https://www.youtube.com/watch?v={video_id}", "thumbnailUrl": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"}


class BestVideoTests(TestCase):
    TOPIC = "Chemical Properties of Acids and Bases"

    def test_prefers_the_board_and_class_among_videos_on_the_topic(self):
        videos = [
            v("aaaaaaaaaaa", "Chemical properties of acids and bases | ICSE class 7"),
            v("bbbbbbbbbbb", "Chemical properties of acids and bases class 10 CBSE"),
            v("ccccccccccc", "Chemical properties of acids and bases NCERT class 9"),
        ]
        self.assertEqual(best_video(videos, self.TOPIC, "Class 10", "CBSE")["videoId"], "bbbbbbbbbbb")
        self.assertEqual(best_video(videos, self.TOPIC, "Class 7", "ICSE")["videoId"], "aaaaaaaaaaa")

    def test_ncert_counts_as_cbse(self):
        videos = [v("aaaaaaaaaaa", "Chemical properties of acids and bases class 10"),
                  v("bbbbbbbbbbb", "Chemical properties of acids and bases NCERT class 10")]
        self.assertEqual(best_video(videos, self.TOPIC, "10", "CBSE")["videoId"], "bbbbbbbbbbb")

    def test_covering_the_topic_beats_matching_the_board(self):
        videos = [v("aaaaaaaaaaa", "Acids CBSE class 10"),
                  v("bbbbbbbbbbb", "Chemical properties of acids and bases explained")]
        self.assertEqual(best_video(videos, self.TOPIC, "10", "CBSE")["videoId"], "bbbbbbbbbbb")

    def test_class_1_does_not_match_class_10(self):
        videos = [v("aaaaaaaaaaa", "Chemical properties of acids and bases class 10"),
                  v("bbbbbbbbbbb", "Chemical properties of acids and bases class 1")]
        self.assertEqual(best_video(videos, self.TOPIC, "Class 1", "")["videoId"], "bbbbbbbbbbb")

    def test_skips_off_topic_and_over_an_hour_and_prefers_short_education_channels(self):
        videos = [
            v("aaaaaaaaaaa", "Funny cat compilation"),
            v("bbbbbbbbbbb", "Chemical properties of acids and bases full chapter", duration="1:30:00"),
            v("ccccccccccc", "Chemical properties of acids and bases", duration="45:00"),
            v("ddddddddddd", "Chemical properties of acids and bases", channel="Magnet Brains", duration="12:00"),
        ]
        self.assertEqual(best_video(videos, self.TOPIC)["videoId"], "ddddddddddd")

    def test_none_when_nothing_fits(self):
        self.assertIsNone(best_video([v("aaaaaaaaaaa", "Funny cat compilation")], self.TOPIC))
        self.assertIsNone(best_video([], self.TOPIC))


class TopicVideoViewTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, data, results):
        calls = []

        def fake_search(query, limit=3):
            calls.append(query)
            return results.pop(0) if results else []

        with patch("ai_services.views.learning.search_videos", side_effect=fake_search):
            resp = self.client.post("/learning/topic-video", data=data, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, calls

    def test_returns_the_one_best_video_searched_for_topic_class_and_board(self):
        found = [v("aaaaaaaaaaa", "Chemical properties of acids and bases"),
                 v("bbbbbbbbbbb", "Chemical properties of acids and bases class 10 CBSE")]
        resp, calls = self._post({"topic": "Chemical Properties of Acids and Bases", "chapter": "Acids, Bases and Salts",
                                  "subject": "Science", "class_name": "Class 10", "board": "CBSE"}, [found])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["video"]["videoId"], "bbbbbbbbbbb")
        self.assertIn("class 10", calls[0])
        self.assertIn("CBSE", calls[0])
        self.assertIn("Acids, Bases and Salts", calls[0])

    def test_searches_again_without_the_board_when_nothing_is_found(self):
        resp, calls = self._post({"topic": "Chemical Properties of Acids and Bases", "class_name": "10", "board": "Odisha"},
                                 [[], [v("aaaaaaaaaaa", "Chemical properties of acids and bases class 10")]])
        self.assertEqual(resp.json()["video"]["videoId"], "aaaaaaaaaaa")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("Odisha", calls[1])

    def test_no_video_and_bad_request(self):
        resp, _ = self._post({"topic": "Chemical Properties of Acids and Bases"}, [[]])
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["video"])
        resp, _ = self._post({"topic": "  "}, [])
        self.assertEqual(resp.status_code, 400)
