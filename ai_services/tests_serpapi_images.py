import os
import unittest
from unittest.mock import Mock, patch

from ai_services.core.serpapi_images import search_google_images


class SerpApiImageSearchTests(unittest.TestCase):
    @patch("ai_services.core.serpapi_images.requests.post")
    def test_returns_safe_original_image_results(self, request_post):
        response = Mock()
        response.json.return_value = {
            "images": [
                {"imageUrl": "https://example.com/diagram.png", "title": "Diagram", "link": "https://example.com/source"},
            ]
        }
        request_post.return_value = response

        with patch.dict(os.environ, {"SERPER_API_KEY": "test-key"}, clear=False):
            results = search_google_images("polynomial graph", 5, "od")

        self.assertEqual(results[0]["imageUrl"], "https://example.com/diagram.png")
        self.assertEqual(len(results), 1)
        self.assertEqual(request_post.call_args.kwargs["json"]["q"], "polynomial graph")
        self.assertEqual(request_post.call_args.kwargs["headers"]["X-API-KEY"], "test-key")
        self.assertEqual(request_post.call_args.kwargs["json"]["hl"], "or")
        self.assertEqual(request_post.call_args.kwargs["json"]["gl"], "in")

    @patch("ai_services.core.serpapi_images._search_wikipedia_images", return_value=[])
    def test_no_key_falls_back_to_wikipedia_not_raises(self, mock_wiki):
        """No Serper key → graceful Wikipedia fallback, no RuntimeError raised."""
        with patch.dict(
            os.environ,
            {"SERPER_API_KEY": "", "SERPER_KEY": "", "SERPAPI_KEY": ""},
            clear=False,
        ):
            results = search_google_images("polynomial graph")
        mock_wiki.assert_called_once()
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
