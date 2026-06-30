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

    def test_requires_api_key(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "", "SERPER_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SERPER_API_KEY or SERPER_KEY"):
                search_google_images("polynomial graph")


if __name__ == "__main__":
    unittest.main()
