import os
import unittest
from unittest.mock import Mock, patch

from ai_services.core.serpapi_images import search_google_images


class SerpApiImageSearchTests(unittest.TestCase):
    @patch("ai_services.core.serpapi_images.requests.get")
    def test_returns_safe_original_image_results(self, request_get):
        response = Mock()
        response.json.return_value = {
            "images_results": [
                {"original": "https://example.com/diagram.png", "title": "Diagram", "unsafe": False},
                {"original": "https://example.com/unsafe.png", "unsafe": True},
            ]
        }
        request_get.return_value = response

        with patch.dict(os.environ, {"SERPAPI_KEY": "test-key"}, clear=False):
            results = search_google_images("polynomial graph", 5, "od")

        self.assertEqual(results[0]["imageUrl"], "https://example.com/diagram.png")
        self.assertEqual(len(results), 1)
        self.assertEqual(request_get.call_args.kwargs["params"]["engine"], "google_images")
        self.assertEqual(request_get.call_args.kwargs["params"]["api_key"], "test-key")
        self.assertEqual(request_get.call_args.kwargs["params"]["hl"], "or")
        self.assertEqual(request_get.call_args.kwargs["params"]["gl"], "in")

    def test_requires_api_key(self):
        with patch.dict(os.environ, {"SERPAPI_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SERPAPI_KEY"):
                search_google_images("polynomial graph")


if __name__ == "__main__":
    unittest.main()
