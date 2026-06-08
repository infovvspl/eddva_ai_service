import os
import tempfile
import unittest
from unittest.mock import patch


class FakeReader:
    def readtext(self, _path):
        return [
            (
                [(10, 10), (40, 10), (40, 30), (10, 30)],
                "HF label",
                0.95,
            )
        ]


class ImageGenerationTests(unittest.TestCase):
    def test_strips_detected_embedded_text_region(self):
        try:
            import cv2
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy unavailable")

        from ai_services.core.image_generation import _strip_embedded_text_from_image

        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[10:30, 10:40] = (0, 0, 0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            cv2.imwrite(path, image)
            with patch.dict(
                os.environ,
                {
                    "NOTES_IMAGE_STRIP_EMBEDDED_TEXT": "true",
                    "NOTES_IMAGE_TEXT_STRIP_MIN_CONFIDENCE": "0.35",
                    "NOTES_IMAGE_TEXT_STRIP_PADDING": "0",
                },
                clear=False,
            ), patch("ai_services.core.image_generation._get_ocr_reader", return_value=FakeReader()):
                result = _strip_embedded_text_from_image(path)

            cleaned = cv2.imread(path)
            self.assertEqual(result["removed"], 1)
            self.assertTrue((cleaned[10:30, 10:40] == 255).all())
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
