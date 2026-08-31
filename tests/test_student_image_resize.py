from __future__ import annotations

import unittest

from verl.utils.dataset.rl_dataset import RLHFDataset


class StudentImageResizeTest(unittest.TestCase):
    def test_max_pixels_is_attached_to_student_image_message(self):
        dataset = object.__new__(RLHFDataset)
        dataset.prompt_key = "prompt"
        dataset.image_key = "images"
        dataset.video_key = "videos"
        dataset.processor = object()
        dataset.image_max_pixels = 4_194_304

        example = {
            "prompt": [{"role": "user", "content": "<image>\nquestion"}],
            "images": [{"path": "/tmp/example.jpg"}],
        }

        messages = dataset._build_messages(example)
        image_item = messages[0]["content"][0]

        self.assertEqual(image_item["type"], "image")
        self.assertEqual(image_item["max_pixels"], 4_194_304)
        self.assertEqual(image_item["image"], "/tmp/example.jpg")


if __name__ == "__main__":
    unittest.main()
