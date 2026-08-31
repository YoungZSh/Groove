from __future__ import annotations

import unittest

import torch
from PIL import Image

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class FakeMultimodalProcessor:
    """Small processor double that models image-token expansion."""

    image_token = "<image>"
    image_token_id = 99

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        del tokenize, add_generation_prompt, kwargs
        parts = []
        for message in messages:
            for item in message["content"]:
                if item["type"] == "image":
                    parts.append("<image>")
                else:
                    parts.append(item["text"])
        return "".join(parts)

    def __call__(self, *, text, images, videos, return_tensors, truncation, **kwargs):
        del videos, return_tensors, kwargs
        self.calls.append({"truncation": truncation, "text": text[0]})
        image_tokens = sum(max(1, image.width * image.height // 100) for image in images or [])
        text_tokens = max(1, len(text[0].replace("<image>", "")))
        ids = [self.image_token_id] * image_tokens + [1] * text_tokens
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class TeacherPromptTest(unittest.TestCase):
    def test_multimodal_prompt_is_resized_without_truncating_image_tokens(self):
        processor = FakeMultimodalProcessor()
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.processor = processor
        original_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_images[0]},
                    {"type": "text", "text": "Inspect the crop."},
                    {"type": "image", "image": original_images[1]},
                ],
            }
        ]

        raw_prompt, model_inputs = trainer._process_teacher_multimodal_prompt(
            messages,
            original_images,
            {},
            max_prompt_len=60,
        )

        self.assertLessEqual(model_inputs["input_ids"].shape[-1], 60)
        self.assertEqual(raw_prompt.count("<image>"), 2)
        self.assertTrue(processor.calls)
        self.assertTrue(all(call["truncation"] is False for call in processor.calls))
        self.assertEqual([image.size for image in original_images], [(100, 100), (100, 100)])

    def test_image_replacement_removes_resize_metadata(self):
        image = Image.new("RGB", (20, 20))
        replacement = Image.new("RGB", (10, 10))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": "/tmp/original.jpg",
                        "max_pixels": 100,
                        "image": image,
                    }
                ],
            }
        ]

        updated = RayPPOTrainer._replace_teacher_message_images(messages, [replacement])
        item = updated[0]["content"][0]
        self.assertIs(item["image"], replacement)
        self.assertNotIn("path", item)
        self.assertNotIn("max_pixels", item)
        self.assertEqual(messages[0]["content"][0]["image"].size, (20, 20))


if __name__ == "__main__":
    unittest.main()
