from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

from groove.verl_trainer import GrooveRayPPOTrainer
from verl import DataProto


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
    def test_missing_evidence_skips_teacher_prompt_construction(self):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = SimpleNamespace(groove={})
        images = np.empty(2, dtype=object)
        images[:] = [[], []]
        batch = DataProto.from_dict(
            tensors={"responses": torch.ones(2, 1, dtype=torch.long)},
            non_tensors={"groove_teacher_images": images},
        )
        teacher, mask, metrics = trainer._build_groove_teacher_batch(batch)
        self.assertIsNone(teacher)
        self.assertEqual(mask.tolist(), [0.0, 0.0])
        self.assertEqual(metrics["groove/teacher_prefix_cache_entries"], 0)

    def test_multimodal_prompt_is_resized_without_truncating_image_tokens(self):
        processor = FakeMultimodalProcessor()
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
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

        updated = GrooveRayPPOTrainer._replace_teacher_message_images(messages, [replacement])
        item = updated[0]["content"][0]
        self.assertIs(item["image"], replacement)
        self.assertNotIn("path", item)
        self.assertNotIn("max_pixels", item)
        self.assertEqual(messages[0]["content"][0]["image"].size, (20, 20))

    def test_teacher_multimodal_prefix_is_processed_once_per_uid_group(self):
        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        processor = FakeMultimodalProcessor()
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.processor = processor
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        groove = AttrDict(
            enabled=True,
            teacher_image_key="teacher_images",
            max_reprompt_len=64,
        )
        trainer.config = SimpleNamespace(
            data=SimpleNamespace(apply_chat_template_kwargs={}, max_prompt_length=64),
            groove=groove,
        )

        image = Image.new("RGB", (10, 10))
        raw_prompt = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        teacher_prompt = [{"role": "user", "content": "<image>\nInspect the image."}]
        raw_prompts = np.empty(2, dtype=object)
        raw_prompts[:] = [raw_prompt, raw_prompt]
        teacher_prompts = np.empty(2, dtype=object)
        teacher_prompts[:] = [teacher_prompt, teacher_prompt]
        teacher_images = np.empty(2, dtype=object)
        teacher_images[:] = [[{"image": image}], [{"image": image}]]

        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.zeros((2, 2), dtype=torch.long),
                "attention_mask": torch.ones((2, 2), dtype=torch.long),
                "responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "response_mask": torch.ones((2, 2), dtype=torch.long),
            },
            non_tensors={
                "uid": np.array(["same-group", "same-group"], dtype=object),
                "raw_prompt": raw_prompts,
                "teacher_prompt": teacher_prompts,
                "teacher_images": teacher_images,
            },
        )

        teacher_batch, evidence_mask, _metrics = trainer._build_groove_teacher_batch(batch)
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(evidence_mask.tolist(), [1.0, 1.0])
        self.assertGreater(teacher_batch.batch["prompts"].shape[-1], 0)
        self.assertEqual(
            teacher_batch.batch["input_ids"][:, -2:].tolist(),
            [[1, 2], [3, 4]],
        )


if __name__ == "__main__":
    unittest.main()
