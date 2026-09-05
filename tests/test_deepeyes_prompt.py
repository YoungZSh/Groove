from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jinja2 import Environment
from omegaconf import OmegaConf

from groove.deepeyes_dataset import DeepEyesReasoningDataset
from groove.deepeyes_prompt import (
    REASONING_SYSTEM_PROMPT,
    configure_deepeyes_response,
    without_empty_think_prefill,
)
from groove.deepeyes_reward import extract_answer
from groove.evidence import build_teacher_prompt_from_student, student_prompt_template


class DeepEyesReasoningPromptTest(unittest.TestCase):
    def make_dataset(self):
        dataset = object.__new__(DeepEyesReasoningDataset)
        dataset.prompt_key = "prompt"
        dataset.image_key = "images"
        dataset.video_key = "videos"
        dataset.processor = object()
        dataset.image_max_pixels = None
        return dataset

    def test_dataset_changes_only_instruction_and_preserves_question_and_image(self):
        example = {
            "prompt": [
                {"role": "system", "content": "Original instruction"},
                {"role": "user", "content": "<image>What color is the umbrella?"},
            ],
            "images": [{"path": "/tmp/original.jpg"}],
            "reward_model": {"ground_truth": "PRIVATE_REFERENCE_SENTINEL"},
            "extra_info": {"image_path": "/tmp/analyzer-only-path.jpg",
                           "answer": "PRIVATE_REFERENCE_SENTINEL", "source_index": 19},
        }
        original = deepcopy(example)
        messages = self.make_dataset()._build_messages(example)
        self.assertEqual(example, original)
        self.assertEqual(messages[0], {"role": "system", "content": [
            {"type": "text", "text": REASONING_SYSTEM_PROMPT},
        ]})
        self.assertEqual(messages[1]["content"], [
            {"type": "image", "path": "/tmp/original.jpg", "image": "/tmp/original.jpg"},
            {"type": "text", "text": "What color is the umbrella?"},
        ])
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", str(messages))
        self.assertNotIn("analyzer-only-path", str(messages))
        self.assertNotIn("<think>", REASONING_SYSTEM_PROMPT)
        teacher = build_teacher_prompt_from_student(
            student_prompt_template(messages), "Inspect the umbrella surface.", 1
        )
        self.assertEqual(teacher[0]["content"], REASONING_SYSTEM_PROMPT)

    def test_dataset_adds_instruction_when_system_message_is_missing(self):
        example = {"prompt": [{"role": "user", "content": "What color?"}], "images": []}
        messages = self.make_dataset()._build_messages(example)
        self.assertEqual(messages[0]["content"], REASONING_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "What color?")
        self.assertEqual(len(example["prompt"]), 1)

    def test_template_keeps_messages_and_generates_plain_assistant_prefix(self):
        template = (
            "{{ messages[0].content }}{{ messages[1].content }}"
            "{% if add_generation_prompt %}"
            "{{- '<|im_start|>assistant\\n' }}"
            "{% if enable_thinking %}{{- '<think>\\n' }}"
            "{% else %}{{- '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
        )
        updated = without_empty_think_prefill(template)
        rendered = Environment().from_string(updated).render(
            messages=[{"content": REASONING_SYSTEM_PROMPT}, {"content": "Question"}],
            add_generation_prompt=True, enable_thinking=False,
        )
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", rendered)
        self.assertIn(REASONING_SYSTEM_PROMPT, rendered)
        self.assertEqual(extract_answer("Visual reasoning.\n<answer>green</answer>"), ("green", True))

    def test_unrecognized_template_fails_instead_of_changing_unrelated_text(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            without_empty_think_prefill("unrelated model template")

    def test_opt_in_configuration_resolves_shared_template_and_dataset(self):
        with TemporaryDirectory() as folder:
            native = "prefix {{- '<think>\\n\\n</think>\\n\\n' }}"
            (Path(folder) / "chat_template.jinja").write_text(native)
            config = OmegaConf.create({
                "data": {"response_format": "reasoning_answer",
                         "apply_chat_template_kwargs": {"enable_thinking": False},
                         "custom_cls": {"path": None, "name": None}},
                "actor_rollout_ref": {"model": {"path": folder, "custom_chat_template": None}},
            })
            configure_deepeyes_response(config)
            self.assertEqual(config.data.custom_cls.name, "DeepEyesReasoningDataset")
            self.assertTrue(Path(config.data.custom_cls.path).is_file())
            self.assertEqual(config.actor_rollout_ref.model.custom_chat_template, "prefix {{- '' }}")
            self.assertEqual((Path(folder) / "chat_template.jinja").read_text(), native)

    def test_original_configuration_is_untouched(self):
        config = OmegaConf.create({"data": {"response_format": "original"}})
        before = OmegaConf.to_container(config)
        configure_deepeyes_response(config)
        self.assertEqual(OmegaConf.to_container(config), before)

    def test_native_thinking_cannot_conflict_with_plain_reasoning_mode(self):
        config = OmegaConf.create({"data": {"response_format": "reasoning_answer",
                                             "apply_chat_template_kwargs": {"enable_thinking": True}}})
        with self.assertRaisesRegex(ValueError, "enable_thinking=false"):
            configure_deepeyes_response(config)


if __name__ == "__main__":
    unittest.main()
