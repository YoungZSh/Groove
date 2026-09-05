from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from groove.analyzer import StaticAnalyzer
from groove.evidence import (
    EvidenceBuilderConfig,
    TeacherEvidenceBuilder,
    build_teacher_prompt_from_student,
    teacher_payload,
    validate_visible_focus,
)
from groove.grounding import StaticGrounder, enlarge_crop
from groove.schemas import FocusProgram, GroupRollout, Rollout


def rollout_group(image_path: Path, correctness: list[float]) -> GroupRollout:
    return GroupRollout(
        uid="group-1",
        question="Which marks should be compared?\n(A) one\n(B) two",
        image_path=image_path,
        rollouts=[
            Rollout(
                rollout_id=index,
                completion=f"reasoning {index} FINAL: {'A' if correct else 'B'}",
                predicted_label="A" if correct else "B",
                is_correct=bool(correct),
            )
            for index, correct in enumerate(correctness)
        ],
    )


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_path = self.root / "source.jpg"
        Image.new("RGB", (200, 100), color=(80, 100, 120)).save(self.image_path)
        self.focus = FocusProgram(
            group_summary="The traces confuse two small marks.",
            visible_focus_instruction="Inspect and compare the two small marks.",
            grounding_queries=["left mark", "right mark"],
            context_margin=0.1,
            confidence=0.9,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_teacher_preserves_student_answer_tag_protocol(self):
        student_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a visual question-answering assistant. Analyze the image and answer "
                    "the question. Put only the final answer inside <answer>...</answer> tags."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": object()},
                    {"type": "text", "text": "What color is the door?"},
                ],
            },
        ]

        teacher_prompt = build_teacher_prompt_from_student(
            student_prompt,
            "Inspect the door color.",
            crop_count=1,
        )

        self.assertEqual(teacher_prompt[0], student_prompt[0])
        self.assertIn("<answer>...</answer>", teacher_prompt[0]["content"])
        self.assertNotIn("FINAL:", str(teacher_prompt))
        self.assertEqual(teacher_prompt[1]["content"].count("<image>"), 2)
        self.assertTrue(teacher_prompt[1]["content"].startswith("<image>What color is the door?"))
        self.assertIn("Hindsight visual focus", teacher_prompt[1]["content"])

    def test_multiple_objects_become_independent_crops(self):
        builder = TeacherEvidenceBuilder(
            StaticAnalyzer(self.focus),
            StaticGrounder([(0.05, 0.1, 0.25, 0.4), (0.7, 0.5, 0.9, 0.9)]),
            EvidenceBuilderConfig(output_dir=self.root / "evidence"),
        )
        result = builder.build(rollout_group(self.image_path, [1.0, 0.0]))
        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.crops), 2)
        self.assertNotEqual(result.crops[0].path, result.crops[1].path)
        prompt, images = teacher_payload(result, question="question")
        self.assertEqual(len(images), 3)  # original plus two independent crops
        self.assertEqual(prompt[0]["content"].count("<image>"), 3)
        self.assertNotIn("training-time privileged context", prompt[0]["content"])
        capped_prompt, capped_images = teacher_payload(
            result,
            question="question",
            max_image_pixels=1024,
        )
        self.assertEqual(capped_prompt, prompt)
        self.assertEqual([image["max_pixels"] for image in capped_images], [1024, 1024, 1024])

    def test_tiny_crop_upscale_is_capped(self):
        tiny = Image.new("RGB", (30, 40), color=(80, 100, 120))
        capped = enlarge_crop(tiny, min_short_side=768, max_scale=10.0)
        self.assertEqual(capped.size, (300, 400))

        ordinary = Image.new("RGB", (100, 120), color=(80, 100, 120))
        self.assertEqual(enlarge_crop(ordinary, min_short_side=768).size, (768, 922))

    def test_uniform_group_receives_evidence_by_default(self):
        builder = TeacherEvidenceBuilder(
            StaticAnalyzer(self.focus),
            StaticGrounder([(0.1, 0.1, 0.5, 0.5)]),
            EvidenceBuilderConfig(output_dir=self.root / "uniform"),
        )
        result = builder.build(rollout_group(self.image_path, [0.0, 0.0]))
        self.assertEqual(result.status, "ready")
        prompt, images = teacher_payload(result, question="question")
        self.assertEqual(len(images), 2)
        self.assertEqual(prompt[0]["content"].count("<image>"), 2)

    def test_mixed_only_compatibility_mode_skips_uniform_group(self):
        builder = TeacherEvidenceBuilder(
            StaticAnalyzer(self.focus),
            StaticGrounder([(0.1, 0.1, 0.5, 0.5)]),
            EvidenceBuilderConfig(
                output_dir=self.root / "mixed-only",
                mixed_groups_only=True,
            ),
        )
        result = builder.build(rollout_group(self.image_path, [1.0, 1.0]))
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "uniform_reward_group")

    def test_visible_answer_leak_is_rejected(self):
        group = rollout_group(self.image_path, [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "answer conclusion"):
            validate_visible_focus(group, "The answer is one mark.")
        with self.assertRaisesRegex(ValueError, "option letter"):
            validate_visible_focus(group, "The correct region supports option A.")
        validate_visible_focus(group, "Compare the one mark with the two mark.")

    def test_builder_sanitizes_answer_leak_without_discarding_crops(self):
        leaking_focus = self.focus.model_copy(
            update={"visible_focus_instruction": "The answer is one mark."}
        )
        builder = TeacherEvidenceBuilder(
            StaticAnalyzer(leaking_focus),
            StaticGrounder([(0.1, 0.1, 0.5, 0.5)]),
            EvidenceBuilderConfig(output_dir=self.root / "sanitized"),
        )

        result = builder.build(rollout_group(self.image_path, [1.0, 0.0]))

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.crops), 1)
        self.assertNotIn("the answer is", result.focus.visible_focus_instruction.lower())
        self.assertIn("zoomed visual evidence", result.focus.visible_focus_instruction.lower())
        validate_visible_focus(
            rollout_group(self.image_path, [1.0, 0.0]),
            result.focus.visible_focus_instruction,
        )

    def test_dot_style_choice_text_leak_is_rejected(self):
        group = rollout_group(self.image_path, [1.0, 0.0]).model_copy(
            update={"question": "Which mark is visible?\nA. alpha mark\nB. beta mark"}
        )
        validate_visible_focus(group, "Inspect the alpha mark closely.")
        with self.assertRaisesRegex(ValueError, "answer conclusion"):
            validate_visible_focus(group, "The answer is alpha mark.")

    def test_comparative_choice_descriptors_are_preserved(self):
        group = rollout_group(self.image_path, [1.0, 0.0]).model_copy(
            update={"question": "Which color is visible?\nA. gold\nB. bronze"}
        )
        validate_visible_focus(
            group,
            "Inspect the frame and compare whether its tone is gold or bronze.",
        )


if __name__ == "__main__":
    unittest.main()
