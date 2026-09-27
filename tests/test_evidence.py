from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

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
from groove.schemas import EvidenceImageConfig, FocusProgram, GroupRollout, Rollout, TeacherEvidence, ToolRegion


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

    def test_focus_replaces_extra_crops_only_and_keeps_selected_regions_and_trace(self):
        regions = [ToolRegion(query="left mark", expanded_box=(10, 10, 50, 40), score=1,
                              source="gemini_native_bbox"),
                   ToolRegion(query="right mark", expanded_box=(140, 50, 180, 90), score=1,
                              source="gemini_native_bbox")]
        analyzer = StaticAnalyzer(self.focus.model_copy(update={"tool_regions": regions}))
        analyzer.last_tool_trace = [{"candidate_id": "selected", "result": {"bbox": [10, 10, 50, 40]}},
                                    {"candidate_id": "unselected", "result": {"bbox": [0, 0, 200, 100]}}]
        grounder = Mock()
        builder = TeacherEvidenceBuilder(analyzer, grounder, EvidenceBuilderConfig(
            output_dir=self.root / "focus", image_config=EvidenceImageConfig(mode="focus")))
        student = [{"role": "system", "content": "Use <answer>...</answer>."},
                   {"role": "user", "content": [{"type": "image", "image": "original"},
                                                {"type": "text", "text": "Compare marks."}]}]
        before = deepcopy(student)
        result = builder.build(rollout_group(self.image_path, [1, 0]), student_prompt=student)
        self.assertEqual(result.status, "ready", result.reason)
        self.assertEqual(student, before)
        grounder.crop_objects.assert_not_called()
        self.assertEqual(result.focus_image.boxes, [r.expanded_box for r in regions])
        self.assertEqual(result.tool_trace, analyzer.last_tool_trace)
        self.assertEqual(len(result.crops), 2)  # Retained for audit only.
        prompt, images = teacher_payload(result, question="unused", max_image_pixels=1024)
        self.assertEqual([i["path"] for i in images], [str(self.image_path), str(result.focus_image.path)])
        self.assertEqual([i["max_pixels"] for i in images], [1024, 1024])
        self.assertEqual(prompt[0], student[0])
        self.assertEqual(prompt[1]["content"].count("<image>"), 2)
        self.assertIn("Full-image visual focus", prompt[1]["content"])
        self.assertIn("red outlines", prompt[1]["content"])
        self.assertNotIn("Zoomed visual evidence", str(prompt))
        self.assertNotIn("unselected", str(prompt))
        record = next((self.root / "focus").rglob("evidence.json"))
        self.assertEqual(TeacherEvidence.model_validate_json(record.read_text()), result)

    def test_focus_cache_isolated_by_mode_and_parameters_and_rebases_prompt(self):
        group = rollout_group(self.image_path, [1, 0])
        output = self.root / "switch"
        grounder = StaticGrounder([(0.1, 0.1, 0.5, 0.5)])
        analyzer = Mock(wraps=StaticAnalyzer(self.focus))
        analyzer.last_tool_trace = []
        def builder(config):
            return TeacherEvidenceBuilder(analyzer, grounder, EvidenceBuilderConfig(output_dir=output, image_config=config))
        crop = builder(EvidenceImageConfig()).build(group)
        old_files = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
        focused_builder = builder(EvidenceImageConfig(mode="focus"))
        focused = focused_builder.build(group)
        stronger = builder(EvidenceImageConfig(mode="focus", blur_alpha=0.75)).build(group)
        wider = builder(EvidenceImageConfig(mode="focus", blur_radius=20)).build(group)
        self.assertEqual(len({focused.focus_image.path, stronger.focus_image.path, wider.focus_image.path}), 3)
        for path, contents in old_files.items():
            self.assertEqual(path.read_bytes(), contents)
        analyzer.analyze.reset_mock()
        self.assertEqual(builder(EvidenceImageConfig()).build(group), crop)
        student = [{"role": "user", "content": "<image>New question protocol."}]
        cached = focused_builder.build(group, student_prompt=student)
        analyzer.analyze.assert_not_called()
        self.assertEqual(cached.focus_image, focused.focus_image)
        self.assertEqual(cached.teacher_prompt[0]["content"].count("<image>"), 2)
        self.assertIn("New question protocol", cached.teacher_prompt[0]["content"])
        self.assertIn("Full-image visual focus", cached.teacher_prompt[0]["content"])

    def test_focus_failure_falls_back_without_sending_partial_images(self):
        builder = TeacherEvidenceBuilder(StaticAnalyzer(self.focus), StaticGrounder([]),
            EvidenceBuilderConfig(output_dir=self.root / "failed", image_config=EvidenceImageConfig(mode="focus")))
        result = builder.build(rollout_group(self.image_path, [1, 0]))
        self.assertEqual(result.status, "error")
        self.assertEqual(teacher_payload(result, question="question")[1], [])

    def test_historical_evidence_without_rendering_fields_still_uses_crops(self):
        builder = TeacherEvidenceBuilder(StaticAnalyzer(self.focus), StaticGrounder([(0.1, 0.1, 0.5, 0.5)]),
                                         EvidenceBuilderConfig(output_dir=self.root / "legacy"))
        result = builder.build(rollout_group(self.image_path, [1, 0]))
        record = result.model_dump(mode="json")
        record.pop("image_config")
        record.pop("focus_image")
        legacy = TeacherEvidence.model_validate(record)
        self.assertEqual(legacy.image_config.mode, "crop")
        self.assertEqual(teacher_payload(legacy, question="question"), teacher_payload(result, question="question"))

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
