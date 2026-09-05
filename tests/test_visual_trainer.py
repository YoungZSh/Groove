from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from groove.verl_trainer import GrooveRayPPOTrainer


class GrooveTrainerTest(unittest.TestCase):
    def test_vanilla_grpo_skips_online_teacher_construction(self):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {"groove": {"enabled": False}}
        trainer._build_online_teacher_columns = lambda *_args: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not build online OPSD evidence")
        )

        batch = object()
        result, metrics = trainer._postprocess_advantages(batch, None, None)

        self.assertIs(result, batch)
        self.assertEqual(metrics, {})

    def test_remote_evidence_builds_groups_concurrently(self):
        class FakeBatch:
            def __init__(self):
                self.batch = {
                    "responses": torch.ones((8, 2), dtype=torch.long),
                    "response_mask": torch.ones((8, 2), dtype=torch.long),
                }
                self.non_tensor_batch = {
                    "uid": np.array(
                        ["group-a"] * 4 + ["group-b"] * 4,
                        dtype=object,
                    ),
                    "extra_info": np.array(
                        [
                            {"question": "What is shown?", "image_path": "/tmp/a.jpg"}
                        ]
                        * 4
                        + [
                            {"question": "What is shown?", "image_path": "/tmp/b.jpg"}
                        ]
                        * 4,
                        dtype=object,
                    ),
                    "raw_prompt": np.array(
                        [
                            [
                                {"role": "system", "content": "Use <answer> tags."},
                                {"role": "user", "content": "<image>What is shown?"},
                            ]
                        ]
                        * 8,
                        dtype=object,
                    ),
                }

        class FakeBuilder:
            def __init__(self, tracker):
                self.tracker = tracker

            def build(self, group, *, student_prompt=None):
                if student_prompt is None:
                    raise AssertionError("Student prompt must be forwarded to the Teacher builder")
                with self.tracker["lock"]:
                    self.tracker["active"] += 1
                    self.tracker["max_active"] = max(
                        self.tracker["max_active"], self.tracker["active"]
                    )
                    self.tracker["groups"].append(group)
                time.sleep(0.03)
                with self.tracker["lock"]:
                    self.tracker["active"] -= 1
                    self.tracker["calls"] += 1
                return SimpleNamespace(
                    status="ready",
                    focus=SimpleNamespace(
                        crucial_evidence_type="visual",
                        visible_focus_instruction="Inspect the object.",
                        tool_regions=[],
                    ),
                    crops=[],
                )

        tracker = {
            "lock": threading.Lock(),
            "active": 0,
            "max_active": 0,
            "calls": 0,
            "groups": [],
        }
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.global_steps = 1
        trainer.tokenizer = SimpleNamespace(
            decode=lambda _tokens, skip_special_tokens=True: "FINAL: A"
        )
        trainer._extra = lambda _batch, index: _batch.non_tensor_batch["extra_info"][index]
        completions = ["useful prefix LOOP LOOP"] + ["FINAL: A"] * 7
        trainer._decode_rollout = lambda _batch, index: completions[index]
        trainer._new_groove_builder = lambda: FakeBuilder(tracker)
        trainer._get_groove_builder = lambda: (_ for _ in ()).throw(
            AssertionError("parallel path must use independent builders")
        )

        batch = FakeBatch()
        with patch.dict(
            os.environ,
            {
                "ANALYZER_GROUNDING_URL": "http://127.0.0.1:8011",
                "ANALYZER_OCR_URL": "http://127.0.0.1:8012",
                "ANALYZER_USE_VISION_TOOLS": "true",
                "GROOVE_MAX_CONCURRENCY": "2",
            },
            clear=False,
        ), patch(
            "groove.verl_trainer.teacher_payload",
            lambda _evidence, question, max_image_pixels: (
                [{"role": "user", "content": question}],
                [{"path": "/tmp/image.jpg", "max_pixels": max_image_pixels}],
            ),
        ):
            metrics = trainer._build_online_teacher_columns(
                batch,
                judge_accuracies=np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32),
                repetition_starts=np.array([13] + [-1] * 7, dtype=np.int64),
            )

        self.assertEqual(tracker["calls"], 2)
        self.assertEqual(tracker["max_active"], 2)
        self.assertEqual(metrics["groove/evidence_concurrency"], 2.0)
        self.assertEqual(metrics["groove/group_count"], 2.0)
        self.assertEqual(metrics["groove/correct_rollout_fraction"], 0.5)
        self.assertEqual(metrics["groove/analyzer_repetition_trimmed_fraction"], 0.125)
        self.assertIn("timing_s/groove/evidence_build_wall", metrics)
        groups = {group.uid: group for group in tracker["groups"]}
        first_group = groups["step-0000001-group-a"]
        self.assertEqual(
            [rollout.is_correct for rollout in first_group.rollouts],
            [True, False, True, False],
        )
        self.assertEqual(first_group.rollouts[0].completion, "useful prefix")


if __name__ == "__main__":
    unittest.main()
