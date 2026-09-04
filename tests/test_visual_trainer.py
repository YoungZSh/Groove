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
        trainer.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(policy_loss={"loss_mode": "vanilla"})
            )
        )
        trainer._build_online_teacher_columns = lambda *_args: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not build online OPSD evidence")
        )

        result = trainer._maybe_build_self_distillation_batch(None, None, None)

        self.assertIsNone(result)

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
                }

        class FakeBuilder:
            def __init__(self, tracker):
                self.tracker = tracker

            def build(self, group):
                del group
                with self.tracker["lock"]:
                    self.tracker["active"] += 1
                    self.tracker["max_active"] = max(
                        self.tracker["max_active"], self.tracker["active"]
                    )
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

        tracker = {"lock": threading.Lock(), "active": 0, "max_active": 0, "calls": 0}
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.global_steps = 1
        trainer.tokenizer = SimpleNamespace(
            decode=lambda _tokens, skip_special_tokens=True: "FINAL: A"
        )
        trainer._extra = lambda _batch, index: _batch.non_tensor_batch["extra_info"][index]
        trainer._decode_rollout = lambda _batch, _index: "FINAL: A"
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
                torch.ones((8, 2), dtype=torch.float32),
            )

        self.assertEqual(tracker["calls"], 2)
        self.assertEqual(tracker["max_active"], 2)
        self.assertEqual(metrics["groove/evidence_concurrency"], 2.0)
        self.assertEqual(metrics["groove/group_count"], 2.0)
        self.assertIn("timing_s/groove/evidence_build_wall", metrics)


if __name__ == "__main__":
    unittest.main()
