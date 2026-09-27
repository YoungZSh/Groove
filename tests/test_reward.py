from __future__ import annotations

import unittest

import numpy as np
import torch

from groove.reward import (
    answers_match,
    compute_score,
    extract_final_answer,
    extract_final_option,
    extract_option,
    format_reward,
)


class RewardTest(unittest.TestCase):
    def test_verl_naive_reward_manager_preserves_components(self):
        from verl.protocol import DataProto
        from verl.workers.reward_manager.naive import NaiveRewardManager

        class Tokenizer:
            eos_token_id = 0

            @staticmethod
            def decode(_, skip_special_tokens=True):
                del skip_special_tokens
                return "Visual rationale.\nFINAL: B"

        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[10, 11]]),
                "responses": torch.tensor([[12, 13, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            },
            non_tensors={
                "data_source": np.array(["vision_opd_6k_groove"], dtype=object),
                "reward_model": np.array([{"ground_truth": "B"}], dtype=object),
                "extra_info": np.array([{}], dtype=object),
            },
        )
        result = NaiveRewardManager(Tokenizer(), 0, compute_score=compute_score)(
            data, return_dict=True
        )
        self.assertEqual(result["reward_tensor"].tolist(), [[0.0, 0.0, 1.0]])
        extra = result["reward_extra_info"]
        self.assertEqual(extra["answer_reward"], [1.0])
        self.assertEqual(extra["format_reward"], [1.0])
        self.assertEqual(extra["weighted_answer_reward"], [0.9])
        self.assertEqual(extra["weighted_format_reward"], [0.1])

    def test_final_answer_has_priority(self):
        text = "I compared A and B. The visible detail supports the latter. FINAL: C"
        self.assertEqual(extract_option(text), "C")
        self.assertEqual(extract_final_answer(text), "C")
        self.assertEqual(extract_final_option(text), "C")
        result = compute_score("vstar", text, "C")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["answer_reward"], 1.0)
        self.assertEqual(result["format_reward"], 1.0)

    def test_correct_answer_without_final_line_receives_only_answer_weight(self):
        result = compute_score("vstar", "The correct choice is (B).", "B")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["final_answer"], "")

    def test_formatted_incorrect_answer_receives_only_format_weight(self):
        result = compute_score("vstar", "Visual reasoning.\nFINAL: D", "A")
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["format_reward"], 1.0)
        self.assertEqual(result["score"], 0.1)

    def test_reward_weights_are_forwarded_by_custom_reward_configuration(self):
        result = compute_score(
            "vstar",
            "Visual reasoning. FINAL: B",
            "B",
            answer_reward_weight=0.9,
            format_reward_weight=0.1,
        )
        self.assertEqual(result["answer_reward_weight"], 0.9)
        self.assertEqual(result["format_reward_weight"], 0.1)
        self.assertEqual(result["weighted_answer_reward"], 0.9)
        self.assertEqual(result["weighted_format_reward"], 0.1)

    def test_groove_logs_both_reward_components(self):
        from groove.verl_trainer import GrooveRayPPOTrainer

        metrics = GrooveRayPPOTrainer._reward_component_metrics(
            {
                "answer_reward": [1.0, 0.0],
                "format_reward": [1.0, 0.0],
                "weighted_answer_reward": [0.9, 0.0],
                "weighted_format_reward": [0.1, 0.0],
            }
        )
        self.assertAlmostEqual(metrics["reward/answer_reward_mean"], 0.5)
        self.assertAlmostEqual(metrics["reward/format_reward_mean"], 0.5)
        self.assertAlmostEqual(metrics["reward/weighted_answer_reward_mean"], 0.45)
        self.assertAlmostEqual(metrics["reward/weighted_format_reward_mean"], 0.05)

    def test_final_answer_format_allows_non_mcq_answer_text(self):
        text = "The object is blue. FINAL: blue"
        self.assertEqual(extract_final_answer(text), "blue")
        self.assertEqual(format_reward(text), 1.0)
        self.assertEqual(compute_score("generic", text, "Blue")["score"], 1.0)
        self.assertTrue(answers_match("**blue**", "blue"))

    def test_trainer_logs_judge_exhaustion_for_each_batch_including_healthy_steps(self):
        from groove.verl_trainer import GrooveRayPPOTrainer

        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {"groove": {"enabled": False}}
        for flags, count in [([1., 0., 0.], 1.), ([0., 0., 0.], 0.)]:
            with self.subTest(flags=flags):
                _, metrics = trainer._postprocess_advantages(object(), None, {
                    "judge_retries_exhausted": flags, "judge_attempts": [6., 1., 6.],
                })
                self.assertEqual(metrics["reward/judge_retries_exhausted_count"], count)
                self.assertAlmostEqual(metrics["reward/judge_retries_exhausted_fraction"], count / 3)
                self.assertAlmostEqual(metrics["reward/judge_attempts_mean"], 13 / 3)
                self.assertEqual(metrics["reward/judge_attempts_max"], 6.)

    def test_rollout_dump_preserves_judge_flags_with_step_and_score(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        with TemporaryDirectory() as folder:
            RayPPOTrainer._write_generations(
                ["question", "question"], ["answer A", "answer B"], ["B", "B"], [0., 1.],
                {"judge_attempts": [6., 1.], "judge_retries_exhausted": [1., 0.]}, folder, 3,
            )
            rows = [json.loads(line) for line in (Path(folder) / "3.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [3, 3])
        self.assertEqual([r["score"] for r in rows], [0., 1.])
        self.assertEqual([r["judge_attempts"] for r in rows], [6., 1.])
        self.assertEqual([r["judge_retries_exhausted"] for r in rows], [1., 0.])

    def test_final_line_must_be_last_nonempty_line(self):
        text = "FINAL: A\nI will revise this later."
        self.assertEqual(extract_option(text), "A")
        self.assertIsNone(extract_final_answer(text))
        self.assertIsNone(extract_final_option(text))
        self.assertEqual(format_reward(text), 0.0)
        self.assertEqual(compute_score("vstar", text, "A")["score"], 0.9)

    def test_missing_answer_is_wrong(self):
        self.assertIsNone(extract_option("The object is difficult to see."))
        self.assertEqual(compute_score("vstar", "unclear", "A")["score"], 0.0)

    def test_reward_diagnostics_are_validation_safe(self):
        result = compute_score("vstar", "unclear", "A")
        self.assertEqual(result["predicted_label"], "")
        self.assertEqual(result["final_answer"], "")


if __name__ == "__main__":
    unittest.main()
