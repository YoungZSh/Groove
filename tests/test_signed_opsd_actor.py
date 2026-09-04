from __future__ import annotations

import unittest

import torch

from groove.verl_trainer import GrooveRayPPOTrainer
from verl import DataProto


class SignedOPSDTrainerTest(unittest.TestCase):
    def test_groove_combines_signed_opsd_with_grpo_before_ppo(self):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {
            "groove": {
                "enabled": True,
                "opsd_advantage_coef": 1.0,
                "opsd_advantage_clip": None,
            }
        }
        trainer._build_online_teacher_columns = lambda *_args: {"groove/group_count": 1.0}
        trainer._build_groove_teacher_batch = lambda _batch: (
            object(),
            torch.ones(2),
            {"groove/teacher_prefix_cache_entries": 1.0},
        )
        teacher_output = DataProto.from_dict(
            tensors={"old_log_probs": torch.tensor([[1.0], [-1.0]])},
            non_tensors={},
        )
        trainer._compute_old_log_prob = lambda _batch: (teacher_output, 0.0)

        batch = DataProto.from_dict(
            tensors={
                "old_log_probs": torch.zeros((2, 1)),
                "advantages": torch.zeros((2, 1)),
                "response_mask": torch.ones((2, 1)),
            },
            non_tensors={},
        )
        result, metrics = trainer._postprocess_advantages(
            batch,
            reward_tensor=torch.zeros((2, 1)),
            reward_extra_infos_dict=None,
        )

        self.assertIs(result, batch)
        self.assertTrue(torch.equal(batch.batch["advantages"], torch.tensor([[1.0], [-1.0]])))
        self.assertIn("actor/groove_opsd_positive_token_fraction", metrics)
        self.assertIn("actor/groove_opsd_negative_token_fraction", metrics)
        self.assertNotIn("actor/groove_opd_loss", metrics)


if __name__ == "__main__":
    unittest.main()
