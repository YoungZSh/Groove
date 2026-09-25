from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from groove.verl_trainer import GrooveRayPPOTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.config import ActorConfig


class RLSDTrainerTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"OPSD_LOG_PROB_DUMP_DIR": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def make_trainer(self, step=1, evidence=(1, 1, 1)):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {"groove": {"enabled": True, "advantage_mode": "rlsd_positive"}}
        trainer.global_steps = step
        trainer.actor_rollout_wg = Mock()
        trainer.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
        trainer._build_online_teacher_columns = Mock(return_value={})
        teacher_batch = SimpleNamespace(meta_info={})
        trainer._build_groove_teacher_batch = Mock(return_value=(teacher_batch, torch.tensor(evidence), {}))
        trainer._compute_old_log_prob = Mock(return_value=(DataProto.from_dict(
            tensors={"old_log_probs": torch.tensor([[-1., -3.], [-1., -3.], [-1., -3.]])},
        ), 0.))
        return trainer

    @staticmethod
    def batch(advantages=None):
        return DataProto.from_dict(tensors={
            "responses": torch.ones(3, 2, dtype=torch.long),
            "old_log_probs": torch.full((3, 2), -2.),
            "advantages": torch.tensor([[1., 1.], [-1., -1.], [0., 0.]]) if advantages is None else advantages,
            "response_mask": torch.ones(3, 2),
        }, non_tensors={"uid": np.array(["a", "a", "b"], dtype=object)})

    def test_positive_only_advantages_reach_ppo_with_frozen_teacher_flag(self):
        trainer = self.make_trainer()
        batch = self.batch()
        teacher_response = trainer._compute_old_log_prob.return_value[0].batch["old_log_probs"]
        teacher_response.requires_grad_()
        grpo = batch.batch["advantages"].clone()
        result, metrics = trainer._postprocess_advantages(batch, torch.zeros(3, 2), {"accuracy": [0, 0, 0]})
        trainer.actor_rollout_wg.sync_rlsd_teacher.assert_called_once_with(0, 10)
        self.assertTrue(trainer._compute_old_log_prob.call_args.args[0].meta_info["rlsd_teacher"])
        torch.testing.assert_close(result.batch["advantages"][0], torch.tensor([1.098, .902]))
        self.assertTrue(torch.equal(result.batch["advantages"][1:], grpo[1:]))
        self.assertFalse(result.batch["advantages"].requires_grad)
        self.assertAlmostEqual(metrics["rlsd/lambda"], .49)
        # The reward accuracy is zero but A>0 remains eligible (format shaping).
        trainer._build_online_teacher_columns.assert_called_once_with(batch, [0, 0, 0], None)
        current = batch.batch["old_log_probs"].clone().requires_grad_()
        config = ActorConfig(
            strategy="fsdp", rollout_n=1, use_dynamic_bsz=True, loss_agg_mode="token-mean",
            global_batch_info={"dp_size": 1, "batch_num_tokens": 6, "global_batch_size": 3},
        )
        loss, _ = compute_policy_loss_vanilla(
            old_log_prob=batch.batch["old_log_probs"], log_prob=current,
            advantages=result.batch["advantages"], response_mask=batch.batch["response_mask"],
            loss_agg_mode="token-mean", config=config,
        )
        loss.backward()
        torch.testing.assert_close(current.grad, -result.batch["advantages"] / 6)
        self.assertIsNone(teacher_response.grad)

    def test_decay_boundary_skips_all_evidence_work_and_releases_once(self):
        for decay_steps in (40, 50):
            with self.subTest(decay_steps=decay_steps):
                trainer = self.make_trainer(step=decay_steps - 1)
                trainer.config["groove"]["rlsd_lambda_decay_steps"] = decay_steps
                before = self.batch()
                original = before.batch["advantages"].clone()
                result, metrics = trainer._postprocess_advantages(before, torch.zeros(3, 2))
                self.assertAlmostEqual(metrics["rlsd/lambda"], .5 / decay_steps)
                self.assertFalse(torch.equal(result.batch["advantages"][0], original[0]))
                trainer._compute_old_log_prob.assert_called_once()
                trainer.actor_rollout_wg.reset_mock()
                trainer._build_online_teacher_columns.reset_mock()
                trainer._compute_old_log_prob.reset_mock()
                for step in (decay_steps, decay_steps + 1, 125):
                    trainer.global_steps = step
                    batch = self.batch()
                    original = batch.batch["advantages"].clone()
                    result, metrics = trainer._postprocess_advantages(batch, torch.zeros(3, 2))
                    self.assertTrue(torch.equal(result.batch["advantages"], original))
                    self.assertEqual(metrics["rlsd/lambda"], 0.)
                trainer.actor_rollout_wg.release_rlsd_teacher.assert_called_once()
                trainer.actor_rollout_wg.sync_rlsd_teacher.assert_not_called()
                trainer._build_online_teacher_columns.assert_not_called()
                trainer._compute_old_log_prob.assert_not_called()

    def test_no_positive_batch_still_syncs_at_boundary(self):
        trainer = self.make_trainer(step=11)
        batch = self.batch(torch.zeros(3, 2))
        result, _ = trainer._postprocess_advantages(batch, torch.ones(3, 2))
        self.assertTrue(torch.equal(result.batch["advantages"], torch.zeros(3, 2)))
        trainer.actor_rollout_wg.sync_rlsd_teacher.assert_called_once_with(10, 10)
        trainer._build_online_teacher_columns.assert_not_called()
        trainer._compute_old_log_prob.assert_not_called()

    def test_absent_evidence_falls_back_to_grpo(self):
        trainer = self.make_trainer(evidence=(0, 0, 0))
        batch = self.batch()
        original = batch.batch["advantages"].clone()
        result, metrics = trainer._postprocess_advantages(batch, torch.zeros(3, 2))
        self.assertTrue(torch.equal(result.batch["advantages"], original))
        trainer._compute_old_log_prob.assert_not_called()
        self.assertEqual(metrics["opsd/fallback_fraction"], 1.)

    def test_audit_saves_actual_multiplicative_credit(self):
        trainer = self.make_trainer()
        batch = self.batch()
        with TemporaryDirectory() as folder, patch.dict(os.environ, {"OPSD_LOG_PROB_DUMP_DIR": folder}):
            trainer._postprocess_advantages(batch, torch.zeros(3, 2))
            audit = torch.load(Path(folder) / "1.rank0.pt", weights_only=True)
        self.assertEqual(audit["advantage_mode"], "rlsd_positive")
        self.assertEqual(audit["rlsd"]["teacher_snapshot_step"], 0)
        torch.testing.assert_close(audit["total_advantages"], audit["grpo_advantages"] * audit["rlsd_weights"])
        torch.testing.assert_close(audit["rlsd_correction"], audit["weighted_opsd_advantages"])


if __name__ == "__main__":
    unittest.main()
