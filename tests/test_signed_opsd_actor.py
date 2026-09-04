from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from groove.verl_trainer import GrooveRayPPOTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


class SignedOPSDTrainerTest(unittest.TestCase):
    @staticmethod
    def _batch(old_log_probs, advantages):
        count, tokens = old_log_probs.shape
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones(count, 1, dtype=torch.long),
                "responses": torch.ones(count, tokens, dtype=torch.long),
                "attention_mask": torch.ones(count, tokens + 1, dtype=torch.long),
                "response_mask": torch.ones_like(old_log_probs),
                "old_log_probs": old_log_probs.detach().clone(),
                "advantages": advantages,
                "ref_log_prob": old_log_probs.detach().clone(),
            },
            non_tensors={"uid": np.array(["same"] * count, dtype=object)},
        )

    @staticmethod
    def _trainer(teacher_scores, evidence_mask):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {"groove": {"enabled": True, "opsd_advantage_coef": 1.0}}
        trainer._build_online_teacher_columns = Mock(return_value={})
        trainer._build_groove_teacher_batch = Mock(return_value=(object(), evidence_mask, {}))
        trainer._compute_old_log_prob = Mock(return_value=(
            DataProto.from_dict(tensors={"old_log_probs": teacher_scores}, non_tensors={}), 0.0
        ))
        return trainer

    def _actor_loss(self, batch, current_scores, *, use_reference_kl=False):
        config = ActorConfig(strategy="fsdp2", rollout_n=2, use_dynamic_bsz=True, use_kl_loss=use_reference_kl)
        data = batch.to_tensordict()
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(batch.batch["response_mask"].sum()),
                             global_batch_size=len(batch))
        # The engine emits a log probability at every packed input position.
        packed_scores = torch.cat([torch.cat((row, row.new_zeros(1))) for row in current_scores])
        policy_fn = Mock(wraps=compute_policy_loss_vanilla)
        with patch("verl.workers.utils.losses.get_policy_loss_fn", return_value=policy_fn):
            loss, metrics = ppo_loss(config, {"log_probs": packed_scores}, data)
        policy_fn.assert_called_once()
        self.assertTrue(torch.equal(policy_fn.call_args.kwargs["advantages"], batch.batch["advantages"]))
        return loss, metrics

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

    def test_combined_advantage_controls_shared_ppo_clipping(self):
        old = torch.full((2, 1), -2.0)
        trainer = self._trainer(old + torch.tensor([[-2.0], [2.0]]), torch.ones(2))
        batch = self._batch(old, torch.tensor([[1.0], [-1.0]]))
        trainer._postprocess_advantages(batch, torch.tensor([[1.0], [0.1]]))
        self.assertTrue(torch.equal(batch.batch["advantages"], torch.tensor([[-1.0], [1.0]])))
        current = (old + torch.tensor([[0.7], [1.3]]).log()).requires_grad_()
        loss, _ = self._actor_loss(batch, current)
        loss.backward()
        # Both total advantages have reached their PPO bound. Separate clipping
        # of the two signals or an extra auxiliary scalar would leave a gradient.
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))

    def test_uniform_group_updates_actor_with_frozen_targets_and_refreshes_next_batch(self):
        actor_logits = torch.nn.Parameter(torch.tensor([-0.2, 0.4]))
        old_scores = actor_logits.detach().log_softmax(-1).unsqueeze(-1)
        trainer = self._trainer(old_scores, torch.ones(2))
        scored_targets = []

        def score_teacher(_batch):
            self.assertFalse(torch.is_grad_enabled())
            scores = (actor_logits + torch.tensor([1.0, -1.0])).log_softmax(-1).unsqueeze(-1)
            scored_targets.append(scores.clone())
            return DataProto.from_dict(tensors={"old_log_probs": scores}, non_tensors={}), 0.0

        trainer._compute_old_log_prob = Mock(side_effect=score_teacher)
        batch = self._batch(old_scores, torch.zeros_like(old_scores))
        rewards = torch.ones_like(old_scores)
        trainer._postprocess_advantages(batch, rewards)
        target = batch.batch["advantages"].clone()
        self.assertFalse(target.requires_grad)
        loss, _ = self._actor_loss(batch, actor_logits.log_softmax(-1).unsqueeze(-1))
        loss.backward()
        self.assertLess(float(actor_logits.grad[0]), 0)
        self.assertGreater(float(actor_logits.grad[1]), 0)
        torch.optim.SGD([actor_logits], lr=0.1).step()
        self.assertTrue(torch.equal(batch.batch["advantages"], target))
        trainer._compute_old_log_prob.assert_called_once()
        self.assertFalse(scored_targets[0].requires_grad)

        next_old = actor_logits.detach().log_softmax(-1).unsqueeze(-1)
        next_batch = self._batch(next_old, torch.zeros_like(next_old))
        trainer._postprocess_advantages(next_batch, rewards)
        self.assertEqual(trainer._compute_old_log_prob.call_count, 2)
        self.assertFalse(torch.equal(scored_targets[0], scored_targets[1]))
        self.assertTrue(torch.equal(rewards, torch.ones_like(rewards)))

    def test_missing_evidence_preserves_grpo_and_reference_kl_without_teacher_scoring(self):
        old = torch.full((2, 1), -2.0)
        original = torch.tensor([[1.0], [-1.0]])
        trainer = self._trainer(torch.full_like(old, float("nan")), torch.zeros(2))
        batch = self._batch(old, original.clone())
        _, metrics = trainer._postprocess_advantages(batch, torch.tensor([[1.0], [0.1]]))
        trainer._compute_old_log_prob.assert_not_called()
        self.assertTrue(torch.equal(batch.batch["advantages"], original))
        self.assertEqual(metrics["opsd/fallback_fraction"], 1.0)
        current = (old + 0.05).requires_grad_()
        joint_loss, joint_metrics = self._actor_loss(batch, current, use_reference_kl=True)
        baseline = self._batch(old, original.clone())
        plain_loss, _ = self._actor_loss(baseline, current, use_reference_kl=True)
        self.assertTrue(torch.equal(joint_loss, plain_loss))
        self.assertIn("kl_loss", joint_metrics)

    def test_reward_changes_do_not_gate_evidence_credit(self):
        old = torch.full((2, 1), -2.0)
        teacher = old + torch.tensor([[0.5], [-0.5]])
        trainer = self._trainer(teacher, torch.ones(2))
        targets = []
        for reward in (0.1, 1.0):
            batch = self._batch(old, torch.zeros_like(old))
            trainer._postprocess_advantages(batch, torch.full_like(old, reward))
            targets.append(batch.batch["advantages"])
        self.assertTrue(torch.equal(targets[0], targets[1]))


if __name__ == "__main__":
    unittest.main()
