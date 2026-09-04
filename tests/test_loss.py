from __future__ import annotations

import unittest

import torch

from groove.losses import combine_grpo_opsd_advantages, groove_opsd_advantages


class SignedOPSDAdvantageTest(unittest.TestCase):
    def test_signed_gap_can_raise_and_lower_sampled_tokens(self):
        student = torch.tensor([[-2.0, -1.0], [-3.0, -0.5]], requires_grad=True)
        teacher = torch.tensor([[-1.0, -2.0], [-2.5, -0.25]], requires_grad=True)
        response_mask = torch.ones_like(student)

        opsd, metrics = groove_opsd_advantages(student, teacher, response_mask)
        expected = teacher.detach() - student.detach()
        self.assertTrue(torch.equal(opsd, expected))
        self.assertGreater(metrics.positive_token_fraction, 0.0)
        self.assertGreater(metrics.negative_token_fraction, 0.0)
        self.assertFalse(opsd.requires_grad)

        grpo = torch.zeros_like(student)
        total = combine_grpo_opsd_advantages(grpo, opsd, opsd_coef=1.0)
        surrogate = -(total * student).mean()
        surrogate.backward()
        self.assertTrue(torch.allclose(student.grad, -expected / student.numel()))
        self.assertIsNone(teacher.grad)

    def test_equal_teacher_and_student_have_zero_opsd_gradient(self):
        student = torch.tensor([[-2.0, -1.0]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        opsd, metrics = groove_opsd_advantages(student, teacher, torch.ones_like(student))
        self.assertTrue(torch.equal(opsd, torch.zeros_like(opsd)))
        self.assertEqual(metrics.zero_token_fraction, 1.0)

        loss = -(opsd * student).mean()
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
        self.assertIsNone(teacher.grad)

    def test_evidence_mask_is_availability_not_correctness(self):
        student = torch.tensor([[-2.0], [-2.0], [-2.0]])
        teacher = torch.tensor([[-1.0], [-3.0], [-1.0]])
        evidence_mask = torch.tensor([1.0, 1.0, 0.0])

        opsd, _ = groove_opsd_advantages(
            student,
            teacher,
            torch.ones_like(student),
            evidence_mask=evidence_mask,
        )
        self.assertEqual(float(opsd[0, 0]), 1.0)
        self.assertEqual(float(opsd[1, 0]), -1.0)
        self.assertEqual(float(opsd[2, 0]), 0.0)

    def test_symmetric_advantage_clipping(self):
        student = torch.tensor([[-4.0, -1.0]])
        teacher = torch.tensor([[-1.0, -4.0]])
        opsd, metrics = groove_opsd_advantages(
            student,
            teacher,
            torch.ones_like(student),
            advantage_clip=2.0,
        )
        self.assertTrue(torch.equal(opsd, torch.tensor([[2.0, -2.0]])))
        self.assertEqual(metrics.clipped_token_fraction, 1.0)

    def test_masked_nonfinite_scores_do_not_pollute_fallback_or_padding(self):
        student = torch.tensor([[-2.0, float("nan")], [float("-inf"), -1.0]])
        teacher = torch.tensor([[-1.0, float("nan")], [float("-inf"), float("nan")]])
        opsd, metrics = groove_opsd_advantages(
            student, teacher, torch.tensor([[1, 0], [1, 1]]),
            evidence_mask=torch.tensor([1, 0]),
        )
        self.assertTrue(torch.equal(opsd, torch.tensor([[1.0, 0.0], [0.0, 0.0]])))
        self.assertAlmostEqual(metrics.active_token_fraction, 1 / 3)
        self.assertEqual(metrics.teacher_gap_mean, 1.0)
        grpo = torch.tensor([[2.0, 0.0], [-1.0, -1.0]])
        total = combine_grpo_opsd_advantages(grpo, opsd)
        self.assertTrue(torch.equal(total[1], grpo[1]))

    def test_nonfinite_active_gap_and_fractional_evidence_are_rejected(self):
        student = torch.tensor([[-2.0]])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            groove_opsd_advantages(student, torch.tensor([[float("nan")]]), torch.ones_like(student))
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            groove_opsd_advantages(student, student, torch.ones_like(student), evidence_mask=torch.tensor([0.5]))

    def test_evidence_and_response_masks_cannot_create_a_gradient_path(self):
        student = torch.tensor([[-2.0]], requires_grad=True)
        teacher = torch.tensor([[-1.0]], requires_grad=True)
        mask = torch.ones_like(student, requires_grad=True)
        evidence = torch.ones(1, requires_grad=True)
        opsd, _ = groove_opsd_advantages(student, teacher, mask, evidence_mask=evidence)
        self.assertFalse(opsd.requires_grad)

    def test_single_rollout_is_a_uniform_group(self):
        import numpy as np
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        rewards = torch.tensor([[1.0, 0.0]])
        advantages, _ = compute_grpo_outcome_advantage(rewards, torch.ones_like(rewards), np.array(["one"]))
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))

    def test_verl_runtime_uses_same_uncentered_signed_advantages(self):
        from verl.trainer.ppo.core_algos import compute_groove_opsd_advantages

        student = torch.tensor([[-2.0, -1.0], [-3.0, -0.5]])
        teacher = torch.tensor([[-1.0, -2.0], [-2.5, -0.25]])
        mask = torch.ones_like(student)
        project_advantages, _ = groove_opsd_advantages(student, teacher, mask)
        runtime_advantages, metrics = compute_groove_opsd_advantages(
            student_log_probs=student,
            teacher_log_probs=teacher,
            response_mask=mask,
            self_distillation_mask=torch.ones(2),
        )
        self.assertTrue(torch.equal(project_advantages, runtime_advantages))
        self.assertIn("actor/groove_opsd_positive_token_fraction", metrics)
        self.assertIn("actor/groove_opsd_negative_token_fraction", metrics)

    def test_runtime_token_mask_counts_only_valid_response_tokens(self):
        from verl.trainer.ppo.core_algos import compute_groove_opsd_advantages

        student = torch.full((1, 4), -2.0)
        advantages, metrics = compute_groove_opsd_advantages(
            student, student + 1, torch.tensor([[1, 1, 1, 0]]),
            self_distillation_mask=torch.tensor([[1, 0, 0, 0]]),
        )
        self.assertTrue(torch.equal(advantages, torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        self.assertAlmostEqual(metrics["actor/groove_opsd_active_token_ratio"], 1 / 3)

    def test_grpo_uniform_non_binary_reward_has_zero_advantage(self):
        """Uniform 0.1 groups must not create a float32 normalization artefact."""
        import numpy as np

        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        rewards = torch.zeros(16, 2, dtype=torch.float32)
        rewards[:8, 0] = 0.1
        rewards[8:, 0] = 1.0
        response_mask = torch.ones_like(rewards)
        groups = np.array(["incorrect"] * 8 + ["correct"] * 8, dtype=object)

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
            index=groups,
            norm_adv_by_std_in_grpo=True,
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))
        self.assertTrue(torch.equal(returns, torch.zeros_like(returns)))


if __name__ == "__main__":
    unittest.main()
