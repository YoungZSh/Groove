from __future__ import annotations

import unittest

import torch

from mmcot_opsd.losses import joint_visual_seed_loss, seed_opd_loss


class SeedOPDLossTest(unittest.TestCase):
    def test_gate_and_gradient_match_seed_equation(self):
        student = torch.tensor([[-2.0, -1.0], [-3.0, -0.5]], requires_grad=True)
        teacher = torch.tensor([[-1.0, -2.0], [-2.5, -0.25]], requires_grad=True)
        mask = torch.ones_like(student)

        loss, metrics = seed_opd_loss(student, teacher, mask, beta=5.0)
        expected_gate = torch.sigmoid(5.0 * (teacher.detach() - student.detach()))
        expected_loss = (expected_gate * (teacher.detach() - student)).mean()
        self.assertTrue(torch.allclose(loss, expected_loss))
        self.assertAlmostEqual(metrics.gate_mean, float(expected_gate.mean()), places=6)

        loss.backward()
        self.assertTrue(torch.allclose(student.grad, -expected_gate / student.numel()))
        self.assertIsNone(teacher.grad)

    def test_evidence_mask_is_not_a_reward_mask(self):
        student = torch.tensor([[-2.0], [-2.0]], requires_grad=True)
        teacher = torch.tensor([[-1.0], [-1.0]])
        # Both a hypothetical correct row and failed row are active.  The API
        # intentionally has no reward or (1-r_i) argument.
        evidence_mask = torch.tensor([1.0, 1.0])
        grpo = student.sum() * 0.0 + 2.0
        joint, opd, _ = joint_visual_seed_loss(
            grpo,
            student,
            teacher,
            torch.ones_like(student),
            evidence_mask=evidence_mask,
        )
        joint.backward()
        self.assertGreater(float(opd.detach()), 0.0)
        self.assertTrue(torch.all(student.grad < 0))

    def test_verl_runtime_uses_the_same_sampled_token_objective(self):
        from verl.trainer.ppo.core_algos import compute_seed_opd_loss

        student = torch.tensor([[-2.0, -1.0], [-3.0, -0.5]], requires_grad=True)
        teacher = torch.tensor([[-1.0, -2.0], [-2.5, -0.25]])
        mask = torch.ones_like(student)
        project_loss, _ = seed_opd_loss(student, teacher, mask, beta=5.0)
        runtime_loss, metrics = compute_seed_opd_loss(
            student_log_probs=student,
            teacher_log_probs=teacher,
            response_mask=mask,
            self_distillation_mask=torch.ones(2),
            gate_beta=5.0,
        )
        self.assertTrue(torch.allclose(project_loss, runtime_loss))
        self.assertIn("actor/seed_opd_gate_mean", metrics)


if __name__ == "__main__":
    unittest.main()
