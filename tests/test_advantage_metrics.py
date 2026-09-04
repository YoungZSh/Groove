from __future__ import annotations

import math
import unittest

import torch

from groove.advantage_metrics import compute_advantage_metrics
from groove.losses import combine_grpo_opsd_advantages, groove_opsd_advantages


class AdvantageMetricsTest(unittest.TestCase):
    def test_masks_group_splits_and_net_credit_use_the_actual_training_tokens(self):
        student = torch.full((6, 2), -4.0)
        gap = torch.tensor([[1., 3.], [-2., 8.], [2., 2.], [-1., -1.], [float("nan"), 9.], [0., 0.]])
        mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        evidence = torch.tensor([1, 1, 1, 1, 0, 0])
        grpo = torch.tensor([[1., 1.], [-1., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        opsd, _ = groove_opsd_advantages(student, student + gap, mask, evidence_mask=evidence)
        total = combine_grpo_opsd_advantages(grpo, opsd, opsd_coef=0.5)
        metrics = compute_advantage_metrics(
            grpo_advantages=grpo, opsd_advantages=opsd, total_advantages=total,
            student_log_probs=student, teacher_log_probs=student + gap,
            response_mask=mask, evidence_mask=evidence, opsd_coef=0.5,
            sequence_rewards=torch.tensor([1., 0.1, 1., 0.1, 0.1, 1.]),
            group_ids=["mixed", "mixed", "correct", "wrong", "wrong", "correct"],
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["opsd/delta_count"], 7)
        self.assertEqual(metrics["opsd/mixed/trajectory_delta_count"], 2)
        self.assertEqual(metrics["opsd/mixed/trajectory_delta_mean"], 0)
        self.assertEqual(metrics["opsd/correct/trajectory_delta_mean"], 2)
        self.assertEqual(metrics["opsd/all_correct/trajectory_credit_mean"], 4)
        self.assertEqual(metrics["opsd/all_wrong/trajectory_credit_mean"], -2)
        self.assertEqual(metrics["opsd/incorrect/trajectory_delta_mean"], -1.5)
        self.assertAlmostEqual(metrics["opsd/advantage_rms_raw"], math.sqrt(24 / 9), places=6)
        self.assertAlmostEqual(metrics["opsd_to_grpo_advantage_rms_ratio"], math.sqrt(2), places=6)
        self.assertEqual(metrics["opsd/grpo_sign_alignment"], 1)
        self.assertTrue(torch.equal(total[4], grpo[4]))

    def test_uniform_batch_reports_undefined_ratio_without_nan(self):
        zeros = torch.zeros(2, 1)
        opsd = torch.tensor([[1.], [-1.]])
        metrics = compute_advantage_metrics(
            grpo_advantages=zeros, opsd_advantages=opsd, total_advantages=opsd,
            student_log_probs=zeros, teacher_log_probs=opsd,
            response_mask=torch.ones_like(zeros), evidence_mask=torch.ones(2), opsd_coef=1,
            sequence_rewards=torch.ones(2), group_ids=["same", "same"],
        )
        self.assertEqual(metrics["opsd_to_grpo_advantage_rms_ratio_defined"], 0)
        self.assertEqual(metrics["opsd/incorrect/delta_count"], 0)
        self.assertEqual(metrics["opsd/advantage_rms_weighted"], 1)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
