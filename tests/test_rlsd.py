from __future__ import annotations

import math
import unittest

import torch

from groove.rlsd import positive_rlsd_advantages, rlsd_lambda
from groove.objective import validate_objective_config


class PositiveRLSDTest(unittest.TestCase):
    def test_positive_only_bounded_weights_and_no_gradient(self):
        base = torch.tensor([[2., 2., 2.], [-1., -1., -1.], [0., 0., 0.]], requires_grad=True)
        student = torch.full_like(base, -3., requires_grad=True)
        teacher = (student.detach() + torch.tensor([[-1000., 0., 1000.]] * 3)).requires_grad_()
        total, weights, metrics = positive_rlsd_advantages(
            base, student, teacher, torch.ones_like(base), torch.ones(3), lam=.5,
        )
        torch.testing.assert_close(total[0], torch.tensor([1.8, 2., 2.2]))
        self.assertTrue(torch.equal(total[1:], base[1:]))
        self.assertTrue(torch.equal(weights[1:], torch.ones(2, 3)))
        self.assertFalse(total.requires_grad)
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(metrics["rlsd/clip_high_fraction"], 1/3)
        self.assertAlmostEqual(metrics["rlsd/clip_low_fraction"], 1/3)

    def test_formula_uses_probability_ratio_without_centering(self):
        base = torch.ones(1, 2)
        student = torch.full_like(base, -3.)
        teacher = student + torch.tensor([[math.log(.9), math.log(1.1)]])
        total, weights, _ = positive_rlsd_advantages(base, student, teacher, base, torch.ones(1), lam=.25)
        torch.testing.assert_close(weights, torch.tensor([[.975, 1.025]]))
        torch.testing.assert_close(total, weights)

    def test_missing_evidence_padding_and_inactive_nan_are_ignored(self):
        base = torch.tensor([[1., 0.], [-1., -1.], [1., 1.]])
        student = torch.zeros_like(base)
        teacher = torch.tensor([[.1, float("nan")], [float("nan"), float("nan")],
                                [float("nan"), float("inf")]])
        total, weights, _ = positive_rlsd_advantages(
            base, student, teacher, torch.tensor([[1, 0], [1, 1], [1, 1]]), torch.tensor([1, 1, 0]), lam=.5,
        )
        self.assertTrue(torch.equal(total[1:], base[1:]))
        self.assertTrue(torch.isfinite(total).all())
        self.assertEqual(float(weights[0, 1]), 1.)
        disabled, _, _ = positive_rlsd_advantages(base, student, teacher, torch.ones_like(base), torch.ones(3), lam=0.)
        self.assertTrue(torch.equal(disabled, base))

    def test_active_nonfinite_and_nonbinary_masks_fail(self):
        base = torch.ones(1, 1)
        for teacher, evidence in ((torch.full_like(base, float("nan")), torch.ones(1)),
                                  (base, torch.tensor([.5]))):
            with self.assertRaises(ValueError):
                positive_rlsd_advantages(base, base, teacher, base, evidence, lam=.5)

    def test_decay_uses_outer_step_and_is_zero_at_boundary(self):
        for step, expected in ((0, .5), (1, .49), (25, .25), (49, .01), (50, 0.), (200, 0.)):
            self.assertAlmostEqual(rlsd_lambda(step), expected)

    def test_invalid_configs_fail_before_workers(self):
        for key, values in {
            "advantage_mode": ["rlsd", "unknown"],
            "rlsd_lambda_initial": [-.1, 1.1, float("nan")],
            "rlsd_clip_range": [0., 1., float("inf")],
            "rlsd_lambda_decay_steps": [0, -1, 2.5, True],
            "rlsd_teacher_sync_interval": [0, -1, 2.5, True],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_objective_config({"groove": {"enabled": True, "advantage_mode": "rlsd_positive", key: value}})


if __name__ == "__main__":
    unittest.main()
