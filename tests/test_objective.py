from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from groove.objective import validate_objective_config


class ObjectiveConfigTest(unittest.TestCase):
    def test_joint_preset_keeps_one_shared_ppo_loss_and_reference_regularization(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="groove", overrides=["groove.enabled=true"])
        validate_objective_config(config)
        self.assertEqual(config.actor_rollout_ref.actor.policy_loss.loss_mode, "vanilla")
        self.assertTrue(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(config.actor_rollout_ref.actor.kl_loss_coef, 0.001)
        self.assertEqual(config.groove.opsd_advantage_coef, 0.01)

    def test_incompatible_objectives_fail_before_evidence_generation(self):
        for override in (
            {"algorithm": {"adv_estimator": "gae"}},
            {"algorithm": {"use_kl_in_reward": True}},
            {"algorithm": {"rollout_correction": {"bypass_mode": True}}},
            {"actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "reinforce"}}}},
            {"distillation": {"enabled": True}},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_objective_config({"groove": {"enabled": True}, **override})
        for coef in (-1, float("nan"), float("inf")):
            with self.subTest(coef=coef), self.assertRaises(ValueError):
                validate_objective_config({"groove": {"enabled": True, "opsd_advantage_coef": coef}})

    def test_disabled_visual_evidence_does_not_constrain_other_training_modes(self):
        validate_objective_config({"groove": {"enabled": False}, "algorithm": {"adv_estimator": "gae"}})
