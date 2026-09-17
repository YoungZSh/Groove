from __future__ import annotations

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from groove import verl_entrypoint


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTest(unittest.TestCase):
    def test_project_preset_inherits_verl_and_accepts_overrides_outside_checkout(self):
        previous_cwd = Path.cwd()
        with TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                config = verl_entrypoint.load_training_config("groove", [
                    "groove.enabled=true",
                    "actor_rollout_ref.actor.kl_loss_coef=0.01",
                    "data.train_batch_size=16",
                    "trainer.experiment_name=unit-project-config",
                ])
            finally:
                os.chdir(previous_cwd)
        self.assertTrue(config.groove.enabled)
        self.assertEqual(config.groove.opsd_advantage_coef, 0.01)
        self.assertEqual(config.actor_rollout_ref.actor.kl_loss_coef, 0.01)
        self.assertEqual(config.actor_rollout_ref.actor.strategy, "fsdp")
        self.assertEqual(config.data.train_batch_size, 16)
        self.assertEqual(config.trainer.experiment_name, "unit-project-config")
        OmegaConf.to_container(config, resolve=True)

    def test_installed_preset_matches_checkout(self):
        expected = verl_entrypoint.load_training_config("groove", [])
        with TemporaryDirectory() as folder:
            prefix = Path(folder)
            config_dir = prefix / "share/groove/configs"
            config_dir.mkdir(parents=True)
            shutil.copy2(ROOT / "configs/groove.yaml", config_dir / "groove.yaml")
            installed_module = prefix / "lib/python/site-packages/groove/verl_entrypoint.py"
            with (
                patch.object(verl_entrypoint, "__file__", str(installed_module)),
                patch.object(verl_entrypoint.sysconfig, "get_path", return_value=str(prefix)),
            ):
                actual = verl_entrypoint.load_training_config("groove", [])
        self.assertEqual(
            OmegaConf.to_container(actual, resolve=True),
            OmegaConf.to_container(expected, resolve=True),
        )

    def test_missing_project_preset_fails_clearly(self):
        with TemporaryDirectory() as folder:
            module = Path(folder) / "src/groove/verl_entrypoint.py"
            with (
                patch.object(verl_entrypoint, "__file__", str(module)),
                patch.object(verl_entrypoint.sysconfig, "get_path", return_value=folder),
                self.assertRaisesRegex(FileNotFoundError, "GROOVE training config"),
            ):
                verl_entrypoint.load_training_config("groove", [])


if __name__ == "__main__":
    unittest.main()
