from pathlib import Path
import json
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from verl.utils.checkpoint.best_checkpoint import (
    BestCheckpointTracker,
    maybe_save_best_checkpoint,
    validate_best_checkpoint_config,
)


METRIC = "val-core/vstar_bench/reward/mean@1"


class BestCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def checkpoint(self, step):
        source = self.root / f"global_step_{step}"
        (source / "actor").mkdir(parents=True)
        for kind in ("model", "optim", "extra_state"):
            for rank in range(2):
                (source / "actor" / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(f"{step}:{kind}".encode())
        (source / "data.pt").write_bytes(b"dataloader")
        return source

    def test_best_survives_source_overwrite_and_rolling_deletion(self):
        tracker = BestCheckpointTracker(self.root, METRIC)
        source = self.checkpoint(10)
        tracker.consider({METRIC: 0.9}, 10, source, Mock())
        best = self.root / "best_checkpoint/global_step_10"
        (source / "actor/model_world_size_2_rank_0.pt").write_bytes(b"overwritten")
        shutil.rmtree(source)
        self.assertEqual((best / "actor/model_world_size_2_rank_0.pt").read_bytes(), b"10:model")
        self.assertEqual(len(list((best / "actor").glob("*.pt"))), 6)
        self.assertEqual((best / "data.pt").read_bytes(), b"dataloader")

    def test_ties_regressions_nan_and_resume_do_not_replace_best(self):
        tracker = BestCheckpointTracker(self.root, METRIC)
        tracker.consider({METRIC: 0.9}, 10, self.checkpoint(10), Mock())
        tracker = BestCheckpointTracker(self.root, METRIC)
        save = Mock(side_effect=AssertionError("Unnecessary checkpoint save"))
        for score in (0.9, 0.5, float("nan"), float("inf")):
            result = tracker.consider({METRIC: score}, 20, self.root / "missing", save)
            self.assertEqual(result["checkpoint/best_step"], 10)
        self.assertEqual(json.loads(tracker.metadata_path.read_text())["value"], 0.9)
        with self.assertRaisesRegex(ValueError, "missing"):
            tracker.consider({}, 20, self.root / "missing", save)

    def test_strict_improvement_replaces_only_previous_best_copy(self):
        tracker = BestCheckpointTracker(self.root, METRIC)
        first = self.checkpoint(10)
        second = self.checkpoint(20)
        tracker.consider({METRIC: 0.8}, 10, first, Mock())
        result = tracker.consider({METRIC: 0.9}, 20, second, Mock())
        self.assertEqual(result, {"checkpoint/best_score": 0.9, "checkpoint/best_step": 20})
        self.assertFalse((tracker.root / "global_step_10").exists())
        self.assertTrue((tracker.root / "global_step_20/actor").is_dir())
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

    def test_minimize_metric_and_zero_initial_score(self):
        tracker = BestCheckpointTracker(self.root, "loss", "min")
        tracker.consider({"loss": 0.0}, 0, self.checkpoint(0), Mock())
        tracker.consider({"loss": -1.0}, 10, self.checkpoint(10), Mock())
        self.assertEqual(tracker.best["step"], 10)

    def test_failed_save_copy_or_publish_preserves_old_best(self):
        tracker = BestCheckpointTracker(self.root, METRIC)
        tracker.consider({METRIC: 0.8}, 10, self.checkpoint(10), Mock())
        source = self.checkpoint(20)
        original = tracker.metadata_path.read_bytes()
        with self.assertRaises(OSError):
            tracker.consider({METRIC: 0.9}, 20, source, Mock(side_effect=OSError("save failed")))
        for operation in ("shutil.copytree", "os.replace"):
            with patch("verl.utils.checkpoint.best_checkpoint." + operation, side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    tracker.consider({METRIC: 0.9}, 20, source, Mock())
            self.assertEqual(tracker.metadata_path.read_bytes(), original)
            self.assertEqual(tracker.best["step"], 10)
            self.assertTrue((tracker.root / "global_step_10/actor").is_dir())
            self.assertFalse((tracker.root / "global_step_20").exists())
            self.assertFalse(list(tracker.root.glob(".pending-*")))

    def config(self):
        return OmegaConf.create({
            "trainer": {
                "default_local_dir": str(self.root), "default_hdfs_dir": None,
                "test_freq": 10, "val_before_train": True, "use_v1": False,
                "max_actor_ckpt_to_keep": 2,
                "best_checkpoint": {"enabled": True, "metric": METRIC, "mode": "max"},
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        })

    def test_disabled_hook_is_inert_and_invalid_config_fails_early(self):
        config = self.config()
        validate_best_checkpoint_config(config)
        config.actor_rollout_ref.actor.checkpoint.async_save = True
        with self.assertRaisesRegex(ValueError, "synchronous checkpoint"):
            validate_best_checkpoint_config(config)
        config.actor_rollout_ref.actor.checkpoint.async_save = False
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_best_checkpoint_config(config)
        config.trainer.best_checkpoint.enabled = False
        trainer = SimpleNamespace(config=config)
        self.assertEqual(maybe_save_best_checkpoint(trainer, {}), {})
        self.assertFalse((self.root / "best_checkpoint").exists())

    def test_native_and_opsd_save_paths_keep_best_outside_real_retention(self):
        from groove.verl_trainer import GrooveRayPPOTrainer
        from verl.trainer.ppo.v1 import PPOTrainerSync
        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        for trainer_type in (PPOTrainerSync, GrooveRayPPOTrainer):
            with self.subTest(trainer=trainer_type.__name__), TemporaryDirectory() as folder:
                trainer = trainer_type.__new__(trainer_type)
                trainer.config = self.config()
                trainer.config.trainer.default_local_dir = folder
                trainer.use_critic = False
                trainer.trainer_mode = "sync"
                trainer.train_dataloader = Mock()
                trainer.train_dataloader.state_dict.return_value = {"position": 16}
                trainer.checkpoint_manager = Mock()
                # Exercise VERL's actual retention implementation without a GPU model.
                retention = BaseCheckpointManager.__new__(BaseCheckpointManager)
                retention.previous_saved_paths = []

                def save_actor(path, remote, step, max_ckpt_to_keep):
                    retention.ensure_checkpoint_capacity(max_ckpt_to_keep)
                    path = Path(path)
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "model.pt").write_text(str(step))
                    (path / "optim.pt").write_text("optimizer")
                    retention.register_checkpoint(str(path), max_ckpt_to_keep)

                trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=Mock(side_effect=save_actor))
                # Initial validation and an off-cadence improvement trigger a save.
                trainer.global_steps = 0
                maybe_save_best_checkpoint(trainer, {METRIC: 0.7})
                trainer.global_steps = 5
                maybe_save_best_checkpoint(trainer, {METRIC: 0.9})
                self.assertEqual(trainer.actor_rollout_wg.save_checkpoint.call_count, 2)
                self.assertEqual(trainer.checkpoint_manager.sleep_replicas.call_count, 2)
                trainer.checkpoint_manager.update_weights.assert_called_with(5)
                # A scheduled save followed by evaluation must not serialize twice.
                for step, score in [(10, 0.8), (20, 0.85)]:
                    trainer.global_steps = step
                    trainer._save_checkpoint()
                    maybe_save_best_checkpoint(trainer, {METRIC: score})
                self.assertEqual(trainer.actor_rollout_wg.save_checkpoint.call_count, 4)
                self.assertFalse((Path(folder) / "global_step_5/actor").exists())
                best = Path(folder) / "best_checkpoint/global_step_5"
                self.assertEqual((best / "actor/model.pt").read_text(), "5")
                self.assertTrue((best / "actor/optim.pt").exists())
                self.assertTrue((best / "data.pt").exists())
                self.assertEqual((Path(folder) / "latest_checkpointed_iteration.txt").read_text(), "20")
                # Recreate the tracker as a resumed trainer would, retaining the threshold.
                del trainer._best_checkpoint_tracker
                self.assertEqual(maybe_save_best_checkpoint(trainer, {METRIC: 0.8})["checkpoint/best_step"], 5)
                trainer.global_steps = 30
                trainer._save_checkpoint()
                maybe_save_best_checkpoint(trainer, {METRIC: 0.95})
                self.assertEqual(trainer.actor_rollout_wg.save_checkpoint.call_count, 5)
                self.assertEqual(trainer.checkpoint_manager.sleep_replicas.call_count, 2)
                self.assertFalse(best.exists())
                self.assertEqual(
                    (Path(folder) / "best_checkpoint/global_step_30/actor/model.pt").read_text(), "30"
                )


if __name__ == "__main__":
    unittest.main()
