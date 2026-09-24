from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from hydra import compose, initialize_config_dir
from verl.trainer.ppo.utils import need_reference_policy


ROOT = Path(__file__).resolve().parents[1]


class TwoGpuNoKlLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / 'project'
        (self.project / 'scripts').mkdir(parents=True)
        self.launcher = self.project / 'scripts/train_a800_2gpu_nokl.sh'
        shutil.copy2(ROOT / 'scripts/train_a800_2gpu_nokl.sh', self.launcher)
        for filename in ('data/vstar_grpo_4000_seed20260917/train.parquet',
                         'data/vstar_bench/validation.parquet'):
            path = self.project / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        binary = Path(self.temporary.name) / 'bin/python'
        binary.parent.mkdir()
        binary.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'print(json.dumps({"args":sys.argv[1:],'
                          '"gpus":os.environ["CUDA_VISIBLE_DEVICES"],'
                          '"wandb":os.environ["WANDB_MODE"]}))\n')
        binary.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items() if key in
                    ('PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH')}
        self.env.update(PYTHON_BIN=str(binary), GROOVE_DRY_RUN='true',
                        EXPERIMENT_NAME='unit-two-gpu-nokl', GROOVE_JUDGE_API_KEY='unit-key')

    def launch(self, overrides=None, args=()):
        return subprocess.run(['bash', str(self.launcher), *args], cwd=self.project,
                              env={**self.env, **(overrides or {})}, capture_output=True, text=True)

    def config(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = json.loads(result.stdout)
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
            return captured, compose(config_name='groove', overrides=captured['args'][2:])

    def test_two_gpu_training_uses_batch_32_and_removes_reference_policy(self):
        captured, config = self.config(self.launch())
        actor = config.actor_rollout_ref.actor
        rollout = config.actor_rollout_ref.rollout
        self.assertEqual(captured['gpus'], '1,2')
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.data.train_batch_size, 32)
        self.assertEqual(actor.ppo_mini_batch_size, 32)
        self.assertEqual(rollout.n, 8)
        self.assertEqual(actor.ppo_epochs, 1)
        self.assertEqual(actor.optim.lr, 1e-6)
        self.assertFalse(actor.use_kl_loss)
        self.assertEqual(actor.kl_loss_coef, 0)
        self.assertFalse(config.algorithm.use_kl_in_reward)
        self.assertEqual(config.algorithm.kl_ctrl.kl_coef, 0)
        self.assertFalse(need_reference_policy(config))
        self.assertEqual(config.algorithm.adv_estimator, 'grpo')
        self.assertTrue(config.algorithm.norm_adv_by_std_in_grpo)
        self.assertFalse(config.algorithm.filter_groups.enable)
        self.assertFalse(config.groove.enabled)
        self.assertTrue(config.trainer.use_v1)
        self.assertEqual(actor.ppo_max_token_len_per_gpu, 32768)
        self.assertEqual(rollout.max_num_batched_tokens, 32768)
        self.assertEqual(rollout.agent.num_workers, 16)
        self.assertEqual(config.reward.num_workers, 4)
        self.assertEqual(captured['wandb'], 'online')
        self.assertEqual(config.data.max_response_length, 1024)
        self.assertIsNone(config.data.val_batch_size)
        self.assertEqual(rollout.val_kwargs.temperature, 0)
        self.assertFalse(rollout.val_kwargs.do_sample)
        self.assertTrue(config.trainer.best_checkpoint.enabled)
        self.assertEqual(config.trainer.test_freq, 5)
        self.assertEqual(config.trainer.save_freq, 5)

    def test_rejects_wrong_gpu_count_and_duplicate_devices(self):
        for overrides in ({'N_GPUS': '4'}, {'CUDA_VISIBLE_DEVICES': '1,2,3'},
                          {'CUDA_VISIBLE_DEVICES': '1,1'}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.launch(overrides).returncode, 0)

    def test_rejects_non_grpo_modes(self):
        for mode in ('dapo', 'grpo_opsd'):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.launch({'TRAINING_MODE': mode}).returncode, 0)

    def test_cli_priority_and_quoted_paths_survive_array_expansion(self):
        path = str(self.project / 'validation outputs')
        captured, config = self.config(self.launch(
            {'VALIDATION_DATA_DIR': path}, ('actor_rollout_ref.actor.optim.lr=5e-7',)))
        self.assertEqual(config.trainer.validation_data_dir, path)
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 5e-7)
        self.assertEqual(captured['args'][-1], 'actor_rollout_ref.actor.optim.lr=5e-7')


if __name__ == '__main__':
    unittest.main()
