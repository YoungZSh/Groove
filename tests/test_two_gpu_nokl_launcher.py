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
from groove.objective import validate_objective_config
from groove.trainer_routing import trainer_backend


ROOT = Path(__file__).resolve().parents[1]


class TwoGpuNoKlLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / 'project'
        (self.project / 'scripts').mkdir(parents=True)
        self.launcher = self.project / 'scripts/train_a800_2gpu_nokl.sh'
        shutil.copy2(ROOT / 'scripts/train_a800_2gpu_nokl.sh', self.launcher)
        (self.project / '.env').write_text('OPENAI_MODEL=gemini-test\n')
        for filename in ('data/vstar_grpo_4000_seed20260917/train.parquet',
                         'data/vstar_opsd_4000_seed20260917/train.parquet',
                         'data/vstar_bench/validation.parquet'):
            path = self.project / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        binary = Path(self.temporary.name) / 'bin/python'
        binary.parent.mkdir()
        binary.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'print(json.dumps({"args":sys.argv[1:],'
                          '"gpus":os.environ["CUDA_VISIBLE_DEVICES"],'
                          '"nccl_cumem_host":os.environ["NCCL_CUMEM_HOST_ENABLE"],'
                          '"nccl_p2p_disable":os.environ["NCCL_P2P_DISABLE"],'
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

    def test_explicit_grpo_uses_batch_32_and_removes_reference_policy(self):
        captured, config = self.config(self.launch({'TRAINING_MODE': 'grpo'}))
        actor = config.actor_rollout_ref.actor
        rollout = config.actor_rollout_ref.rollout
        self.assertEqual(captured['gpus'], '0,3')
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.data.train_batch_size, 32)
        self.assertEqual(actor.ppo_mini_batch_size, 32)
        self.assertEqual(rollout.n, 8)
        self.assertEqual(actor.ppo_epochs, 1)
        self.assertEqual(actor.optim.lr, 1e-6)
        self.assertEqual(actor.entropy_coeff, 0.001)
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_PROVIDER, 'qwen')
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_BASE_URL,
                         'http://127.0.0.1:8005/v1')
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_MODEL, 'Qwen3.8-27B')
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_ENV_FILE,
                         str(self.project / '.env'))
        self.assertEqual(trainer_backend(config), 'verl_v1_sync')
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

    def test_default_rlsd_explicit_mode_and_alias_use_two_gpu_batch_32_without_reference_kl(self):
        for mode in (None, 'grpo_opsd', 'groove'):
            with self.subTest(mode=mode):
                overrides = {} if mode is None else {'TRAINING_MODE': mode}
                captured, config = self.config(self.launch(overrides))
                validate_objective_config(config)
                self.assertEqual(trainer_backend(config), 'groove_opsd')
                self.assertEqual(config.actor_rollout_ref.model.path,
                                 '/data/home/yangzesheng/models/ckpts/Qwen3.5-2B')
                actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
                self.assertEqual(captured['gpus'], '0,3')
                self.assertEqual(captured['nccl_cumem_host'], '0')
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_HOST_ENABLE, '0')
                self.assertEqual(captured['nccl_p2p_disable'], '1')
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE, '1')
                self.assertEqual(config.trainer.n_gpus_per_node, 2)
                self.assertEqual(config.data.train_batch_size, 32)
                self.assertEqual(actor.ppo_mini_batch_size, 32)
                self.assertEqual(rollout.n, 8)
                self.assertEqual(config.data.train_batch_size * rollout.n, 256)
                self.assertEqual(actor.optim.lr, 1e-6)
                self.assertEqual(actor.entropy_coeff, 0)
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_PROVIDER, 'qwen')
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_BASE_URL,
                                 'http://127.0.0.1:8002/v1')
                self.assertEqual(actor.ppo_epochs, 1)
                self.assertEqual(actor.clip_ratio_low, .2)
                self.assertEqual(actor.clip_ratio_high, .2)
                self.assertFalse(actor.use_kl_loss)
                self.assertEqual(actor.kl_loss_coef, 0)
                self.assertFalse(config.algorithm.use_kl_in_reward)
                self.assertEqual(config.algorithm.kl_ctrl.kl_coef, 0)
                self.assertFalse(need_reference_policy(config))
                self.assertTrue(config.groove.enabled)
                self.assertEqual(config.groove.advantage_mode, 'rlsd_positive')
                self.assertEqual(config.groove.teacher_evidence_mode, 'focus')
                self.assertEqual(config.groove.focus_blur_alpha, 0.5)
                self.assertEqual(config.groove.focus_blur_radius, 12.0)
                self.assertEqual(config.groove.rlsd_lambda_initial, .5)
                self.assertEqual(config.groove.rlsd_lambda_decay_steps, 40)
                self.assertEqual(config.groove.rlsd_clip_range, .2)
                self.assertEqual(config.groove.rlsd_teacher_sync_interval, 5)
                self.assertFalse(config.trainer.use_v1)
                self.assertFalse(config.algorithm.filter_groups.enable)
                self.assertEqual(captured['wandb'], 'offline')
                self.assertEqual(config.data.train_files,
                                 [str(self.project / 'data/vstar_opsd_4000_seed20260917/train.parquet')])
                self.assertEqual(config.data.val_files,
                                 [str(self.project / 'data/vstar_bench/validation.parquet')])
                self.assertEqual(actor.ppo_max_token_len_per_gpu, 32768)
                self.assertTrue(rollout.log_prob_use_dynamic_bsz)
                self.assertEqual(rollout.log_prob_max_token_len_per_gpu, 32768)
                self.assertEqual(rollout.max_num_batched_tokens, 32768)
                self.assertEqual(rollout.agent.num_workers, 16)
                self.assertEqual(config.reward.num_workers, 4)
                self.assertEqual(config.data.max_response_length, 1024)
                self.assertIsNone(config.data.val_batch_size)
                self.assertEqual(config.trainer.test_freq, 5)
                self.assertEqual(config.trainer.save_freq, 5)
                self.assertTrue(config.trainer.best_checkpoint.enabled)

    def test_rlsd_overrides_and_additive_comparison_keep_no_kl_profile(self):
        for advantage_mode in ('rlsd_positive', 'opsd'):
            with self.subTest(advantage_mode=advantage_mode):
                _, config = self.config(self.launch({
                    'TRAINING_MODE': 'grpo_opsd', 'OPSD_ADVANTAGE_MODE': advantage_mode,
                    'RLSD_LAMBDA_INITIAL': '0.3', 'RLSD_LAMBDA_DECAY_STEPS': '60',
                    'RLSD_CLIP_RANGE': '0.1', 'RLSD_TEACHER_SYNC_INTERVAL': '20',
                }, ('groove.rlsd_teacher_sync_interval=5',)))
                validate_objective_config(config)
                self.assertEqual(config.groove.advantage_mode, advantage_mode)
                self.assertEqual(config.groove.rlsd_lambda_initial, .3)
                self.assertEqual(config.groove.rlsd_lambda_decay_steps, 60)
                self.assertEqual(config.groove.rlsd_clip_range, .1)
                self.assertEqual(config.groove.rlsd_teacher_sync_interval, 5)
                self.assertEqual(config.data.train_batch_size, 32)
                self.assertFalse(need_reference_policy(config))

    def test_teacher_evidence_switch_and_cli_priority(self):
        for mode in ('crop', 'focus'):
            with self.subTest(mode=mode):
                _, config = self.config(self.launch({'TEACHER_EVIDENCE_MODE': mode,
                    'FOCUS_BLUR_ALPHA': '0.7', 'FOCUS_BLUR_RADIUS': '9'}))
                validate_objective_config(config)
                self.assertEqual(config.groove.teacher_evidence_mode, mode)
                self.assertEqual(config.groove.focus_blur_alpha, 0.7)
                self.assertEqual(config.groove.focus_blur_radius, 9)
        _, config = self.config(self.launch({'TEACHER_EVIDENCE_MODE': 'focus'},
            ('groove.teacher_evidence_mode=crop', 'groove.focus_blur_alpha=0.25')))
        self.assertEqual(config.groove.teacher_evidence_mode, 'crop')
        self.assertEqual(config.groove.focus_blur_alpha, 0.25)

    def test_rejects_unsupported_modes(self):
        for mode in ('dapo', 'unknown'):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.launch({'TRAINING_MODE': mode}).returncode, 0)

    def test_grpo_optimizer_environment_overrides_and_cli_priority(self):
        overrides = {'TRAINING_MODE': 'grpo', 'LEARNING_RATE': '1e-6', 'ENTROPY_COEFF': '0'}
        _, config = self.config(self.launch(overrides))
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1e-6)
        self.assertEqual(config.actor_rollout_ref.actor.entropy_coeff, 0)
        captured, config = self.config(self.launch(overrides, (
            'actor_rollout_ref.actor.optim.lr=3e-7',
            'actor_rollout_ref.actor.entropy_coeff=0.002',
        )))
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 3e-7)
        self.assertEqual(config.actor_rollout_ref.actor.entropy_coeff, 0.002)
        self.assertEqual(captured['args'][-1], 'actor_rollout_ref.actor.entropy_coeff=0.002')
        self.assertFalse(need_reference_policy(config))

    def test_judge_provider_override_and_missing_gemini_credentials_file(self):
        _, config = self.config(self.launch({'TRAINING_MODE': 'grpo', 'GROOVE_JUDGE_PROVIDER': 'gemini',
                                             'LEARNING_RATE': '5e-7'}))
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.GROOVE_JUDGE_PROVIDER, 'gemini')
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 5e-7)
        self.assertEqual(config.actor_rollout_ref.actor.entropy_coeff, .001)
        result = self.launch({'TRAINING_MODE': 'grpo', 'GROOVE_JUDGE_PROVIDER': 'gemini',
                              'GROOVE_JUDGE_ENV_FILE': '/missing/judge.env'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing Gemini Judge env file', result.stderr)

    def test_qwen_endpoint_and_model_overrides_reach_workers_without_credentials(self):
        _, config = self.config(self.launch({'TRAINING_MODE': 'grpo',
            'GROOVE_JUDGE_BASE_URL': 'http://127.0.0.1:9005/v1',
            'GROOVE_JUDGE_MODEL': 'test-qwen'}))
        worker_env = config.ray_kwargs.ray_init.runtime_env.env_vars
        self.assertEqual(worker_env.GROOVE_JUDGE_BASE_URL, 'http://127.0.0.1:9005/v1')
        self.assertEqual(worker_env.GROOVE_JUDGE_MODEL, 'test-qwen')
        self.assertNotIn('GROOVE_JUDGE_API_KEY', worker_env)

    def test_cli_priority_and_quoted_paths_survive_array_expansion(self):
        path = str(self.project / 'validation outputs')
        model_path = str(self.project / 'model weights')
        captured, config = self.config(self.launch(
            {'VALIDATION_DATA_DIR': path, 'MODEL_PATH': model_path},
            ('actor_rollout_ref.actor.optim.lr=5e-7',)))
        self.assertEqual(config.trainer.validation_data_dir, path)
        self.assertEqual(config.actor_rollout_ref.model.path, model_path)
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 5e-7)
        self.assertEqual(captured['args'][-1], 'actor_rollout_ref.actor.optim.lr=5e-7')


if __name__ == '__main__':
    unittest.main()
