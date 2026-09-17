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


ROOT = Path(__file__).resolve().parents[1]


class FourGpuLauncherTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.project = Path(self.folder.name) / "project"
        (self.project / "scripts").mkdir(parents=True)
        for name in ("run_2b_4gpu.sh", "run_grpo_2b.sh", "run_grpo_opsd_2b.sh", "run_groove.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.project / "scripts" / name)
        for name in ("vstar_grpo_2200_seed20260904", "vstar_opsd_2200_seed20260904"):
            self.make_data(self.project / "data" / name)
        self.validation = self.project / "data/vstar_bench/validation.parquet"
        self.validation.parent.mkdir(parents=True)
        self.validation.touch()
        self.binary = Path(self.folder.name) / "bin/python"
        self.binary.parent.mkdir()
        self.binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if sys.argv[1:] == ['-']:\n"
            "    sys.stdin.read()\n"
            "    sys.exit(int(os.environ.get('TEST_GPU_CHECK_EXIT', '0')))\n"
            "keys = ('CUDA_VISIBLE_DEVICES', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE',\n"
            "        'NO_PROXY', 'no_proxy', 'WANDB_MODE', 'RAY_NODE_MEMORY_CAP_GIB')\n"
            "print(json.dumps({'args': sys.argv[1:],\n"
            "                  'env': {key: os.environ.get(key) for key in keys}}))\n"
            "sys.exit(int(os.environ.get('TEST_TRAINING_EXIT', '0')))\n"
        )
        self.binary.chmod(0o755)
        self.env = dict(os.environ)
        for key in (
            "DATA_DIR", "VALIDATION_FILE", "MODEL_PATH", "SEED", "N_GPUS", "CUDA_VISIBLE_DEVICES",
            "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NO_PROXY", "no_proxy",
            "TRAINING_MODE", "TEST_GPU_CHECK_EXIT", "TEST_TRAINING_EXIT",
            "VAL_BATCH_SIZE", "ROLLOUT_AGENT_NUM_WORKERS", "REWARD_NUM_WORKERS", "OMP_NUM_THREADS",
            "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU", "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
            "WANDB_MODE", "RAY_NODE_MEMORY_CAP_GIB",
        ):
            self.env.pop(key, None)
        self.env.update(
            PYTHON_BIN=str(self.binary), GROOVE_DRY_RUN="true",
            EXPERIMENT_NAME="unit-four-gpu", GROOVE_JUDGE_API_KEY="unit-judge-key",
        )

    def make_data(self, path):
        path.mkdir(parents=True)
        (path / "train.parquet").touch()

    def run_launcher(self, *, overrides=None, args=()):
        return subprocess.run(
            ["bash", str(self.project / "scripts/run_2b_4gpu.sh"), *args],
            cwd=self.project, env={**self.env, **(overrides or {})},
            capture_output=True, text=True,
        )

    def resolved_config(self, captured):
        with initialize_config_dir(
            version_base=None, config_dir=str(ROOT / "src/verl/trainer/config")
        ):
            return compose(config_name="groove", overrides=captured["args"][2:])

    def test_both_modes_resolve_four_gpus_and_matching_training_settings(self):
        for mode, enabled, split in (
            ("grpo", False, "vstar_grpo_2200_seed20260904"),
            ("grpo_opsd", True, "vstar_opsd_2200_seed20260904"),
        ):
            with self.subTest(mode=mode):
                result = self.run_launcher(overrides={"TRAINING_MODE": mode})
                self.assertEqual(result.returncode, 0, result.stderr)
                captured = json.loads(result.stdout)
                config = self.resolved_config(captured)
                self.assertEqual(config.trainer.n_gpus_per_node, 4)
                self.assertEqual(config.trainer.nnodes, 1)
                self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertEqual(captured["env"]["RAY_NODE_MEMORY_CAP_GIB"], "null")
                self.assertEqual(config.groove.enabled, enabled)
                self.assertEqual(config.data.train_batch_size, 16)
                self.assertEqual(config.data.max_response_length, 1024)
                self.assertEqual(config.data.response_format, "reasoning_answer")
                self.assertFalse(config.data.apply_chat_template_kwargs.enable_thinking)
                self.assertEqual(config.data.train_files, [str(self.project / "data" / split / "train.parquet")])
                self.assertEqual(config.data.val_files, [str(self.validation)])
                self.assertIsNone(config.data.val_batch_size)
                self.assertEqual(config.data.val_max_samples, -1)
                self.assertEqual(config.data.max_prompt_length, 9216)
                actor = config.actor_rollout_ref.actor
                rollout = config.actor_rollout_ref.rollout
                self.assertEqual(actor.ppo_mini_batch_size, 16)
                self.assertTrue(actor.use_dynamic_bsz)
                self.assertEqual(actor.ppo_max_token_len_per_gpu, 65536)
                self.assertEqual(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu, 65536)
                self.assertEqual(rollout.log_prob_max_token_len_per_gpu, 65536)
                self.assertEqual(rollout.max_num_batched_tokens, 65536)
                self.assertEqual(rollout.agent.num_workers, 16)
                self.assertEqual(config.reward.num_workers, 4)
                self.assertEqual(actor.policy_loss.loss_mode, "vanilla")
                self.assertEqual(actor.loss_agg_mode, "token-mean")
                self.assertEqual(actor.optim.lr, 1e-6)
                self.assertEqual(actor.kl_loss_coef, 0.01)
                self.assertEqual(rollout.n, 8)
                self.assertEqual(rollout.max_model_len, 10240)
                self.assertEqual(rollout.tensor_model_parallel_size, 1)
                self.assertEqual(rollout.data_parallel_size, 1)
                self.assertEqual(rollout.pipeline_model_parallel_size, 1)
                self.assertEqual(config.actor_rollout_ref.model.path, "/root/siton-tmp/yzs/ckpts/Qwen3.5-2B")
                self.assertEqual(rollout.temperature, 1.0)
                self.assertEqual(rollout.val_kwargs.temperature, 0)
                self.assertFalse(rollout.val_kwargs.do_sample)
                self.assertEqual(config.data.seed, 20260904)
                self.assertEqual(rollout.seed, config.data.seed)
                self.assertEqual(actor.data_loader_seed, config.data.seed)
                self.assertEqual(actor.fsdp_config.seed, config.data.seed)
                self.assertEqual(captured["env"]["WANDB_MODE"], "offline" if enabled else "online")
                worker_env = config.ray_kwargs.ray_init.runtime_env.env_vars
                self.assertEqual(worker_env.NCCL_SOCKET_IFNAME, "lo")
                self.assertEqual(worker_env.NCCL_IB_DISABLE, "1")
                self.assertEqual(worker_env.OMP_NUM_THREADS, "4")
                for key in ("NO_PROXY", "no_proxy"):
                    self.assertTrue({"127.0.0.1", "localhost", "::1"}.issubset(set(worker_env[key].split(","))))
                if enabled:
                    self.assertEqual(config.groove.opsd_advantage_coef, 0.01)
                    self.assertIsNone(config.groove.opsd_advantage_clip)
        self.assertFalse((self.project / "outputs").exists())

    def test_grpo_can_explicitly_keep_wandb_offline(self):
        result = self.run_launcher(overrides={"TRAINING_MODE": "grpo", "WANDB_MODE": "offline"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["env"]["WANDB_MODE"], "offline")

    def test_explicit_devices_model_seed_data_and_network_reach_both_modes(self):
        data = Path(self.folder.name) / "existing split"
        self.make_data(data)
        for mode in ("grpo", "grpo_opsd"):
            with self.subTest(mode=mode):
                result = self.run_launcher(overrides={
                    "TRAINING_MODE": mode, "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                    "MODEL_PATH": "/models/merged-grpo", "SEED": "77", "DATA_DIR": str(data),
                    "NCCL_SOCKET_IFNAME": "=eth0,eth1", "NCCL_IB_DISABLE": "0",
                    "NO_PROXY": "existing.internal", "no_proxy": "lower.internal",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                captured = json.loads(result.stdout)
                config = self.resolved_config(captured)
                self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
                self.assertEqual(config.actor_rollout_ref.model.path, "/models/merged-grpo")
                self.assertEqual(config.data.seed, 77)
                self.assertEqual(config.data.train_files, [str(data / "train.parquet")])
                self.assertEqual(config.data.val_files, [str(self.validation)])
                worker_env = config.ray_kwargs.ray_init.runtime_env.env_vars
                self.assertEqual(worker_env.NCCL_SOCKET_IFNAME, "=eth0,eth1")
                self.assertEqual(worker_env.NCCL_IB_DISABLE, "0")
                for key in ("NO_PROXY", "no_proxy"):
                    self.assertTrue({"existing.internal", "lower.internal"}.issubset(set(worker_env[key].split(","))))

    def test_requires_an_explicit_experiment_name(self):
        result = self.run_launcher(overrides={"EXPERIMENT_NAME": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set a new EXPERIMENT_NAME", result.stderr)

    def test_worker_and_token_budgets_can_be_tuned_without_changing_global_batch(self):
        for mode in ("grpo", "grpo_opsd"):
            with self.subTest(mode=mode):
                result = self.run_launcher(overrides={
                    "TRAINING_MODE": mode, "ROLLOUT_AGENT_NUM_WORKERS": "8",
                    "REWARD_NUM_WORKERS": "3", "OMP_NUM_THREADS": "2",
                    "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "98304",
                    "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "49152",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                config = self.resolved_config(json.loads(result.stdout))
                self.assertEqual(config.data.train_batch_size, 16)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 16)
                self.assertEqual(config.actor_rollout_ref.rollout.n, 8)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu, 98304)
                self.assertEqual(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu, 98304)
                self.assertEqual(config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu, 98304)
                self.assertEqual(config.actor_rollout_ref.rollout.max_num_batched_tokens, 49152)
                self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 8)
                self.assertIsNone(config.data.val_batch_size)
                self.assertEqual(config.reward.num_workers, 3)
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS, "2")

    def test_explicit_validation_batch_is_independent_of_worker_count(self):
        for mode in ("grpo", "grpo_opsd"):
            with self.subTest(mode=mode):
                result = self.run_launcher(overrides={
                    "TRAINING_MODE": mode, "ROLLOUT_AGENT_NUM_WORKERS": "8",
                    "VAL_BATCH_SIZE": "64",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                config = self.resolved_config(json.loads(result.stdout))
                self.assertEqual(config.data.val_batch_size, 64)
                self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 8)
                self.assertEqual(config.data.train_batch_size, 16)

    def test_rejects_invalid_mode_and_gpu_topology_before_launching(self):
        cases = (
            ({"TRAINING_MODE": "unknown"}, ()),
            ({"N_GPUS": "2"}, ()),
            ({"CUDA_VISIBLE_DEVICES": "0,1"}, ()),
            ({"CUDA_VISIBLE_DEVICES": "0,1,1,3"}, ()),
            ({"CUDA_VISIBLE_DEVICES": "0,1,,3"}, ()),
            ({}, ("trainer.nnodes=2",)),
            ({}, ("trainer.n_gpus_per_node=2",)),
        )
        for overrides, args in cases:
            with self.subTest(overrides=overrides, args=args):
                result = self.run_launcher(overrides=overrides, args=args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_failed_cuda_check_stops_before_training_and_log_creation(self):
        result = self.run_launcher(overrides={
            "GROOVE_DRY_RUN": "false", "TEST_GPU_CHECK_EXIT": "31",
        })
        self.assertEqual(result.returncode, 31)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.project / "outputs").exists())

    def test_training_log_is_preserved_and_failure_exit_code_propagates(self):
        log = self.project / "outputs/logs/unit-four-gpu.log"
        log.parent.mkdir(parents=True)
        log.write_text("existing diagnostic\n")
        result = self.run_launcher(overrides={
            "GROOVE_DRY_RUN": "false", "TEST_TRAINING_EXIT": "7",
        })
        self.assertEqual(result.returncode, 7)
        self.assertEqual(log.read_text(), "existing diagnostic\n" + result.stdout)
        self.assertEqual(json.loads(result.stdout)["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3")


if __name__ == "__main__":
    unittest.main()
