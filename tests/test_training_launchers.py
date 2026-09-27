from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingLauncherTest(unittest.TestCase):
    def test_both_launchers_select_plain_reasoning_and_preserve_objectives(self):
        self._check_launchers(override_data=False)

    def test_both_launchers_accept_an_existing_data_directory(self):
        self._check_launchers(override_data=True)

    def test_both_launchers_accept_an_independent_validation_file(self):
        self._check_launchers(override_data=False, override_validation=True)

    def test_all_modes_keep_repetition_processing_by_default(self):
        self._check_launchers(override_data=False, repetition_override=None)

    def test_all_modes_accept_a_separate_validation_rollout_directory(self):
        self._check_launchers(override_data=False, override_validation_dump=True)

    def test_teacher_evidence_accepts_crop_and_custom_blending(self):
        self._check_launchers(override_data=False, evidence_override=True)

    def _check_launchers(self, *, override_data, override_validation=False, repetition_override="false",
                         override_validation_dump=False, evidence_override=False):
        with TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            (project / "scripts").mkdir(parents=True)
            for name in ["train_siton_2gpu.sh"]:
                shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
            data_paths = {
                "grpo": project / "data/vstar_grpo_4000_seed20260917",
                "dapo": project / "data/vstar_grpo_4000_seed20260917",
                "grpo_opsd": project / "data/vstar_opsd_4000_seed20260917",
            }
            if override_data:
                data_paths = dict.fromkeys(data_paths, Path(folder) / "existing split")
            for data in set(data_paths.values()):
                data.mkdir(parents=True)
                (data / "train.parquet").touch()
            validation = project / "data/vstar_bench/validation.parquet"
            if override_validation:
                validation = Path(folder) / "custom validation.parquet"
            validation.parent.mkdir(parents=True, exist_ok=True)
            validation.touch()
            binary = Path(folder) / "bin" / "python"
            binary.parent.mkdir()
            binary.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'reward_env': {\n"
                "    key: value for key, value in os.environ.items()\n"
                "    if key.startswith(('GROOVE_JUDGE_', 'GROOVE_REPETITION_'))\n"
                "}}))\n"
            )
            binary.chmod(0o755)
            for name, enabled in [("grpo", "false"),
                                  ("dapo", "false"),
                                  ("grpo_opsd", "true")]:
                with self.subTest(launcher=name):
                    env = {
                        **os.environ, "PYTHON_BIN": str(binary),
                        "EXPERIMENT_NAME": "unit-launcher-" + name, "TRAINING_MODE": name,
                        "GROOVE_DRY_RUN": "true", "PYTHONPATH": str(ROOT / "src"),
                        "GROOVE_JUDGE_API_KEY": "unit-judge-key",
                        "GROOVE_REPETITION_MIN_REPEATS": "7",
                    }
                    for key in (
                        "DATA_DIR", "VALIDATION_FILE", "MODEL_PATH", "SEED", "N_GPUS", "CUDA_VISIBLE_DEVICES",
                        "VAL_BATCH_SIZE", "ROLLOUT_AGENT_NUM_WORKERS", "REWARD_NUM_WORKERS",
                        "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU", "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
                        "GROOVE_REPETITION_ZERO_REWARD",
                        "SAVE_BEST_CHECKPOINT", "BEST_CHECKPOINT_METRIC",
                        "VALIDATION_DATA_DIR",
                        "DAPO_MAX_INFLIGHT_GEN_BATCHES", "ROLLOUT_ENFORCE_EAGER", "STEP_TIMING_DIR",
                        "OPSD_ADVANTAGE_MODE", "RLSD_LAMBDA_INITIAL", "RLSD_LAMBDA_DECAY_STEPS",
                        "RLSD_CLIP_RANGE", "RLSD_TEACHER_SYNC_INTERVAL",
                        "TEACHER_EVIDENCE_MODE", "FOCUS_BLUR_ALPHA", "FOCUS_BLUR_RADIUS",
                    ):
                        env.pop(key, None)
                    if repetition_override is not None:
                        env["GROOVE_REPETITION_ZERO_REWARD"] = repetition_override
                    if evidence_override:
                        env.update(TEACHER_EVIDENCE_MODE="crop", FOCUS_BLUR_ALPHA="0.7", FOCUS_BLUR_RADIUS="9")
                    if override_data:
                        env["DATA_DIR"] = str(data_paths[name])
                    if override_validation:
                        env["VALIDATION_FILE"] = str(validation)
                    validation_dump = project / "outputs/validation" / ("unit-launcher-" + name)
                    if override_validation_dump:
                        validation_dump = Path(folder) / "all validation responses" / name
                        env["VALIDATION_DATA_DIR"] = str(validation_dump)
                    result = subprocess.run(
                        ["bash", str(project / "scripts/train_siton_2gpu.sh")], cwd=project,
                        env=env,
                        check=True, capture_output=True, text=True,
                    )
                    captured = json.loads(result.stdout)
                    args = captured["args"]
                    self.assertIn("data.response_format=reasoning_answer", args)
                    self.assertIn("data.apply_chat_template_kwargs.enable_thinking=false", args)
                    self.assertIn("groove.enabled=" + enabled, args)
                    self.assertIn("groove.advantage_mode=rlsd_positive", args)
                    self.assertIn("groove.teacher_evidence_mode=" + ("crop" if evidence_override else "focus"), args)
                    self.assertIn("groove.focus_blur_alpha=" + ("0.7" if evidence_override else "0.5"), args)
                    self.assertIn("groove.focus_blur_radius=" + ("9" if evidence_override else "12.0"), args)
                    self.assertIn("groove.rlsd_lambda_initial=0.5", args)
                    self.assertIn("groove.rlsd_lambda_decay_steps=50", args)
                    self.assertIn("groove.rlsd_clip_range=0.2", args)
                    self.assertIn("groove.rlsd_teacher_sync_interval=10", args)
                    self.assertIn("trainer.use_v1=" + ("false" if enabled == "true" else "true"), args)
                    self.assertIn("algorithm.filter_groups.enable=" + ("true" if name == "dapo" else "false"), args)
                    self.assertIn("trainer.best_checkpoint.enabled=true", args)
                    self.assertIn("trainer.best_checkpoint.metric=val-core/vstar_bench/reward/mean@1", args)
                    self.assertIn("trainer.best_checkpoint.mode=max", args)
                    self.assertIn(f"trainer.validation_data_dir={validation_dump}", args)
                    self.assertIn("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla", args)
                    self.assertIn("reward.custom_reward_function.reward_kwargs.format_reward_weight=0.2", args)
                    self.assertIn("reward.custom_reward_function.reward_kwargs.answer_reward_weight=1.0", args)
                    self.assertIn("++reward.reward_kwargs.overlong_buffer_cfg.enable=false", args)
                    self.assertIn("data.train_batch_size=16", args)
                    self.assertIn("actor_rollout_ref.rollout.n=8", args)
                    self.assertIn("data.max_response_length=1024", args)
                    self.assertIn("data.max_prompt_length=9216", args)
                    self.assertIn("actor_rollout_ref.rollout.max_model_len=10240", args)
                    self.assertIn("data.val_batch_size=8", args)
                    self.assertIn("actor_rollout_ref.rollout.agent.num_workers=8", args)
                    self.assertIn("reward.num_workers=1", args)
                    self.assertIn("algorithm.filter_groups.max_inflight_gen_batches=1", args)
                    self.assertIn("actor_rollout_ref.rollout.enforce_eager=true", args)
                    self.assertIn("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768", args)
                    self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=32768", args)
                    self.assertIn(f"data.train_files=['{data_paths[name] / 'train.parquet'}']", args)
                    self.assertIn(f"data.val_files=['{validation}']", args)
                    self.assertIn(
                        f"reward.custom_reward_function.path={project / 'src/groove/semantic_reward.py'}",
                        args,
                    )
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_API_KEY"], "unit-judge-key")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_MODEL"], "Qwen3.8-27B")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_BASE_URL"], "http://127.0.0.1:8002/v1")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_CONCURRENCY"], "128")
                    self.assertEqual(captured["reward_env"]["GROOVE_REPETITION_ZERO_REWARD"], repetition_override or "true")
                    self.assertEqual(captured["reward_env"]["GROOVE_REPETITION_MIN_REPEATS"], "7")


if __name__ == "__main__":
    unittest.main()
