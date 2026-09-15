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

    def _check_launchers(self, *, override_data):
        with TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            (project / "scripts").mkdir(parents=True)
            for name in ["run_groove.sh", "run_grpo_2b.sh", "run_grpo_opsd_2b.sh"]:
                shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
            data_paths = {
                "run_grpo_2b.sh": project / "data/vstar_grpo_2200_seed20260904",
                "run_grpo_opsd_2b.sh": project / "data/vstar_opsd_2200_seed20260904",
            }
            if override_data:
                data_paths = dict.fromkeys(data_paths, Path(folder) / "existing split")
            for data in set(data_paths.values()):
                data.mkdir(parents=True)
                for split in ["train.parquet", "validation.parquet"]:
                    (data / split).touch()
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
            for name, enabled in [("run_grpo_2b.sh", "false"),
                                  ("run_grpo_opsd_2b.sh", "true")]:
                with self.subTest(launcher=name):
                    env = {
                        **os.environ, "PYTHON_BIN": str(binary),
                        "EXPERIMENT_NAME": "unit-launcher-" + name,
                        "GROOVE_DRY_RUN": "true", "PYTHONPATH": str(ROOT / "src"),
                        "GROOVE_JUDGE_API_KEY": "unit-judge-key",
                        "GROOVE_REPETITION_ZERO_REWARD": "false",
                        "GROOVE_REPETITION_MIN_REPEATS": "7",
                    }
                    env.pop("DATA_DIR", None)
                    if override_data:
                        env["DATA_DIR"] = str(data_paths[name])
                    result = subprocess.run(
                        ["bash", str(project / "scripts" / name)], cwd=project,
                        env=env,
                        check=True, capture_output=True, text=True,
                    )
                    captured = json.loads(result.stdout)
                    args = captured["args"]
                    self.assertIn("data.response_format=reasoning_answer", args)
                    self.assertIn("data.apply_chat_template_kwargs.enable_thinking=false", args)
                    self.assertIn("groove.enabled=" + enabled, args)
                    self.assertIn("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla", args)
                    self.assertIn("reward.custom_reward_function.reward_kwargs.format_reward_weight=0.2", args)
                    self.assertIn("data.train_batch_size=16", args)
                    self.assertIn("actor_rollout_ref.rollout.n=8", args)
                    self.assertIn("data.max_response_length=1024", args)
                    self.assertIn(f"data.train_files=['{data_paths[name] / 'train.parquet'}']", args)
                    self.assertIn(f"data.val_files=['{data_paths[name] / 'validation.parquet'}']", args)
                    self.assertIn(
                        f"reward.custom_reward_function.path={project / 'src/groove/semantic_reward.py'}",
                        args,
                    )
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_API_KEY"], "unit-judge-key")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_MODEL"], "Qwen3.8-27B")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_BASE_URL"], "http://127.0.0.1:8002/v1")
                    self.assertEqual(captured["reward_env"]["GROOVE_JUDGE_CONCURRENCY"], "128")
                    self.assertEqual(captured["reward_env"]["GROOVE_REPETITION_ZERO_REWARD"], "false")
                    self.assertEqual(captured["reward_env"]["GROOVE_REPETITION_MIN_REPEATS"], "7")


if __name__ == "__main__":
    unittest.main()
