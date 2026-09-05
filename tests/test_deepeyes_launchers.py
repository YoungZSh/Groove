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


class DeepEyesLauncherTest(unittest.TestCase):
    def test_both_launchers_select_plain_reasoning_and_preserve_objectives(self):
        with TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            (project / "scripts").mkdir(parents=True)
            for name in ["run_groove.sh", "run_grpo_2b.sh", "run_deepeyes_vstar_opsd_2b.sh"]:
                shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
            for name in ["deepeyes_vstar_grpo_2200_seed20260904", "deepeyes_vstar_opsd_2200_seed20260904"]:
                data = project / "data" / name
                data.mkdir(parents=True)
                for split in ["train.parquet", "validation.parquet"]:
                    (data / split).touch()
            binary = Path(folder) / "bin" / "python"
            binary.parent.mkdir()
            binary.write_text(
                f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            binary.chmod(0o755)
            for name, enabled in [("run_grpo_2b.sh", "false"),
                                  ("run_deepeyes_vstar_opsd_2b.sh", "true")]:
                with self.subTest(launcher=name):
                    result = subprocess.run(
                        ["bash", str(project / "scripts" / name)], cwd=project,
                        env={**os.environ, "PYTHON_BIN": str(binary),
                             "EXPERIMENT_NAME": "unit-launcher-" + name,
                             "GROOVE_DRY_RUN": "true", "PYTHONPATH": str(ROOT / "src")},
                        check=True, capture_output=True, text=True,
                    )
                    args = json.loads(result.stdout)
                    self.assertIn("data.response_format=reasoning_answer", args)
                    self.assertIn("data.apply_chat_template_kwargs.enable_thinking=false", args)
                    self.assertIn("groove.enabled=" + enabled, args)
                    self.assertIn("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla", args)
                    self.assertIn("reward.custom_reward_function.reward_kwargs.format_reward_weight=0.2", args)
                    self.assertIn("data.train_batch_size=16", args)
                    self.assertIn("actor_rollout_ref.rollout.n=8", args)
                    self.assertIn("data.max_response_length=1024", args)


if __name__ == "__main__":
    unittest.main()
