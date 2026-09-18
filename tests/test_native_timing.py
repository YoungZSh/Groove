import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.timing import agent_loop_timing_metrics, append_step_timing


class NativeTimingTest(unittest.TestCase):
    def test_request_timings_exclude_padding_and_keep_stages_separate(self):
        result = agent_loop_timing_metrics([
            {"generate_sequences": 10, "compute_score": 2},
            {"generate_sequences": 20, "compute_score": 4},
            {"generate_sequences": 1000, "compute_score": 1000},
        ], [True, True, False])
        self.assertEqual(result["timing_s/retained_agent/generate_sequences/mean"], 15)
        self.assertEqual(result["timing_s/retained_agent/compute_score/mean"], 3)
        self.assertAlmostEqual(result["timing_s/retained_agent/compute_score/p95"], 3.9)
        self.assertEqual(agent_loop_timing_metrics([{}], [False]), {})
        with self.assertRaises(ValueError):
            agent_loop_timing_metrics([{}], [])

    def test_step_record_includes_both_loggers_and_final_step_without_extra_wandb_calls(self):
        from verl.trainer.ppo.v1 import trainer_base as module
        from verl.trainer.ppo.v1 import PPOTrainerSync

        clock = [0.0]

        @contextmanager
        def timer(name, timings, **kwargs):
            start = clock[0]
            yield
            timings[name] = clock[0] - start

        def consume(seconds):
            def callback(*args, **kwargs):
                clock[0] += seconds
            return callback

        with TemporaryDirectory() as folder:
            trainer = PPOTrainerSync.__new__(PPOTrainerSync)
            trainer.config = OmegaConf.create({"trainer": {"step_timing_dir": folder, "logger": ["wandb"],
                                                           "rollout_data_dir": "rollouts"}})
            trainer.global_steps = 5
            trainer.timing_raw = {"step": 30.0}
            clock[0] = 30.0
            trainer._compute_metrics = Mock(side_effect=consume(2))

            def dump(*args):
                clock[0] += 3
                trainer.timing_raw["dump_rollout_generations"] = 3.0

            trainer._log_rollout_data = Mock(side_effect=dump)
            trainer.logger = Mock()
            trainer.logger.log.side_effect = consume(7)
            trainer.dapo_filtered_reward_logger = Mock()
            trainer.dapo_filtered_reward_logger.log.side_effect = consume(11)
            batch = SimpleNamespace(keys=["one"], partition_id="train")
            metrics = {module.DAPO_FILTERED_REWARD_COUNTS_KEY: {1.0: 8}}
            with patch.object(module, "marked_timer", timer), \
                 patch.object(module.time, "perf_counter", side_effect=lambda: clock[0]), \
                 patch.object(module.tq, "kv_clear", side_effect=consume(5)), patch("builtins.print"):
                trainer._finish_step_logging(batch, metrics, epoch=0, iteration_started=0.0)

            trainer.logger.log.assert_called_once()
            trainer.dapo_filtered_reward_logger.log.assert_called_once_with(["wandb"], {1.0: 8}, 5)
            self.assertEqual(metrics["timing_s/compute_metrics"], 2)
            self.assertEqual(metrics["timing_s/dump_rollout_generations"], 3)
            self.assertEqual(metrics["timing_s/clear_transfer_queue"], 5)
            self.assertNotIn("timing_s/log_metrics", metrics)
            rows = [json.loads(line) for line in (Path(folder) / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["step"], 5)
            timings = rows[0]["timing_s"]
            self.assertEqual(timings["log_metrics"], 7)
            self.assertEqual(timings["log_filtered_reward_table"], 11)
            self.assertEqual(timings["iteration"], 58)
            append_step_timing(folder, 6, {"iteration": 1.0})
            self.assertEqual(len((Path(folder) / "steps.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
