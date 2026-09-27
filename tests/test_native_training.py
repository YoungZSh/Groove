from __future__ import annotations

import asyncio
from importlib.util import find_spec
from unittest.mock import MagicMock, Mock, patch
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from groove.reward_manager import VisualQARewardManager
from groove.semantic_reward import compute_score
from groove.trainer_routing import task_runner_class, trainer_backend
from verl import DataProto
from verl.trainer.ppo.reward_metrics import reward_columns, reward_extra_metrics


class TrainerRoutingTest(unittest.TestCase):
    def config(self, *, opsd=False, v1=True, filtering=False):
        return OmegaConf.create({
            "groove": {"enabled": opsd},
            "trainer": {"use_v1": v1, "v1": {"trainer_mode": "sync"}},
            "algorithm": {"filter_groups": {"enable": filtering}},
        })

    def test_pure_modes_use_native_runner_and_joint_mode_uses_opsd_runner(self):
        from verl.trainer.main_ppo import TaskRunnerV1
        from verl.trainer.main_ppo_v0 import TaskRunner

        with patch("groove.trainer_routing.find_spec", return_value=object()):
            for filtering in (False, True):
                backend = trainer_backend(self.config(filtering=filtering))
                self.assertEqual(backend, "verl_v1_sync")
                self.assertIs(task_runner_class(backend), TaskRunnerV1)
        self.assertIs(task_runner_class(trainer_backend(self.config(v1=False))), TaskRunner)
        backend = trainer_backend(self.config(opsd=True, v1=False))
        self.assertEqual(backend, "groove_opsd")
        self.assertEqual(task_runner_class(backend).__ray_metadata__.modified_class.__name__, "GrooveTaskRunner")

    def test_unsupported_filtering_and_opsd_combinations_fail_before_ray(self):
        for config in (
            self.config(v1=False, filtering=True),
            self.config(opsd=True, v1=True),
            self.config(opsd=True, v1=False, filtering=True),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                trainer_backend(config)

    def test_missing_native_dependency_and_wrong_runtime_fail_early(self):
        with patch("groove.trainer_routing.find_spec", return_value=None), self.assertRaisesRegex(RuntimeError, "TransferQueue"):
            trainer_backend(self.config())
        config = self.config()
        config.trainer.v1.trainer_mode = "colocate_async"
        with self.assertRaisesRegex(ValueError, "sync"):
            trainer_backend(config)


class VisualQARewardManagerTest(unittest.IsolatedAsyncioTestCase):
    def config(self, *, enabled=True, buffer=128, penalty=1.0):
        return OmegaConf.create({"reward": {"reward_kwargs": {
            "max_resp_len": 1024,
            "overlong_buffer_cfg": {"enable": enabled, "len": buffer, "penalty_factor": penalty},
        }}})

    def batch(self, length=950, *, validation=False, split=None):
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones((1, 1), dtype=torch.long),
                "responses": torch.ones((1, length), dtype=torch.long),
                "attention_mask": torch.ones((1, length + 1), dtype=torch.long),
            },
            non_tensors={
                "data_source": np.array(["vstar_bench" if validation else "vstar_grpo"], dtype=object),
                "reward_model": np.array([{"ground_truth": "D"}], dtype=object),
                "extra_info": np.array([{
                    "question": "What material?", "choices": {"A": "rubber", "D": "leather"},
                    "split": split or ("validation" if validation else "train"),
                }], dtype=object),
            },
        )

    def manager(self, config=None, scorer=None):
        tokenizer = Mock()
        tokenizer.decode.return_value = "<answer>D</answer>"
        return VisualQARewardManager(
            config or self.config(), tokenizer,
            scorer or (lambda **kwargs: {"score": 0.8, "accuracy": 1.0, "format_reward": -1.0}),
        )

    async def test_training_length_shaping_preserves_accuracy_and_filter_uses_actual_reward(self):
        manager = self.manager()
        for length, penalty in ((128, 0), (896, 0), (950, -0.421875), (1024, -1)):
            with self.subTest(length=length):
                result = await manager.run_single(self.batch(length))
                self.assertAlmostEqual(result["reward_score"], 0.8 + penalty)
                extra = result["reward_extra_info"]
                self.assertEqual(extra["score"], 0.8)
                self.assertEqual(extra["accuracy"], 1.0)
                self.assertEqual(extra["format_reward"], -1.0)
                self.assertEqual(extra["overlong_reward"], penalty)
                self.assertEqual(extra["training_reward"], result["reward_score"])

    async def test_benchmark_uses_semantic_judge_without_length_or_training_penalties(self):
        with patch("groove.semantic_reward._judge_one", return_value={"score": 1.0, "accuracy": 1.0}) as judge:
            result = await self.manager(scorer=compute_score).run_single(self.batch(1024, validation=True))
        self.assertEqual(result["reward_score"], 1.0)
        self.assertEqual(result["reward_extra_info"]["overlong_reward"], 0.0)
        self.assertEqual(result["reward_extra_info"]["semantic_judge"], 1.0)
        self.assertFalse(judge.call_args.kwargs["apply_training_shaping"])
        self.assertEqual(judge.call_args.kwargs["format_reward_weight"], 0.0)

    async def test_validation_and_training_can_share_a_manager_concurrently(self):
        manager = self.manager()
        train, val = await asyncio.gather(
            manager.run_single(self.batch(1024)),
            manager.run_single(self.batch(1024, split="validation")),
        )
        self.assertAlmostEqual(train["reward_score"], -0.2)
        self.assertEqual(val["reward_score"], 0.8)
        self.assertTrue(manager.overlong_enabled)

    async def test_grpo_and_opsd_rewards_are_unchanged_when_length_shaping_is_disabled(self):
        result = await self.manager(config=self.config(enabled=False)).run_single(self.batch(1024))
        self.assertEqual(result["reward_score"], 0.8)
        self.assertEqual(result["reward_extra_info"]["overlong_reward"], 0)

    async def test_shared_reward_keeps_format_and_repetition_at_the_generation_limit(self):
        manager = self.manager(config=self.config(enabled=False), scorer=compute_score)
        repeated = "looping answer token sequence " * 10
        cases = (
            ("<answer>D</answer>", 1, 1.0, 0),
            ("<answer>A</answer>", 0, 0.0, 0),
            ("D", 1, 0.8, 0),
            ("A", 0, -0.2, 0),
            (repeated + "<answer>D</answer>", 1, 0.0, 1),
            (repeated + "D", 1, -0.2, 1),
        )
        response = MagicMock()
        response.__enter__.return_value = response
        environment = {
            "GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "0",
            "GROOVE_REPETITION_ZERO_REWARD": "true", "GROOVE_REPETITION_MIN_REPEATS": "4",
            "GROOVE_REPETITION_MIN_TOTAL_CHARACTERS": "80",
        }
        with patch.dict("os.environ", environment), \
             patch("groove.semantic_reward.urllib.request.urlopen", return_value=response), \
             patch("groove.semantic_reward.json.load") as judge:
            for length in (128, 896, 1024):
                for output, accuracy, expected_score, repetition in cases:
                    with self.subTest(length=length, output=output):
                        manager.tokenizer.decode.return_value = output
                        judge.return_value = {"choices": [{"message": {"content": str(accuracy)}}]}
                        result = await manager.run_single(self.batch(length))
                        self.assertAlmostEqual(result["reward_score"], expected_score)
                        extra = result["reward_extra_info"]
                        self.assertEqual(extra["accuracy"], accuracy)
                        self.assertEqual(extra["severe_repetition"], repetition)
                        self.assertEqual(extra["overlong_reward"], 0)
                        self.assertEqual(extra["training_reward"], result["reward_score"])

    async def test_invalid_length_settings_are_rejected(self):
        for config in (self.config(buffer=0), self.config(buffer=1025), self.config(penalty=-1), self.config(penalty=float("nan"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.manager(config=config)


class RewardDiagnosticsTest(unittest.TestCase):
    def test_judge_exhaustion_counts_ignore_padding_and_absent_judge_fields(self):
        fields = [
            {"reward_extra_info": {"judge_retries_exhausted": 1., "judge_attempts": 6.}},
            {"reward_extra_info": {"judge_retries_exhausted": 0., "judge_attempts": 6.}},
            {"reward_extra_info": {"judge_retries_exhausted": 1., "judge_attempts": 100.}},
            {"reward_extra_info": {"accuracy": 1.}},
        ]
        metrics = reward_extra_metrics(fields, [True, True, False, True])
        self.assertEqual(metrics["reward/judge_retries_exhausted_count"], 1.)
        self.assertEqual(metrics["reward/judge_retries_exhausted_fraction"], 0.5)
        self.assertEqual(metrics["reward/judge_attempts_mean"], 6.)
        self.assertEqual(metrics["reward/judge_attempts_max"], 6.)
        self.assertEqual(reward_extra_metrics(fields, [False] * 4), {})

    def test_native_tensordict_field_access_preserves_per_row_reward_objects(self):
        from verl.utils.tensordict_utils import get_tensordict

        fields = [{"reward_extra_info": {"accuracy": 1.0}}, {"reward_extra_info": {"accuracy": 0.0}}]
        data = get_tensordict({"tokens": torch.tensor([[1], [2]]), "extra_fields": fields})
        # TensorDict indexing may unwrap stacks into LinkedList; pop/get retain
        # the NonTensorStack API used by the native trainer's logging path.
        restored = data.pop("extra_fields").tolist()
        self.assertEqual(restored, fields)
        self.assertEqual(reward_extra_metrics(restored, [True, True])["reward/accuracy_mean"], 0.5)

    def test_diagnostics_remain_aligned_and_padding_does_not_bias_averages(self):
        fields = [
            {"reward_extra_info": {"accuracy": 1.0, "answer_reward": 1.0, "format_valid": True}},
            {"reward_extra_info": {"accuracy": 0.0, "answer_reward": 0.0}},
            {"reward_extra_info": {"accuracy": 99.0, "answer_reward": 99.0}},
        ]
        columns = reward_columns(fields, [1, 0])
        self.assertEqual(columns["accuracy"], [0.0, 1.0])
        self.assertEqual(columns["format_valid"], [None, True])
        metrics = reward_extra_metrics(fields, [True, True, False])
        self.assertEqual(metrics["reward/answer_reward_mean"], 0.5)
        self.assertEqual(metrics["reward/accuracy_mean"], 0.5)
        self.assertEqual(metrics["reward/format_valid_mean"], 1.0)
        with self.assertRaises(ValueError):
            reward_extra_metrics(fields, [True])


@unittest.skipUnless(find_spec("transfer_queue"), "Install native-training extra to exercise native replay buffer")
class NativeDynamicSamplingTest(unittest.TestCase):
    def test_native_rollout_dump_preserves_reward_components_and_optimized_score(self):
        from types import SimpleNamespace
        from verl.trainer.ppo.v1 import trainer_base as module
        from verl.trainer.ppo.v1 import PPOTrainerSync

        def column(values):
            return SimpleNamespace(tolist=lambda: values, to_padded_tensor=lambda **kwargs: torch.tensor(values))

        data = {
            "uid": column(["b", "a"]), "prompts": column([[1], [2]]),
            "responses": column([[3], [4]]), "rm_scores": torch.tensor([[0.25], [0.75]]),
            "reward_model": column([{"ground_truth": "B"}, {"ground_truth": "A"}]),
            "extra_fields": column([
                {"reward_extra_info": {"accuracy": 0.0, "score": 0.5}},
                {"reward_extra_info": {"accuracy": 1.0, "score": 1.0}},
            ]),
        }
        trainer = PPOTrainerSync.__new__(PPOTrainerSync)
        trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: str(ids.tolist()))
        trainer._dump_generations = Mock()
        batch = SimpleNamespace(keys=["b_0_0", "a_0_0"], partition_id="train")
        with patch.object(module.tq, "kv_batch_get", return_value=data):
            trainer._log_rollout_data(batch, {}, "/unused-test-output")
        dumped = trainer._dump_generations.call_args.kwargs
        self.assertEqual(dumped["scores"], [0.75, 0.25])
        self.assertEqual(dumped["reward_extra_infos_dict"]["accuracy"], [1.0, 0.0])
        self.assertEqual(dumped["reward_extra_infos_dict"]["reward_function_score"], [1.0, 0.5])
        self.assertNotIn("score", dumped["reward_extra_infos_dict"])

    def test_native_sampler_filters_uniform_groups_refills_and_keeps_exact_batch(self):
        from verl.trainer.ppo.v1 import replay_buffer as module

        metadata = {"train": {}}
        values = {}

        def add_group(uid, rewards):
            metadata["train"][uid] = {"is_prompt": True, "status": "finished", "global_steps": 0}
            for i, reward in enumerate(rewards):
                key = f"{uid}_{i}_0"
                metadata["train"][key] = {"min_global_steps": 0, "max_global_steps": 0, "is_padding": False}
                values[key] = {"reward_extra_info": {"training_reward": reward}}

        add_group("uniform", [1.0, 1.0])
        add_group("mixed", [0.0, 1.0])

        def clear(*, keys, partition_id, **kwargs):
            for key in keys:
                metadata[partition_id].pop(key, None)
                values.pop(key, None)

        def refill(count):
            for i in range(count):
                add_group(f"refill{i}", [0.0, 0.8])

        callback = Mock(side_effect=refill)
        sampler = module.ReplayBuffer(
            trainer_mode="sync", trainer_config=OmegaConf.create({}),
            max_off_policy_threshold=1, max_off_policy_strategy="drop", sampler_kwargs=OmegaConf.create({}),
            refill_fn=callback, filter_groups_metric="training_reward",
            train_batch_size=2, gen_batch_size=1, poll_interval=0,
        )
        with patch.object(module.tq, "kv_list", side_effect=lambda: metadata), \
             patch.object(module.tq, "kv_batch_get", side_effect=lambda *, keys, **kwargs: {"extra_fields": np.array([values[k] for k in keys], dtype=object)}), \
             patch.object(module.tq, "kv_clear", side_effect=clear), \
             patch.object(sampler, "_materialize_batch", side_effect=lambda partition, uids, snapshot: uids):
            selected, metrics = sampler.sample(global_steps=0, partition_id="train", batch_size=2)
        self.assertEqual(len(selected), 2)
        self.assertNotIn("uniform", selected)
        callback.assert_called_once_with(2)
        self.assertEqual(metrics["training/filter_groups/evicted_samples"], 1)
        self.assertEqual(metrics["training/filter_groups/discarded_surplus_samples"], 1)
        self.assertEqual(sampler._dapo_filtered_keys("val"), (set(), {}))


if __name__ == "__main__":
    unittest.main()
