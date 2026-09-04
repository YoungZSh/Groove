from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class GRPOOnlyActorTest(unittest.TestCase):
    def test_null_self_distillation_runs_only_the_student_forward(self):
        config = AttrDict(
            policy_loss={"loss_mode": "vanilla"},
            self_distillation=None,
            use_dynamic_bsz=False,
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            calculate_entropy=False,
            entropy_coeff=0.0,
            loss_agg_mode="token-mean",
            use_kl_loss=False,
            global_batch_info={},
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = config
        actor.use_prefix_grouper = False
        actor.use_ulysses_sp = False
        actor.ulysses_sequence_parallel_size = 1
        actor.use_dynamic_bsz = False
        actor.actor_optimizer = SimpleNamespace(zero_grad=lambda: None)
        actor.scaler = None
        actor.actor_module = SimpleNamespace(train=lambda: None)

        forward_calls = []

        def forward(*_args, **kwargs):
            forward_calls.append(kwargs)
            return {"log_probs": torch.zeros((2, 2), requires_grad=True)}

        actor._forward_micro_batch = forward
        actor._optimizer_step = lambda: torch.tensor(1.0)
        actor._update_teacher = lambda: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not update a Teacher")
        )

        data = DataProto.from_dict(
            tensors={
                "responses": torch.ones((2, 2), dtype=torch.long),
                "response_mask": torch.ones((2, 2)),
                "input_ids": torch.ones((2, 2), dtype=torch.long),
                "attention_mask": torch.ones((2, 2)),
                "position_ids": torch.ones((2, 2), dtype=torch.long),
                "old_log_probs": torch.zeros((2, 2)),
                "advantages": torch.ones((2, 2)),
            },
            non_tensors={},
        )
        data.meta_info = {"temperature": 1.0, "pad_token_id": 0}

        with patch(
            "verl.workers.actor.dp_actor.get_device_id",
            return_value=torch.device("cpu"),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(len(forward_calls), 1)
        self.assertFalse(any("teacher" in key or "opd" in key for key in metrics))


if __name__ == "__main__":
    unittest.main()
