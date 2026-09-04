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


class SignedOPSDActorTest(unittest.TestCase):
    def test_groove_combines_signed_opsd_with_grpo_before_ppo(self):
        self_distillation = AttrDict(
            teacher_regularization="ema",
            teacher_model_source="current",
            full_logit_distillation=False,
            distillation_topk=None,
            log_prob_dump_dir=None,
            opsd_advantage_coef=1.0,
            opsd_advantage_clip=None,
        )
        config = AttrDict(
            policy_loss={"loss_mode": "groove"},
            self_distillation=self_distillation,
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
            clip_ratio_c=3.0,
        )
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = config
        actor.use_prefix_grouper = False
        actor.use_ulysses_sp = False
        actor.ulysses_sequence_parallel_size = 1
        actor.use_dynamic_bsz = False
        actor.use_fused_kernels = False
        actor.actor_optimizer = SimpleNamespace(zero_grad=lambda: None)
        actor.scaler = None
        actor.actor_module = SimpleNamespace(train=lambda: None)
        actor.teacher_module = None
        actor._update_teacher = lambda: None

        student_log_probs = torch.zeros((2, 1), requires_grad=True)
        teacher_log_probs = torch.tensor([[1.0], [-1.0]])
        forward_count = 0

        def forward(*_args, **_kwargs):
            nonlocal forward_count
            forward_count += 1
            if forward_count == 1:
                return {"log_probs": student_log_probs}
            return {"log_probs": teacher_log_probs}

        captured_gradient = None

        def optimizer_step():
            nonlocal captured_gradient
            captured_gradient = student_log_probs.grad.detach().clone()
            return torch.tensor(1.0)

        actor._forward_micro_batch = forward
        actor._optimizer_step = optimizer_step

        data = DataProto.from_dict(
            tensors={
                "responses": torch.ones((2, 1), dtype=torch.long),
                "response_mask": torch.ones((2, 1)),
                "input_ids": torch.ones((2, 1), dtype=torch.long),
                "attention_mask": torch.ones((2, 1)),
                "position_ids": torch.ones((2, 1), dtype=torch.long),
                "old_log_probs": torch.zeros((2, 1)),
                "advantages": torch.zeros((2, 1)),
                "teacher_input_ids": torch.ones((2, 1), dtype=torch.long),
                "teacher_attention_mask": torch.ones((2, 1)),
                "teacher_position_ids": torch.ones((2, 1), dtype=torch.long),
                "teacher_response_start_idx": torch.zeros(2, dtype=torch.long),
                "self_distillation_mask": torch.ones(2),
            },
            non_tensors={},
        )
        data.meta_info = {"temperature": 1.0, "pad_token_id": 0}

        with patch(
            "verl.workers.actor.dp_actor.get_device_id",
            return_value=torch.device("cpu"),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(forward_count, 2)
        self.assertTrue(torch.allclose(captured_gradient, torch.tensor([[-0.5], [0.5]])))
        self.assertIn("actor/groove_opsd_positive_token_fraction", metrics)
        self.assertIn("actor/groove_opsd_negative_token_fraction", metrics)
        self.assertNotIn("actor/groove_opd_loss", metrics)


if __name__ == "__main__":
    unittest.main()
