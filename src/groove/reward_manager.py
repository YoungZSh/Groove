"""Visual-QA reward adapter shared by native VERL and GRPO + OPSD."""

from __future__ import annotations

import math

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class VisualQARewardManager(NaiveRewardManager):
    """Apply optional DAPO length shaping to training, preserving raw accuracy.

    ``training_reward`` is the final scalar actually optimized, so native VERL's
    group filter sees format, repetition and length shaping consistently.
    Benchmark validation always retains the custom scorer's original reward.
    """

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score, **kwargs)
        reward_kwargs = config.reward.get("reward_kwargs", {}) or {}
        buffer = reward_kwargs.get("overlong_buffer_cfg", {}) or {}
        self.overlong_enabled = bool(buffer.get("enable", False))
        self.max_response_length = int(reward_kwargs.get("max_resp_len", 1024))
        self.buffer_length = int(buffer.get("len", 128))
        self.penalty_factor = float(buffer.get("penalty_factor", 1.0))
        if self.overlong_enabled:
            if not 0 < self.buffer_length <= self.max_response_length:
                raise ValueError("DAPO buffer must be positive and no longer than the response limit")
            if not math.isfinite(self.penalty_factor) or self.penalty_factor < 0:
                raise ValueError("DAPO penalty factor must be finite and non-negative")

    async def run_single(self, data):
        result = await super().run_single(data)
        item = data[-1:][0]
        extra = item.non_tensor_batch.get("extra_info", {}) or {}
        is_validation = (
            item.non_tensor_batch.get("data_source") == "vstar_bench"
            or extra.get("split") in {"validation", "val", "test"}
            or bool(data.meta_info.get("validate", False))
        )
        penalty = 0.0
        if self.overlong_enabled and not is_validation:
            capacity = item.batch["responses"].shape[-1]
            length = int(item.batch["attention_mask"][-capacity:].sum()) if capacity else 0
            threshold = self.max_response_length - self.buffer_length
            penalty = min(-(length - threshold) / self.buffer_length * self.penalty_factor, 0.0)
            result["reward_score"] = float(result["reward_score"]) + penalty
        result["reward_extra_info"]["training_reward"] = float(result["reward_score"])
        result["reward_extra_info"]["overlong_reward"] = penalty
        result["reward_extra_info"]["overlong"] = float(penalty < 0)
        return result
