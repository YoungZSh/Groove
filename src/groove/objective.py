"""Configuration contract for joint outcome and visual-evidence learning."""

from __future__ import annotations

import math

from .rlsd import rlsd_lambda


def validate_objective_config(config) -> None:
    groove = config.get("groove", {}) or {}
    if not groove.get("enabled", False):
        return
    mode = groove.get("advantage_mode", "opsd")
    if mode not in {"opsd", "rlsd_positive"}:
        raise ValueError("groove.advantage_mode must be opsd or rlsd_positive")
    if mode == "rlsd_positive":
        rlsd_lambda(0, initial=float(groove.get("rlsd_lambda_initial", 0.5)),
                    decay_steps=groove.get("rlsd_lambda_decay_steps", 50))
        weight_clip = float(groove.get("rlsd_clip_range", 0.2))
        if not math.isfinite(weight_clip) or not 0 < weight_clip < 1:
            raise ValueError("groove.rlsd_clip_range must be finite and in (0, 1)")
        interval = groove.get("rlsd_teacher_sync_interval", 10)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("groove.rlsd_teacher_sync_interval must be a positive integer")
        actor = (config.get("actor_rollout_ref", {}) or {}).get("actor", {}) or {}
        if actor.get("strategy", "fsdp") not in {"fsdp", "fsdp2"}:
            raise ValueError("Frozen RLSD Teacher currently supports fsdp and fsdp2 only")
        if actor.get("checkpoint", {}).get("async_save", False):
            raise ValueError("Frozen RLSD Teacher requires synchronous checkpoints")
        if (config.get("trainer", {}) or {}).get("default_hdfs_dir") is not None:
            raise ValueError("Frozen RLSD Teacher checkpoints currently require local/shared storage")
    coef = float(groove.get("opsd_advantage_coef", 0.01))
    clip = groove.get("opsd_advantage_clip")
    if not math.isfinite(coef) or coef < 0:
        raise ValueError("groove.opsd_advantage_coef must be finite and non-negative")
    if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0):
        raise ValueError("groove.opsd_advantage_clip must be finite and positive when set")

    algorithm = config.get("algorithm", {}) or {}
    if algorithm.get("adv_estimator", "grpo") != "grpo":
        raise ValueError("Visual evidence advantages require algorithm.adv_estimator=grpo")
    if not algorithm.get("norm_adv_by_std_in_grpo", True):
        raise ValueError("Visual evidence training requires group-standardized GRPO advantages")
    if algorithm.get("use_kl_in_reward", False):
        raise ValueError("Visual evidence training keeps terminal rewards unchanged; use reference KL in the loss")
    correction = algorithm.get("rollout_correction", {}) or {}
    if correction.get("bypass_mode", False):
        raise ValueError("Visual evidence scoring requires recomputed pre-update actor log probabilities")
    actor = (config.get("actor_rollout_ref", {}) or {}).get("actor", {}) or {}
    if (actor.get("policy_loss", {}) or {}).get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Visual evidence advantages require one vanilla PPO policy loss")
    if (config.get("distillation", {}) or {}).get("enabled", False):
        raise ValueError("Visual evidence training uses its own sampled-token targets; disable the separate distillation loss")
