"""Hydra entrypoint that swaps verl's trainer for :class:`GrooveRayPPOTrainer`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ray
import verl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from .verl_trainer import GrooveRayPPOTrainer


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def configure_ray_memory_guard(config) -> dict[str, float | int]:
    """Configure Ray's node-wide OOM guard before ``ray.init`` starts raylet.

    Ray's memory monitor is a soft, node-wide guard.  The enclosing cgroup is
    still the hard limit; the guard deliberately fires earlier so the kernel
    does not choose an arbitrary process to OOM-kill.
    """
    from ray._private.utils import get_system_memory

    gib = 1024**3
    total_bytes = int(get_system_memory())
    cap_gib = _env_float("RAY_NODE_MEMORY_CAP_GIB", 220.0)
    headroom_gib = _env_float("RAY_MEMORY_GUARD_HEADROOM_GIB", 4.0)
    threshold_ceiling = _env_float("RAY_MEMORY_USAGE_THRESHOLD_CEILING", 0.95)
    object_store_gib = _env_float("RAY_OBJECT_STORE_GIB", 8.0)
    refresh_ms = int(_env_float("RAY_MEMORY_MONITOR_REFRESH_MS", 100.0))

    if cap_gib <= 0:
        raise ValueError("RAY_NODE_MEMORY_CAP_GIB must be positive")
    if headroom_gib < 0 or headroom_gib >= cap_gib:
        raise ValueError(
            "RAY_MEMORY_GUARD_HEADROOM_GIB must be non-negative and smaller than the cap"
        )
    if not 0 < threshold_ceiling < 1:
        raise ValueError("RAY_MEMORY_USAGE_THRESHOLD_CEILING must be between 0 and 1")
    if object_store_gib <= 0:
        raise ValueError("RAY_OBJECT_STORE_GIB must be positive")
    if refresh_ms <= 0:
        raise ValueError("RAY_MEMORY_MONITOR_REFRESH_MS must be positive")

    requested_cap_bytes = int(cap_gib * gib)
    guarded_cap_bytes = min(requested_cap_bytes, total_bytes)
    trigger_bytes = guarded_cap_bytes - int(headroom_gib * gib)
    if trigger_bytes <= 0:
        raise ValueError("Ray memory guard trigger must be positive")
    threshold = min(threshold_ceiling, trigger_bytes / total_bytes)
    object_store_bytes = int(object_store_gib * gib)
    if object_store_bytes >= trigger_bytes:
        raise ValueError("Ray object store must be smaller than the memory guard trigger")

    # These names are intentionally mixed-case: they are Ray's documented
    # raylet environment variables, not project-local settings.
    os.environ["RAY_memory_usage_threshold"] = f"{threshold:.9f}"
    os.environ["RAY_memory_monitor_refresh_ms"] = str(refresh_ms)
    OmegaConf.update(
        config,
        "ray_kwargs.ray_init.object_store_memory",
        object_store_bytes,
        force_add=True,
    )
    OmegaConf.update(
        config,
        "ray_kwargs.ray_init.include_dashboard",
        False,
        force_add=True,
    )

    return {
        "ray_total_bytes": total_bytes,
        "requested_cap_bytes": requested_cap_bytes,
        "trigger_bytes": int(threshold * total_bytes),
        "threshold": threshold,
        "headroom_bytes": int(headroom_gib * gib),
        "object_store_bytes": object_store_bytes,
        "refresh_ms": refresh_ms,
    }


def validate_full_time_sharing(config) -> int:
    """Require the memory-release path promised by the two-GPU launcher."""
    from verl.third_party.vllm import VLLM_SLEEP_LEVEL

    if not _env_bool("GROOVE_REQUIRE_SLEEP_LEVEL_2", True):
        return int(VLLM_SLEEP_LEVEL)

    actor_fsdp_config = config.actor_rollout_ref.actor.fsdp_config
    actor_uses_fsdp2_cpu_offload = (
        str(config.actor_rollout_ref.actor.strategy) == "fsdp2"
        and bool(actor_fsdp_config.offload_policy)
    )
    required = {
        "hybrid engine": config.actor_rollout_ref.hybrid_engine,
        # FSDP1 uses the explicit inter-phase offload helpers.  FSDP2's native
        # CPUOffloadPolicy owns parameters, gradients, and optimizer state, so
        # the two legacy booleans must not be required in that configuration.
        "actor parameter offload": actor_uses_fsdp2_cpu_offload or actor_fsdp_config.param_offload,
        "actor optimizer offload": actor_uses_fsdp2_cpu_offload or actor_fsdp_config.optimizer_offload,
        "reference parameter offload": config.actor_rollout_ref.ref.fsdp_config.param_offload,
        "vLLM free-cache engine": config.actor_rollout_ref.rollout.free_cache_engine,
        "vLLM sleep mode": config.actor_rollout_ref.rollout.enable_sleep_mode,
        "vLLM eager mode": config.actor_rollout_ref.rollout.enforce_eager,
    }
    disabled = [name for name, enabled in required.items() if not bool(enabled)]
    if disabled:
        raise ValueError(
            "Full GPU time-sharing requires these settings: " + ", ".join(disabled)
        )
    if bool(config.actor_rollout_ref.rollout.layered_summon):
        raise ValueError("layered_summon forces vLLM sleep level 1; disable it for full release")
    if int(VLLM_SLEEP_LEVEL) != 2:
        raise RuntimeError(
            f"Full GPU time-sharing requires vLLM sleep level 2, got {VLLM_SLEEP_LEVEL}"
        )
    return int(VLLM_SLEEP_LEVEL)


class GrooveTaskRunner(TaskRunner):
    def run(self, config):
        # TaskRunner resolves this module global only inside the Ray driver process.
        import verl.trainer.main_ppo as main_ppo

        main_ppo.RayPPOTrainer = GrooveRayPPOTrainer
        return super().run(config)


def main() -> None:
    # ``verl`` is vendored in this project's ``src`` tree.  Resolve the
    # configuration relative to the imported package so the entrypoint works
    # both from a checkout and from an installed wheel, without a sibling
    # upstream runtime or an external patch step.
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"Bundled verl config was not found at {config_dir}"
        )
    config_name = os.environ.get("VERL_CONFIG_NAME", "groove")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=sys.argv[1:])
    memory_guard = configure_ray_memory_guard(config)
    sleep_level = validate_full_time_sharing(config)
    auto_set_device(config)
    if os.environ.get("GROOVE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}:
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        reward_kwargs = config.custom_reward_function.get("reward_kwargs", {})
        optimizer_overrides = config.actor_rollout_ref.actor.optim.get(
            "override_optimizer_config", {}
        ) or {}
        policy_loss_mode = str(config.actor_rollout_ref.actor.policy_loss.loss_mode)
        opsd_enabled = policy_loss_mode in {"vopd", "groove"}
        self_distillation_cfg = config.actor_rollout_ref.actor.get("self_distillation") or {}
        print(
            "groove config valid:",
            policy_loss_mode,
            config.actor_rollout_ref.rollout.n,
            config.trainer.n_gpus_per_node,
            f"opsd_enabled={opsd_enabled}",
            f"opsd_advantage_coef={self_distillation_cfg.get('opsd_advantage_coef')}",
            f"opsd_advantage_clip={self_distillation_cfg.get('opsd_advantage_clip')}",
            f"vllm_sleep_level={sleep_level}",
            "full_time_sharing=enabled",
            f"optimizer={config.actor_rollout_ref.actor.optim.optimizer_impl}."
            f"{config.actor_rollout_ref.actor.optim.optimizer}",
            f"fused_adamw={optimizer_overrides.get('fused', False)}",
            f"train_batch_size={config.data.train_batch_size}",
            f"ppo_mini_batch_size={config.actor_rollout_ref.actor.ppo_mini_batch_size}",
            f"ray_total_gib={memory_guard['ray_total_bytes'] / 1024**3:.3f}",
            f"ray_guard_gib={memory_guard['trigger_bytes'] / 1024**3:.3f}",
            f"ray_threshold={memory_guard['threshold']:.6f}",
            f"ray_object_store_gib={memory_guard['object_store_bytes'] / 1024**3:.3f}",
            f"total_epochs={config.trainer.total_epochs}",
            f"total_training_steps={config.trainer.total_training_steps}",
            f"train_files={list(config.data.train_files)}",
            f"val_files={list(config.data.val_files)}",
            f"reward_manager={config.reward_manager.name}",
            f"reward_async={config.reward_model.launch_reward_fn_async}",
            f"reward_fn={config.custom_reward_function.path}:"
            f"{config.custom_reward_function.name}",
            f"answer_reward_weight={reward_kwargs.get('answer_reward_weight')}",
            f"format_reward_weight={reward_kwargs.get('format_reward_weight')}",
        )
        return
    runner = ray.remote(num_cpus=1)(GrooveTaskRunner)
    run_ppo(config, task_runner_class=runner)


if __name__ == "__main__":
    main()
