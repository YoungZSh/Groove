"""Choose native VERL for baseline algorithms and the OPSD extension for joint runs."""

from __future__ import annotations

from importlib.util import find_spec


def trainer_backend(config) -> str:
    """Validate the requested path before any Ray workers are started."""
    opsd = bool((config.get("groove", {}) or {}).get("enabled", False))
    v1 = bool(config.trainer.get("use_v1", False))
    filtering = bool((config.algorithm.get("filter_groups", {}) or {}).get("enable", False))
    if opsd:
        if v1 or filtering:
            raise ValueError(
                "GRPO + OPSD uses the project trainer with trainer.use_v1=false and group filtering disabled; "
                "uniform-reward groups may carry OPSD signal"
            )
        return "groove_opsd"
    if filtering and not v1:
        raise ValueError("Dynamic group filtering requires trainer.use_v1=true; the legacy loop ignores it")
    if v1:
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("These single-node visual-QA experiments require trainer.v1.trainer_mode=sync")
        if find_spec("transfer_queue") is None:
            raise RuntimeError("Native VERL V1 requires TransferQueue; install the project's native-training extra")
        return "verl_v1_sync"
    return "verl_legacy"


def task_runner_class(backend: str):
    if backend == "verl_v1_sync":
        from verl.trainer.main_ppo import TaskRunnerV1

        return TaskRunnerV1
    if backend == "verl_legacy":
        from verl.trainer.main_ppo_v0 import TaskRunner

        return TaskRunner
    if backend == "groove_opsd":
        import ray
        from groove.verl_entrypoint import GrooveTaskRunner

        return ray.remote(num_cpus=1)(GrooveTaskRunner)
    raise ValueError(f"Unknown training backend: {backend}")
