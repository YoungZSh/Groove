"""Keep one independent, validation-selected snapshot alongside rolling checkpoints."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile


logger = logging.getLogger(__name__)


def validate_best_checkpoint_config(config) -> None:
    settings = config.trainer.get("best_checkpoint", {}) or {}
    if not settings.get("enabled", False):
        return
    if not settings.get("metric") or settings.get("mode", "max") not in {"min", "max"}:
        raise ValueError("best_checkpoint requires a metric and mode=max or min")
    if config.trainer.get("test_freq", -1) <= 0 and not config.trainer.get("val_before_train", False):
        raise ValueError("best_checkpoint requires validation to be enabled")
    for worker in (config.actor_rollout_ref.actor, config.get("critic", {})):
        if (worker.get("checkpoint", {}) or {}).get("async_save", False):
            raise ValueError("best_checkpoint requires synchronous checkpoint saving (async_save=false)")
    if config.trainer.get("use_v1", False) and config.trainer.v1.trainer_mode != "sync":
        raise ValueError("best_checkpoint currently supports synchronous training only")


class BestCheckpointTracker:
    """Publish metadata only after copying a complete checkpoint; ties keep the old best.

    Files are copied, not symlinked/hardlinked to the rolling checkpoint: both
    retention deletion and a later overwrite of that source must be harmless.
    The persisted metadata also restores the comparison threshold on resume.
    """

    def __init__(self, root: str | Path, metric: str, mode: str = "max"):
        self.root = Path(root) / "best_checkpoint"
        self.metric = metric
        self.mode = mode
        self.metadata_path = self.root / "metadata.json"
        self.best = None
        if self.metadata_path.exists():
            self.best = json.loads(self.metadata_path.read_text())
            if self.best["metric"] != metric or self.best["mode"] != mode:
                raise ValueError("Existing best checkpoint uses a different metric or comparison mode")
            if self.best["path"] != f"global_step_{int(self.best['step'])}":
                raise ValueError("Invalid best checkpoint path in metadata")
            self._validate_snapshot(self.root / self.best["path"])

    @staticmethod
    def _validate_snapshot(path: Path) -> None:
        if not (path / "data.pt").is_file() or not (path / "actor").is_dir():
            raise RuntimeError(f"Incomplete checkpoint for best snapshot: {path}")
        if not any(p.is_file() for p in (path / "actor").rglob("*")):
            raise RuntimeError(f"Empty actor checkpoint for best snapshot: {path}")

    def consider(self, metrics: dict, step: int, source: Path, save_current) -> dict:
        if self.metric not in metrics:
            raise ValueError(f"Best-checkpoint metric {self.metric!r} is missing from validation results")
        score = float(metrics[self.metric])
        if not math.isfinite(score):
            logger.warning("Skipping non-finite best-checkpoint metric %s=%s", self.metric, score)
            return self.metrics()
        if self.best is not None:
            better = score > self.best["value"] if self.mode == "max" else score < self.best["value"]
            # Revalidating the same weights on resume must not replace their snapshot.
            if not better or step == self.best["step"]:
                return self.metrics()

        save_current()
        self._validate_snapshot(source)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"global_step_{step}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing best snapshot: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        metadata_tmp = staging.with_suffix(".json")
        published_snapshot = False
        record = {"metric": self.metric, "mode": self.mode, "value": score, "step": step,
                  "path": destination.name}
        try:
            shutil.copytree(source, staging, dirs_exist_ok=True)
            self._validate_snapshot(staging)
            staging.rename(destination)
            published_snapshot = True
            with metadata_tmp.open("x") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(metadata_tmp, self.metadata_path)
        except Exception:
            shutil.rmtree(destination if published_snapshot else staging, ignore_errors=True)
            metadata_tmp.unlink(missing_ok=True)
            raise

        previous = self.best
        self.best = record
        if previous is not None:
            try:
                shutil.rmtree(self.root / previous["path"])
            except OSError:
                logger.warning("Could not remove superseded best snapshot", exc_info=True)
        logger.info("Best checkpoint: step=%s %s=%s path=%s", step, self.metric, score, destination)
        return self.metrics()

    def metrics(self) -> dict:
        if self.best is None:
            return {}
        return {"checkpoint/best_step": self.best["step"], "checkpoint/best_score": self.best["value"]}


def maybe_save_best_checkpoint(trainer, metrics: dict) -> dict:
    """Shared validation hook for native V1 and the legacy/OPSD trainer."""
    settings = trainer.config.trainer.get("best_checkpoint", {}) or {}
    if not settings.get("enabled", False):
        return {}
    if not hasattr(trainer, "_best_checkpoint_tracker"):
        validate_best_checkpoint_config(trainer.config)
        trainer._best_checkpoint_tracker = BestCheckpointTracker(
            trainer.config.trainer.default_local_dir, settings["metric"], settings.get("mode", "max")
        )

    def save_current():
        if getattr(trainer, "_last_saved_checkpoint_step", None) == trainer.global_steps:
            return
        # An initial/off-cadence validation may improve without a scheduled save.
        # Release rollout memory for serialization, then restore the rollout weights.
        trainer.checkpoint_manager.sleep_replicas()
        try:
            trainer._save_checkpoint()
        finally:
            trainer.checkpoint_manager.update_weights(trainer.global_steps)

    source = Path(trainer.config.trainer.default_local_dir) / f"global_step_{trainer.global_steps}"
    return trainer._best_checkpoint_tracker.consider(metrics, trainer.global_steps, source, save_current)
