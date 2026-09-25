"""Project-only worker extension for periodically frozen RLSD teachers."""

from pathlib import Path

import torch

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker

from .rlsd_teacher import FrozenTeacher


class RLSDActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlsd_teacher = FrozenTeacher()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_rlsd_teacher(self, completed_steps: int, interval: int):
        expected = completed_steps // interval * interval
        if self.rlsd_teacher.state is not None and self.rlsd_teacher.step == expected:
            return
        with self.actor.engine.eval_mode():
            self.rlsd_teacher.sync(self.actor.engine.module, completed_steps, interval)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rlsd_teacher(self):
        self.rlsd_teacher = FrozenTeacher()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data):
        if not tu.pop(data, "rlsd_teacher", default=False):
            return super().compute_log_prob(data)
        # The inner infer context must keep parameters resident until Student
        # restoration; the outer context owns the normal offload lifecycle.
        tu.assign_non_tensor(data, disable_auto_offload=True)
        with self.actor.engine.eval_mode():
            with self.rlsd_teacher.apply(self.actor.engine.module):
                return super().compute_log_prob(data)

    def _teacher_path(self, local_path):
        return Path(local_path) / f"rlsd_teacher_world_size_{self.world_size}_rank_{self.rank}.pt"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        path = self._teacher_path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(self.rlsd_teacher.state_dict(), temporary)
        temporary.replace(path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        path = self._teacher_path(local_path)
        self.rlsd_teacher = FrozenTeacher()
        if path.exists():
            # Trusted local training checkpoints include PyTorch sharded tensors.
            self.rlsd_teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
