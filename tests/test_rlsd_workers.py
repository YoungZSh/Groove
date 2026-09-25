from contextlib import contextmanager
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from groove.rlsd_teacher import FrozenTeacher
from groove.rlsd_workers import RLSDActorRolloutRefWorker
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker


class RLSDWorkerTest(unittest.TestCase):
    def worker(self):
        worker = RLSDActorRolloutRefWorker.__new__(RLSDActorRolloutRefWorker)
        worker._rank, worker._world_size = 0, 1
        model = torch.nn.Linear(1, 1, bias=False)
        events = []

        @contextmanager
        def eval_mode():
            events.append("load")
            try:
                yield
            finally:
                events.append("offload")

        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=model, eval_mode=eval_mode))
        worker.rlsd_teacher = FrozenTeacher()
        return worker, model, events

    def test_scoring_dispatch_restores_actor_before_offload(self):
        worker, model, events = self.worker()
        worker.sync_rlsd_teacher(0, 10)
        original = model.weight.detach().clone()
        with torch.no_grad(): model.weight.add_(2.)
        current = model.weight.detach().clone()

        def score(_worker, data):
            torch.testing.assert_close(model.weight, original)
            self.assertTrue(tu.get(data, "disable_auto_offload"))
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("forward failure")

        data = tu.get_tensordict({}, {"rlsd_teacher": True})
        with patch.object(ActorRolloutRefWorker, "compute_log_prob", score):
            with self.assertRaisesRegex(RuntimeError, "forward failure"):
                worker.compute_log_prob(data)
        torch.testing.assert_close(model.weight, current)
        self.assertEqual(events, ["load", "offload", "load", "offload"])

    def test_actor_checkpoint_saves_and_loads_teacher_sidecar(self):
        worker, model, _ = self.worker()
        worker.sync_rlsd_teacher(0, 10)
        original = model.weight.detach().clone()
        with TemporaryDirectory() as folder:
            with patch.object(ActorRolloutRefWorker, "save_checkpoint") as save:
                worker.save_checkpoint(folder, global_step=5)
                save.assert_called_once_with(folder, None, 5, None)
            worker.release_rlsd_teacher()
            with torch.no_grad(): model.weight.add_(3.)
            with patch.object(ActorRolloutRefWorker, "load_checkpoint"):
                worker.load_checkpoint(folder)
            self.assertEqual(worker.rlsd_teacher.step, 0)
            with worker.rlsd_teacher.apply(model):
                torch.testing.assert_close(model.weight, original)


if __name__ == "__main__":
    unittest.main()
