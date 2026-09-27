from __future__ import annotations

from io import BytesIO
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from groove.rlsd_teacher import FrozenTeacher


def _check_sharded_teacher(rank, folder):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{folder}/init", timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = torch.nn.Linear(2, 4, bias=False)
        model.weight = torch.nn.Parameter(distribute_tensor(torch.arange(8.).reshape(4, 2), mesh, [Shard(0)]))
        original = model.weight.to_local().detach().clone()
        teacher = FrozenTeacher()
        teacher.sync(model, 0, 10)
        path = Path(folder) / f"teacher-{rank}.pt"
        torch.save(teacher.state_dict(), path)
        restored = FrozenTeacher()
        restored.load_state_dict(torch.load(path, weights_only=False))
        with torch.no_grad():
            model.weight.add_(3.)
        with restored.apply(model):
            torch.testing.assert_close(model.weight.to_local(), original)
        torch.testing.assert_close(model.weight.to_local(), original + 3.)
        assert restored.step == 0
    finally:
        dist.destroy_process_group()


class FrozenTeacherTest(unittest.TestCase):
    def test_five_step_schedule_keeps_initial_teacher_through_step_five(self):
        model = torch.nn.Linear(1, 1, bias=False)
        teacher = FrozenTeacher()
        teacher.sync(model, 0, 5)
        initial = model.weight.detach().clone()
        for step in range(1, 11):
            teacher.sync(model, step - 1, 5)
            expected = initial if step <= 5 else after_five
            with teacher.apply(model):
                torch.testing.assert_close(model.weight, expected)
            with torch.no_grad():
                model.weight.add_(1.)
            if step == 5:
                after_five = model.weight.detach().clone()
            if step % 5 == 0:
                teacher.sync(model, step, 5)
        self.assertEqual(teacher.step, 10)

    def test_two_rank_cpu_shards_roundtrip_and_restore_independently(self):
        with TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(_check_sharded_teacher, args=(folder,), nprocs=2, join=True)

    def test_freezes_between_syncs_restores_student_and_preserves_optimizer(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("counter", torch.tensor(1.))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        model(torch.ones(2)).sum().backward()
        optimizer.step()
        teacher = FrozenTeacher()
        self.assertTrue(teacher.sync(model, 0, 10))
        original = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(2.)
            model.counter.add_(1.)
        self.assertFalse(teacher.sync(model, 9, 10))
        current = {k: v.clone() for k, v in model.state_dict().items()}
        parameter_ids = [id(p) for p in model.parameters()]
        optimizer_steps = [s["step"].clone() for s in optimizer.state.values()]
        with self.assertRaisesRegex(RuntimeError, "scoring failed"):
            with teacher.apply(model):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[key])
                raise RuntimeError("scoring failed")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, current[key])
        self.assertTrue(model.training)
        self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])
        for actual, expected in zip(optimizer.state.values(), optimizer_steps):
            torch.testing.assert_close(actual["step"], expected)
        self.assertTrue(teacher.sync(model, 10, 10))
        with teacher.apply(model):
            torch.testing.assert_close(model.weight, current["weight"])

    def test_checkpoint_roundtrip_retains_frozen_version(self):
        model = torch.nn.Linear(1, 1)
        teacher = FrozenTeacher()
        teacher.sync(model, 10, 10)
        original = model.weight.detach().clone()
        stream = BytesIO()
        torch.save(teacher.state_dict(), stream)
        stream.seek(0)
        restored = FrozenTeacher()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad(): model.weight.add_(3.)
        self.assertFalse(restored.sync(model, 15, 10))
        with restored.apply(model):
            torch.testing.assert_close(model.weight, original)
        with self.assertRaisesRegex(RuntimeError, "Missing RLSD Teacher"):
            FrozenTeacher().sync(model, 15, 10)


if __name__ == "__main__":
    unittest.main()
